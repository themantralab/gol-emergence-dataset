# Stage 1 — Data Pipeline

## Overview

**Goal**: generate, simulate, and save the N_SEEDS-seed training dataset with
10-signal behavioral signatures and an initial reference library.
N_SEEDS is a CLI parameter (`--n-seeds`, default 1,000,000; production 1,000,000).

**Production target (1.5M seeds)**: pool_size = 1.2× = 1.8M seeds.
Key RAM peaks (mmap-based generation + in-place normalization):
- Pool generation: workers write directly to mmap files on disk; per-worker peak
  = 1 batch of trajectories ≈ BATCH_SIZE × (T+1) × 128² ≈ 840 MB × n_workers.
  No concatenation step — workers write to non-overlapping mmap slices.
- Stratification: pool mmap is read on demand; selected signatures_raw (15.4 GB)
  loaded into RAM. Pool mmap files (~22 GB on disk) deleted after stratification.
- Normalization: in-place — overwrites signatures_raw buffer, no second copy.
  Classification runs on raw values BEFORE the in-place overwrite.
- FFT: chunked 100k seeds at a time; peak ~1.5 GB extra on top of the
  pre-allocated sig_reference (7.7 GB).
- True system peak: ~25–27 GB during signatures_norm + sig_reference overlap.
- `signatures_raw.npy` is NOT saved — it is normalized in-place and saved once
  as `signatures_norm.npy`. Stage 2 does not need the raw values.

**Inputs**: none — generates all data from scratch using NumPy.

**Outputs**:
- `data/` — 13 `.npy` files consumed by all later stages
- `data/diagnostics/` — 7 diagnostic outputs for pipeline verification

**Files** (write in this order):
1. `simulator.py` — standalone GoL engine; also imported by Stage 4
2. `generate_data.py` — all remaining Stage 1 logic (seeds, signals, normalization, clustering, saving)

**Dependency**: nothing. Must complete before Stage 2 begins.

---

## Data Pipeline Design

### Simulation setup

- **Seed size**: 16×16 binary cells
- **Grid size**: 128×128 (seed embedded centered, surrounding cells dead)
- **Embedding offset**: row 56, col 56 (`SEED_OFFSET = (128-16)//2 = 56`)
- **Timesteps**: T = 256 generations per seed
- **Boundary condition**: fixed-zero (non-toroidal). Cells outside the 128×128
  grid are treated as permanently dead. Implemented by zeroing border neighbor
  counts after np.roll operations.

### Seed generation

Seeds are generated using **4 density bands**, each contributing 25% of the pool.
Workers rotate through bands sequentially (batch 0 → band 0, batch 1 → band 1, …):

| Band | Range        | Behavioral emphasis                          |
|------|-------------|----------------------------------------------|
| 0    | [0.03, 0.08] | Sparse — gliders, single-cell oscillators    |
| 1    | [0.08, 0.15] | Low-density still lifes and small oscillators |
| 2    | [0.15, 0.22] | Medium density — balanced dynamics           |
| 3    | [0.22, 0.30] | Denser patterns — complex oscillators        |

The ceiling of 0.30 avoids overcrowding (> 0.30 collapses to chaotic soup;
all false-positive gliders in the prior run came from 38–45% density seeds).
The floor of 0.03 avoids trivially empty seeds.

Each seed is independently randomly generated using numpy default_rng.

### Lifespan stratification

Pure random seeds at ~35% density produce ~65-70% long-lived patterns and
very few dying/short patterns, but the distribution within long-lived is
heavily biased toward simple still lifes and period-2 oscillators.
Stratification by lifespan bucket with oversampling ensures the model sees
a representative range of behavioral complexity.

**Lifespan definition**: index of the last timestep at which any cell changed
state (XOR of consecutive grids). Patterns still changing at step T get
lifespan = T. Patterns that never change after t=0 get lifespan = 0.
Using cell-state XOR (not population change) ensures constant-population
oscillators like blinkers land in the correct bucket rather than bucket 0.

**Bucket definitions** (for T=256):

| Bucket  | Range        | Description              | Target share |
|---------|-------------|--------------------------|--------------|
| dying   | [0, 20)     | Collapses quickly         | 25% (N/4)   |
| short   | [20, 60)    | Brief activity            | 25% (N/4)   |
| medium  | [60, 130)   | Moderate duration         | 25% (N/4)   |
| long    | [130, T]    | Long-lived, interesting   | 25% (N/4)   |

Stratification uses **equal bucket allocation** — exactly N/4 seeds per bucket,
sampling with replacement when a bucket is underrepresented in the pool. This
replaces the prior weight-based scheme (dying=3.0×, …, long=1.0×), which
produced a 57% dying bias when the density range was lowered to [0.03, 0.30]
(sparse seeds die quickly, making dying the dominant pool bucket at the old weights).

Equal allocation guarantees the model sees equal behavioral coverage across all
lifespan strata and ensures `sig_reference` is balanced — no behavioral class
is spuriously flagged as novel solely due to underrepresentation.

### Training pairs

Each seed produces 256 consecutive (grid_t, grid_{t+1}) training pairs.
Step index t is stored alongside each pair so the model can distinguish
early-trajectory dynamics from late-trajectory attractor behavior.

### Dataset scale

- **Target**: N_SEEDS × T training pairs (default/production 1,000,000 × 256 = 256M)
- **Environment**: Python + NumPy, local CPU, 32GB RAM
- **Batch simulation**: seeds simulated in batches of 200 workers-side for memory efficiency

### Memory scaling — always streaming

Trajectories are discarded immediately after per-batch signature extraction.
Per-worker peak RAM: `BATCH_SIZE × (T+1) × 128 × 128 × 1 byte ≈ 850 MB`.

Pool size: N_SEEDS × 1.2 = 1,200,000 seeds.

FFT is chunked in 100,000-seed batches to bound peak memory during sig_reference
computation (each chunk: `100k × 257 × 10 × 4 bytes ≈ 1 GB` input + similar output).

`data_loader.py` (Stage 2) re-simulates on demand using `simulator.simulate()` to
generate (grid_t, grid_{t+1}) training pairs at training time — no trajectory files
are written to disk.

---

## 10-Signal Behavioral Trajectory Signature

This section is the **canonical definition** of the 10-signal signature.
Stage 2 trains on normalized signatures as identity targets. Stages 3 and 4
use the FFT-magnitude sig_reference for novelty scoring. All cross-stage use
loads the saved `.npy` files from `data/` — `sig_reference.npy`,
`sig_mean.npy`, `sig_std.npy` — which are produced here.

### Motivation

The behavioral signature serves as the ground truth target for the identity
objective. It must be:
- Computable from simulation alone (fully unsupervised)
- Discriminating across all major GoL behavioral classes
- Temporally structured (full trajectory, not a collapsed mean)
- Position-invariant (two identical patterns at different grid positions give
  the same signature)
- Scale-normalized across the dataset

### Signal definitions

All signals computed at each timestep t = 0..T:

```
sig[t] = [P(t), Δcx(t), Δcy(t), V(t), E(t), N_cc(t), S_lag_2(t), S_lag_4(t), S_lag_8(t), S_lag_16(t)]
```

**P(t)** — Population: number of alive cells at step t.
- Discriminates: dying patterns decay to 0; still lifes and oscillators are
  flat or periodic; gliders are approximately constant.

**Δcx(t)** = cx(t) - cx(0) — Center-of-mass x displacement (column axis).
- Displacement removes absolute position dependence.
- Discriminates: gliders show steady linear drift; stationary patterns are ~0.

**Δcy(t)** = cy(t) - cy(0) — Center-of-mass y displacement (row axis).
- Together Δcx and Δcy encode glider direction and speed. Kept separate (not
  collapsed to magnitude) so the attractor head can represent directed motion.

**V(t)** — Spatial variance: mean squared distance of alive cells from
current center of mass.
- V(t) = Σ((row−cy)² + (col−cx)²) × alive / P(t)
- Discriminates: asymmetric oscillators show periodic V variation; still lifes
  are constant.

**E(t)** — Motion energy: number of cells that changed state since previous step.
- E(t) = |grid[t] XOR grid[t-1]|.sum() for t ≥ 1; E(0) = 0.
- Discriminates: still life = 0 after settling; oscillator = constant nonzero;
  glider = constant nonzero; dying = decreasing toward 0.

**N_cc(t)** — Connected component count (8-connectivity).
- Computed via scipy.ndimage.label with a 3×3 all-ones structure.
- Discriminates: single coherent patterns (still life, glider) have N_cc=1;
  oscillators may fragment; chaotic soups have high, variable N_cc.

**S_lag_L(t)** — Temporal self-similarity at lag L steps (four signals: L ∈ {2, 4, 8, 16}).
- S_lag_L(t) = (grid[t] AND grid[min(t+L, T)]).sum() / P(t)
- Each lag targets a different periodicity class:
  - **lag-2**: catches period-2 oscillators (blinker, toad) — the most common GoL oscillator
  - **lag-4**: critical glider discriminator — a glider at ¼-cell/step repeats every 4 steps;
    period-4 oscillators also score high, but N_cc and population-constancy separate them
  - **lag-8**: general self-similarity; period-8 structures and slow oscillators
  - **lag-16**: long-period structures (p14, p15 oscillators, slow methuselahs settling)
- Still lifes have S_lag ≈ 1 for all lags; chaos has S_lag ≈ 0 for all lags.
- Four lags provide richer temporal structure in FFT space than a single lag,
  improving LOF discrimination between similar pattern classes.

### Normalization

Per-signal mean and std computed across all N×(T+1) timesteps (flattened).

Before being used as training targets:
1. Compute per-signal mean and std across all N×(T+1) timesteps
2. Clip std floor to 1e-6 (handles exactly-constant signals)
3. Standardize: sig_norm = (sig − mean) / std
4. Clip normalized output to [−5, 5] to bound attractor head regression targets.

The mean and std vectors (shape: 10,) are saved as `sig_mean.npy` and
`sig_std.npy` and must be used consistently at inference time.

### sig_reference: FFT magnitude spectrum

`sig_reference.npy` is the novelty reference set for LOF scoring in Stages 3
and 4. It is derived from the normalized trajectories:

1. Apply 1D rfft to each of the 10 signal channels over the 257-timestep axis
2. Take magnitude → 129 frequency bins per channel
3. Concatenate → 1290-D vector per seed

Shape: (N, 1290) float32 — ~4.8 GB at 1M seeds.

**Why FFT magnitude?** It is phase-invariant: two blinkers at opposite phases
produce the same frequency spectrum and zero novelty distance. The four S_lag
channels dramatically improve frequency-domain class separation:
- Still life → DC spike in all 10 channels
- Period-2 oscillator → DC + Nyquist in E/N_cc/S_lag_2, regardless of phase
- Glider → DC in P/V/E, strong low-freq ramp in Δcx/Δcy, periodic in S_lag_4
- Dying → decaying DC, rest near zero
- Chaos → broad flat spectrum across all channels

**FFT is chunked** (100k seeds per chunk) to bound peak RAM to ~3 GB/chunk
before concatenating into the final (N, 1290) array.

### Behavioral fingerprints

| Class      | P(t)      | Δcx/Δcy       | V(t)      | E(t)             | N_cc(t)   | S_lag_2  | S_lag_4  | S_lag_8  | S_lag_16 |
|------------|-----------|---------------|-----------|------------------|-----------|----------|----------|----------|----------|
| Still life | flat      | flat (~0)     | flat      | 0 after settling | 1, stable | ~1       | ~1       | ~1       | ~1       |
| Oscillator | flat      | flat (~0)     | periodic  | constant nonzero | variable  | periodic | periodic | periodic | periodic |
| Glider     | flat      | linear drift  | mild osc  | constant nonzero | 1, stable | varies   | ~1 (p4) | periodic | periodic |
| Dying      | decays→0  | flattens      | shrinks→0 | decreasing → 0   | 0 at end  | drops    | drops    | drops    | drops    |

### Full signature shape

Each seed produces a signature matrix of shape (257, 10) for T=256.
Stored as (257, 10) — the trajectory head learns temporal patterns within the
signature rather than treating all timesteps independently.

---

## Implementation

### `simulator.py`

Core GoL simulation. No dependencies on other project files. Must be a
standalone module because `emergence/validity.py` imports it in Stage 4.

Functions to implement:
- `step(grid: np.ndarray) -> np.ndarray`
  - Input: (128, 128) uint8 grid
  - Apply B3/S23 rule with fixed-zero boundaries
  - Implementation: use `np.roll` to compute 8-neighbor counts, then zero the
    border row/column counts to enforce non-toroidal edges, then apply birth
    (count==3 & dead) and survival (count in {2,3} & alive) masks
  - Output: (128, 128) uint8 next state
- `simulate(grid, steps=256) -> np.ndarray`
  - Output: (steps+1, 128, 128) uint8 trajectory including t=0
- `step_batch(grids) -> np.ndarray`
  - Input: (N, H, W) uint8 — apply one B3/S23 step to all N grids simultaneously
  - Output: (N, H, W) uint8
  - Uses numpy batch operations (roll + masking over the N dimension) — much
    faster than calling `step()` in a Python loop; used internally by `simulate_batch`
- `simulate_batch(grids, steps=256) -> np.ndarray`
  - Input: (N, 128, 128) uint8
  - Output: (N, steps+1, 128, 128) uint8; uses `step_batch` for all N simultaneously
- `run_all_verifications()`
  - Embed canonical patterns and assert expected behavior:
    - Block (2×2 still life): population constant, grid unchanged after 1 step
    - Blinker (3-cell oscillator): period 2, population constant
    - Glider: population constant, center of mass drifts ~1 cell per 4 steps
    - `step_batch` output matches `step()` for block/blinker/glider
    - Birth rule: dead cell with exactly 3 alive neighbors becomes alive
    - Overcrowding: alive cell with 4+ neighbors dies
    - Fixed-zero boundary: cells at border never become alive
  - Print PASS/FAIL for each; raise AssertionError on any failure

Run `python simulator.py` to execute verifications before proceeding.

---

### `generate_data.py`

Single script containing all seed generation, signal extraction, normalization,
clustering, saving, and validation. Imports only `simulator.py` and standard
libraries. Nothing in Stages 2–4 imports this file — they load `.npy` files
directly.

#### Seed generation functions

- Seed generation and embedding are **inline in `_worker_generate`** — no standalone
  `make_seed` or `embed_seed` functions. Each mini-batch samples density from the
  current band, generates seeds with `rng.random < density`, and embeds them directly
  into zero (128×128) grids at `[56:72, 56:72]` before simulation.
- `measure_lifespans_batch(trajectories) -> np.ndarray`
  - Input: (N, T+1, 128, 128) uint8
  - Computes cell-state XOR step-by-step (`trajectories[:, t+1] != trajectories[:, t]`)
    to find the last timestep where any cell changed; returns (N,) int32 lifespans
  - Uses a loop over T steps (not N) to avoid materialising a (N, T, H, W) bool
    intermediate that would cost ~6.7 GB per worker at BATCH_SIZE=200
  - Cell-state XOR (not population) ensures constant-population oscillators like
    blinkers land in the correct bucket rather than bucket 0
- `assign_buckets(lifespans) -> np.ndarray`
  - 0 = dying [0, 20), 1 = short [20, 60), 2 = medium [60, 130),
    3 = long [130, T]; vectorised over (N,) input
- `_worker_generate(args) -> tuple`
  - Top-level function (required for ProcessPoolExecutor pickling)
  - args = `(n_seeds, rng_seed, T_steps, batch_size, log_file)`
  - `log_file`: if set, worker immediately redirects its own stdout/stderr to that
    file (append, line-buffered) — required because subprocess stdout is detached
    from the main process's sys.stdout reassignment
  - `DENSITY_BANDS` is a module-level constant, not passed as an argument
  - Generates seeds in mini-batches; rotates through density bands per batch;
    for each: simulate → extract sigs → measure lifespans → `del trajectories`
  - Returns (seeds (N,16,16), lifespans (N,), sigs (N,T+1,10)) — grids NOT returned
  - Per-worker peak RAM: BATCH_SIZE × (T+1) × 128 × 128 × 1 byte ≈ 850 MB
- `generate_pool_parallel(pool_size, T_steps, rng_seed, n_workers=8, log_file=None) -> tuple`
  - Splits pool_size evenly across n_workers ProcessPoolExecutor workers;
    each worker gets a unique rng_seed (rng_seed + worker_idx + 1)
  - Prints per-worker progress heartbeats (every **1,000** seeds) and completion with RSS
  - Collects results into indexed slots by worker index for deterministic concatenation
    order (reproducible across same --rng-seed); concatenation happens once after all
    workers complete
  - Returns (seeds, lifespans, signatures) concatenated across workers
  - Grids and trajectories never leave workers — only seeds + signatures returned
- `stratify_seeds(pool_seeds, pool_lifespans, pool_sigs, target_n, rng) -> tuple`
  - Equal bucket allocation: exactly target_n // 4 seeds per bucket
  - Sampling with replacement when a bucket has fewer than quota (warns to stdout)
  - Skips empty buckets gracefully (warns; actual N may be < target_n)
  - Returns (seeds, lifespans, signatures, buckets) for selected indices
- `bucket_report(buckets, lifespans, label="")`
  - Print per-bucket count, percentage, and lifespan range to stdout

#### Signal extraction functions

- `extract_signature(trajectory) -> np.ndarray`
  - Input: (T+1, 128, 128) uint8 trajectory
  - At each timestep t compute 10 signals:
    - P(t): count of alive cells
    - cx(t), cy(t): column and row center of mass weighted by alive cells
    - Δcx(t) = cx(t) - cx(0), Δcy(t) = cy(t) - cy(0)
    - V(t): mean squared distance of alive cells from current center of mass
    - E(t): cells that changed state since previous step (XOR count); 0 at t=0
    - N_cc(t): connected component count via scipy.ndimage.label (8-connectivity);
      **optimization**: if E(t)==0 (grid unchanged), N_cc(t) = N_cc(t-1), skipping label call
    - S_lag_L(t) for L in {2, 4, 8, 16}: (grid[t] AND grid[min(t+L,T)]).sum() / P(t)
    - If P(t)==0: all signals = 0.0 for that timestep
  - Output: (T+1, 10) float32
- `extract_signatures_batch(trajectories) -> np.ndarray`
  - Input: (N, T+1, 128, 128) uint8
  - Output: (N, T+1, 10) float32; loop over N
- `compute_normalization_stats(signatures) -> tuple`
  - Input: (N, T+1, 10) float32
  - Compute per-signal mean and std across all N×(T+1) timesteps flattened
  - Clip std floor to 1e-6
  - Return: (mean (10,), std (10,)) float32
- `normalize_signatures(signatures, sig_mean, sig_std) -> np.ndarray`
  - Standardize: (sig − mean) / std; clip output to [−5, 5]
  - **In-place ops** (`np.subtract/divide/clip` with `out=` parameter) to avoid
    chained float32 intermediates — each naive operator allocates a new (N, 257, 10)
    array (~10.3 GB at 1M seeds); three chained ops would peak at ~31 GB before
    the output is even assigned. In-place reduces peak to `signatures + out` = 20.6 GB.
  - Output same shape as input
- `compute_sig_reference(signatures_norm, chunk_size=100_000) -> np.ndarray`
  - Input: (N, T+1, 10) float32 normalized trajectories
  - Process in chunks of chunk_size to bound peak RAM
  - Apply rfft along time axis (axis=1) → 129 bins per channel → magnitude
  - Flatten → (chunk, 1290) float32; concatenate chunks → (N, 1290)
  - This is the phase-invariant novelty reference used by Stages 3 and 4
- `classify_trajectory(sig, T, tol=0.05) -> str`
  - Heuristic classifier for t-SNE visualization only — not used in training
  - Returns one of: 'dying', 'still_life', 'oscillator', 'glider'
  - Logic (on raw, unnormalized sig):
    - **Active window**: find first t where P(t) < 0.9 × P(0). Call that index
      `active_end` (or len(sig) if P never drops that far). All glider checks
      operate only on `sig[:active_end]`. Using < 90% of P(0) (not == 0) is
      necessary because GoL gliders in a fixed-zero 128×128 grid typically hit the
      boundary and shatter into lower-P debris rather than dying cleanly to P=0;
      the 90% threshold isolates the translating phase in both cases.
    - **glider**: all three of these over the active window:
        (a) `(P.max() − P.min()) / P.mean() < 0.10` — population nearly constant
            (a translating periodic pattern preserves cell count at every phase)
        (b) `|Δcx[active_end−1]| + |Δcy[active_end−1]| > 3.0` — net displacement
            at least 3 cells (rules out stationary patterns with zero drift)
        (c) `max(|corr(t, Δcx)|, |corr(t, Δcy)|) > 0.95` — displacement has
            strong linear correlation with time (rules out random-walk drift;
            a staircase glider at ¼ cell/step gives r ≈ 0.999 over any window
            ≥ 8 steps; active window must be ≥ 8 steps for this check to apply)
    - **dying**: P at final step < tol × P at step 0 (catches patterns that
      decayed but weren't classified as gliders above)
    - **oscillator**: `std(V[T//2:]) > tol OR mean(fdiff[T//2:]) > tol`
        - First branch catches asymmetric oscillators (V varies)
        - Second branch catches symmetric oscillators like blinker (V is
          rotation-invariant and constant, but fdiff is nonzero every step)
        - This resolves the previously accepted blinker=still_life limitation
    - **still_life**: fallthrough
- `classify_batch(signatures, T) -> list`
  - Return list of str labels for each signature

#### Main execution (`if __name__ == '__main__'`)

1. Parse CLI args:
   - `--n-seeds INT` (default 1,000,000)
   - `--rng-seed INT` (default 42) for reproducibility
   - `--workers INT` (default min(8, cpu_count()))
2. Create `data/` and `data/diagnostics/` directories if not present
3. pool_size = N_SEEDS × 1.2
4. `generate_pool_parallel(pool_size, T, rng_seed, n_workers)` — all 8 cores
   - Workers generate (density band rotation per batch), simulate, extract
     signatures, discard trajectories; grids never returned from workers
   - Returns (seeds, lifespans, signatures) for the full pool
   - Print per-worker progress heartbeats and completion with RSS
5. Stratify to N_SEEDS (equal N//4 per bucket); print bucket report before and after
   - Pool arrays deleted immediately after stratification to free RAM
6. Signatures for the stratified subset are indexed directly from pool_sigs —
   no re-simulation required
   - Output: signatures_raw (N, T+1, 10) float32
7. Reconstruct grids from seeds: embed each (16,16) seed into (128,128) grid
   - Save grids.npy immediately; `del grids` to free RAM before FFT
8. Compute normalization stats (mean + std); normalize signatures
   - Output: sig_mean (10,), sig_std (10,), signatures_norm (N, T+1, 10)
9. Classify all trajectories (heuristic, visualization only)
   - Output: labels (N,) as U20 string array; print class distribution
10. Compute chunked FFT sig_reference: normalize_signatures → rfft → magnitude → flatten
    - chunked 100k seeds at a time to bound peak RAM
    - Output: sig_reference (N, 1290) float32 — phase-invariant novelty basis
11. Cluster sig_reference (PCA 50D first, then Ward k=4 on 10k subsample):
    - Fit on subsample; compute silhouette score on subsample (printed inline and in report)
    - Assign all N points to nearest centroid via cdist
    - Cluster centroids stored as per-cluster medians
    - Clustering is visualization-only; never consumed by Stages 2–4
    - Output: cluster_labels (N,) int32, cluster_centroids (4, 1290) float32
12. Save all outputs to `data/`:
    - `seeds.npy`             — (N, 16, 16) uint8
    - `grids.npy`             — (N, 128, 128) uint8
    - `lifespans.npy`         — (N,) int32
    - `buckets.npy`           — (N,) int32
    - `signatures_raw.npy`    — (N, 257, 10) float32
    - `signatures_norm.npy`   — (N, 257, 10) float32
    - `sig_mean.npy`          — (10,) float32
    - `sig_std.npy`           — (10,) float32
    - `sig_reference.npy`     — (N, 1290) float32  (FFT magnitude; phase-invariant novelty basis)
    - `labels.npy`            — (N,) str (U20)
    - `cluster_centroids.npy` — (4, 1290) float32  (per-cluster medians; visualization only)
    - `cluster_labels.npy`    — (N,) int32
    - `n_seeds.npy`           — scalar int32 (N; loaded by downstream stages)
13. Generate diagnostic outputs to `data/diagnostics/`:

    **`lifespan_hist.png`** — two side-by-side histograms of lifespan
    distribution before and after stratification. Verifies bucket weighting
    took effect. Bins at bucket boundaries [0, 20, 60, 130, T].

    **`signal_distributions.png`** — for each of the 10 raw signals: violin
    plot of values across all seeds, split by heuristic class (dying /
    still_life / oscillator / glider). Verifies signals discriminate between
    classes. Should show clear separation especially for P (dying vs rest),
    Δcx/Δcy (glider vs rest), and S_lag_4 (glider peak vs oscillator periodic).

    **`signal_samples.png`** — for 3 randomly selected seeds per heuristic
    class (up to 12 seeds total): plot all 10 signals over 257 timesteps as
    separate subplots. Verifies signal extraction looks correct per class:
    - dying: P decays to 0, all S_lag channels drop
    - still_life: all signals flat, S_lag ≈ 1 throughout
    - oscillator: V and S_lag channels show periodicity
    - glider: Δcx or Δcy shows linear drift, S_lag_4 ≈ 1

    **`normalization_check.png`** — for each signal: overlaid density plots
    of raw vs normalized values. After normalization, mean should be near 0
    and bulk of mass between −2 and +2. Verifies mean+std scaling worked.

    **`cluster_summary.png`** — bar chart of cluster sizes with dominant
    behavioral class labeled per cluster. Verifies clusters align with known
    classes (each cluster should be dominated by one class).

    **`tsne_signatures.png`** — t-SNE of `sig_reference` (N, 1290) reduced via
    PCA(50) then t-SNE to 2D, colored by heuristic class label. Verifies
    behavioral classes are separable in FFT-fingerprint space before encoding.
    Expected: dying and glider clearly separated; still_life/oscillator overlap
    acceptable. Subsampled to 10,000 points via **proportional sampling** —
    each class receives slots proportional to its natural frequency in the
    dataset (e.g. at 0.76% gliders → ~76 of 10k points). This faithfully
    represents the true data distribution; legend counts show exact n per class.
    Rationale: equal-per-class sampling would misrepresent prevalence and could
    mislead Stage 3 LOF threshold calibration decisions. This is the Stage 1
    equivalent of the Stage 2 latent-space t-SNE.

    **`canonical_check.txt`** — embed block, blinker, glider as 16×16 seeds;
    simulate for T=256 steps; extract signatures; classify with heuristic
    classifier. Expected:
    - block → still_life
    - blinker → oscillator (fdiff=4 every step; distinguished from block)
    - glider → glider
    Print PASS if block and glider are in different classes; OK if blinker
    differs from block (expected with fdiff signal).

14. Print Stage 1 completion report to stdout:
    - N_SEEDS requested and N_actual produced
    - Grid configuration (T, GRID_SIZE, density bands, signals, lags)
    - sig_reference dimensions
    - Stratification: seeds per bucket (N_actual // 4)
    - Silhouette score from clustering subsample
    - Canonical pattern check verdict (PASS/FAIL)
    - Total data size on disk
    - Path to diagnostics directory
    - Total elapsed time

Run: `python generate_data.py [--n-seeds 1000000] [--rng-seed 42]`

---

## Completion Criteria

- [ ] `simulator.py` verifications all pass (`python simulator.py`)
- [ ] `generate_data.py` runs end-to-end without errors
- [ ] All N_SEEDS seeds and 13 dataset files saved to `data/`
- [ ] Normalization stats (mean + std) saved
- [ ] Cluster library initialized using agglomerative clustering (not KMeans);
      centroids saved as per-cluster medians
- [ ] `signal_samples.png` inspected: signal shapes match expected per-class
      behavior (dying decays, glider drifts, oscillator shows periodicity)
- [ ] `signal_distributions.png` inspected: P and Δcx/Δcy show clear
      class separation
- [ ] `normalization_check.png` inspected: normalized signals centered near 0
- [ ] `cluster_summary.png` inspected: each cluster dominated by one behavioral class
- [ ] `canonical_check.txt` shows block=still_life, blinker=oscillator, glider=glider
- [ ] t-SNE plot shows dying and glider visually separated (oscillator/
      still_life overlap is acceptable due to blinker edge case)
