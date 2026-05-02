# GoL Emergence Discovery System — Overarching Design

This document covers project-wide goals, architecture, design rationale,
and constraints. Stage-specific design details live in each stage's own
design document — this file is intentionally free of implementation specifics.

---

## 1. Project Goal

Build a two-model system that:
1. Learns a structured latent representation of Conway's Game of Life dynamics
2. Uses a conditioned latent diffusion model to sample novel regions of that
   latent space
3. Filters samples through an emergence pipeline to discover GoL configurations
   that are both mechanically valid (obey B3/S23) and behaviorally novel (do
   not match any known behavioral class in the training distribution)

The core research hypothesis: **a learned non-Gaussian sampler operating on a
structured, decorrelated latent space will discover emergent GoL structures more
efficiently than a Gaussian-prior VAE**, because GoL's interesting configurations
occupy thin, curved, non-Gaussian manifolds in configuration space that a
Gaussian prior systematically smooths over.

### System-wide no-Gaussian constraint

The system avoids Gaussian priors, assumptions, and structures in all **core
components**. This constraint is absolute for the three components where a
Gaussian assumption would directly bias which configurations are discovered:

1. **Latent space regularization** — no KL divergence, no unit Gaussian prior,
   no reparameterization trick (VICReg replaces all of these)
2. **Novelty scoring** — no centroid distance, no Gaussian cluster shape assumed;
   k-NN Euclidean distance against raw data points only
3. **Novelty scoring and emergence gate** — k-NN Euclidean distance against
   raw sig_reference data points; no cluster shape or distribution assumed

In **peripheral areas** (training utilities, visualization clustering,
normalization), Gaussian-adjacent methods are permitted when strictly necessary
or abundantly convenient, provided an explicit justification appears in the
relevant stage design document.

**Accepted Gaussian-adjacent elements and their justifications:**

| Element | Stage | Justification |
|---------|-------|---------------|
| Gaussian noise in diffusion chain | 3 | Computational mechanism for training stability only; not a structural assumption about latent space |
| Ward linkage clustering | 1 | Minimizes within-cluster variance — a second-moment criterion, not a distributional assumption; chaining-resistant; visualization only |
| BatchNorm in encoder | 2 | Per-batch normalization for training stability; does not impose a Gaussian prior on learned representations |
| z_prior N(μ, diag(σ²)) for diffusion start | 3 | Eliminates inference-time z cloud dependency; valid approximation because VICReg already pushes z toward decorrelated near-unit-variance dimensions |

Any method not in the above table that assumes spherical clusters, unit Gaussian
priors, or normally distributed signal statistics in a core component violates
this constraint and must not be used without an explicit justification added to
the table above.

---

## 2. System Architecture Overview

The system has four sequential stages, each fully completed before the next begins:

```
[Stage 1 — Data Pipeline]        simulator.py, generate_data.py
           ↓
[Stage 2 — Core World Model]     model/, data_loader.py, train_core.py
           ↓  (frozen z cloud)
[Stage 3 — Latent Diffusion]     sampler/, train_sampler.py
           ↓  (z̃ samples)
[Stage 4 — Emergence Pipeline]   emergence/
```

### Why sequential and frozen

The core model (Stage 2) is frozen before the sampler (Stage 3) sees any data.
This means:
- The sampler cannot corrupt the world model's learned dynamics
- The latent space geometry is fixed and stable before the sampler learns it
- The two training objectives never interfere with each other

Stage 4 is inference-only — no training occurs; both frozen models are consumed.

### Key shared artifact: z_cloud.npy

`data/z_cloud.npy` is a (N_SEEDS, 128) matrix of encoded latent vectors,
one per training seed. It is produced at the start of Stage 3 by passing all
training grids through the frozen Stage 2 encoder. LOF density scoring against
this matrix — via the saved `checkpoints/lof_pipeline.pkl` — is the sole
novelty mechanism across all stages. No cluster centroids, no Gaussian distance
assumptions.

The novelty chain from raw simulation to novelty signal:
```
simulate → grid → encoder → z ∈ ℝ¹²⁸
                                 ↓
               LOF (local density vs neighbours in z space)
                                 ↓
                     scalar novelty score
```

`data/sig_reference.npy` is a (N_SEEDS, 1290) matrix of FFT magnitude
fingerprints produced in Stage 1. It is retained as a behavioural archive
for analysis and visualisation but is not part of the novelty pipeline.

Why z-cloud LOF over FFT-fingerprint LOF: the Stage 2 encoder is trained
on trajectory dynamics with a temporal contrastive loss that explicitly
separates behavioural classes. Gliders produce a drifting latent trajectory;
oscillators produce a closed loop; still lifes converge to a fixed point.
These geometric differences are directly exploited by LOF in z space. FFT
fingerprints in contrast compress everything through a population-level PC1
that conflates gliders and oscillators (Stage 1 finding: gliders scored as
LOF outliers at only 5.6% vs the 5% baseline).

N_SEEDS is set via `--n-seeds` CLI arg in generate_data.py (default 10,000;
production 1,500,000). All downstream stages read N from `data/n_seeds.npy` and
require no CLI args to adapt to different dataset sizes.

---

## 3. Project Folder Structure

All code lives at the project root. Design documents also live at the project root.

```
gol/
├── overarching_design.md       — this file
├── s1_design.md                — Data Pipeline: simulator + generate_data
├── s2_design.md                — Core World Model: encoder/transition/decoder/trajectory_head
├── s3_design.md                — Latent Diffusion Sampler: denoiser + CFG
├── s4_design.md                — Emergence Pipeline: validity/novelty/gate/run
│
├── simulator.py                — [Stage 1] standalone GoL engine; also used by Stage 4
├── generate_data.py            — [Stage 1] full data pipeline
├── data_loader.py              — [Stage 2] PyTorch Dataset/DataLoader
├── train_core.py               — [Stage 2] 3-phase training loop
├── train_sampler.py            — [Stage 3] denoiser training loop
│
├── model/                      — [Stage 2] encoder, decoder, transition, attractor, losses
├── sampler/                    — [Stage 3] schedule, novelty, denoiser, diffusion
├── emergence/                  — [Stage 4] validity, classifier, gate, run
│
├── data/                       — [Stage 1 output] all .npy training files
│   ├── diagnostics/            — [Stage 1] validation plots and canonical check
│   ├── z_cloud.npy             — [Stage 3] (N,128) encoded training z vectors
│   ├── novelty_scores.npy      — [Stage 3] (N,) pre-computed per-z novelty scores
│   ├── z_prior.npy             — [Stage 3] (2,128) diagonal Gaussian fit to z cloud
│   └── sig_reference.npy       — [Stages 1→4] normalized; grows as discoveries are logged
│
├── checkpoints/                — [Stage 2+3] model weight snapshots
└── discoveries/                — [Stage 4] PNG grids + log.jsonl
```

**Decision**: all design docs and code files live at the project root with
standard Python import paths so `from model.encoder import Encoder` and
`import simulator` work without path manipulation.

---

## 4. Stage Summaries

| Stage | Goal | Primary files | Inputs | Outputs |
|-------|------|---------------|--------|---------|
| 1 | Generate N_SEEDS dataset with 10-signal signatures (default/production 1M) | `simulator.py`, `generate_data.py` | none | `data/*.npy`, diagnostics |
| 2 | Train encoder / transition / decoder / trajectory head via progressive rollout curriculum | `model/`, `train_core.py` | `data/` | `checkpoints/` |
| 3 | Train novelty-conditioned denoiser on frozen z cloud | `sampler/`, `train_sampler.py` | `checkpoints/`, `sig_reference.npy` | `z_cloud.npy`, `novelty_scores.npy`, `z_prior.npy`, denoiser checkpoint |
| 4 | Batch inference → filter → log discoveries | `emergence/` | both frozen models | `discoveries/`, updated `sig_reference.npy` |

Full specifications — architecture, function signatures, completion criteria —
are in each stage's design document.

---

## 5. Environment and Dependencies

- Python 3.12.3
- NumPy 2.4.3 (available)
- scikit-learn (available) — AgglomerativeClustering, TSNE, silhouette_score
- matplotlib (available) — validation plots and diagnostics
- PyTorch (available, CPU only) — used from Stage 2 onward
- hdbscan — NOT available → use AgglomerativeClustering (Ward linkage)
- umap-learn — NOT available → use sklearn TSNE

**No GPU.** All training is on CPU. This affects batch sizes and training time
but not architecture or correctness of any component.

---

## 6. Key Design Decisions and Rationale

### Why no Gaussian prior?

Gaussian priors force the latent space into a unit Gaussian shape regardless
of what the data actually looks like. GoL behavioral classes occupy thin,
curved, non-Gaussian manifolds — a glider lives on a helix (limit cycle +
drift), an oscillator on a closed loop, a still life at a point attractor.
Forcing these onto a Gaussian prior destroys the geometric structure that
makes the latent space useful for sampling. VICReg enforces decorrelation
and prevents collapse without imposing any distributional shape.

### Why conditioned diffusion over blind diffusion?

Blind diffusion will oversample high-density regions of the z cloud — the
common patterns like simple still lifes and blinkers — and undersample the
sparse regions where interesting and rare structures live. The novelty
conditioning signal directly addresses this: it steers the reverse diffusion
process toward underexplored regions.

### Why short diffusion chain (T≈100) with a fitted z_prior?

The chain is short to avoid collapsing the starting distribution to pure N(0,I)
noise, which would force the model to learn a Gaussian approximation of the
true z distribution as a side effect of training. Instead, inference starts
from N(μ, diag(σ²)) fitted to the empirical z cloud — a lightweight two-row
file (`data/z_prior.npy`) that captures the actual per-dimension mean and
spread without requiring the full z cloud at inference time. This is a
deliberate peripheral Gaussian approximation: VICReg already drives the z cloud
toward decorrelated near-unit-variance dimensions, so the diagonal Gaussian fit
is a close approximation of the true marginal distribution. The Gaussian noise
within the chain is a computational mechanism (training stability), not a
structural statement about the latent space geometry.

### Why f_θ operates in latent space rather than pixel space?

Predicting next GoL states in pixel space directly would make f_θ responsible
for learning the encoder+rule+decoder simultaneously. In latent space, f_θ
only needs to learn the rule — the encoder and decoder handle the rest.
This separation of concerns makes each component learnable independently and
allows the contrastive and identity losses to operate directly on z vectors
rather than having to backpropagate through pixel-space reconstruction.

### Why separate training phases?

Mechanics must be accurate before identity training begins — the attractor head
trains on rollouts of f_θ, so if f_θ is wrong the identity signal is garbage.
The contrastive loss requires stable rollouts to produce meaningful triplets.
The phased curriculum enforces correct dependency ordering and prevents each
training objective from corrupting the others.

### Why 250 timesteps instead of 100?

250 timesteps gives:
- More room for slow methuselahs and complex patterns to develop
- Richer 10-signal signature curves with more temporal structure
- Better separation of behavioral classes that look similar at step 100
  (some period-N oscillators need N steps to complete a cycle)
- Longer lifespan to detect truly persistent structures

The cost is larger dataset files and slower signature extraction — manageable
on 32GB RAM with batch-wise processing.

### Why embed in 128×128 instead of simulating in 16×16?

Patterns need room to grow and travel. A glider moves ~1 cell per 4 steps;
over 250 steps it travels ~60 cells. A 16×16 grid would cause it to hit the
boundary immediately. The 128×128 grid gives patterns space to evolve naturally.
The 24-cell border around the 16×16 seed acts as a buffer.

### Why LOF on z cloud instead of FFT fingerprints?

The FFT fingerprint approach (computing `rfft` of the (257, 10) trajectory →
(1290,) vector → PCA(50) → LOF) was the original design. A sanity check at
the end of Stage 1 showed it failed at the primary goal: gliders scored as
outliers at only 5.6%, indistinguishable from the 5% inlier baseline. PC1 of
PCA(50) on FFT fingerprints captured population level ("alive vs dead"), causing
gliders and oscillators to collapse into the same density region.

Z-cloud LOF was chosen as the replacement because the Stage 2 encoder is
explicitly trained, via temporal contrastive loss, to produce geometrically
distinct trajectories for each behavioral class (glider → helix, oscillator →
closed loop, still life → fixed point). LOF in that space correctly separates
them.

**Known tradeoffs of z-cloud LOF (accepted):**

| Tradeoff | Mitigation |
|----------|------------|
| Entirely dependent on Stage 2 encoder quality — if encoder fails to separate classes, novelty signal fails | Stage 2 completion gate: t-SNE of z_cloud must show visual cluster separation before Stage 3 begins |
| "Novel in z space" may not equal "behaviorally novel" — encoder trained on training distribution may map unseen patterns near familiar ones | Temporal contrastive loss specifically trains for behavioral separation, not visual similarity |
| z̃ from denoiser is not encoded from a grid — off-manifold z̃ may score as novel without being valid GoL | GoL validity check (Check 1) filters off-manifold z̃ before LOF gate |
| Novelty scores unavailable until Stage 2 complete — FFT scores could have been computed at end of Stage 1 | Accepted; no mitigation needed |

### Why LOF over k-NN distance?

Raw k-NN distance conflates sparsity with novelty: a point in a sparse but
internally consistent cluster scores high purely because the cluster is sparse,
not because the point is unusual relative to its neighbours. LOF compares each
point's local density to its neighbours' local densities — a point in a
consistently sparse neighbourhood scores ~1 (not novel), while a genuinely
isolated point scores high. This is the correct signal for discovery.

### Resolved — glider/oscillator separation

A LOF sanity check on `sig_reference` at the end of Stage 1 revealed that
gliders score as outliers at only 5.6% — indistinguishable from the 5% inlier
baseline. PC1 of the PCA reduction (70.8% of variance) captures population
level rather than behavioural dynamics, causing gliders and oscillators to
occupy the same density region in FFT space.

**Resolution**: the novelty pipeline was redesigned to use LOF on the Stage 2
z cloud rather than FFT fingerprints. The encoder's temporal contrastive loss
explicitly trains it to separate behavioural classes in latent space, so gliders
and oscillators will occupy distinct z-cloud regions. LOF in z space will
reliably score novel spaceships as outliers. No changes to Stage 1 or the
dataset are required.

---

## 7. Glossary

| Term | Definition |
|------|-----------|
| z | Latent vector, ∈ ℝ¹²⁸, output of encoder |
| z cloud | Full set of z vectors produced by encoding training dataset |
| z̃ | Novel z vector produced by diffusion sampler |
| f_θ | Latent transition function MLP, z_t → z_{t+1} |
| T | Number of simulation timesteps (256 for simulation; 100 for diffusion chain) |
| 10-signal | [P, Δcx, Δcy, V, E, N_cc, S_lag_2, S_lag_4, S_lag_8, S_lag_16] trajectory signature at each timestep |
| trajectory head | Per-timestep MLP, z_t → ℝ¹⁰; predicts normalized 10-signal values at each rollout step |
| sig_reference | (N, 1290) FFT magnitude spectra of normalized (257,10) trajectories; behavioral archive produced in Stage 1, not used in novelty pipeline |
| novelty score | LOF score of a query z vector against z_cloud in ℝ¹²⁸; higher = more unusual relative to neighbours |
| GoL validity | Whether decode(z̃) → simulate 1 step ≈ decode(f_θ(z̃)) |
| behavioral class | dying / still_life / oscillator / glider / other |
| emergence gate | novelty score > threshold → confirmed discovery |
| VICReg | Variance-Invariance-Covariance Regularization (no Gaussian prior) |
| CFG | Classifier-free guidance — inference-time conditioning scale |
| lambda_cfg | CFG guidance scale; swept over {1.0, 2.0, 3.0, 5.0, 7.5} |
| lifespan | Last step index where population changed; T if never stabilized |
| bucket | Lifespan stratum: dying [0,20) / short [20,60) / medium [60,130) / long [130,T] |
| B3/S23 | GoL rule: Born with 3 neighbors, Survives with 2 or 3 |
| Ward linkage | Agglomerative clustering criterion used for visualization clustering |
