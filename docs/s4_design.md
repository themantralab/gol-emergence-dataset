# Stage 4 — Emergence Pipeline

## Overview

**Goal**: batch inference loop that samples novel GoL configurations, filters
them through validity and novelty checks, logs confirmed discoveries, and
grows the reference library so the novelty bar rises over time.

**Inputs**:
- Frozen Stage 2 model (encoder, transition, decoder, trajectory head)
- Frozen Stage 3 denoiser + chosen lambda_cfg
- `data/z_cloud.npy` — (N, 128) training z vectors (LOF reference set)
- `data/z_discovered.npy` — (M, 128) confirmed discovery z vectors (starts empty, grows here)
- `checkpoints/lof_pipeline.pkl` — fitted PCA + LOF pipeline on z_cloud from Stage 3
- `data/z_prior.npy` — (2, 128) inference starting distribution

**Outputs**:
- `discoveries/` — one PNG per confirmed emergent pattern
- `discoveries/log.jsonl` — z̃ vector, LOF score, batch index, trajectory head label per discovery
- Updated `data/z_discovered.npy` — appended after each confirmed discovery
- Updated `checkpoints/lof_pipeline.pkl` — re-fit on z_cloud + z_discovered every 100 discoveries

**Files** (write in this order):
1. `emergence/__init__.py`
2. `emergence/validity.py`
3. `emergence/classifier.py`
4. `emergence/gate.py`
5. `emergence/run.py`

**Dependencies**: Stage 2 and Stage 3 complete, both model sets frozen.
`emergence/validity.py` imports `simulator.step` directly from `simulator.py`.

---

## Emergence Pipeline Design

### Two-check filter

Two sequential checks. Earlier checks are cheaper and gate the later ones.

**Check 1 — GoL validity** (cheap, runs first):
- Decode z̃ → candidate grid
- Simulate one GoL step (`simulator.step`)
- Decode f_θ(z̃) → model's predicted next state
- If cell agreement < 90%: discard as off-manifold
- Cost: 1 decoder call + 1 f_θ call + 1 simulation step

**Check 2 — LOF emergence gate** (near-free, z̃ already available):
- PCA-transform z̃; call `lof.predict()` → +1 (inlier) or -1 (outlier)
- -1 means z̃ falls outside the boundary set by `contamination=0.05`
  at LOF fit time — more novel than 95% of the training z distribution
- No threshold constant in application code; boundary lives in `lof.offset_`
- Cost: one PCA transform + one LOF query (microseconds; z̃ is already computed)

**Post-gate logging** (confirmed discoveries only):
- Simulate 64 steps from decoded grid → save PNG montage to `discoveries/`
- Run trajectory head on rollout z vectors → behavioural label (glider / oscillator / etc.)
- Append z̃ to z_discovered; re-fit LOF every 100 discoveries
- Cost: 64 simulation steps + trajectory head inference (only for confirmed discoveries)

### Why contamination over a hardcoded threshold

`contamination=0.05` is set once at LOF fit time in Stage 3. Stage 4 calls
`lof.predict()` — a single boolean with no scalar constant in application code.
To adjust the bar, change `contamination` in Stage 3 and re-fit. The parameter
has a concrete meaning: *"the most novel 5% of the training z distribution
defines the discovery boundary."*

### Sample outcome taxonomy

| GoL valid | lof.predict()   | Outcome                                   |
|-----------|-----------------|-------------------------------------------|
| No        | —               | Discard (off-manifold)                    |
| Yes       | +1 (inlier)     | Log for coverage metrics only             |
| Yes       | -1 (outlier)    | EMERGENT DISCOVERY — log + update library |

### Reference library update and LOF re-fit

When a discovery is confirmed:
1. Append z̃ to `data/z_discovered.npy` (M+1, 128)
2. Keep the LOF pipeline in memory (stale but still valid between re-fits)
3. Every 100 discoveries: re-fit PCA + LOF (contamination=0.05) on
   z_cloud + z_discovered; save updated `checkpoints/lof_pipeline.pkl`

The denoiser does NOT need retraining — it learned to condition on a novelty
scalar. As z_discovered grows, confirmed discovery regions become denser and
subsequent variants score as inliers, naturally raising the effective bar.

### Operational loop

Run in batches of 1000 samples. Monitor per batch:
- GoL validity rate (should stay > 80% at chosen lambda_cfg)
- Mean LOF score of valid samples
- Discovery rate (will decline as library fills — expected and correct)
- Pairwise fingerprint diversity among discoveries

If discovery rate drops to zero:
- Validity still high → library saturated (normal stopping condition)
- Validity also dropped → denoiser drifting off-manifold; retrain Stage 3

---

## Implementation

### `emergence/__init__.py`
Empty init.

### `emergence/validity.py`
GoL validity check — Check 1.
- `check_validity(z_tilde, decoder, transition, threshold=0.90) -> bool`
  - Decode z̃ → candidate grid; simulate one GoL step; decode f_θ(z̃)
  - Return cell agreement >= threshold
- `check_validity_batch(z_batch, decoder, transition, threshold=0.90) -> np.ndarray`
  - Return (N,) bool

### `emergence/classifier.py`
Behavioural labelling via trajectory head — post-gate logging only.
- `label_discovery(z_tilde, transition, trajectory_head, steps=64) -> str`
  - Unroll f_θ from z̃ for `steps` steps; apply trajectory_head at each step
  - Average predicted 10-signal over last 32 steps; classify as in Stage 1 heuristic
  - Return label string: 'dying' / 'still_life' / 'oscillator' / 'glider' / 'other'
  - Called only for confirmed discoveries — not in the gate

### `emergence/gate.py`
LOF emergence gate — Check 2.
- `load_pipeline(checkpoint_dir) -> dict`
  - Load `checkpoints/lof_pipeline.pkl`; return {'pca': PCA, 'lof': LOF}
- `predict_novelty(z_tilde, pipeline) -> bool`
  - z_tilde: (128,) — PCA transform then lof.predict(); return True if -1 (outlier)
- `predict_novelty_batch(z_batch, pipeline) -> np.ndarray`
  - z_batch: (N, 128); return (N,) bool
- `update_library(z_tilde, z_discovered, pipeline, checkpoint_dir,
                  z_cloud, discovery_count) -> tuple`
  - Append z̃ to z_discovered → (M+1, 128)
  - If discovery_count % 100 == 0: re-fit PCA+LOF on z_cloud + z_discovered; save pipeline
  - Return (updated_z_discovered, updated_pipeline)

### `emergence/run.py`
Batch inference and discovery loop.
- Load: frozen core model, denoiser, LOF pipeline, z_cloud, z_discovered, z_prior
- Threshold encoded in lof.offset_ — no percentile constant needed
- CLI args: `--n-batches INT`, `--batch-size INT` (default 1000),
  `--lambda-cfg FLOAT`
- For each batch:
  1. Sample z_T ~ N(μ, diag(σ²)) from z_prior; run reverse diffusion → z̃ batch
  2. Check 1: validity filter on all z̃ (decode → simulate 1 step); log validity rate
  3. Check 2: lof.predict() on valid z̃ batch; keep where result == -1
  4. For each emergent discovery:
     - Decode z̃ → grid; simulate 64 steps; save PNG montage to `discoveries/`
     - Call label_discovery() → behavioural label
     - Append to `discoveries/log.jsonl`: z̃, LOF score, batch index, label
     - Call update_library(); increment discovery_count
  5. Save updated `data/z_discovered.npy` to disk after each batch
  6. Print batch report: total / valid / novel / emergent counts + mean LOF score

---

## Completion Criteria

- [ ] Pipeline runs on batches of 1000 without errors
- [ ] GoL validity rate > 80% at chosen lambda_cfg
- [ ] LOF pipeline loaded; lof.predict() on z̃ used directly with no hardcoded threshold constant
- [ ] At least one confirmed emergent discovery logged with PNG saved and behavioural label
- [ ] z_discovered grows by one row per discovery; LOF pipeline re-fit on z_cloud + z_discovered at 100 discoveries
- [ ] Discovery rate per batch logged and shows expected decline as library fills
