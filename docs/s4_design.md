# Stage 4 — Emergence Pipeline

## Overview

**Goal**: batch inference loop that samples novel GoL configurations, filters
them through validity and novelty checks, logs confirmed discoveries, and
grows the reference library so the novelty bar rises over time.

**Inputs**:
- Frozen Stage 2 model (encoder, transition, decoder) — trajectory head not
  needed at inference; behavioral classification uses exact simulation signals
- `checkpoints/denoiser.pt` — Stage 3 denoiser, warm-retrained every 100
  discoveries; loaded fresh after each retrain
- `data/sig_reference.npy` — (N, 1290) training behavioral fingerprints;
  the traj-sig LOF reference corpus; grows every discovery
- `data/sig_mean.npy`, `data/sig_std.npy` — normalization parameters from
  Stage 1; applied to candidate signals before fingerprint computation
- `checkpoints/traj_lof.pkl` — LOF model fitted on sig_reference; re-fitted
  every 100 discoveries
- `data/z_cloud.npy` — (N, 128) encoded z vectors; grows every discovery;
  used to re-encode discoveries and trigger Stage 3 retrains
- `data/z_prior.npy` — (2, 128) diagonal Gaussian fit to z_cloud; inference
  starting distribution for the denoiser
- `checkpoints/lof_pipeline.pkl` — Stage 3 conditioning LOF; refit alongside
  traj-sig LOF every 100 discoveries

**Outputs**:
- `discoveries/` — one PNG per confirmed emergent pattern (initial grid +
  64-step simulation montage)
- `discoveries/log.jsonl` — z̃ vector, traj-sig LOF score, batch index,
  behavioral label, and (1290,) trajectory fingerprint per discovery
- `data/sig_reference.npy` — one fingerprint row appended per discovery
- `data/z_cloud.npy` — one encoded z row appended per discovery
- `checkpoints/traj_lof.pkl` — refit every 100 discoveries
- `checkpoints/lof_pipeline.pkl` — refit every 100 discoveries
- `checkpoints/denoiser.pt` — warm-retrained every 100 discoveries

**Files** (write in this order):
1. `emergence/__init__.py`
2. `emergence/validity.py`
3. `emergence/classifier.py`
4. `emergence/gate.py`
5. `emergence/run.py`

**Dependencies**: Stage 2 and Stage 3 complete, both model sets frozen.
`emergence/validity.py` imports `simulator.step` directly from `simulator.py`.
`emergence/run.py` imports `simulator.simulate` for the full 256-step rollout.

---

## Emergence Pipeline Design

### Discovery pipeline

Every candidate z̃ from the denoiser passes through three sequential steps.
Earlier steps are cheaper and gate the later ones.

**Step 1 — GoL validity** (cheap, runs first):
- Decode z̃ → candidate grid (128×128)
- Simulate one GoL step via `simulator.step`
- Decode f_θ(z̃) → model's predicted next state
- If cell agreement < 90%: discard as off-manifold
- Cost: 1 decoder call + 1 f_θ call + 1 simulation step

**Step 2 — Full trajectory simulation** (valid candidates only):
- Run `simulator.simulate(candidate_grid, steps=256)` → exact (257, 128, 128)
  trajectory under exact GoL physics
- Compute the 10-signal array (257, 10) from the exact trajectory — same
  signal extraction code as Stage 1 (population, Δcx, Δcy, velocity, motion
  energy, connected components, four self-similarity lags)
- Normalize: sig_norm = (sig_raw − sig_mean) / sig_std
- Compute rfft of sig_norm along the time axis (axis=0) → take magnitudes →
  flatten → (1290,) trajectory fingerprint
- This is the exact same fingerprint computation used to produce sig_reference
  in Stage 1. No approximation; no trajectory head involved.
- Cost: 256 simulation steps + signal extraction (≈ 1–2 seconds per candidate)

**Step 3 — Trajectory-signature LOF** (novelty gate):
- Score the candidate's (1290,) fingerprint against sig_reference using the
  fitted LOF model
- If lof.predict(fingerprint) == −1 (outlier): DISCOVERY
- The boundary is set by contamination=0.05 at LOF fit time — the most
  behaviorally novel 5% of training patterns defines the discovery threshold
- Cost: one Ball Tree query (milliseconds)

**Post-gate logging** (confirmed discoveries only):
- Save decoded initial grid + 64-step simulation montage PNG to `discoveries/`
- Classify behavioral class from the exact 10-signal trajectory (classifier.py)
- Append the (1290,) fingerprint to sig_reference; re-fit LOF every 100 discoveries
- Append record to `discoveries/log.jsonl`

### Trajectory-signature LOF setup

LOF is fitted once at Stage 4 startup on the full sig_reference (1,500,000 × 1290):

```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    algorithm='ball_tree',
    metric='euclidean',
    novelty=True,   # required for predict() on new points
)
lof.fit(sig_reference)
```

`novelty=True` is required so that `lof.predict()` can score new candidates
not seen during fitting.

Fit time on 1.5M × 1290 is expensive on CPU (minutes to hours) but is a
one-time startup cost. Query time is O(k log N) via Ball Tree — milliseconds
per candidate. The Ball Tree index is serialised inside traj_lof.pkl so the
full re-fit at every 100 discoveries is the only repeated cost.

### Why contamination over a hardcoded threshold

`contamination=0.05` is set once at LOF fit time. Stage 4 application code
calls `lof.predict()` with no scalar constant — the boundary lives in
`lof.offset_`. To adjust the discovery bar: change `contamination` and re-fit
at Stage 4 startup. The parameter has a concrete meaning: *"the most
behaviorally novel 5% of the training distribution defines the discovery
boundary."*

### Reference library update

When a discovery is confirmed:
1. Append the (1290,) fingerprint to `data/sig_reference.npy` → (N+1, 1290)
2. Keep the current LOF in memory between re-fits (stale but still valid)
3. Every 100 discoveries: re-fit LOF on the full (expanded) sig_reference;
   save updated `checkpoints/traj_lof.pkl`

As discoveries accumulate, previously-discovered signature regions become
denser. Subsequent variants of the same discovery class score lower LOF
(they now have close neighbours in sig space) and eventually fall below the
threshold. This naturally raises the effective novelty bar without any manual
intervention.

### Sample outcome taxonomy

| GoL valid | LOF predict | Outcome                                    |
|-----------|-------------|--------------------------------------------|
| No        | —           | Discard (off-manifold)                     |
| Yes       | +1 (inlier) | Log for coverage metrics; not a discovery  |
| Yes       | -1 (outlier)| EMERGENT DISCOVERY — log + update library  |

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
GoL validity check — Step 1.
- `check_validity(z_tilde, decoder, transition, threshold=0.90) -> bool`
  - Decode z̃ → candidate grid; simulate one GoL step via simulator.step;
    decode f_θ(z̃) → predicted next state
  - Compute cell agreement (fraction matching); return agreement >= threshold
- `check_validity_batch(z_batch, decoder, transition, threshold=0.90) -> np.ndarray`
  - Return (N,) bool array

### `emergence/classifier.py`
Behavioral classification from exact simulation signals — post-gate only.
The trajectory head is not used here. Classification is derived entirely from
the exact 10-signal trajectory computed in Step 2.

- `classify_trajectory(sig_raw) -> str`
  - `sig_raw`: (257, 10) float32, unnormalized signals from exact simulation
  - Returns one of: 'dying' / 'still_life' / 'oscillator' / 'glider' / 'other'
  - Classification heuristic (same logic as Stage 1 labelling):
    - **dying**: P(t=256) == 0 (population reaches zero)
    - **still_life**: P(t) stable across last 64 steps, E(t) < 0.5, |Δcx|+|Δcy| < 0.1
    - **oscillator**: max(S_lag_2, S_lag_4, S_lag_8, S_lag_16) > 0.9 and
                      |Δcx|+|Δcy| < 0.1 over last 128 steps
    - **glider**: mean(|Δcx| + |Δcy|) > 0.3 over last 128 steps
    - **other**: none of the above

### `emergence/gate.py`
Trajectory-signature LOF gate — Step 3 and library management.

- `fit_lof(sig_reference) -> LocalOutlierFactor`
  - Fit LOF (novelty=True, n_neighbors=20, contamination=0.05,
    algorithm='ball_tree') on sig_reference (N, 1290)
  - Return fitted lof object

- `load_lof(checkpoint_dir) -> LocalOutlierFactor`
  - Load `checkpoints/traj_lof.pkl`; return lof object

- `save_lof(lof, checkpoint_dir)`
  - Save lof to `checkpoints/traj_lof.pkl`

- `compute_fingerprint(sig_norm) -> np.ndarray`
  - `sig_norm`: (257, 10) normalized signal array
  - rfft along axis=0 → take magnitude → flatten → (1290,) float32
  - Same computation as Stage 1 sig_reference generation

- `score_novelty(fingerprint, lof) -> float`
  - `fingerprint`: (1290,) array
  - Return raw LOF score (higher = more novel relative to neighbours)

- `predict_novelty(fingerprint, lof) -> bool`
  - Return True if lof.predict(fingerprint[None])[0] == -1 (outlier)

- `update_library(fingerprint, z_discovery, sig_reference_path, z_cloud_path,
                  lof, checkpoint_dir, data_dir, discovery_count) -> LocalOutlierFactor`
  - Append fingerprint to sig_reference.npy (one new traj-sig row)
  - Append z_discovery (encoder(decode(z̃))) to z_cloud.npy (one new z row)
  - If discovery_count % 100 == 0:
    - Refit traj-sig LOF on expanded sig_reference; save traj_lof.pkl
    - Call sampler/novelty.refit_and_save() on expanded z_cloud;
      saves updated lof_pipeline.pkl and novelty_scores.npy
    - Call train_sampler.retrain() on expanded z_cloud (warm start);
      saves updated denoiser.pt
    - Return updated traj-sig lof
  - Else: return lof unchanged

### `emergence/run.py`
Batch inference and discovery loop.

Startup:
- Load frozen Stage 2 model (encoder, transition, decoder)
- Load denoiser from checkpoints/denoiser.pt; set to eval mode
- Load sig_reference.npy (mmap), z_cloud.npy (mmap), sig_mean.npy,
  sig_std.npy, z_prior.npy
- Load or fit traj-sig LOF (traj_lof.pkl); fit if absent, load if present

CLI args: `--n-batches INT`, `--batch-size INT` (default 1000),
`--lambda-cfg FLOAT`

For each batch:
1. Sample z_T ~ N(μ, diag(σ²)) from z_prior; run reverse diffusion → z̃ batch (B, 128)
2. **Step 1**: validity filter
   - check_validity_batch(z̃_batch, decoder, transition) → valid mask
   - Log validity rate; discard invalid z̃ vectors
3. **Step 2**: full trajectory simulation for valid candidates
   - For each valid z̃:
     - Decode z̃ → candidate_grid (128×128)
     - simulator.simulate(candidate_grid, steps=256) → (257, 128, 128)
     - Extract 10-signal array (257, 10) from trajectory (sig_raw)
     - sig_norm = (sig_raw − sig_mean) / sig_std
     - compute_fingerprint(sig_norm) → (1290,) fingerprint
4. **Step 3**: novelty gate
   - predict_novelty(fingerprint, traj_lof) for each candidate
   - Keep candidates where result is True (outlier)
5. For each confirmed discovery:
   - Decode z̃ → grid; simulate 64 steps; save PNG montage to `discoveries/`
   - classify_trajectory(sig_raw) → behavioral label
   - z_discovery = encoder(decode(z̃)) — re-encode for z_cloud consistency
   - Append to `discoveries/log.jsonl`:
     `{z_tilde, lof_score, batch_idx, label, fingerprint}`
   - update_library(fingerprint, z_discovery, ...); increment discovery_count
   - If discovery_count % 100 == 0:
     - update_library triggers: traj-sig LOF refit, conditioning LOF refit,
       denoiser retrain (warm start)
     - Reload denoiser from updated checkpoints/denoiser.pt
6. Print batch report: total / valid / novel / discovery counts + mean LOF score
   + discovery rate + z_cloud size

---

## Completion Criteria

- [ ] Pipeline runs on batches of 1000 without errors
- [ ] GoL validity rate > 80% at chosen lambda_cfg
- [ ] traj-sig LOF fitted on sig_reference at startup; traj_lof.pkl saved
- [ ] At least one confirmed discovery logged with PNG saved and behavioral label
- [ ] Every discovery: sig_reference grows by one fingerprint row;
      z_cloud grows by one encoder(decode(z̃)) row
- [ ] Every 100 discoveries: traj-sig LOF refit, conditioning LOF refit,
      denoiser warm-retrained — all three in the same pass; denoiser reloaded
- [ ] Discovery rate per batch logged and shows expected decline as library fills
- [ ] z_cloud size reported in each batch report
