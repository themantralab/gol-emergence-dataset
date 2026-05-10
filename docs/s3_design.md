# Stage 3 — Conditioned Latent Diffusion Sampler

## Overview

**Goal**: train a denoiser MLP on the frozen z cloud, conditioned on
pre-computed LOF novelty scores, so that at inference time the reverse diffusion
process can be steered toward underexplored regions of latent space.

**Inputs**:
- Frozen Stage 2 model checkpoints (`checkpoints/`)
- `data/grids.npy` — (N, 128, 128) uint8 training grids (for initial z cloud encoding)

**Outputs** (initial, from first Stage 3 run):
- `data/z_cloud.npy` — (N, 128) float32 encoded training z vectors; grows in Stage 4
- `data/novelty_scores.npy` — (N,) float32 pre-computed conditioning LOF scores
- `data/z_prior.npy` — (2, 128) diagonal Gaussian fit to z cloud for inference
- `checkpoints/lof_pipeline.pkl` — fitted PCA(50) + LOF on z_cloud; updated every
  100 discoveries in Stage 4
- `checkpoints/denoiser.pt` — denoiser weights; retrained every 100 discoveries

**Stage 4 retraining**: Stage 3 is not a one-shot stage. Every 100 confirmed
discoveries, Stage 4 triggers a Stage 3 retrain: z_cloud has grown by 100 rows
(one `encoder(decode(z̃))` per discovery); novelty scores are recomputed for all
z vectors; the conditioning LOF is refit; the denoiser is retrained on the
expanded cloud. This causes the sampler to explore progressively further from all
previously known territory.

**Files** (write in this order):
1. `sampler/__init__.py`
2. `sampler/schedule.py`
3. `sampler/novelty.py`
4. `sampler/denoiser.py`
5. `sampler/diffusion.py`
6. `train_sampler.py`

**Dependencies**: Stage 2 complete, model frozen.
Must complete before Stage 4 begins.

---

## Sampling Model Design

### Design principles

The sampling model operates entirely on the frozen z cloud — it never touches
the core model weights or the raw GoL grids. It learns the geometry of the
latent space and generates new z vectors that correspond to novel GoL
configurations.

**Physics consistency loss**: The denoiser training loss includes a physics
consistency term alongside the standard denoising MSE. After each denoising
step produces a predicted z̃, the frozen decoder and transition function are
used to compute a validity signal:

```
z̃          = (z_t − √(1−ᾱ_t) × ε_pred) / √(ᾱ_t)     # clean z̃ estimate
grid_pred  = (decode(z̃) > 0).to(torch.uint8)         # threshold logits → binary
grid_sim   = simulate_one_step(grid_pred)             # one real GoL step
grid_model = decode(f_θ(z̃))                          # model-predicted next state
L_physics  = BCE(grid_sim.float(), torch.sigmoid(grid_model))
```

Weight schedule: L_denoising × 1.0 + L_physics × 0.1 initially; increase
physics weight to 0.3 if validity rate at chosen λ_cfg drops below 85%.

**Conditioning novelty scores**: LOF scores are pre-computed on z_cloud before
each denoiser training run and used as the conditioning signal `c`. On the
initial Stage 3 run, z_cloud is fixed and scores are stable. On each Stage 4
retrain, z_cloud has grown by 100 new discovery vectors — scores are
recomputed for all N+100k vectors before training resumes.

**Inference starting distribution**: Diffusion chain T=256 steps, matching
the simulation horizon. At inference, the reverse chain starts from N(μ, diag(σ²))
fitted to the training z cloud — stored as `data/z_prior.npy` (shape (2, 128):
row 0 = μ, row 1 = σ). Peripheral Gaussian approximation permitted: VICReg
pushes z toward decorrelated near-unit-variance dimensions, making the diagonal
Gaussian a reliable fit.

---

### Novelty scoring: LOF on z cloud

The novelty signal is a **Local Outlier Factor (LOF)** score computed directly
in the 128-dimensional latent z space produced by the Stage 2 encoder.

**Why z-cloud LOF over FFT-fingerprint LOF:**

A LOF sanity check at the end of Stage 1 showed that gliders scored as outliers
at only 5.6% in FFT space — indistinguishable from the 5% inlier baseline.
PC1 of PCA(50) on FFT fingerprints captures population level ("alive vs dead"),
causing gliders and oscillators to collapse onto the same density region.

The Stage 2 encoder is trained differently. Its temporal contrastive loss
explicitly requires that: (a) near timesteps of the same trajectory are close,
(b) far timesteps of the same trajectory are closer than cross-trajectory
negatives, and (c) temporal ordering is preserved in distance. This means a
glider (linear CoM drift → helix in z space) and an oscillator (closed loop
in z space) will occupy geometrically distinct regions. LOF in z space
correctly scores novel spaceships as outliers because they live far from both
the glider helix and the oscillator loops.

**Why LOF over raw k-NN distance:**

Raw k-NN distance conflates two things: a point in a sparse but internally
consistent cluster (e.g. a rare oscillator type) scores high purely because the
cluster is sparse — not because it is behaviourally unusual relative to its
neighbours. LOF compares each point's local density to its neighbours' local
densities. A point in a sparse cluster surrounded by equally sparse neighbours
scores ~1 (not novel). A point genuinely isolated from everything scores high.

**Fitting the LOF pipeline:**

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import joblib

# PCA to 50D — z_cloud is 128D which is tractable, but PCA improves k-NN
# efficiency and removes near-zero-variance dimensions (VICReg makes this mild)
pca = PCA(n_components=50).fit(z_cloud)
z_reduced = pca.transform(z_cloud)               # (N, 50)

lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
lof.fit(z_reduced)

# lof.offset_ carries the calibrated threshold (top 5% = outlier)
joblib.dump({'pca': pca, 'lof': lof}, 'checkpoints/lof_pipeline.pkl')
```

`contamination=0.05` encodes the threshold in `lof.offset_` at fit time.
Stage 4 calls `lof.predict()` directly: +1 = inlier, -1 = emergent.
No percentile constant in application code.

**Scoring a new z̃ at inference:**

```python
pipeline  = joblib.load('checkpoints/lof_pipeline.pkl')
z_reduced = pipeline['pca'].transform(z_tilde.reshape(1, -1))
is_novel  = pipeline['lof'].predict(z_reduced)[0] == -1  # -1 = outlier = novel
```

---

### Conditioning signal

The conditioning signal `c` fed to the denoiser is the **LOF score** of a
training z vector against the z cloud:

```
z_cloud[i]  →  PCA(50D)  →  LOF  →  c_i  (scalar, higher = more novel)
```

Pre-computed for all N training seeds before denoiser training begins.
Stored in `data/novelty_scores.npy` (N,).

At training time, z vectors are drawn from the z cloud and their pre-computed
LOF scores are looked up. The denoiser learns: *high c → this z lives in a
sparse, unusual region of latent space → steer the reverse chain toward it.*

---

---

### Denoiser architecture

```
Input: concatenate([z_t (128,), t_embed (64,), c_embed (32,)])
                                                   ↓
                                    4-6 residual MLP layers
                                    hidden dim: 512
                                    activation: SiLU
                                                   ↓
                                    Output: predicted noise ε (128,)
```

Time embedding: sinusoidal positional encoding of scalar t → 64d.
Conditioning embedding: linear projection of scalar c → 32d.

---

### Noise schedule

Cosine schedule preferred over linear:

```python
def cosine_schedule(T, s=0.008):
    steps = np.arange(T + 1) / T
    alphas_cumprod = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0, 0.999)
```

---

### Update cadence — synchronized with Stage 4

Every 100 confirmed discoveries, Stage 4 triggers the following in order:

1. **Refit traj-sig LOF** on expanded `sig_reference` → save `traj_lof.pkl`
   (Stage 4's discovery gate; handled entirely in `emergence/gate.py`)
2. **Refit conditioning LOF** on expanded `z_cloud` → save `lof_pipeline.pkl`
   and recompute `novelty_scores.npy` for all vectors
3. **Retrain denoiser** on expanded `z_cloud` with updated novelty scores →
   save `checkpoints/denoiser.pt`

Steps 2 and 3 are Stage 3 responsibilities; Stage 4's `run.py` invokes
`train_sampler.retrain()` as a subprocess after the LOF refit completes.
Both LOF refits (traj-sig and conditioning) happen in the same pass so the
system never operates with a stale gate and a fresh sampler or vice versa.

---

## Implementation

### `sampler/__init__.py`
Empty init.

### `sampler/schedule.py`
- `cosine_schedule(T=256, s=0.008) -> dict`
- Returns: betas, alphas, alphas_cumprod (all numpy, length T)

### `sampler/novelty.py`
LOF novelty scoring pipeline.

- `fit_lof_pipeline(z_cloud, n_pca=50, n_neighbors=20, contamination=0.05) -> dict`
  - Fit PCA(50) then LOF(novelty=True, contamination=contamination) on z_cloud
  - contamination encodes the decision threshold in lof.offset_ — no downstream percentile needed
  - Return {'pca': fitted PCA, 'lof': fitted LOF}
- `score_novelty(z_vectors, pipeline) -> np.ndarray`
  - z_vectors: (N, 128) latent vectors
  - Return (N,) float32 LOF scores (negated: higher = more novel)
- `fit_z_prior(z_cloud) -> np.ndarray`
  - z_cloud: (N, 128) float32
  - Return (2, 128): row 0 = per-dim mean, row 1 = per-dim std
- `encode_and_save(data_dir, model_dir)`
  - Load frozen encoder; encode all training grids → z_cloud (N, 128)
  - Save `data/z_cloud.npy`
  - Fit z_prior; save `data/z_prior.npy`
  - (Called once at initial Stage 3 setup only — encoding uses grids.npy)

- `refit_and_save(z_cloud, data_dir, model_dir)`
  - Fit PCA(50) + LOF on z_cloud (handles both initial fit and post-discovery refits)
  - Save `checkpoints/lof_pipeline.pkl`
  - Compute LOF scores for all z vectors in z_cloud
  - Save `data/novelty_scores.npy` (len(z_cloud),)
  - Called at initial Stage 3 setup and every 100 discoveries thereafter

### `sampler/denoiser.py`
Residual MLP denoiser: (z_t, t, c) → predicted noise ε ∈ ℝ¹²⁸.
- Input: concatenate [z_t (128,), t_embed (64,), c_embed (32,)] → 224-dim
- 4–6 residual MLP layers, hidden dim 512, SiLU
- Output: predicted noise ε (128,)
- Conditioning always active — no dropout

### `sampler/diffusion.py`
- `forward(z_0, t, schedule) -> (z_t, noise)`
- `reverse_step(z_t, t, eps_pred, schedule) -> z_{t-1}`
- `sample(denoiser, c, schedule, z_prior) -> z_0_pred`
  - Draw z_T ~ N(μ, diag(σ²)); add schedule noise at t=T
  - Run full conditioned reverse chain T→0

### `train_sampler.py`
Two entry points: initial training and post-discovery retrain.

`train(data_dir, model_dir, epochs)` — initial Stage 3 training:
- Load frozen core model; load z_cloud and novelty_scores
- For each step:
  1. Sample z from z_cloud; look up pre-computed LOF score c
  2. Sample t; add noise via forward()
  3. Predict noise conditioned on c; compute denoising MSE
  4. Compute physics consistency loss; add with weight 0.1
  5. Backpropagate; update denoiser only
- Save `checkpoints/denoiser.pt`

`retrain(data_dir, model_dir, epochs)` — called by Stage 4 every 100 discoveries:
- Load expanded z_cloud (now N + 100k rows) and updated novelty_scores
- Re-initialise denoiser from last saved checkpoint (warm start, not scratch)
- Run same training loop on expanded cloud
- Save updated `checkpoints/denoiser.pt`
- Warm-start prevents forgetting the geometry learned on original z_cloud;
  the expanded cloud extends that knowledge rather than replacing it

---

## Completion Criteria

Initial Stage 3 run:
- [ ] `data/z_cloud.npy` shape (N, 128); one vector per training seed
- [ ] `data/z_prior.npy` shape (2, 128); row 1 (σ) near 1.0 per dim (VICReg effect)
- [ ] Conditioning LOF fit on z_cloud; saved to `checkpoints/lof_pipeline.pkl`
- [ ] `data/novelty_scores.npy` shape (N,); higher scores correspond to rarer z vectors
- [ ] t-SNE of z_cloud shows gliders, oscillators, still lifes in visually distinct regions
- [ ] Denoiser training loss converges; saved to `checkpoints/denoiser.pt`
- [ ] Samples conditioned on high novelty score produce more novel grids than low-score conditioning
- [ ] GoL validity rate > 80% at chosen lambda_cfg

Post-discovery retrains (every 100 discoveries):
- [ ] z_cloud grows by 100 rows (encoder(decode(z̃)) per discovery)
- [ ] novelty_scores.npy recomputed for all N+100k vectors
- [ ] lof_pipeline.pkl refit on expanded z_cloud
- [ ] denoiser.pt warm-retrained on expanded z_cloud; validity rate remains > 80%
