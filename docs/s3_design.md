# Stage 3 — Conditioned Latent Diffusion Sampler

## Overview

**Goal**: train a denoiser MLP on the frozen z cloud, conditioned on
pre-computed LOF novelty scores, so that at inference time the reverse diffusion
process can be steered toward underexplored regions of latent space.

**Inputs**:
- Frozen Stage 2 model checkpoints (`checkpoints/`)
- `data/grids.npy` — (N, 128, 128) uint8 training grids (for encoding to z cloud)

**Outputs**:
- `data/z_cloud.npy` — (N, 128) float32 encoded training z vectors
- `data/novelty_scores.npy` — (N,) float32 pre-computed LOF novelty scores
- `data/z_prior.npy` — (2, 128) diagonal Gaussian fit to z cloud for inference
- `checkpoints/lof_pipeline.pkl` — fitted PCA + LOF pipeline on z_cloud (frozen after Stage 3)
- Denoiser checkpoint

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

**Static novelty scores**: LOF scores are pre-computed once before training
on the fixed sig_reference and remain frozen throughout denoiser training.
Because the encoder is frozen and sig_reference is not modified until Stage 4,
re-computing would produce identical results — the scores are stable
conditioning inputs.

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

### Reference library interaction

`data/sig_reference.npy` starts as (N=1M, 1290) from Stage 1 and grows in
Stage 4 as discoveries are confirmed. The LOF pipeline is re-fit periodically
(every 100 discoveries) to incorporate new fingerprints. Between re-fits,
Stage 4 uses the most recently saved pipeline.

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
- `precompute_and_save(data_dir, model_dir)`
  - Load frozen encoder; encode all training grids → z_cloud (N, 128)
  - Save `data/z_cloud.npy`
  - Fit LOF pipeline on z_cloud
  - Save `checkpoints/lof_pipeline.pkl`
  - Compute LOF scores for all z vectors
  - Save `data/novelty_scores.npy` (N,)
  - Fit z_prior; save `data/z_prior.npy`

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
- Load frozen core model; load pre-computed novelty scores
- For each step:
  1. Sample z from z_cloud; look up pre-computed LOF score c
  2. Sample t; add noise via forward()
  3. Predict noise conditioned on c; compute denoising MSE
  4. Compute physics consistency loss; add with weight 0.1
  5. Backpropagate; update denoiser only
- Conditioning is always active — no dropout, no CFG

---

## Completion Criteria

- [ ] LOF pipeline fit on z_cloud (N, 128) with contamination=0.05; saved to `checkpoints/lof_pipeline.pkl`
- [ ] `data/novelty_scores.npy` shape (N,); higher scores correspond to rarer z vectors
- [ ] `data/z_cloud.npy` shape (N, 128)
- [ ] `data/z_prior.npy` shape (2, 128); row 1 (σ) near 1.0 per dim (VICReg effect)
- [ ] t-SNE of z_cloud shows gliders, oscillators, still lifes in visually distinct regions (confirming LOF will separate them)
- [ ] Denoiser training loss converges
- [ ] Samples conditioned on high novelty score produce more novel grids than low-score conditioning
- [ ] GoL validity rate > 80%
