# Stage 2 — Core World Model

## Overview

**Goal**: train encoder, transition function, decoder, and trajectory head on
the Stage 1 dataset. The resulting model is the fixed foundation everything
else builds on — it is frozen after Stage 2 and never modified again.

**Inputs**: all `.npy` files from `data/` (Stage 1 output)

**Outputs**: trained model checkpoints in `checkpoints/`

**Files** (write in this order):
1. `model/__init__.py`
2. `model/encoder.py`
3. `model/decoder.py`
4. `model/transition.py`
5. `model/trajectory_head.py`
6. `model/vicreg.py`
7. `model/contrastive.py`
8. `data_loader.py`
9. `train_core.py`

**Dependencies**: Stage 1 complete, `data/` populated.
Must complete before Stage 3 begins. Freeze all weights after training.

**Cross-reference**: The identity training target is the normalized 10-signal
signature defined in `s1_design.md`. Use `sig_mean.npy` and `sig_std.npy`
from `data/` consistently for any inference-time normalization.

---

## Core World Model Design

### Design philosophy

The core model has **two simultaneous objectives** that are in tension:

1. **Mechanics objective**: learn the B3/S23 rule as a latent-space transition.
   This is a local constraint — it operates on cell neighborhoods and has exact
   ground truth.

2. **Identity objective**: learn behavioral class — what kind of thing is this
   configuration in the long run? This is a global constraint — it operates on
   the full 256-step trajectory and has approximate ground truth from the
   10-signal signature.

The encoder must satisfy both. Mechanics wants local neighborhood information.
Identity wants global trajectory class. This tension forces the encoder to
develop a representation that captures both, rather than collapsing to either
static appearance or average behavior.

### Architecture

**Encoder**: convolutional network
- Input: 128×128 binary grid (uint8, treated as float32 in [0,1])
- Architecture: 4 conv layers with batch norm, stride-2 downsampling
- Output: z ∈ ℝ¹²⁸ (latent dimension = 128)
- No reparameterization trick, no sampling, purely deterministic
- BatchNorm is a training stability mechanism, not a distributional assumption
  about the latent space — permitted under the project's no-Gaussian policy

**Transition function f_θ**: small MLP
- Input: z_t ∈ ℝ¹²⁸
- Architecture: 3-layer MLP, hidden dimension 256, SiLU activations
- Output: z_{t+1} ∈ ℝ¹²⁸
- Deliberately small so it cannot memorize grid states — forced to learn
  the transition rule abstractly
- Applied repeatedly (unrolled) during rollout training

**Decoder**: convolutional network mirroring encoder
- Input: z ∈ ℝ¹²⁸
- Architecture: transposed convolutions mirroring encoder
- Output: 128×128 binary grid logits (sigmoid applied for loss, argmax for
  sampling)

**Trajectory head**: per-timestep MLP
- Input: z_t ∈ ℝ¹²⁸ — the latent state at any step t of a rollout
- Architecture: 3-layer MLP, hidden dim 128, output ℝ¹⁰
- Output: predicted normalized 10-signal values [P, Δcx, Δcy, V, E, N_cc, S_lag_2, S_lag_4, S_lag_8, S_lag_16]
  at timestep t
- Applied at every step of the rollout during training; loss accumulates
  across all steps
- This replaces the prior "attractor head" design (which predicted the full
  trajectory from a single z_100 snapshot). Per-timestep prediction provides
  dense supervision across the entire rollout rather than a single endpoint,
  and naturally integrates with the progressive rollout curriculum — as rollout
  depth increases, the head receives gradient signal from more timesteps.

### Latent space regularization — VICReg

No Gaussian prior. No KL divergence. No reparameterization.

VICReg enforces three soft constraints on the batch of z vectors:

```python
def vicreg_loss(z, lambda_var=25.0, lambda_cov=25.0):
    # z shape: (batch, 128)

    # 1. Variance: each dimension should have std near 1
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    loss_var = torch.mean(torch.relu(1 - std))

    # 2. Covariance: off-diagonal terms should be near 0
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    off_diag_mask = ~torch.eye(128, dtype=bool)
    loss_cov = (cov ** 2)[off_diag_mask].mean()

    return lambda_var * loss_var + lambda_cov * loss_cov
```

This prevents:
- Dimensional collapse (all z vectors become identical)
- Feature redundancy (dimensions encoding the same information)
- Dead dimensions (dimensions with near-zero variance)

### Anti-blob training — multiscale temporal contrastive loss

This is the most important addition beyond basic reconstruction + VICReg.
Without it, the encoder learns "what do GoL grids look like" rather than
"what trajectory class does this configuration belong to" — producing a
featureless cloud where all patterns mix together.

**Mechanism**: for every training batch, sample triplets (anchor, near, far,
negative) from the trajectory sequences:
- anchor: z at timestep t
- near: z at timestep t+k (small k, same trajectory)
- far: z at timestep t+K (large K, same trajectory)
- negative: z from a different trajectory entirely

```python
def temporal_contrastive_loss(z_anchor, z_near, z_far, z_neg,
                               margin_near=0.5, margin_far=1.0,
                               margin_order=0.3):
    d = torch.nn.functional.pairwise_distance

    # Near pairs should be close (short-term coherence)
    loss_near = torch.relu(
        d(z_anchor, z_near) - d(z_anchor, z_neg) + margin_near
    ).mean()

    # Far pairs should be closer than negatives (long-term identity)
    loss_far = torch.relu(
        d(z_anchor, z_far) - d(z_anchor, z_neg) + margin_far
    ).mean()

    # Near should be closer than far (temporal ordering preserved)
    loss_order = torch.relu(
        d(z_anchor, z_near) - d(z_anchor, z_far) + margin_order
    ).mean()

    return loss_near + loss_far + loss_order
```

This simultaneously enforces short-term dynamics, long-term trajectory
identity, and the temporal ordering relationship between them.

### Training losses

| Loss             | Formula                                              | Phase active |
|------------------|------------------------------------------------------|--------------|
| L_mechanics      | BCE(decode(f_θ^k(z_0)), grid_k) for sampled k        | 1, 2, 3      |
| L_trajectory     | MSE(traj_head(z_k), sig_norm[k]) for k in rollout    | 2, 3         |
| L_contrastive    | temporal triplet on z sequence                       | 3 only       |
| VICReg           | var + cov on batch z                                 | 1, 2, 3      |

**Mechanics loss** uses per-cell binary cross-entropy. Ground truth is exact
GoL simulation — no approximation. The completion criterion is >95% accuracy
on alive cells specifically (not overall accuracy, which is dominated by the
sparse dead cells). Applied at every sampled rollout depth k — not just k=1.

**Trajectory loss** uses MSE between traj_head(z_k) and sig_norm[k] for each
step k in the rollout. The target sig_norm[k] is the pre-computed normalized
10-signal value at timestep k from `signatures_norm.npy`. The loss is averaged
across all k steps in the rollout. This provides dense behavioral supervision
throughout the trajectory: every latent state z_k must contain enough
information to predict the observable physics at that exact moment.

**Weight schedule** (approximate — tune based on validation metrics):
- Phase 1: L_mechanics=1.0, L_trajectory=0.0, L_contrastive=0.0, VICReg=0.05
- Phase 2: L_mechanics=1.0, L_trajectory=0.2, L_contrastive=0.0, VICReg=0.05
- Phase 3: L_mechanics=1.0, L_trajectory=0.5, L_contrastive=0.2, VICReg=0.05

Mechanics loss never goes below 40% of total weight. If trajectory loss
dominates early, the model learns to predict behavioral class by cheating on
grid reconstruction.

### Training curriculum

**Phase 1 — Rule learning** (mechanics only, progressive rollout k = 1 → 32):
- Loss: L_mechanics × 1.0 + VICReg × 0.05
- **Teacher forcing active**: use real encoded z_t as input to f_θ (not
  predicted); prevents early error compounding before f_θ is reliable.
- **Progressive rollout**: begin at k=1. Once alive-cell accuracy on k=1 exceeds
  95% on validation, increase max rollout depth: 1 → 4 → 8 → 16 → 32. At each
  step, randomly sample k ∈ [1, k_max] per training step and supervise
  decode(f_θ^k(z_0)) against grid_k. This directly trains f_θ to compose —
  z_0 must encode everything needed to reconstruct the grid at any depth up to
  k_max without teacher forcing intermediate states.
- **Retrospective reliance**: because the decoder must reproduce grid_k from
  z_k = f_θ^k(z_0) alone, the encoder is forced to pack all mechanistically
  necessary information into z_0 at encoding time. Short-horizon errors compound
  into long-horizon failures, so the encoder cannot drop any detail that matters.
- Do not proceed to Phase 2 until k=32 rollout drift has plateaued on
  validation (see Rollout drift monitoring below).

**Phase 2 — Trajectory supervision** (add identity, rollout k = 1 → 64,
teacher forcing linearly decays 100% → 0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.2 + VICReg × 0.05
- **Trajectory head introduced**: at each rollout step k, apply traj_head(z_k)
  and supervise against sig_norm[k]. Loss averaged across all k steps sampled
  in the rollout.
- Progressive rollout continues from k=32 (end of Phase 1) up to k=64.
  Randomly sample k ∈ [1, k_max] per training step.
- **Gradual teacher forcing schedule**: linearly interpolate from 100% teacher
  forcing (Phase 1 end) to 0% teacher forcing (Phase 2 end) over Phase 2
  epochs. At p_teacher probability, use real z_t; otherwise use predicted z_t.
- The trajectory loss provides dense behavioral signal: as k grows, the head
  must predict signal[k] from z_k derived from a free rollout — this couples
  f_θ accuracy directly to behavioral prediction quality.
- Trigger to Phase 3: trajectory head predictions are stable at k=64 rollout
  and teacher forcing has reached 0%.

**Phase 3 — Full joint** (all losses, rollout k = 1 → 256, teacher forcing = 0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.5 + L_contrastive × 0.2 + VICReg × 0.05
- Progressive rollout extends to full T=256. Randomly sample k ∈ [1, 256]
  per training step.
- **Contrastive loss added at full weight**: hard negative mining enabled.
  (Hard negatives require a partially trained encoder to be meaningful —
  Phase 2 has provided this.)
- Free rollout only (teacher forcing = 0%).
- Key validation: t-SNE of encoded held-out patterns shows visually separable
  clusters for still lifes, oscillators, gliders, dying patterns.
- Do not proceed to Stage 3 until t-SNE shows separation.

### Rollout drift monitoring

When f_θ is unrolled k times, small per-step errors compound. The diagnostic:
plot ||decode(f_θ^k(z_0)) − grid_{t+k}|| for k in {1, 5, 10, 25, 50, 100, 256}
at each training checkpoint. Should plateau, not grow monotonically.

Progressive rollout (random k per step) is active throughout all phases as the
primary drift mitigation — it is not a fallback but a standard part of training.
If drift still occurs despite progressive rollout, secondary mitigations are:
1. Reduce maximum rollout depth for the current phase
2. Slow the teacher-forcing decay schedule in Phase 2

---

## Implementation

### `model/__init__.py`
Empty init to make `model/` a package.

### `model/encoder.py`
Convolutional encoder: (128, 128) binary grid → z ∈ ℝ¹²⁸.
- 4 conv layers, stride-2 downsampling, BatchNorm after each, ReLU activations
- After 4 stride-2 layers: spatial size = 128/16 = 8×8, then flatten + linear → 128
- No reparameterization, no sampling — purely deterministic

### `model/decoder.py`
Convolutional decoder mirroring encoder: z ∈ ℝ¹²⁸ → (128, 128) logits.
- Linear → reshape to (C, 8, 8), then 4 transposed conv layers, stride-2 upsampling
- Output: (1, 128, 128) logits (no sigmoid — apply in loss, argmax for sampling)

### `model/transition.py`
Latent transition MLP: z_t ∈ ℝ¹²⁸ → z_{t+1} ∈ ℝ¹²⁸.
- 3-layer MLP, hidden dim 256, SiLU activations
- Deliberately small to prevent memorization of grid states

### `model/trajectory_head.py`
Per-timestep trajectory prediction head: z_t ∈ ℝ¹²⁸ → ℝ¹⁰.
- 3-layer MLP, hidden dim 128, output 10
- Applied at each step t of the rollout; no recurrence, no sequence dependency
- Output is the predicted normalized 10-signal value at timestep t
- Loss: MSE against sig_norm[t] from `signatures_norm.npy`

### `model/vicreg.py`
VICReg regularization loss (variance + covariance terms).
- Exactly as specified above: lambda_var=25.0, lambda_cov=25.0

### `model/contrastive.py`
Temporal contrastive triplet loss with hard negative mining.
- Exactly as specified above: margin_near=0.5, margin_far=1.0, margin_order=0.3
- Triplet sampling: anchor at t, near at t+k (small k ~5), far at t+K (K ~50)
- **Hard negative mining**: rather than drawing negatives randomly from other
  trajectories in the batch, select the hardest negative — the z vector from a
  different trajectory that is closest to the anchor in current latent space.
  This produces negatives that are genuinely confusable with the anchor,
  sharpening behavioral cluster boundaries and improving novelty discrimination.
- `sample_hard_negatives(z_batch, trajectory_ids) -> z_neg`
  - z_batch: (B, 128) latent vectors; trajectory_ids: (B,) int seed indices
  - For each anchor i: compute pairwise distances to all j where trajectory_ids[j] ≠ trajectory_ids[i]
  - Select j with minimum distance as the hard negative
  - Return (B, 128) hard negative vectors
- Hard negatives used in Phase 3 only; random negatives used in Phase 2
  (hard negatives require a partially trained encoder to be meaningful)

### `data_loader.py`
PyTorch Dataset and DataLoader for the Stage 1 dataset.
- Load metadata npy files from `data/` at init (seeds, grids, signatures_norm,
  labels, buckets, sig_mean, sig_std)
- **On-demand simulation**: trajectories are not stored for large N_SEEDS datasets.
  `__getitem__` calls `simulator.simulate(grid, steps=256)` to generate the
  trajectory for the requested seed at access time. This keeps RAM usage
  proportional to batch size, not dataset size.
- `__getitem__` returns dict: trajectory (full (257, 128, 128) uint8),
  sig_norm (257, 10) float32, trajectory_id, bucket
- The training loop encodes raw grids to z vectors at each step — the
  dataloader never calls the encoder. Returning z vectors would be wrong
  because the encoder weights change every step during training.
- The full trajectory is returned (from LRU cache or freshly simulated) so
  the training loop can derive (grid_t, grid_{t+k}) pairs and index into
  sig_norm[k] for any rollout depth k during training without calling the
  simulator again.
- Support 90/10 train/validation split by seed index
- Cache recently simulated trajectories (LRU, configurable size) to avoid
  re-simulating the same seed within a training epoch

### `train_core.py`
Training loop with 3-phase progressive rollout curriculum.

Phase 1 (mechanics only, progressive rollout k=1→32, teacher forcing=100%):
- Loss: L_mechanics × 1.0 + VICReg × 0.05
- Rollout schedule: begin at k_max=1; advance to 4, 8, 16, 32 once alive-cell
  accuracy on current k_max exceeds 95% on validation
- Randomly sample k ∈ [1, k_max] per training step
- Teacher forcing: always use real z_t as f_θ input

Phase 2 (add trajectory head, progressive rollout k=1→64, teacher forcing
100%→0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.2 + VICReg × 0.05
- Continue rollout progression from k_max=32 up to k_max=64
- Apply traj_head(z_k) at each rollout step; supervise against sig_norm[k]
- Teacher forcing decays linearly from 100% to 0% over Phase 2 epochs
- Hard negatives: disabled

Phase 3 (full joint, progressive rollout k=1→256, teacher forcing=0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.5 + L_contrastive × 0.2 + VICReg × 0.05
- Rollout extends to full T=256; randomly sample k ∈ [1, 256] per step
- Hard negatives: enabled
- Stop when t-SNE of encoded held-out patterns shows cluster separation

Checkpointing: save model state_dict after each phase and every N epochs.
Monitor rollout drift at each checkpoint.

---

## Completion Criteria

- [ ] 1-step reconstruction accuracy on alive cells > 95% on validation set
- [ ] f_θ correctly predicts next state for canonical patterns (block,
      blinker, glider) verified by visual inspection of decoded outputs
- [ ] Rollout drift plot plateaus — does not grow monotonically to k=256
- [ ] Trajectory head predictions qualitatively match expected per-class signal
      shapes at k=64 and k=256 (dying decays, glider drifts, oscillator periodic)
- [ ] t-SNE of encoded held-out patterns shows behavioral cluster separation
      (dying and glider clearly separated; still_life/oscillator overlap acceptable)
- [ ] Model checkpoints saved to `checkpoints/`; weights frozen before Stage 3
