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
- Channel progression: 1→32→64→128→256, stride-2 at each layer
  - After 4 stride-2 layers: spatial size 128→64→32→16→8
  - Flatten: 256×8×8 = 16,384 → linear → 128
- BatchNorm + ReLU after each conv layer (standard pairing; BN removes the
  dying-neuron risk that makes ReLU problematic without normalisation)
- No activation after the final linear → 128; z is raw, VICReg shapes its
  distribution during training
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
- Linear → reshape to (256, 8, 8), then 4 transposed conv layers stride-2,
  channel progression 256→128→64→32→1
- BatchNorm + ReLU after each transposed conv except the last
- Output: (1, 128, 128) logits — sigmoid applied for loss, threshold at 0.5
  for grid reconstruction

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
| L_contrastive    | temporal triplet on z sequence                       | 2, 3         |
| VICReg           | var + cov on batch z_0 (see VICReg target note)      | 1, 2, 3      |

**VICReg target**: applied to z_0 vectors only (one per trajectory in the
accumulated batch of 256). This directly regularizes the distribution that
z_cloud represents. Mid-trajectory z_t vectors are structured by the
contrastive loss instead. Running VICReg on trajectory frames would penalize
the correct still-life behavior of constant z_t and conflate temporal
correlation with feature redundancy.

**Mechanics loss** uses per-cell binary cross-entropy with `pos_weight=50` on
alive cells. Ground truth is exact GoL simulation — no approximation. The
completion criterion is >95% accuracy on alive cells specifically (not overall
accuracy, which is dominated by the sparse dead cells).

The pos_weight is required because grids are ~99.74% dead cells (dead:alive
ratio ~378:1 in the 1.5M-seed dataset). Without it the model minimises BCE by
predicting all-dead — achieving near-zero loss and near-zero alive-cell
accuracy simultaneously. pos_weight=50 gives alive cells ~13% of total gradient
mass (up from 0.26%), sufficient to force the model to learn alive cell patterns
without destabilising training. The natural weight of 378 was confirmed too
aggressive in initial training runs (plateau at 40% alive-cell accuracy after
20k steps with no weight).

Applied at the sampled rollout depth k only — not at every step.

**Trajectory loss** uses MSE between traj_head(z_k) and sig_norm[k] for each
step k in the rollout. The target sig_norm[k] is the pre-computed normalized
10-signal value at timestep k from `signatures_norm.npy`. The loss is averaged
across all k steps in the rollout. This provides dense behavioral supervision
throughout the trajectory: every latent state z_k must contain enough
information to predict the observable physics at that exact moment.

**Weight schedule**:
- Phase 1: L_mechanics=1.0, L_trajectory=0.0, L_contrastive=0.0,  VICReg=0.01
- Phase 2: L_mechanics=1.0, L_trajectory=0.2, L_contrastive=0.1,  VICReg=0.05
- Phase 3: L_mechanics=1.0, L_trajectory=0.5, L_contrastive=0.2,  VICReg=0.05

Phase 1 VICReg weight is 0.01 (not 0.05) to prevent VICReg from competing
with the mechanics gradient at low LR. At 0.05, VICReg dominated mechanics
once ReduceLROnPlateau decayed the LR, causing alive-cell accuracy regression.
0.01 still prevents collapse while keeping mechanics as the dominant signal.
Phases 2 and 3 restore VICReg to 0.05 as the full loss suite provides
sufficient gradient mass.

Mechanics loss never goes below 40% of total weight. If trajectory loss
dominates early, the model learns to predict behavioral class by cheating on
grid reconstruction.

### Training curriculum

#### Rollout depth advancement — decaying threshold

k_max advances through fixed levels: 1 → 2 → 4 → 8 → 16 → 32 → 48 → 64 → 96
→ 128 → 192 → 256. Advancement from level L to the next requires alive-cell
accuracy on k=L to exceed the **advancement threshold** for 2 consecutive
validation checks. The threshold decays linearly with depth but never drops
below 95%:

```
threshold(k_max) = max(0.975 - 0.04 × (k_max / 256), 0.95)
```

Representative values:

| k_max | Required accuracy |
|-------|------------------|
| 1     | 97.5%            |
| 32    | 97.0%            |
| 64    | 96.5%            |
| 96    | 96.0%            |
| 128   | 95.5%            |
| 192   | 95.0%            |
| 256   | 95.0%            |

The rationale for the decay: compounding rollout errors make perfect accuracy
physically impossible at long horizons even for a well-trained model. The
threshold relaxes to reflect this without ever accepting reconstruction quality
below 95% — the GoL rule is simple enough that the model should stay well above
this floor throughout training.

Phase transitions are tied to k_max milestones, not to separate criteria:
- **Phase 1 → Phase 2**: when k_max first reaches **96**
- **Phase 2 → Phase 3**: when k_max first reaches **192**

**Phase 1 — Rule learning** (mechanics only, progressive rollout k = 1 → 96):
- Loss: L_mechanics × 1.0 + VICReg × 0.01
- **Teacher forcing at 90%** (p_teacher=0.9): 90% of rollout steps use the
  real encoded z_t as input to f_θ; 10% use f_θ's own predicted z_t
  (scheduled sampling). The 10% free-rollout exposure prevents the transition
  from over-specialising to exact encoder outputs — with TF=100%, val accuracy
  systematically peaked at step ~500 then regressed to ~84% by step 13k as
  the training and eval distributions diverged. 90% TF still provides strong
  supervision for learning GoL physics while closing that gap.
- **Progressive rollout**: begin at k_max=1. Advance through depth levels
  (1→2→4→…→96) using the decaying threshold rule above. At each training
  step, randomly sample k ∈ [1, k_max] and supervise decode(f_θ^k(z_0))
  against grid_k. This directly trains f_θ to compose — z_0 must encode
  everything needed to reconstruct the grid at any depth up to k_max without
  teacher forcing intermediate states.
- **Retrospective reliance**: because the decoder must reproduce grid_k from
  z_k = f_θ^k(z_0) alone, the encoder is forced to pack all mechanistically
  necessary information into z_0 at encoding time. Short-horizon errors compound
  into long-horizon failures, so the encoder cannot drop any detail that matters.

**Phase 2 — Trajectory supervision** (add identity + contrastive, rollout
k = 1 → 192, teacher forcing linearly decays 100% → 0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.2 + L_contrastive × 0.1 + VICReg × 0.05
- **Trajectory head introduced**: at each rollout step k, apply traj_head(z_k)
  and supervise against sig_norm[k]. Loss averaged across all k steps sampled
  in the rollout.
- Progressive rollout continues from k_max=96 (end of Phase 1) up to k_max=192,
  using the same decaying threshold advancement rule.
  Randomly sample k ∈ [1, k_max] per training step.
- **Gradual teacher forcing schedule**: linearly interpolate from 100% teacher
  forcing (Phase 1 end) to 0% teacher forcing (Phase 2 end) over Phase 2
  training steps. At p_teacher probability, use real z_t; otherwise use
  predicted z_t.
- The trajectory loss provides dense behavioral signal: as k grows, the head
  must predict signal[k] from z_k derived from a free rollout — this couples
  f_θ accuracy directly to behavioral prediction quality.

**Phase 3 — Full joint** (all losses, rollout k = 1 → 256, teacher forcing = 0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.5 + L_contrastive × 0.2 + VICReg × 0.05
- Progressive rollout continues from k_max=192 to k_max=256 using the same
  decaying threshold advancement rule. Randomly sample k ∈ [1, k_max] per step.
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
- 3-layer MLP, hidden dim 128, SiLU activations, output 10
- Applied at each step t of the rollout; no recurrence, no sequence dependency
- Output is the predicted normalized 10-signal value at timestep t
- Loss: MSE against sig_norm[t] from `signatures_norm.npy`
- **Training instrument only.** Forces the encoder to pack behavioral
  information into z by providing dense per-timestep supervision. Not used
  for novelty scoring at inference time — Stage 4 decodes z̃ to a grid,
  re-simulates exactly using simulator.py, computes signals from exact
  physics, and scores against sig_reference via trajectory-signature LOF.

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
- **Phase 2**: random negatives (any z from a different trajectory in the batch)
- **Phase 3**: hard negatives — the z from a different trajectory that is
  geometrically closest to the anchor in current latent space
- Hard negatives require a partially trained encoder to be meaningful; random
  negatives in Phase 2 provide early structural pressure without this risk
- Near lag k=5, far lag K=50, both fixed throughout training

### `data_loader.py`
PyTorch Dataset and DataLoader for the Stage 1 dataset.
- Load metadata npy files from `data/` at init (seeds, grids, signatures_norm,
  labels, buckets, sig_mean, sig_std). Large arrays (grids, signatures_norm)
  opened with mmap_mode='r' — no full load into RAM.
- `__getitem__` returns a lightweight dict: `grid_t0` (128×128 uint8),
  `sig_norm` (257, 10) float32, `trajectory_id` (int), `bucket` (int).
  No simulation happens inside `__getitem__`.
- **Batch-level simulation via collate function**: a custom collate function
  assembles a batch from `__getitem__` results and calls
  `simulator.simulate_batch(grids, steps=256)` once, producing
  `(B, 257, 128, 128)` uint8 in a single vectorized pass. This amortizes
  numpy overhead and is substantially faster than per-item simulation.
- The training loop receives `(B, 257, 128, 128)` uint8 trajectories and
  `(B, 257, 10)` float32 sig_norm. It encodes raw grids to z vectors at each
  rollout step — the dataloader never calls the encoder. Returning z vectors
  would be wrong because encoder weights change every step during training.
- **Stratified batch sampler**: maintains one index pool per behavioral bucket
  (dying / short / medium / long). Each batch draws proportionally from all
  four buckets, ensuring VICReg and the contrastive loss always see behavioral
  variety regardless of the underlying class imbalance.
- Support 90/10 train/validation split by seed index.
- No LRU cache — at N=1,500,000 any cache provides negligible hit rate and
  adds complexity without benefit.

### `train_core.py`
Training loop with 3-phase progressive rollout curriculum.

**Epoch structure**: one epoch = one complete shuffled pass over all 1,500,000
training seeds (46,875 steps at batch_size=32). Training runs for as many
epochs as needed until Phase 3 completion criteria are met. Validation runs
every 500 steps. Checkpoints saved every 1,000 steps.

**Resume behaviour**: on startup, automatically loads the latest checkpoint in
`checkpoints/` if one exists. Pass `--fresh` to start from scratch regardless.
This makes crash recovery transparent — just restart the process.

**DataLoader parallelism**: `num_workers=4` worker processes each run
`simulate_batch` independently, parallelising simulation across cores.
Main process uses 4 PyTorch threads for model ops. Workers use 1 thread each
(`worker_init_fn` sets `OMP_NUM_THREADS=1`). `persistent_workers=True` avoids
worker respawn overhead between epochs.

**Optimizer**: AdamW, lr=3e-4, weight_decay=1e-4. Gradient clipping: max_norm=1.0.

**LR schedule — per-k independent ReduceLROnPlateau**:
Each k level has its own `ReduceLROnPlateau(mode='min', patience=10, factor=0.5,
min_lr=1e-5)` scheduler tracking that level's mech-loss history independently.

- When k=1 stagnates (10 val intervals without improvement), k=1's LR halves.
  k=2's LR is unaffected.
- When k_max advances to a new level, that level starts fresh at lr=3e-4 with
  a new scheduler. All other levels keep their independent LR and history.
- Phase transitions reset all per-k LRs to 3e-4 and clear all scheduler state.

Rationale: CosineAnnealingLR caused catastrophic regression — it cycled LR
back to 3e-4 at step 200k, destroying learned representations. ReduceLROnPlateau
is monotonically decreasing. Per-k independence prevents a new (harder) task
from stalling at a low LR inherited from a completed (easier) task.

**VICReg**: computed every step on the current z_0 batch (B=32). No gradient
accumulation buffer — per-step computation on B=32 is sufficient to maintain
variance/covariance constraints.

Phase 1 (mechanics only, progressive rollout k_max=1→96, teacher forcing=100%):
- Loss: L_mechanics × 1.0 + VICReg × 0.01
- Rollout schedule: begin at k_max=1; advance through levels
  [1,2,4,8,16,32,48,64,96] once alive-cell accuracy at current k_max exceeds
  threshold(k_max) for 2 consecutive validation checks
- Randomly sample k ∈ [1, k_max] per training step
- Teacher forcing: always use real z_t as f_θ input
- Phase ends and Phase 2 begins when k_max first reaches 96

Phase 2 (add trajectory head, progressive rollout k_max=96→192, teacher
forcing 100%→0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.2 + L_contrastive × 0.1 + VICReg × 0.05
- Continue rollout progression from k_max=96 through [128, 192]
- Apply traj_head(z_k) at each rollout step; supervise against sig_norm[k]
- Teacher forcing decays linearly from 100% to 0% over Phase 2 training steps
- Contrastive loss: random negatives (no hard negative mining yet)
- Phase ends and Phase 3 begins when k_max first reaches 192

Phase 3 (full joint, progressive rollout k_max=192→256, teacher forcing=0%):
- Loss: L_mechanics × 1.0 + L_trajectory × 0.5 + L_contrastive × 0.2 + VICReg × 0.05
- Continue rollout progression from k_max=192 to k_max=256
- Hard negatives: enabled
- Stop when t-SNE of encoded held-out patterns shows cluster separation

Rollout advancement threshold (all phases):
  threshold(k_max) = max(0.975 - 0.04 × (k_max / 256), 0.95)
2 consecutive validation checks above threshold required to advance.

---

## Resolved Implementation Decisions

### Batch size and memory (Q1 — resolved)

- **batch_size = 32** per gradient step
- VICReg computed every step on B=32 z_0 vectors — no accumulation buffer.
  Per-step B=32 is sufficient for variance/covariance constraints.
- Trajectories are retained as **uint8 numpy arrays** from simulate_batch;
  only the specific frames needed for the current rollout step (grid_0 and grid_k)
  are converted to float32 tensors. This prevents peak RAM from scaling with T.
- Peak RAM budget: ~134 MB for uint8 trajectory batch + float32 model activations
  for two frames + model weights. Well within 32 GB.

### DataLoader simulation strategy (Q4 — resolved)

- `__getitem__` returns `(grid_t0, sig_norm, trajectory_id, bucket)` without
  simulating. A custom **collate function** calls
  `simulator.simulate_batch(grids, steps=256)` once per batch, producing
  `(B, 257, 128, 128)` uint8 in a single vectorized pass.
- No LRU cache. At N=1,500,000 any cache provides negligible hit rate.
- Stratified batch sampler ensures each batch draws proportionally from all
  four behavioral buckets regardless of class imbalance.

### Phase advancement confidence (Q5 — resolved)

- **2 consecutive validation checks** above threshold required before advancing
  rollout depth or transitioning phases.

### Optimizer, LR, schedule (Q2 — resolved)

- **AdamW**, lr=3e-4, weight_decay=1e-4
- **ReduceLROnPlateau** per k level, independently. patience=10 val intervals,
  factor=0.5, min_lr=1e-5. Resets on phase transition. New k levels start at
  3e-4. See LR schedule section in train_core.py notes above.
- Gradient clipping: max_norm=1.0

### Epoch structure and training budget (Q3 — resolved)

- Full dataset epochs: 1,500,000 seeds, 46,875 steps/epoch at batch_size=32
- No step ceiling per phase — training continues until criteria are met
- Validation every 500 steps; checkpoint every 1,000 steps
- Rollout advancement threshold: threshold(k) = max(0.99 - 0.04×(k/256), 0.95)
- Phase 1→2 at k_max=96; Phase 2→3 at k_max=192

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
