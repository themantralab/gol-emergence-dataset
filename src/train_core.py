"""
Stage 2 training loop — Core World Model.

Three-phase progressive rollout curriculum:
  Phase 1: mechanics only,  k_max 1  -> 96,  teacher forcing 100%
  Phase 2: + trajectory head + contrastive (random neg), k_max 96 -> 192, TF 90%->0%,
           auxiliary losses ramped from 0 to full weight over PHASE2_RAMP_STEPS (30k steps)
  Phase 3: full joint + hard negatives,  k_max 192 -> 256, teacher forcing 0%

k_max advances through fixed levels when alive-cell accuracy at current k_max
exceeds threshold(k_max) = max(0.99 - 0.04*(k_max/256), 0.95) for 2 consecutive
validation checks.

Phase transitions are tied to k_max milestones:
  Phase 1 -> 2 when k_max first reaches 96
  Phase 2 -> 3 when k_max first reaches 192

VICReg regularises z_0 every step (B=32 batch).

Optimizer: AdamW, lr=3e-4, weight_decay=1e-4, single ReduceLROnPlateau(mode='max',
           patience=10, factor=0.5, min_lr=1e-5) stepping on val accuracy, reset on
           k_max advance and phase transitions. Gradient clip max_norm=1.0.
"""

import argparse
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Encoder, Decoder, Transition, TrajectoryHead
from model.vicreg import VICReg
from model.contrastive import TemporalContrastiveLoss
from data_loader import make_loaders
import simulator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_DIM   = 256
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
LOG_EVERY    = 100     # train metrics written every N steps
VAL_EVERY    = 500     # validation check every N steps
CKPT_EVERY   = 1000   # checkpoint every N steps
VAL_FRACTION = 0.1
NUM_WORKERS  = 2   # DataLoader workers for parallel simulate_batch
TORCH_THREADS    = 6    # PyTorch threads in main process (workers get 1 each)
ALIVE_POS_WEIGHT = 50.0 # BCE upweight for alive cells (dead:alive ratio ~378:1 in data;
                         # natural weight ~378 is too aggressive, 50 gives ~13% gradient share)

# k_max advancement levels
K_LEVELS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]

# Phase transition milestones
PHASE2_K = 96
PHASE3_K = 192

# Contrastive lag parameters
NEAR_K = 5
FAR_K  = 50

# Loss weights per phase (Phase 2 target weights — ramped in gradually)
PHASE_WEIGHTS = {
    1: dict(mechanics=1.0, trajectory=0.0, contrastive=0.0, vicreg=0.01),
    2: dict(mechanics=1.0, trajectory=0.2, contrastive=0.1, vicreg=0.05),
    3: dict(mechanics=1.0, trajectory=0.5, contrastive=0.2, vicreg=0.05),
}

# Steps to linearly ramp Phase 2 auxiliary losses from 0 to full weight.
# Prevents gradient conflict from hitting the encoder all at once at phase entry.
PHASE2_RAMP_STEPS = 30_000


def get_loss_weights(phase: int, step: int, phase2_start_step: int | None) -> dict:
    """Return loss weights, with Phase 2 auxiliary losses ramped in gradually."""
    if phase != 2 or phase2_start_step is None:
        return PHASE_WEIGHTS[phase]
    frac = min(1.0, (step - phase2_start_step) / max(1, PHASE2_RAMP_STEPS))
    w1, w2 = PHASE_WEIGHTS[1], PHASE_WEIGHTS[2]
    return dict(
        mechanics=w2['mechanics'],
        trajectory=w2['trajectory']  * frac,
        contrastive=w2['contrastive'] * frac,
        vicreg=w1['vicreg'] + (w2['vicreg'] - w1['vicreg']) * frac,
    )


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Appends JSON-lines records to metrics.jsonl in the checkpoint directory.

    Each record has a 'type' field: 'train', 'val', or 'event'.
    Multiple runs append to the same file; run_metadata.json records each
    run's configuration and demarcates runs by run_id.
    """

    def __init__(self, log_dir: Path):
        self._path = log_dir / 'metrics.jsonl'
        self._fh   = open(self._path, 'a')

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + '\n')
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def write_run_metadata(log_dir: Path, args: argparse.Namespace, run_id: str, resume_step: int | None) -> None:
    """
    Write run_metadata.json capturing the full configuration for this training run.
    Each resume creates a new file under run_id so configs never overwrite each other.
    """
    meta = {
        'run_id':        run_id,
        'start_time':    time.strftime('%Y-%m-%dT%H:%M:%S'),
        'hostname':      socket.gethostname(),
        'git_sha':       _git_sha(),
        'resumed_from':  args.resume,
        'resume_step':   resume_step,
        'args':          vars(args),
        'constants': {
            'LATENT_DIM':    LATENT_DIM,
            'BATCH_SIZE':    BATCH_SIZE,
            'LR':            LR,
            'WEIGHT_DECAY':  WEIGHT_DECAY,
            'GRAD_CLIP':     GRAD_CLIP,
            'LOG_EVERY':     LOG_EVERY,
            'VAL_EVERY':     VAL_EVERY,
            'CKPT_EVERY':    CKPT_EVERY,
            'VAL_FRACTION':  VAL_FRACTION,
            'K_LEVELS':      K_LEVELS,
            'PHASE2_K':      PHASE2_K,
            'PHASE3_K':      PHASE3_K,
            'NEAR_K':        NEAR_K,
            'FAR_K':         FAR_K,
            'PHASE_WEIGHTS':    PHASE_WEIGHTS,
            'ALIVE_POS_WEIGHT': ALIVE_POS_WEIGHT,
        },
    }
    path = log_dir / f'run_metadata_{run_id}.json'
    with open(path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'  [meta] {path.name}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def advancement_threshold(k_max: int, phase: int) -> float:
    """Minimum alive-cell accuracy required to advance from k_max to next level."""
    if phase == 1:
        return max(0.95 - 0.04 * (k_max / 256), 0.90)
    elif phase == 2:
        return max(0.85 - 0.04 * (k_max / 256), 0.80)
    else:
        return max(0.90 - 0.04 * (k_max / 256), 0.85)


def alive_cell_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Accuracy on alive cells only.

    Args:
        logits:  (B, 1, H, W) or (B, H, W) raw decoder output
        targets: (B, H, W) uint8 ground truth
    Returns:
        fraction of alive (target==1) cells correctly predicted
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    pred  = (logits > 0).long()
    alive = targets.long()
    mask  = alive == 1
    if mask.sum() == 0:
        return 1.0
    return (pred[mask] == alive[mask]).float().mean().item()


def encode_grid(encoder: Encoder, grid: torch.Tensor) -> torch.Tensor:
    """Encode a (B, H, W) uint8 grid to z (B, latent_dim)."""
    return encoder(grid.unsqueeze(1).float())


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def build_rollout(
    encoder:    Encoder,
    transition: Transition,
    traj:       torch.Tensor,
    k:          int,
    p_teacher:  float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a k-step latent rollout from t=0.

    Teacher forcing: at each step t, with probability p_teacher use the real
    encoded grid z_t; otherwise use the predicted z_t from f_θ. Applied
    independently at each step.

    Args:
        traj:      (B, T, H, W) uint8 trajectory from the batch
        k:         rollout depth to unroll
        p_teacher: probability of using real z_t at each step [0, 1]
    Returns:
        z_traj: (B, k+1, latent_dim) latent trajectory (z_0 through z_k)
        z0:     (B, latent_dim)      z_0 (in computation graph for VICReg)
    """
    B = traj.shape[0]
    z_pred = encode_grid(encoder, traj[:, 0])   # z_0, always from real grid
    z0     = z_pred
    z_list = [z_pred]

    for t in range(1, k + 1):
        z_next_pred = transition(z_pred)

        if p_teacher > 0.0 and t < traj.shape[1]:
            use_real = torch.rand(B, device=z_pred.device) < p_teacher
            if use_real.any():
                z_real = encode_grid(encoder, traj[:, t])
                mask   = use_real.unsqueeze(1).float()
                z_pred = mask * z_real + (1 - mask) * z_next_pred
            else:
                z_pred = z_next_pred
        else:
            z_pred = z_next_pred

        z_list.append(z_pred)

    z_traj = torch.stack(z_list, dim=1)  # (B, k+1, latent_dim)
    return z_traj, z0


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def mechanics_loss(
    decoder: Decoder,
    z_traj:  torch.Tensor,
    traj:    torch.Tensor,
    k:       int,
) -> torch.Tensor:
    """
    BCE reconstruction loss at rollout depth k.

    Decodes z_k = z_traj[:, k] and compares against the real grid at step k.
    Applied at the sampled depth only (not every step) to keep compute bounded.

    pos_weight upweights alive cells to counteract the severe class imbalance
    (~378 dead cells per alive cell). Without it, the model minimises loss by
    predicting all-dead, producing near-zero BCE but ~0% alive-cell accuracy.
    """
    z_k      = z_traj[:, k]
    logits_k = decoder(z_k)                       # (B, 1, H, W)
    target_k = traj[:, k].float()                 # (B, H, W)
    pw = torch.tensor([ALIVE_POS_WEIGHT], device=logits_k.device)
    return F.binary_cross_entropy_with_logits(
        logits_k.squeeze(1), target_k,
        pos_weight=pw,
        reduction='mean',
    )


def trajectory_loss(
    traj_head: TrajectoryHead,
    z_traj:    torch.Tensor,
    sig_norm:  torch.Tensor,
    k:         int,
) -> torch.Tensor:
    """
    MSE between trajectory head predictions and ground-truth normalised signals.

    Averaged across all k+1 steps in the rollout (includes z_0).

    Args:
        z_traj:   (B, k+1, latent_dim)
        sig_norm: (B, 257, 10) pre-computed normalised signals
        k:        rollout depth
    """
    z_steps      = z_traj[:, :k + 1]              # (B, k+1, latent_dim)
    pred_signals = traj_head(z_steps)             # (B, k+1, 10)
    true_signals = sig_norm[:, :k + 1]            # (B, k+1, 10)
    return F.mse_loss(pred_signals, true_signals)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    encoder:    Encoder,
    decoder:    Decoder,
    transition: Transition,
    val_loader,
    k_max:      int,
    device:     torch.device,
    n_batches:  int = 50,
) -> tuple[float, dict[int, float]]:
    """
    Estimate alive-cell accuracy at current k_max on the validation set.

    Runs n_batches batches using pure free rollout (p_teacher=0).

    Returns:
        (overall_acc, per_bucket_acc)  where per_bucket_acc maps bucket_id -> accuracy.
    """
    encoder.eval(); decoder.eval(); transition.eval()

    all_accs: list[float] = []
    bucket_accs: dict[int, list[float]] = {}

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        traj    = batch['trajectories'].to(device)
        buckets = batch['buckets']
        k       = min(k_max, traj.shape[1] - 1)

        z_traj, _ = build_rollout(encoder, transition, traj, k, p_teacher=0.0)
        z_k       = z_traj[:, k]
        logits_k  = decoder(z_k)
        target_k  = traj[:, k]

        # Overall accuracy across batch
        all_accs.append(alive_cell_accuracy(logits_k, target_k))

        # Per-item accuracy (one value per sample)
        if logits_k.dim() == 4:
            lk = logits_k.squeeze(1)
        else:
            lk = logits_k
        pred  = (lk > 0).long()
        alive = target_k.long()
        for j in range(traj.shape[0]):
            mask = alive[j] == 1
            if mask.sum() == 0:
                acc_j = 1.0
            else:
                acc_j = (pred[j][mask] == alive[j][mask]).float().mean().item()
            b = int(buckets[j].item())
            bucket_accs.setdefault(b, []).append(acc_j)

    encoder.train(); decoder.train(); transition.train()

    overall = float(np.mean(all_accs)) if all_accs else 0.0
    per_bucket = {b: float(np.mean(v)) for b, v in bucket_accs.items()}
    return overall, per_bucket


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_dir:          Path,
    encoder:           Encoder,
    decoder:           Decoder,
    transition:        Transition,
    traj_head:         TrajectoryHead,
    optimizer:         torch.optim.Optimizer,
    scheduler:         ReduceLROnPlateau,
    lr_current:        float,
    step:              int,
    phase:             int,
    k_max:             int,
    phase2_start_step: int | None = None,
    tag:               str = '',
):
    path = ckpt_dir / f'core_step{step:07d}{("_" + tag) if tag else ""}.pt'
    torch.save({
        'step':              step,
        'phase':             phase,
        'k_max':             k_max,
        'phase2_start_step': phase2_start_step,
        'encoder':           encoder.state_dict(),
        'decoder':           decoder.state_dict(),
        'transition':        transition.state_dict(),
        'traj_head':         traj_head.state_dict(),
        'optimizer':         optimizer.state_dict(),
        'scheduler':         scheduler.state_dict(),
        'lr_current':        lr_current,
    }, path)
    print(f'  [ckpt] saved {path.name}')


def load_checkpoint(
    path:      str,
    encoder:   Encoder,
    decoder:   Decoder,
    transition: Transition,
    traj_head: TrajectoryHead,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
) -> tuple[int, int, int, float, int | None]:
    """Load checkpoint. Returns (step, phase, k_max, lr_current, phase2_start_step).

    Backward-compatible: old checkpoints without phase2_start_step return None,
    which the caller must handle (e.g. estimate from known Phase 2 start).
    """
    ckpt = torch.load(path, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    transition.load_state_dict(ckpt['transition'])
    traj_head.load_state_dict(ckpt['traj_head'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    lr_current = ckpt.get('lr_current', LR)
    phase2_start_step = ckpt.get('phase2_start_step', None)
    return ckpt['step'], ckpt['phase'], ckpt['k_max'], lr_current, phase2_start_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the highest-step checkpoint in ckpt_dir, or None if none exist."""
    ckpts = sorted(ckpt_dir.glob('core_step*.pt'))
    return ckpts[-1] if ckpts else None


def _worker_init_fn(worker_id: int) -> None:
    """Limit each DataLoader worker to 1 thread so 4 workers + 4 main-process
    threads = 8 cores total without oversubscription."""
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS']   = '1'
    os.environ['MKL_NUM_THREADS']   = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def train(args):
    # Pin main-process torch ops to TORCH_THREADS; workers use 1 thread each
    n_workers = args.num_workers
    main_threads = max(1, os.cpu_count() - n_workers) if n_workers > 0 else os.cpu_count()
    torch.set_num_threads(min(main_threads, TORCH_THREADS))
    torch.set_num_interop_threads(2)

    device   = torch.device('cpu')
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Models ---
    encoder    = Encoder(LATENT_DIM).to(device)
    decoder    = Decoder(LATENT_DIM).to(device)
    transition = Transition(LATENT_DIM, hidden_dim=LATENT_DIM * 2).to(device)
    traj_head  = TrajectoryHead(LATENT_DIM, hidden_dim=LATENT_DIM).to(device)

    vicreg      = VICReg(LATENT_DIM).to(device)
    contrastive = TemporalContrastiveLoss(near_k=NEAR_K, far_K=FAR_K)

    params = (
        list(encoder.parameters())    +
        list(decoder.parameters())    +
        list(transition.parameters()) +
        list(traj_head.parameters())
    )
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # --- Learning rate management ---
    # Single ReduceLROnPlateau steps on val accuracy (mode='max').
    # Reset to full LR on k_max advance and phase transitions so each new
    # curriculum stage starts fresh without carrying a depleted LR.
    lr_current = LR

    def make_scheduler() -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-5
        )

    scheduler = make_scheduler()

    # --- Data ---
    train_loader, val_loader = make_loaders(
        args.data_dir,
        batch_size=BATCH_SIZE,
        num_workers=n_workers,
        val_fraction=VAL_FRACTION,
        worker_init_fn=_worker_init_fn if n_workers > 0 else None,
        persistent_workers=True,
    )

    # --- State ---
    step         = 0
    phase        = 1
    k_max        = K_LEVELS[0]
    k_level_idx  = 0
    above_thresh = 0
    p_teacher    = 1.0
    val_acc_buf: list[float] = []   # rolling window for advancement decisions

    phase2_start_step  = None
    phase2_total_steps = args.phase2_steps


    # --- Resume (explicit path, or auto-detect latest checkpoint) ---
    resume_step = None
    resume_path = args.resume
    if resume_path is None and not args.fresh:
        resume_path = latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f'Auto-resuming from latest checkpoint: {resume_path.name}')

    if resume_path:
        step, phase, k_max, lr_current, ckpt_phase2_start = load_checkpoint(
            resume_path, encoder, decoder, transition, traj_head,
            optimizer, scheduler
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr_current
        resume_step = step
        k_level_idx = K_LEVELS.index(k_max)
        p_teacher   = 0.0 if phase == 3 else 1.0
        if phase == 2:
            if ckpt_phase2_start is not None:
                phase2_start_step = ckpt_phase2_start
            else:
                # Old checkpoint: estimate from known Phase 2 start milestone
                phase2_start_step = 57500
                print(f'  [warn] phase2_start_step not in checkpoint, defaulting to {phase2_start_step}')
        print(f'Resumed from {resume_path} at step={step} phase={phase} k_max={k_max} lr={lr_current:.2e} TF_start={phase2_start_step}')

    # --- Metadata + logger ---
    run_id = f'{time.strftime("%Y%m%d_%H%M%S")}_{os.urandom(2).hex()}'
    write_run_metadata(ckpt_dir, args, run_id, resume_step)
    logger = MetricsLogger(ckpt_dir)
    logger.log({'type': 'event', 'event': 'train_start', 'run_id': run_id,
                'step': step, 'phase': phase, 'k_max': k_max,
                'wall_time_s': 0.0})

    train_start = time.time()
    step_start  = time.time()

    print(f'Threads: main={torch.get_num_threads()} torch  workers={n_workers}×1  cores={os.cpu_count()}')
    print(f'Starting training — phase={phase}, k_max={k_max}, steps_per_epoch≈{len(train_loader)}')
    print(f'Metrics -> {ckpt_dir / "metrics.jsonl"}')

    # --- Main loop ---
    encoder.train(); decoder.train(); transition.train(); traj_head.train()

    while True:
        for batch in train_loader:
            step += 1

            traj     = batch['trajectories'].to(device)  # (B, 257, 128, 128) uint8
            sig_norm = batch['sig_norm'].to(device)       # (B, 257, 10)

            # --- Teacher forcing probability ---
            if phase == 1:
                p_teacher = 0.9   # 10% free-rollout exposure prevents specialisation
            elif phase == 2:
                if phase2_start_step is None:
                    phase2_start_step = step
                frac = min(1.0, (step - phase2_start_step) / max(1, phase2_total_steps))
                p_teacher = 0.9 * (1.0 - frac)
            else:
                p_teacher = 0.0

            # --- Sample rollout depth k ---
            k = int(torch.randint(1, k_max + 1, (1,)).item())

            # --- Build rollout ---
            z_traj, z0 = build_rollout(encoder, transition, traj, k, p_teacher)

            # --- Losses ---
            w = get_loss_weights(phase, step, phase2_start_step)

            loss_mech = mechanics_loss(decoder, z_traj, traj, k)
            loss = w['mechanics'] * loss_mech

            if w['trajectory'] > 0:
                loss_traj = trajectory_loss(traj_head, z_traj, sig_norm, k)
                loss = loss + w['trajectory'] * loss_traj
                loss_traj_val: float | None = loss_traj.item()
            else:
                loss_traj_val = None

            # Contrastive: requires T >= FAR_K + NEAR_K + 1 = 56
            if w['contrastive'] > 0 and (k + 1) >= (FAR_K + NEAR_K + 1):
                hard_neg    = (phase == 3)
                loss_contr  = contrastive(z_traj, hard_negatives=hard_neg)
                loss        = loss + w['contrastive'] * loss_contr
                loss_contr_val: float | None = loss_contr.item()
            else:
                loss_contr_val = None

            # VICReg: regularise z_0 distribution every step
            loss_vicreg = vicreg(z0)
            loss        = loss + w['vicreg'] * loss_vicreg

            # --- NaN / inf guard ---
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                logger.log({
                    'type': 'event', 'event': 'non_finite_loss', 'run_id': run_id,
                    'step': step, 'phase': phase, 'k_max': k_max, 'k': k,
                    'loss_total':   loss_val,
                    'loss_mech':    loss_mech.item(),
                    'loss_traj':    loss_traj_val,
                    'loss_contr':   loss_contr_val,
                    'loss_vicreg':  loss_vicreg.item(),
                    'wall_time_s':  time.time() - train_start,
                })
                raise RuntimeError(
                    f'Non-finite loss at step {step}: total={loss_val:.4f} '
                    f'mech={loss_mech.item():.4f} vicreg={loss_vicreg.item():.4f}'
                )

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params, GRAD_CLIP).item()
            optimizer.step()

            # --- Step timing ---
            now        = time.time()
            step_time  = now - step_start
            wall_time  = now - train_start
            step_start = now

            # --- Logging ---
            if step % LOG_EVERY == 0:
                z0_d         = z0.detach()
                z0_norm_mean = z0_d.norm(dim=1).mean().item()
                z0_std_min   = z0_d.std(dim=0).min().item()

                record = {
                    'type':          'train',
                    'run_id':        run_id,
                    'step':          step,
                    'phase':         phase,
                    'k_max':         k_max,
                    'k':             k,
                    'p_teacher':     round(p_teacher, 4),
                    'p2_ramp':       round(min(1.0, (step - phase2_start_step) / max(1, PHASE2_RAMP_STEPS)), 4) if phase == 2 and phase2_start_step else None,
                    'loss_total':    round(loss_val,               4),
                    'loss_mech':     round(loss_mech.item(),       4),
                    'loss_traj':     round(loss_traj_val,    4) if loss_traj_val    is not None else None,
                    'loss_contr':    round(loss_contr_val,   4) if loss_contr_val   is not None else None,
                    'loss_vicreg':   round(loss_vicreg.item(),     4),
                    'lr':            round(lr_current,             8),
                    'grad_norm':     round(grad_norm,              4),
                    'z0_norm_mean':  round(z0_norm_mean,           4),
                    'z0_std_min':    round(z0_std_min,             4),
                    'step_time_s':   round(step_time,              3),
                    'wall_time_s':   round(wall_time,              1),
                }
                logger.log(record)

                print(
                    f'step={step:7d} ph={phase} k_max={k_max:3d} k={k:3d} '
                    f'TF={p_teacher:.2f} '
                    f'mech={loss_mech.item():.4f} '
                    f'traj={loss_traj_val if loss_traj_val is not None else "-":>7} '
                    f'contr={loss_contr_val if loss_contr_val is not None else "-":>7} '
                    f'vicreg={loss_vicreg.item():.2f} '
                    f'gnorm={grad_norm:.3f} '
                    f'total={loss_val:.4f}'
                )

            # --- Checkpoint ---
            if step % CKPT_EVERY == 0:
                save_checkpoint(
                    ckpt_dir, encoder, decoder, transition, traj_head,
                    optimizer, scheduler, lr_current, step, phase, k_max,
                    phase2_start_step=phase2_start_step,
                )
                logger.log({'type': 'event', 'event': 'checkpoint', 'run_id': run_id,
                            'step': step, 'phase': phase, 'k_max': k_max,
                            'wall_time_s': round(time.time() - train_start, 1)})

            # --- Validation and advancement ---
            if step % VAL_EVERY == 0:
                val_start = time.time()
                acc, bucket_acc = validate(
                    encoder, decoder, transition, val_loader, k_max, device
                )
                val_time = time.time() - val_start
                thresh   = advancement_threshold(k_max, phase)

                # Compute rolling average before logging/printing
                val_acc_buf.append(acc)
                if len(val_acc_buf) > 3:
                    val_acc_buf.pop(0)
                smooth_acc = float(np.mean(val_acc_buf))

                val_record = {
                    'type':          'val',
                    'run_id':        run_id,
                    'step':          step,
                    'phase':         phase,
                    'k_max':         k_max,
                    'val_acc':       round(acc,      4),
                    'threshold':     round(thresh,   4),
                    'above_thresh':  above_thresh,
                    'smooth_acc':    round(float(np.mean(val_acc_buf)) if val_acc_buf else acc, 4),
                    'bucket_acc':    {str(b): round(v, 4) for b, v in sorted(bucket_acc.items())},
                    'val_time_s':    round(val_time, 2),
                    'wall_time_s':   round(time.time() - train_start, 1),
                }
                logger.log(val_record)

                bucket_str = '  '.join(
                    f'b{b}={v:.4f}' for b, v in sorted(bucket_acc.items())
                )
                print(
                    f'  [val] step={step} k_max={k_max} acc={acc:.4f} smooth={smooth_acc:.4f} '
                    f'thresh={thresh:.4f} above={above_thresh}/2  {bucket_str}'
                )

                # Step scheduler only in Phase 3 (TF=0, val acc is stable signal).
                # During Phase 2 TF decay, val acc oscillates structurally and
                # would cause premature LR reductions unrelated to real plateaus.
                if phase == 3:
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_current
                    scheduler.step(acc)
                    lr_current = optimizer.param_groups[0]['lr']

                if smooth_acc >= thresh:
                    above_thresh += 1
                else:
                    above_thresh = 0

                if above_thresh >= 2:
                    above_thresh = 0
                    if k_level_idx + 1 < len(K_LEVELS):
                        prev_k      = k_max
                        k_level_idx += 1
                        k_max       = K_LEVELS[k_level_idx]
                        print(f'  [advance] k_max {prev_k} -> {k_max}')
                        val_acc_buf.clear()

                        # Reset LR, scheduler, and TF on each k_max advance
                        lr_current = LR
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr_current
                        scheduler = make_scheduler()
                        if phase == 2:
                            phase2_start_step = step
                        print(f'  [lr+TF] reset at k_max={k_max}  lr={LR:.2e}  TF->0.9')

                        logger.log({
                            'type': 'event', 'event': 'k_max_advance', 'run_id': run_id,
                            'step': step, 'k_max_prev': prev_k, 'k_max_new': k_max,
                            'lr_new': LR,
                            'wall_time_s': round(time.time() - train_start, 1),
                        })

                        # Phase transitions — reset all per-k LRs for new phase
                        if k_max == PHASE2_K and phase == 1:
                            phase = 2
                            phase2_start_step = step
                            lr_current = LR
                            for pg in optimizer.param_groups:
                                pg['lr'] = lr_current
                            scheduler = make_scheduler()
                            print(f'  [phase] -> Phase 2 (trajectory + contrastive random)')
                            logger.log({
                                'type': 'event', 'event': 'phase_transition', 'run_id': run_id,
                                'step': step, 'from_phase': 1, 'to_phase': 2, 'k_max': k_max,
                                'wall_time_s': round(time.time() - train_start, 1),
                            })

                        elif k_max == PHASE3_K and phase == 2:
                            phase     = 3
                            p_teacher = 0.0
                            lr_current = LR
                            for pg in optimizer.param_groups:
                                pg['lr'] = lr_current
                            scheduler = make_scheduler()
                            print(f'  [phase] -> Phase 3 (full joint, hard negatives)')
                            logger.log({
                                'type': 'event', 'event': 'phase_transition', 'run_id': run_id,
                                'step': step, 'from_phase': 2, 'to_phase': 3, 'k_max': k_max,
                                'wall_time_s': round(time.time() - train_start, 1),
                            })

                    else:
                        print(f'  [complete] k_max={k_max} reached 256 with acc={acc:.4f}')
                        logger.log({
                            'type': 'event', 'event': 'training_complete', 'run_id': run_id,
                            'step': step, 'k_max': k_max, 'final_acc': round(acc, 4),
                            'wall_time_s': round(time.time() - train_start, 1),
                        })

                if phase == 3 and k_max == 256:
                    print(
                        f'  [Phase 3 running] step={step} acc@256={acc:.4f} — '
                        f'stop when t-SNE shows cluster separation'
                    )

        print(f'--- epoch complete at step={step} ---')
        logger.log({'type': 'event', 'event': 'epoch_complete', 'run_id': run_id,
                    'step': step, 'phase': phase, 'k_max': k_max,
                    'wall_time_s': round(time.time() - train_start, 1)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Stage 2 Core World Model')
    parser.add_argument('--data-dir',       default='data',        help='Path to data/ directory')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Where to save checkpoints')
    parser.add_argument('--resume',  default=None, help='Path to specific checkpoint to resume from')
    parser.add_argument('--fresh',   action='store_true', help='Start from scratch even if checkpoints exist')
    parser.add_argument(
        '--phase2-steps', type=int, default=100_000,
        help='Steps over which teacher forcing decays from 0.9 to 0.0 in Phase 2'
    )
    parser.add_argument(
        '--num-workers', type=int, default=NUM_WORKERS,
        help='DataLoader worker processes for parallel simulation (default: 4)'
    )
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
