"""Stage 1 data pipeline: generate, simulate, and save the GoL training dataset."""

# Must be set before numpy is imported to prevent OpenBLAS+fork deadlock.
# Workers fork off and initialize OpenBLAS threads; when they exit, the parent's
# OpenBLAS semaphore state is corrupted, causing the next BLAS call to deadlock.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import label as _ndlabel
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulator

T          = 256
GRID_SIZE  = 128
SEED_SIZE  = 16
SEED_OFFSET = 56   # (128 - 16) // 2
N_WORKERS  = min(8, multiprocessing.cpu_count())
BATCH_SIZE = 200   # 200 × 257 × 128² ≈ 840 MB/worker
CLUSTER_K  = 4
LAGS       = [2, 4, 8, 16]   # temporal self-similarity lags
N_SIGS     = 6 + len(LAGS)   # P Δcx Δcy V E N_cc + one S_lag per lag = 10
SIG_NAMES  = ["P", "Δcx", "Δcy", "V", "E", "N_cc"] + [f"S_lag_{l}" for l in LAGS]

# Four density bands — each worker rotates through them uniformly per batch,
# so every band contributes exactly 25 % of generated seeds.
DENSITY_BANDS = [
    (0.03, 0.08),   # very sparse  — isolated patterns, rare spaceships
    (0.08, 0.15),   # sparse       — small pattern interactions
    (0.15, 0.22),   # medium       — complex oscillators
    (0.22, 0.30),   # denser       — methuselahs, large complexes
]

_STRUCT_8CONN = np.ones((3, 3), dtype=bool)
_row_idx, _col_idx = (a.astype(np.float32) for a in np.indices((GRID_SIZE, GRID_SIZE)))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def _phase(name):
    print(f"\n[{name}]  RSS={_rss_mb():.0f} MB", flush=True)
    return time.time()

def _done(t0):
    print(f"  done ({time.time() - t0:.1f}s)  RSS={_rss_mb():.0f} MB", flush=True)


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------


def measure_lifespans_batch(trajectories):
    """(N, T+1, H, W) uint8 → (N,) int32 lifespans.

    Uses cell-state XOR (not population) so constant-population oscillators
    like blinkers get the correct lifespan rather than landing in bucket 0.
    Computed step-by-step to avoid materialising an (N, T, H, W) bool
    intermediate (~6.7 GB per 8 workers at BATCH_SIZE=200).
    """
    N, T_plus1 = trajectories.shape[:2]
    T_len = T_plus1 - 1
    changed = np.empty((N, T_len), dtype=bool)
    for t in range(T_len):
        changed[:, t] = (trajectories[:, t + 1] != trajectories[:, t]).any(axis=(1, 2))
    has_change = changed.any(axis=1)
    last_idx = T_len - 1 - np.argmax(changed[:, ::-1], axis=1)
    return np.where(has_change, last_idx + 1, 0).astype(np.int32)


def assign_buckets(lifespans):
    """Vectorised: (N,) int32 → (N,) int32  (0=dying 1=short 2=medium 3=long)."""
    return np.digitize(lifespans, [20, 60, 130]).astype(np.int32)


def stratify_seeds(pool_seeds, pool_lifespans, pool_sigs, target_n, rng):
    """Equal representation across lifespan buckets: target_n // 4 per bucket.

    Sampling with replacement when a bucket has fewer than n_per_bucket seeds
    (warns to stdout). Resulting dataset has balanced behavioural coverage
    regardless of how the raw pool is distributed.
    """
    buckets = assign_buckets(pool_lifespans)
    n_per  = target_n // 4
    names  = ["dying", "short", "medium", "long"]
    parts  = []
    for b in range(4):
        b_idx = np.where(buckets == b)[0]
        if len(b_idx) == 0:
            print(f"  WARNING: bucket '{names[b]}' has 0 seeds, skipping", flush=True)
            continue
        replace = len(b_idx) < n_per
        if replace:
            print(f"  WARNING: bucket '{names[b]}' has {len(b_idx)} seeds, "
                  f"sampling with replacement to reach {n_per}", flush=True)
        parts.append(rng.choice(b_idx, size=n_per, replace=replace))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return pool_seeds[idx], pool_lifespans[idx], pool_sigs[idx], buckets[idx]


def bucket_report(buckets, lifespans, label=""):
    names = ["dying", "short", "medium", "long"]
    print(f"  Bucket distribution {label}:", flush=True)
    for b in range(4):
        mask = buckets == b
        n    = mask.sum()
        ls   = lifespans[mask]
        info = f"  lifespan [{ls.min()}, {ls.max()}]  mean={ls.mean():.1f}" if n else ""
        print(f"    {names[b]:8s}: {n:7d} ({100*n/len(buckets):5.1f}%){info}", flush=True)


# ---------------------------------------------------------------------------
# Signal extraction — 10 signals: [P, Δcx, Δcy, V, E, N_cc, S_lag_2/4/8/16]
# ---------------------------------------------------------------------------

def extract_signature(trajectory):
    """(T+1, H, W) uint8 → (T+1, 10) float32."""
    T_len  = len(trajectory)
    traj_f = trajectory.astype(np.float32)
    pops   = traj_f.sum(axis=(1, 2))
    alive  = pops > 0
    p_safe = np.where(alive, pops, 1.0)

    cy = (_row_idx * traj_f).sum(axis=(1, 2)) / p_safe
    cx = (_col_idx * traj_f).sum(axis=(1, 2)) / p_safe

    cy_bc = cy[:, np.newaxis, np.newaxis]
    cx_bc = cx[:, np.newaxis, np.newaxis]
    V = ((_row_idx - cy_bc) ** 2 + (_col_idx - cx_bc) ** 2) * traj_f
    V = V.sum(axis=(1, 2)) / p_safe

    E = np.zeros(T_len, dtype=np.float32)
    E[1:] = (trajectory[1:] != trajectory[:-1]).sum(axis=(1, 2))

    # N_cc: skip label() when E[t]==0 — grid unchanged → N_cc unchanged
    N_cc = np.zeros(T_len, dtype=np.float32)
    for t in range(T_len):
        if alive[t]:
            if t > 0 and E[t] == 0:
                N_cc[t] = N_cc[t - 1]
            else:
                _, n = _ndlabel(trajectory[t], structure=_STRUCT_8CONN)
                N_cc[t] = float(n)

    # S_lag at each lag: fractional overlap with the grid `lag` steps ahead
    s_lags = []
    for lag in LAGS:
        t2      = np.minimum(np.arange(T_len) + lag, T_len - 1)
        overlap = (trajectory & trajectory[t2]).sum(axis=(1, 2)).astype(np.float32)
        s_lags.append(overlap / p_safe)

    sig          = np.stack([pops, cx - cx[0], cy - cy[0], V, E, N_cc] + s_lags, axis=1)
    sig[~alive]  = 0.0
    return sig.astype(np.float32)


def extract_signatures_batch(trajectories):
    """(N, T+1, H, W) uint8 → (N, T+1, 10) float32."""
    N   = len(trajectories)
    out = np.empty((N, trajectories.shape[1], N_SIGS), dtype=np.float32)
    for i in range(N):
        out[i] = extract_signature(trajectories[i])
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_normalization_stats(signatures):
    """Per-signal mean and std from (N, T+1, 10) array."""
    flat     = signatures.reshape(-1, N_SIGS)
    sig_mean = flat.mean(axis=0).astype(np.float32)
    sig_std  = np.maximum(flat.std(axis=0), 1e-6).astype(np.float32)
    return sig_mean, sig_std


def normalize_signatures(signatures, sig_mean, sig_std):
    """Standardise and clip to [−5, 5] IN PLACE — overwrites input array.

    Avoids allocating a second (N, 257, 10) copy (~15 GB at 1.5M seeds).
    Caller must not use the original raw values after this call.
    """
    np.subtract(signatures, sig_mean, out=signatures)
    np.divide(signatures, sig_std, out=signatures)
    np.clip(signatures, -5.0, 5.0, out=signatures)
    return signatures  # same buffer, now normalized


# ---------------------------------------------------------------------------
# sig_reference: chunked FFT magnitude spectrum (phase-invariant novelty basis)
# ---------------------------------------------------------------------------

def compute_sig_reference(signatures_norm, chunk_size=100_000):
    """(N, T+1, 10) float32 → (N, 129*10=1290) float32 FFT magnitude.

    Processed in chunks of 100k to bound peak RAM to ~3 GB per chunk
    regardless of N — mandatory at 1M+ seeds.
    """
    N        = len(signatures_norm)
    n_bins   = signatures_norm.shape[1] // 2 + 1   # 129 for T=256
    n_sig    = signatures_norm.shape[2]
    out      = np.empty((N, n_bins * n_sig), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        chunk = signatures_norm[start:end]
        fft   = np.abs(np.fft.rfft(chunk, axis=1))  # (chunk, 129, 10)
        out[start:end] = fft.reshape(len(chunk), -1)
    return out


# ---------------------------------------------------------------------------
# Heuristic classifier (diagnostics only)
# ---------------------------------------------------------------------------

def classify_trajectory(sig, T, tol=0.05):
    """Classify raw (T+1, 10) sig → dying / still_life / oscillator / glider."""
    p_start = float(sig[0, 0])
    if p_start == 0:
        return "dying"

    pops      = sig[:, 0]
    drop_mask = pops < 0.9 * p_start
    active_end = int(np.argmax(drop_mask)) if drop_mask.any() else len(sig)
    active     = sig[:active_end]

    # Glider check on a trailing window — junk typically dies in <100 steps,
    # leaving a clean spaceship. Checking the last 100 steps avoids the noisy
    # early phase corrupting the CoM linearity test.
    WIN = min(100, len(sig) // 2)
    win = sig[-WIN:]
    wp  = win[:, 0]
    if wp.min() > 0:                       # still alive throughout the window
        wp_mean = float(wp.mean())
        wp_var  = (float(wp.max()) - float(wp.min())) / wp_mean
        if wp_var < 0.20:                  # stable population in window
            t_win  = np.arange(WIN, dtype=np.float64)
            cx_win = win[:, 1].astype(np.float64)
            cy_win = win[:, 2].astype(np.float64)
            disp_win = (abs(float(cx_win[-1] - cx_win[0]))
                        + abs(float(cy_win[-1] - cy_win[0])))
            if disp_win > 8.0:             # moved ≥ 8 cells within the window
                corr_x = (abs(float(np.corrcoef(t_win, cx_win)[0, 1]))
                          if cx_win.std() > 1e-6 else 0.0)
                corr_y = (abs(float(np.corrcoef(t_win, cy_win)[0, 1]))
                          if cy_win.std() > 1e-6 else 0.0)
                # N_cc <= 30 in window: allows residual background junk.
                # S_lag_4 > 0.05: the drifting core persists across 4-step lag.
                ncc_ok   = int(win[:, 5].max()) <= 30
                slag4_ok = float(win[:, 7].mean()) > 0.05
                if max(corr_x, corr_y) > 0.95 and ncc_ok and slag4_ok:
                    return "glider"

    if pops[-1] < tol * p_start:
        return "dying"

    half = T // 2
    if np.std(sig[half:, 3]) > tol or sig[half:, 4].mean() > tol:
        return "oscillator"

    return "still_life"


def classify_batch(signatures, T):
    return [classify_trajectory(signatures[i], T) for i in range(len(signatures))]


# ---------------------------------------------------------------------------
# Parallel pool generation
# ---------------------------------------------------------------------------

def _worker_generate(args):
    """Write each batch directly into pre-allocated mmap slices — zero RAM accumulation.

    Workers write into non-overlapping regions of shared mmap files so no
    synchronisation is needed. Peak RAM per worker = 1 batch of trajectories
    (BATCH_SIZE × 257 × 128² bytes ≈ 840 MB) plus tiny sigs/seeds.
    """
    import sys
    n_seeds, rng_seed, offset, T_steps, batch_size, seeds_path, ls_path, sigs_path, log_file = args
    if log_file:
        sys.stdout = open(log_file, "a", buffering=1)
        sys.stderr = sys.stdout
    rng  = np.random.default_rng(rng_seed)
    # Open the pre-allocated mmap files for writing into this worker's slice
    seeds_mm = np.load(seeds_path, mmap_mode="r+")
    ls_mm    = np.load(ls_path,    mmap_mode="r+")
    sigs_mm  = np.load(sigs_path,  mmap_mode="r+")
    done  = 0
    b_num = 0
    t0    = time.time()
    while done < n_seeds:
        bs       = min(batch_size, n_seeds - done)
        lo, hi   = DENSITY_BANDS[b_num % len(DENSITY_BANDS)]
        densities = rng.uniform(lo, hi, size=bs)
        seeds_b  = (rng.random((bs, SEED_SIZE, SEED_SIZE)) < densities[:, None, None]).astype(np.uint8)
        grids_b  = np.zeros((bs, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        grids_b[:, SEED_OFFSET:SEED_OFFSET + SEED_SIZE,
                   SEED_OFFSET:SEED_OFFSET + SEED_SIZE] = seeds_b
        trajs_b  = simulator.simulate_batch(grids_b, T_steps)
        del grids_b
        pos = offset + done
        seeds_mm[pos:pos + bs] = seeds_b
        ls_mm[pos:pos + bs]    = measure_lifespans_batch(trajs_b)
        sigs_mm[pos:pos + bs]  = extract_signatures_batch(trajs_b)
        del trajs_b
        done  += bs
        b_num += 1
        if done % 1_000 == 0 or done >= n_seeds:
            el   = time.time() - t0
            rate = done / el if el > 0 else 0
            eta  = (n_seeds - done) / max(rate, 0.01)
            print(f"  worker {rng_seed}: {done}/{n_seeds}  "
                  f"({rate:.0f} seeds/s  eta {eta:.0f}s)", flush=True)
    return True  # all data written to disk; no large payload returned


def generate_pool_parallel(pool_size, T_steps, rng_seed, n_workers=N_WORKERS,
                           data_dir=None, log_file=None):
    """Pre-allocate mmap pool files, spawn workers to fill them, return mmap arrays.

    Peak RAM during generation: n_workers × 1 batch of trajectories ≈ n_workers × 840 MB.
    No concatenation step — workers write directly to disk-backed mmap slices.
    """
    tmp_dir = (data_dir or Path("data")) / "_tmp_pool"
    tmp_dir.mkdir(exist_ok=True)
    seeds_path = str(tmp_dir / "pool_seeds.npy")
    ls_path    = str(tmp_dir / "pool_ls.npy")
    sigs_path  = str(tmp_dir / "pool_sigs.npy")

    # Pre-allocate on disk — workers will mmap-write into their slice
    np.lib.format.open_memmap(seeds_path, mode="w+", dtype=np.uint8,
                              shape=(pool_size, SEED_SIZE, SEED_SIZE))
    np.lib.format.open_memmap(ls_path,    mode="w+", dtype=np.int32,
                              shape=(pool_size,))
    np.lib.format.open_memmap(sigs_path,  mode="w+", dtype=np.float32,
                              shape=(pool_size, T_steps + 1, N_SIGS))

    n_per     = pool_size // n_workers
    remainder = pool_size % n_workers
    offsets   = [sum(n_per + (1 if j < remainder else 0) for j in range(i))
                 for i in range(n_workers)]
    w_args    = [
        (n_per + (1 if i < remainder else 0),
         rng_seed + i + 1,
         offsets[i],
         T_steps, BATCH_SIZE,
         seeds_path, ls_path, sigs_path,
         log_file)
        for i in range(n_workers)
    ]
    print(f"  Spawning {n_workers} workers ({n_per}–{n_per+1} seeds each, "
          f"mmap pool {pool_size:,})...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker_generate, a): i for i, a in enumerate(w_args)}
        for fut in as_completed(futs):
            i = futs[fut]
            fut.result()  # raises if worker raised
            print(f"  Worker {i+1}/{n_workers} done  RSS={_rss_mb():.0f} MB  "
                  f"t={time.strftime('%H:%M:%S')}", flush=True)

    # Return as read-only mmap — pages are loaded on demand, not all at once
    return (
        np.load(seeds_path, mmap_mode="r"),
        np.load(ls_path,    mmap_mode="r"),
        np.load(sigs_path,  mmap_mode="r"),
        tmp_dir,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _diag_lifespan_hist(ls_before, bk_before, ls_after, bk_after, diag_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bounds = [0, 20, 60, 130, T + 1]
    for ax, ls, title in [(axes[0], ls_before, "Before stratification"),
                          (axes[1], ls_after,  "After stratification")]:
        ax.hist(ls, bins=bounds, edgecolor="black", color="steelblue")
        ax.set_title(title); ax.set_xlabel("Lifespan"); ax.set_ylabel("Count")
        ax.set_xticks([0, 20, 60, 130, T])
    fig.suptitle("Lifespan distribution before/after stratification")
    fig.tight_layout()
    fig.savefig(diag_dir / "lifespan_hist.png", dpi=100)
    plt.close(fig)


def _diag_signal_distributions(signatures_raw, labels, rng, diag_dir):
    classes     = ["dying", "still_life", "oscillator", "glider"]
    labels_arr  = np.array(labels)
    max_cls     = 5_000
    fig, axes   = plt.subplots(1, N_SIGS, figsize=(3.5 * N_SIGS, 5))
    for si, (ax, sname) in enumerate(zip(axes, SIG_NAMES)):
        data, present = [], []
        for cls in classes:
            idx = np.where(labels_arr == cls)[0]
            if not len(idx): continue
            if len(idx) > max_cls: idx = rng.choice(idx, max_cls, replace=False)
            data.append(signatures_raw[idx, :, si].ravel())
            present.append(cls)
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=7)
        ax.set_title(sname, fontsize=8)
    fig.suptitle("Raw signal distributions by class")
    fig.tight_layout()
    fig.savefig(diag_dir / "signal_distributions.png", dpi=100)
    plt.close(fig)


def _diag_signal_samples(signatures_raw, labels, rng, diag_dir):
    classes    = ["dying", "still_life", "oscillator", "glider"]
    labels_arr = np.array(labels)
    selected   = []
    for cls in classes:
        idx = np.where(labels_arr == cls)[0]
        if not len(idx): continue
        for i in rng.choice(idx, min(3, len(idx)), replace=False):
            selected.append((cls, int(i)))
    if not selected: return
    fig, axes = plt.subplots(len(selected), N_SIGS,
                             figsize=(3.5 * N_SIGS, 3 * len(selected)))
    if len(selected) == 1: axes = axes[np.newaxis, :]
    for row, (cls, i) in enumerate(selected):
        sig = signatures_raw[i]
        for col, sname in enumerate(SIG_NAMES):
            axes[row, col].plot(sig[:, col], lw=0.7)
            if row == 0: axes[row, col].set_title(sname, fontsize=7)
            if col == 0: axes[row, col].set_ylabel(f"{cls}\n#{i}", fontsize=7)
    fig.suptitle("Signal samples by class (raw)")
    fig.tight_layout()
    fig.savefig(diag_dir / "signal_samples.png", dpi=100)
    plt.close(fig)


def _diag_normalization_check(signatures_raw, signatures_norm, rng, diag_dir):
    max_s  = 20_000
    idx    = rng.choice(len(signatures_raw), min(max_s, len(signatures_raw)), replace=False)
    fig, axes = plt.subplots(1, N_SIGS, figsize=(3.5 * N_SIGS, 4))
    for si, (ax, sname) in enumerate(zip(axes, SIG_NAMES)):
        raw  = signatures_raw[idx, :, si].ravel()
        norm = signatures_norm[idx, :, si].ravel()
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
        ax.hist(np.clip(raw, lo, hi), bins=50, alpha=0.5, label="raw",
                density=True, color="steelblue")
        ax.hist(np.clip(norm, -5, 5), bins=50, alpha=0.5, label="norm",
                density=True, color="orange")
        ax.set_title(sname, fontsize=8); ax.legend(fontsize=6)
    fig.suptitle("Raw vs normalized signal densities")
    fig.tight_layout()
    fig.savefig(diag_dir / "normalization_check.png", dpi=100)
    plt.close(fig)


def _diag_cluster_summary(sig_reference, cluster_labels, cluster_centroids, labels, diag_dir):
    k          = len(cluster_centroids)
    labels_arr = np.array(labels)
    classes    = ["dying", "still_life", "oscillator", "glider"]
    sizes      = [int((cluster_labels == c).sum()) for c in range(k)]
    dominant   = []
    for c in range(k):
        mask = cluster_labels == c
        if mask.any():
            counts = {cls: int((labels_arr[mask] == cls).sum()) for cls in classes}
            dominant.append(max(counts, key=counts.get))
        else:
            dominant.append("—")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(k), sizes, color="steelblue", edgecolor="black")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"{c}\n({dominant[c]})" for c in range(k)])
    ax.set_title(f"Cluster sizes (k={k}, dominant class labeled)")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(diag_dir / "cluster_summary.png", dpi=100)
    plt.close(fig)


def _diag_tsne(sig_reference, labels, rng, diag_dir):
    """t-SNE of sig_reference coloured by class.

    Proportional subsampling: each class receives slots proportional to its
    natural frequency in the dataset, capped at max_pts total. This faithfully
    represents the true data distribution — rare classes (e.g. gliders) appear
    as sparse points reflecting their actual prevalence. Legend counts show
    how many of each class are in the plot.
    """
    classes    = ["dying", "still_life", "oscillator", "glider"]
    colors     = ["red", "blue", "green", "orange"]
    labels_arr = np.array(labels)
    max_pts    = 10_000

    class_idx = {c: np.where(labels_arr == c)[0] for c in classes}
    n_total   = len(sig_reference)

    selected = []
    for cls in classes:
        idx_c = class_idx[cls]
        if len(idx_c) == 0:
            continue
        # Proportional quota: class share of dataset × max_pts, floored at 1
        take = max(1, int(round(max_pts * len(idx_c) / n_total)))
        take = min(take, len(idx_c))
        selected.append(rng.choice(idx_c, take, replace=False))

    idx = np.concatenate(selected) if selected else np.arange(min(max_pts, n_total))
    pts = sig_reference[idx]
    lbl = labels_arr[idx]
    n_total = len(sig_reference)
    suffix     = f" (subsample {len(idx)}/{n_total})" if len(idx) < n_total else ""

    # PCA → 50D before t-SNE
    if pts.shape[1] > 50:
        pts = PCA(n_components=50).fit_transform(pts)

    print("  Running t-SNE...", flush=True)
    emb = TSNE(n_components=2, random_state=17,
               perplexity=min(30, len(pts) - 1)).fit_transform(pts)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, col in zip(classes, colors):
        mask  = lbl == cls
        n_cls = int(mask.sum())
        if n_cls > 0:
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=4 if n_cls > 10 else 20, c=col,
                       label=f"{cls} (n={n_cls})", alpha=0.6)
        else:
            ax.scatter([], [], c=col, label=f"{cls} (n=0)", s=10)
    ax.legend()
    ax.set_title(f"t-SNE of sig_reference (FFT space) — proportional subsample{suffix}")
    fig.tight_layout()
    fig.savefig(diag_dir / "tsne_signatures.png", dpi=100)
    plt.close(fig)


def _diag_canonical_check(sig_mean, sig_std, diag_dir):
    patterns = {
        "block":   np.array([[1, 1], [1, 1]], dtype=np.uint8),
        "blinker": np.array([[1, 1, 1]], dtype=np.uint8),
        "glider":  np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
    }
    lines, classes = [], {}
    for name, pat in patterns.items():
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        grid[SEED_OFFSET:SEED_OFFSET + pat.shape[0],
             SEED_OFFSET:SEED_OFFSET + pat.shape[1]] = pat
        sig_raw  = extract_signature(simulator.simulate(grid, T))
        cls      = classify_trajectory(sig_raw, T)
        mean_norm = np.clip((sig_raw.mean(axis=0) - sig_mean) / sig_std, -5.0, 5.0)
        lines.append(f"{name}: class={cls}, mean_sig_norm={mean_norm.round(3).tolist()}")
        classes[name] = cls
    verdict      = "PASS" if classes["block"] != classes["glider"] else "FAIL"
    blinker_note = ("WARN: blinker same class as block"
                    if classes["blinker"] == classes["block"]
                    else "OK: blinker differs from block")
    lines += ["", f"Verdict: {verdict} (block={classes['block']}, glider={classes['glider']})",
              blinker_note]
    text = "\n".join(lines)
    print("\nCanonical check:", flush=True)
    print(text, flush=True)
    (diag_dir / "canonical_check.txt").write_text(text)
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys, json
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds",  type=int, default=1_500_000)
    parser.add_argument("--rng-seed", type=int, default=None,
                        help="Master RNG seed. Omit to generate from OS entropy.")
    parser.add_argument("--workers",  type=int, default=N_WORKERS)
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write all stdout/stderr to this file (appended)")
    args = parser.parse_args()

    if args.log_file:
        log_fh = open(args.log_file, "a", buffering=1)
        sys.stdout = log_fh
        sys.stderr = log_fh

    # Resolve RNG seed — use OS entropy if not supplied so every run is unique.
    # Always log to data/seeds.json for reproducibility.
    import secrets
    rng_seed = args.rng_seed if args.rng_seed is not None else secrets.randbits(32)

    N_SEEDS  = args.n_seeds
    rng      = np.random.default_rng(rng_seed)
    t_total  = time.time()
    data_dir = Path("data")
    diag_dir = data_dir / "diagnostics"
    data_dir.mkdir(exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    # Derive a separate seed for t-SNE subsampling from the master seed so it
    # is also deterministic and logged.
    tsne_seed = int(np.random.default_rng(rng_seed).integers(0, 2**31))

    # Log all seeds immediately — before any work starts — so the run is always
    # reproducible even if it crashes partway through.
    seeds_record = {
        "rng_seed":  rng_seed,
        "tsne_seed": tsne_seed,
        "n_seeds":   N_SEEDS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (data_dir / "seeds.json").write_text(json.dumps(seeds_record, indent=2))
    print(f"Seeds logged → data/seeds.json  "
          f"rng_seed={rng_seed}  tsne_seed={tsne_seed}", flush=True)

    # 1.2× pool — density bands give good bucket coverage without needing heavy oversampling.
    # Pool files are mmap'd to disk; peak RAM during generation ≈ n_workers × 840 MB.
    pool_size = int(N_SEEDS * 1.2)
    pool_gb   = pool_size * (T + 1) * N_SIGS * 4 / 1e9
    print(f"N_SEEDS={N_SEEDS}  pool_size={pool_size}  workers={args.workers}  "
          f"density_bands={len(DENSITY_BANDS)}  signals={N_SIGS}  "
          f"pool_sigs_disk={pool_gb:.1f} GB  t={time.strftime('%H:%M:%S')}", flush=True)
    print(f"Density bands: {DENSITY_BANDS}", flush=True)

    # --- Pool generation (mmap-based: workers write directly to disk) ---
    t0 = _phase("Pool generation")
    print(f"  Generating pool of {pool_size:,} seeds (mmap)...", flush=True)
    pool_seeds, pool_lifespans, pool_sigs, tmp_dir = generate_pool_parallel(
        pool_size, T, rng_seed, args.workers,
        data_dir=data_dir, log_file=args.log_file)
    pool_buckets = assign_buckets(pool_lifespans)
    bucket_report(pool_buckets, pool_lifespans, "(pool, before stratification)")
    _diag_ls_before = np.array(pool_lifespans)   # copy out of mmap for diagnostics
    _diag_bk_before = np.array(pool_buckets)
    _done(t0)

    # --- Stratification: equal representation per bucket ---
    t0 = _phase("Stratification")
    print(f"  Stratifying to {N_SEEDS:,} seeds (equal buckets = {N_SEEDS//4:,} each)...",
          flush=True)
    seeds, lifespans, signatures_raw, buckets = stratify_seeds(
        pool_seeds, pool_lifespans, pool_sigs, N_SEEDS, rng)
    bucket_report(buckets, lifespans, "(after stratification)")
    # Release mmap references and delete tmp pool files (~pool_gb GB freed from disk)
    del pool_seeds, pool_lifespans, pool_sigs, pool_buckets
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    _done(t0)

    # --- Grid reconstruction (saved immediately to free RAM before FFT) ---
    t0 = _phase("Grid reconstruction")
    N_actual = len(seeds)
    print(f"  Embedding {N_actual} seeds into {GRID_SIZE}×{GRID_SIZE} grids...", flush=True)
    grids = np.zeros((N_actual, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    grids[:, SEED_OFFSET:SEED_OFFSET + SEED_SIZE,
             SEED_OFFSET:SEED_OFFSET + SEED_SIZE] = seeds
    np.save(data_dir / "grids.npy", grids)
    print(f"  grids.npy saved ({(data_dir/'grids.npy').stat().st_size/1e6:.0f} MB)", flush=True)
    del grids, seeds  # free RAM before normalization + FFT
    _done(t0)

    # --- Normalization stats ---
    t0 = _phase("Normalization")
    sig_mean, sig_std = compute_normalization_stats(signatures_raw)
    print(f"  mean: {sig_mean.round(3)}", flush=True)
    print(f"  std:  {sig_std.round(3)}", flush=True)
    _done(t0)

    # --- Classification (must run on raw values before in-place normalization) ---
    t0 = _phase("Classification")
    labels     = classify_batch(signatures_raw, T)
    labels_arr = np.array(labels, dtype="U20")
    for cls in ["dying", "still_life", "oscillator", "glider"]:
        n = (labels_arr == cls).sum()
        print(f"    {cls:12s}: {n:7d} ({100*n/N_actual:.1f}%)", flush=True)
    _done(t0)

    # --- Sample raw data for diagnostics BEFORE overwriting the buffer ---
    # 10k seeds covers all diagnostic plots; avoids holding full raw + norm simultaneously.
    DIAG_N    = min(N_actual, 10_000)
    diag_idx  = rng.choice(N_actual, DIAG_N, replace=False)
    diag_raw  = signatures_raw[diag_idx].copy()          # (≤10k, 257, 10) ≈ 103 MB
    diag_norm = normalize_signatures(diag_raw.copy(), sig_mean, sig_std)
    diag_lbls = labels_arr[diag_idx]

    # --- Normalize in-place — overwrites signatures_raw buffer ---
    # After this call signatures_raw and signatures_norm alias the same memory.
    # Peak RAM = 1× (N, 257, 10) instead of 2×.
    # IMPORTANT: sig_mean/sig_std must be computed from the stratified data above,
    # not from the pool. Normalizing with pool stats produces wrong z-scores on
    # the balanced dataset (lag signals especially — pool is dying-dominated).
    signatures_norm = normalize_signatures(signatures_raw, sig_mean, sig_std)
    del signatures_raw  # drop the alias; buffer is now signatures_norm

    # --- Verify normalisation is correct on a sample ---
    _verify_sample = signatures_norm[:min(50_000, len(signatures_norm))].reshape(-1, N_SIGS)
    _vmean = _verify_sample.mean(axis=0)
    _vstd  = _verify_sample.std(axis=0)
    _bad   = np.where((np.abs(_vmean) > 1.0) | (np.abs(_vstd - 1.0) > 0.5))[0]
    if len(_bad):
        raise RuntimeError(
            f"Normalisation sanity check failed for signal indices {_bad.tolist()}.\n"
            f"  mean: {_vmean.round(3)}\n  std:  {_vstd.round(3)}\n"
            "sig_mean/sig_std were likely computed on the wrong (pre-stratification) data."
        )
    print(f"  Normalisation check OK — mean: {_vmean.round(2)}  std: {_vstd.round(2)}", flush=True)
    del _verify_sample, _vmean, _vstd, _bad

    # --- FFT sig_reference (chunked to bound peak RAM) ---
    t0 = _phase("FFT sig_reference (chunked)")
    sig_reference = compute_sig_reference(signatures_norm)
    print(f"  sig_reference shape: {sig_reference.shape}", flush=True)
    _done(t0)

    # --- Clustering ---
    t0 = _phase(f"Clustering (Ward k={CLUSTER_K})")
    N_actual = len(sig_reference)
    sub_n   = min(N_actual, 3_000)   # Ward is O(n² log n); 3k ≈ 1-2s, 10k hangs
    sub_idx = rng.choice(N_actual, sub_n, replace=False)
    sub_pts = sig_reference[sub_idx]
    sub_pca = PCA(n_components=min(50, sub_pts.shape[1])).fit_transform(sub_pts)
    sub_lbl = AgglomerativeClustering(n_clusters=CLUSTER_K,
                                      linkage="ward").fit_predict(sub_pca)
    # Silhouette on a 2k sub-subsample — full 10k is O(n²) pairwise distances (~800 MB,
    # tens of minutes). 2k gives a reliable estimate in under a second.
    sil_n   = min(2_000, len(sub_pca))
    sil_idx = rng.choice(len(sub_pca), sil_n, replace=False)
    sil_score = silhouette_score(sub_pca[sil_idx], sub_lbl[sil_idx])
    print(f"  Silhouette score (subsample n={sil_n}): {sil_score:.4f}", flush=True)
    cluster_centroids = np.array(
        [np.median(sub_pts[sub_lbl == c], axis=0) for c in range(CLUSTER_K)],
        dtype=np.float32)
    cluster_labels = cdist(sig_reference, cluster_centroids).argmin(axis=1).astype(np.int32)
    _done(t0)

    # --- Save ---
    t0 = _phase("Saving dataset files")
    saves = {
        "lifespans.npy":         lifespans.astype(np.int32),
        "buckets.npy":           buckets.astype(np.int32),
        "signatures_norm.npy":   signatures_norm,   # raw was normalized in-place; no dup
        "sig_mean.npy":          sig_mean,
        "sig_std.npy":           sig_std,
        "sig_reference.npy":     sig_reference,
        "labels.npy":            labels_arr,
        "cluster_centroids.npy": cluster_centroids,
        "cluster_labels.npy":    cluster_labels,
        "n_seeds.npy":           np.int32(N_actual),
    }
    for fname, arr in saves.items():
        path = data_dir / fname
        np.save(path, arr)
        print(f"  {fname}: shape={getattr(arr,'shape','scalar')} "
              f"({path.stat().st_size/1e6:.0f} MB)", flush=True)
    _done(t0)

    # --- Diagnostics ---
    t0 = _phase("Diagnostics")
    print("  lifespan_hist.png",         flush=True)
    _diag_lifespan_hist(_diag_ls_before, _diag_bk_before, lifespans, buckets, diag_dir)
    print("  signal_distributions.png",  flush=True)
    _diag_signal_distributions(diag_raw, diag_lbls, rng, diag_dir)
    print("  signal_samples.png",        flush=True)
    _diag_signal_samples(diag_raw, diag_lbls, rng, diag_dir)
    print("  normalization_check.png",   flush=True)
    _diag_normalization_check(diag_raw, diag_norm, rng, diag_dir)
    print("  cluster_summary.png",       flush=True)
    _diag_cluster_summary(sig_reference, cluster_labels, cluster_centroids, labels, diag_dir)
    print("  tsne_signatures.png",       flush=True)
    _diag_tsne(sig_reference, labels, np.random.default_rng(tsne_seed), diag_dir)
    print("  canonical_check.txt",       flush=True)
    verdict = _diag_canonical_check(sig_mean, sig_std, diag_dir)
    _done(t0)

    # --- Completion report ---
    total_mb  = (data_dir / "grids.npy").stat().st_size / 1e6
    total_mb += sum((data_dir / f).stat().st_size for f in saves) / 1e6
    elapsed   = time.time() - t_total
    ref_dim   = sig_reference.shape[1]
    print(f"\n{'='*60}\nSTAGE 1 COMPLETE\n{'='*60}", flush=True)
    print(f"  N_SEEDS:         {N_SEEDS}", flush=True)
    print(f"  T:               {T}", flush=True)
    print(f"  Grid:            {GRID_SIZE}×{GRID_SIZE}", flush=True)
    print(f"  Density bands:   {DENSITY_BANDS}", flush=True)
    print(f"  Signals:         {N_SIGS}  {SIG_NAMES}", flush=True)
    print(f"  Lags:            {LAGS}", flush=True)
    print(f"  sig_reference:   (N, {ref_dim}) FFT magnitude", flush=True)
    print(f"  Stratification:  equal ({N_actual//4} per bucket, {N_actual} total)", flush=True)
    print(f"  Silhouette:      {sil_score:.4f}", flush=True)
    print(f"  Canonical check: {verdict}", flush=True)
    print(f"  Total data size: {total_mb:.0f} MB", flush=True)
    print(f"  Diagnostics:     {diag_dir}", flush=True)
    print(f"  Total time:      {elapsed/3600:.2f} hr", flush=True)


if __name__ == "__main__":
    main()
