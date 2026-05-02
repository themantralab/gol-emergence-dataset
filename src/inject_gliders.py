"""Inject known glider seeds into existing Stage 1 data and re-run t-SNE diagnostic.

Usage:
    python3 inject_gliders.py [--n-gliders N]

Loads data/*.npy, appends N glider seeds with correct signatures, overwrites
tsne_signatures.png. Does not touch any other diagnostic or .npy file.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import simulator
from generate_data import (
    T, GRID_SIZE, SEED_SIZE, SEED_OFFSET,
    extract_signature, normalize_signatures, compute_sig_reference,
    classify_trajectory,
)

# ---------------------------------------------------------------------------
# Canonical glider pattern (3×3, travels diagonally in B3/S23)
# .O.
# ..O
# OOO
# ---------------------------------------------------------------------------
_GLIDER = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
], dtype=np.uint8)


def make_glider_seed(offset_r=4, offset_c=4):
    """Place glider at (offset_r, offset_c) inside a 16×16 seed."""
    seed = np.zeros((SEED_SIZE, SEED_SIZE), dtype=np.uint8)
    r, c = offset_r, offset_c
    seed[r:r + 3, c:c + 3] = _GLIDER
    return seed


def embed_seed(seed):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    grid[SEED_OFFSET:SEED_OFFSET + SEED_SIZE,
         SEED_OFFSET:SEED_OFFSET + SEED_SIZE] = seed
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gliders", type=int, default=20,
                        help="Number of glider variants to inject (default 20)")
    args = parser.parse_args()

    data_dir = Path("data")
    diag_dir = data_dir / "diagnostics"

    # --- Load existing data ---
    sig_reference = np.load(data_dir / "sig_reference.npy")
    labels_arr    = np.load(data_dir / "labels.npy")
    sig_mean      = np.load(data_dir / "sig_mean.npy")
    sig_std       = np.load(data_dir / "sig_std.npy")
    print(f"Loaded {len(sig_reference)} existing seeds from data/", flush=True)
    print(f"Class distribution before injection:", flush=True)
    for cls in ["dying", "still_life", "oscillator", "glider"]:
        n = int((labels_arr == cls).sum())
        print(f"  {cls:12s}: {n}", flush=True)

    # --- Generate glider variants (different placement offsets) ---
    offsets = [(r, c)
               for r in range(1, 10, 3)
               for c in range(1, 10, 3)]  # 9 positions in the 16×16 seed
    glider_sigs  = []
    glider_norms = []
    glider_refs  = []
    glider_lbls  = []
    added = 0
    for i in range(args.n_gliders):
        r, c   = offsets[i % len(offsets)]
        seed   = make_glider_seed(r, c)
        grid   = embed_seed(seed)
        traj   = simulator.simulate(grid, T)            # (T+1, H, W)
        sig    = extract_signature(traj)                # (T+1, 10)
        cls    = classify_trajectory(sig, T)
        norm   = normalize_signatures(sig[np.newaxis], sig_mean, sig_std)[0]  # (T+1,10)
        ref    = compute_sig_reference(norm[np.newaxis])[0]                   # (1290,)
        glider_sigs.append(sig)
        glider_norms.append(norm)
        glider_refs.append(ref)
        glider_lbls.append(cls)
        added += 1
        print(f"  Glider {i+1}/{args.n_gliders}: offset=({r},{c})  class={cls}", flush=True)

    glider_refs_arr = np.stack(glider_refs).astype(np.float32)
    glider_lbls_arr = np.array(glider_lbls, dtype="U20")

    # --- Append to existing arrays ---
    sig_ref_combined = np.concatenate([sig_reference, glider_refs_arr], axis=0)
    labels_combined  = np.concatenate([labels_arr,   glider_lbls_arr], axis=0)

    print(f"\nAfter injection: {len(sig_ref_combined)} seeds", flush=True)
    for cls in ["dying", "still_life", "oscillator", "glider"]:
        n = int((labels_combined == cls).sum())
        print(f"  {cls:12s}: {n}", flush=True)

    # --- Re-run t-SNE ---
    rng = np.random.default_rng(42)
    classes = ["dying", "still_life", "oscillator", "glider"]
    colors  = ["red", "blue", "green", "orange"]
    max_pts = 5_000

    glider_idx = np.where(labels_combined == "glider")[0]
    other_pool = np.where(labels_combined != "glider")[0]
    budget     = max(0, max_pts - len(glider_idx))
    other_idx  = rng.choice(other_pool, min(budget, len(other_pool)), replace=False)
    idx        = np.concatenate([glider_idx, other_idx])
    pts        = sig_ref_combined[idx]
    lbl        = labels_combined[idx]

    if pts.shape[1] > 50:
        pts = PCA(n_components=50).fit_transform(pts)

    print(f"\nRunning t-SNE on {len(pts)} points...", flush=True)
    emb = TSNE(n_components=2, random_state=42,
               perplexity=min(30, len(pts) - 1)).fit_transform(pts)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, col in zip(classes, colors):
        mask  = lbl == cls
        n_cls = int(mask.sum())
        sz    = 50 if cls == "glider" else (4 if n_cls > 10 else 20)
        mk    = "*" if cls == "glider" else "o"
        if n_cls > 0:
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=sz, c=col, marker=mk,
                       label=f"{cls} (n={n_cls})", alpha=0.8, zorder=5 if cls=="glider" else 1)
        else:
            ax.scatter([], [], c=col, label=f"{cls} (n=0)", s=10)
    ax.legend()
    ax.set_title(f"t-SNE of sig_reference — gliders injected (n={added})")
    fig.tight_layout()
    out = diag_dir / "tsne_glider_injection.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
