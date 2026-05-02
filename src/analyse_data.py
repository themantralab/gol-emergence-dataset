"""Graphical and empirical analysis of Stage 1 dataset.

Generates data/diagnostics/analysis_*.png and prints statistics to stdout.
Uses mmap for large arrays (sig_reference, signatures_raw) to avoid OOM.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

DATA  = Path("data")
DIAG  = DATA / "diagnostics"
DIAG.mkdir(exist_ok=True)

SIG_NAMES = ["P", "Δcx", "Δcy", "V", "E", "N_cc",
             "S_lag_2", "S_lag_4", "S_lag_8", "S_lag_16"]
CLASSES   = ["dying", "still_life", "oscillator", "glider"]
COLORS    = {"dying": "red", "still_life": "blue",
             "oscillator": "green", "glider": "orange"}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
lifespans     = np.load(DATA / "lifespans.npy")
buckets       = np.load(DATA / "buckets.npy")
labels_arr    = np.load(DATA / "labels.npy")
cluster_lbls  = np.load(DATA / "cluster_labels.npy")
sig_mean      = np.load(DATA / "sig_mean.npy")
sig_std       = np.load(DATA / "sig_std.npy")
cluster_cents = np.load(DATA / "cluster_centroids.npy")
N             = len(lifespans)

# mmap large arrays — never fully loaded into RAM
sig_ref_mm  = np.load(DATA / "sig_reference.npy",   mmap_mode="r")  # (N, 1290)
sigs_raw_mm = np.load(DATA / "signatures_raw.npy",  mmap_mode="r")  # (N, 257, 10)

print(f"N={N:,}  sig_reference={sig_ref_mm.shape}  dtype={sig_ref_mm.dtype}", flush=True)

# ── Empirical summary ─────────────────────────────────────────────────────────
print("\n=== CLASS DISTRIBUTION ===")
for cls in CLASSES:
    n = int((labels_arr == cls).sum())
    print(f"  {cls:14s}: {n:8,}  ({100*n/N:.2f}%)")

print("\n=== BUCKET DISTRIBUTION ===")
bnames = ["dying [0,20)", "short [20,60)", "medium [60,130)", "long [130,256]"]
for b, name in enumerate(bnames):
    n = int((buckets == b).sum())
    print(f"  {name:22s}: {n:8,}  ({100*n/N:.2f}%)")

print("\n=== LIFESPAN STATS PER CLASS ===")
for cls in CLASSES:
    idx = np.where(labels_arr == cls)[0]
    if not len(idx): continue
    ls  = lifespans[idx]
    print(f"  {cls:14s}: mean={ls.mean():.1f}  median={int(np.median(ls))}  "
          f"min={ls.min()}  max={ls.max()}  std={ls.std():.1f}")

print("\n=== SIGNAL NORMALIZATION STATS ===")
print(f"  {'Signal':10s}  {'mean':>10s}  {'std':>10s}")
for i, name in enumerate(SIG_NAMES):
    print(f"  {name:10s}  {sig_mean[i]:10.4f}  {sig_std[i]:10.4f}")

print("\n=== CLUSTER × CLASS CROSS-TAB ===")
K = len(cluster_cents)
header = f"  {'':14s}" + "".join(f"  C{k}" for k in range(K))
print(header)
for cls in CLASSES:
    row = f"  {cls:14s}"
    idx = np.where(labels_arr == cls)[0]
    if not len(idx):
        row += "  (none)"
    else:
        for k in range(K):
            cnt = int((cluster_lbls[idx] == k).sum())
            row += f"  {cnt:5,}"
    print(row)

# ── Subsample for plots ───────────────────────────────────────────────────────
rng    = np.random.default_rng(42)
NSUB   = 2_000           # per-class subsample for trajectory plots
NREF   = min(N, 30_000)  # subsample for PCA / correlation

ref_idx = rng.choice(N, NREF, replace=False)
ref_sub = sig_ref_mm[ref_idx].astype(np.float32)      # (NREF, 1290)
lbl_sub = labels_arr[ref_idx]

class_idx = {cls: np.where(labels_arr == cls)[0] for cls in CLASSES}

# ── Plot 1: class + bucket bar charts ─────────────────────────────────────────
print("\nPlot 1: class & bucket distributions...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cls_counts = [(labels_arr == c).sum() for c in CLASSES]
axes[0].bar(CLASSES, cls_counts, color=[COLORS[c] for c in CLASSES], edgecolor="black")
axes[0].set_title("Class distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(cls_counts):
    axes[0].text(i, v + max(cls_counts)*0.01, f"{v:,}", ha="center", fontsize=8)

bkt_counts = [(buckets == b).sum() for b in range(4)]
axes[1].bar(bnames, bkt_counts, color="steelblue", edgecolor="black")
axes[1].set_title("Bucket distribution (after stratification)")
axes[1].set_ylabel("Count")
axes[1].set_xticklabels(bnames, rotation=15, ha="right")
for i, v in enumerate(bkt_counts):
    axes[1].text(i, v + max(bkt_counts)*0.01, f"{v:,}", ha="center", fontsize=8)

fig.suptitle(f"Dataset overview  N={N:,}")
fig.tight_layout()
fig.savefig(DIAG / "analysis_1_distributions.png", dpi=100)
plt.close(fig)

# ── Plot 2: lifespan violin per class ─────────────────────────────────────────
print("Plot 2: lifespan violins...", flush=True)
fig, ax = plt.subplots(figsize=(10, 5))
data_vln, labels_vln = [], []
for cls in CLASSES:
    idx = class_idx[cls]
    if not len(idx): continue
    sidx = rng.choice(idx, min(10_000, len(idx)), replace=False)
    data_vln.append(lifespans[sidx])
    labels_vln.append(cls)
parts = ax.violinplot(data_vln, showmedians=True, showextrema=True)
for pc, cls in zip(parts["bodies"], labels_vln):
    pc.set_facecolor(COLORS[cls]); pc.set_alpha(0.6)
ax.set_xticks(range(1, len(labels_vln)+1))
ax.set_xticklabels(labels_vln)
ax.set_ylabel("Lifespan (steps)")
ax.set_title("Lifespan distribution per class")
ax.axhline(256, color="gray", linestyle="--", linewidth=0.8, label="T=256")
ax.legend()
fig.tight_layout()
fig.savefig(DIAG / "analysis_2_lifespan_violin.png", dpi=100)
plt.close(fig)

# ── Plot 3: mean signal trajectories per class ────────────────────────────────
print("Plot 3: mean signal trajectories...", flush=True)
T_len = sigs_raw_mm.shape[1]   # 257
t_ax  = np.arange(T_len)
fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
axes = axes.ravel()
for si, (ax, sname) in enumerate(zip(axes, SIG_NAMES)):
    for cls in CLASSES:
        idx = class_idx[cls]
        if not len(idx): continue
        sidx  = rng.choice(idx, min(NSUB, len(idx)), replace=False)
        chunk = sigs_raw_mm[sidx, :, si].astype(np.float32)  # (sub, T+1)
        mn    = chunk.mean(axis=0)
        sd    = chunk.std(axis=0)
        ax.plot(t_ax, mn, color=COLORS[cls], linewidth=1.2, label=cls)
        ax.fill_between(t_ax, mn-sd, mn+sd, color=COLORS[cls], alpha=0.12)
    ax.set_title(sname, fontsize=9)
    ax.set_xlabel("t", fontsize=7)
    if si == 0:
        ax.legend(fontsize=7)
fig.suptitle(f"Mean ± 1σ raw signal trajectories per class  (subsample {NSUB}/class)")
fig.tight_layout()
fig.savefig(DIAG / "analysis_3_signal_trajectories.png", dpi=100)
plt.close(fig)

# ── Plot 4: PCA of sig_reference (2D) + scree ─────────────────────────────────
print("Plot 4: PCA of sig_reference...", flush=True)
pca50  = PCA(n_components=50).fit(ref_sub)
evr    = pca50.explained_variance_ratio_
emb2   = pca50.transform(ref_sub)[:, :2]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scree
axes[0].bar(range(1, 21), evr[:20]*100, color="steelblue", edgecolor="black")
axes[0].plot(range(1, 21), np.cumsum(evr[:20])*100, "o-r", linewidth=1.2,
             label="Cumulative")
axes[0].set_xlabel("PC")
axes[0].set_ylabel("Explained variance (%)")
axes[0].set_title("PCA scree (top 20 components)")
axes[0].legend()
axes[0].axhline(100*np.cumsum(evr)[49], color="gray", linestyle="--",
                label=f"PC50 = {100*np.cumsum(evr)[49]:.1f}%")

# 2D scatter
for cls in CLASSES:
    mask = lbl_sub == cls
    if not mask.any(): continue
    n = mask.sum()
    sidx2 = np.where(mask)[0]
    if n > 3000:
        sidx2 = rng.choice(sidx2, 3000, replace=False)
    axes[1].scatter(emb2[sidx2, 0], emb2[sidx2, 1],
                    s=2, c=COLORS[cls], label=f"{cls} ({n:,})", alpha=0.5)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title(f"PCA-2D of sig_reference  (subsample {NREF:,})")
axes[1].legend(markerscale=4)

fig.tight_layout()
fig.savefig(DIAG / "analysis_4_pca.png", dpi=100)
plt.close(fig)

# ── Plot 5: cluster purity heatmap ─────────────────────────────────────────────
print("Plot 5: cluster purity heatmap...", flush=True)
cross = np.zeros((len(CLASSES), K), dtype=int)
for ci, cls in enumerate(CLASSES):
    idx = np.where(labels_arr == cls)[0]
    for k in range(K):
        cross[ci, k] = int((cluster_lbls[idx] == k).sum())

cross_pct = cross / cross.sum(axis=0, keepdims=True) * 100  # col-normalise

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(cross, aspect="auto", cmap="Blues")
axes[0].set_xticks(range(K)); axes[0].set_xticklabels([f"C{k}" for k in range(K)])
axes[0].set_yticks(range(len(CLASSES))); axes[0].set_yticklabels(CLASSES)
axes[0].set_title("Cluster × class (raw counts)")
for ci in range(len(CLASSES)):
    for k in range(K):
        axes[0].text(k, ci, f"{cross[ci,k]:,}", ha="center", va="center", fontsize=7)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(cross_pct, aspect="auto", cmap="Oranges", vmin=0, vmax=100)
axes[1].set_xticks(range(K)); axes[1].set_xticklabels([f"C{k}" for k in range(K)])
axes[1].set_yticks(range(len(CLASSES))); axes[1].set_yticklabels(CLASSES)
axes[1].set_title("Cluster × class (% of cluster)")
for ci in range(len(CLASSES)):
    for k in range(K):
        axes[1].text(k, ci, f"{cross_pct[ci,k]:.1f}%", ha="center", va="center", fontsize=7)
plt.colorbar(im1, ax=axes[1])

fig.suptitle("Cluster purity: how well Ward clusters align with heuristic classes")
fig.tight_layout()
fig.savefig(DIAG / "analysis_5_cluster_purity.png", dpi=100)
plt.close(fig)

# ── Plot 6: FFT fingerprint correlation heatmap ────────────────────────────────
print("Plot 6: inter-signal FFT correlation...", flush=True)
# Each signal contributes 129 FFT bins; average within each signal block → (NREF, 10) summary
n_bins = sig_ref_mm.shape[1] // len(SIG_NAMES)  # 129
sig_power = np.stack([
    ref_sub[:, i*n_bins:(i+1)*n_bins].mean(axis=1)
    for i in range(len(SIG_NAMES))
], axis=1)  # (NREF, 10)

corr = np.corrcoef(sig_power.T)  # (10, 10)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(10)); ax.set_xticklabels(SIG_NAMES, rotation=45, ha="right")
ax.set_yticks(range(10)); ax.set_yticklabels(SIG_NAMES)
ax.set_title("Pearson correlation of mean FFT power per signal\n"
             f"(subsample {NREF:,} seeds)")
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                fontsize=6, color="black" if abs(corr[i,j]) < 0.7 else "white")
plt.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(DIAG / "analysis_6_signal_correlation.png", dpi=100)
plt.close(fig)

# ── Plot 7: novelty proxy — nearest-neighbour distance distribution ─────────────
print("Plot 7: pairwise distance distributions...", flush=True)
from sklearn.neighbors import NearestNeighbors

pca50_model = PCA(n_components=50).fit(ref_sub)
ref_50      = pca50_model.transform(ref_sub)   # (NREF, 50)

nn = NearestNeighbors(n_neighbors=6, algorithm="ball_tree").fit(ref_50)
dists, _ = nn.kneighbors(ref_50)
knn5_dist = dists[:, 1:].mean(axis=1)   # mean k=5 NN distance (exclude self)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(knn5_dist, bins=80, color="steelblue", edgecolor="none")
axes[0].set_xlabel("Mean k=5 NN distance (PCA-50 space)")
axes[0].set_ylabel("Count")
axes[0].set_title("k-NN distance distribution (all classes)")

for cls in CLASSES:
    mask = lbl_sub == cls
    if not mask.any(): continue
    axes[1].hist(knn5_dist[mask], bins=60, alpha=0.5,
                 color=COLORS[cls], label=f"{cls} (n={mask.sum():,})", density=True)
axes[1].set_xlabel("Mean k=5 NN distance (PCA-50 space)")
axes[1].set_ylabel("Density")
axes[1].set_title("k-NN distance by class (novelty proxy)")
axes[1].legend()

fig.suptitle("Nearest-neighbour distances in PCA-50 sig_reference space\n"
             "Higher = more isolated = higher LOF novelty score")
fig.tight_layout()
fig.savefig(DIAG / "analysis_7_knn_distances.png", dpi=100)
plt.close(fig)

print("\n=== k-NN DISTANCE STATS PER CLASS (novelty proxy) ===")
for cls in CLASSES:
    mask = lbl_sub == cls
    if not mask.any(): continue
    d = knn5_dist[mask]
    print(f"  {cls:14s}: mean={d.mean():.4f}  median={np.median(d):.4f}  "
          f"p95={np.percentile(d,95):.4f}  max={d.max():.4f}")

print(f"\nAll plots saved to {DIAG}/", flush=True)
print("Done.", flush=True)
