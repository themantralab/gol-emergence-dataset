# Data Files

> **Full dataset (including large array files):**
> [huggingface.co/datasets/themantralab/gol-emergence-dataset](https://huggingface.co/datasets/themantralab/gol-emergence-dataset)

## Files in this repo

| File | Shape | Size | Description |
|---|---|---|---|
| `labels.npy` | (1.5M,) str | 115 MB | Behavioral class label per seed: `still_life`, `oscillator`, `dying`, `glider` |
| `lifespans.npy` | (1.5M,) int32 | 5.8 MB | Per-seed lifespan — last timestep where any cell changed state |
| `buckets.npy` | (1.5M,) int32 | 5.8 MB | Stratification bucket: 0=dying [0,20), 1=short [20,60), 2=medium [60,130), 3=long [130,256] |
| `sig_mean.npy` | (10,) float32 | < 1 KB | Per-signal population mean — required to invert normalization |
| `sig_std.npy` | (10,) float32 | < 1 KB | Per-signal population std — required to invert normalization |
| `n_seeds.npy` | scalar int32 | < 1 KB | Total seed count (1,500,000) |
| `seeds.npy` | (1.5M, 16, 16) uint8 | ~230 MB | Raw 16×16 binary seed grids |
| `seeds.json` | — | < 1 KB | RNG seeds for full reproducibility |

## Large files — Hugging Face only

These files exceed GitHub's size limits and are available on Hugging Face:
**[huggingface.co/datasets/themantralab/gol-emergence-dataset](https://huggingface.co/datasets/themantralab/gol-emergence-dataset)**

| File | Shape | Size | Description |
|---|---|---|---|
| `grids.npy` | (1.5M, 128, 128) uint8 | 23 GB | Full 128×128 embedded initial conditions |
| `signatures_norm.npy` | (1.5M, 257, 10) float32 | 15 GB | Normalized 10-signal behavioral trajectories |
| `sig_reference.npy` | (1.5M, 1290) float32 | ~7 GB | FFT-magnitude novelty reference (phase-invariant) |

```python
from huggingface_hub import hf_hub_download
import numpy as np

grids = np.load(hf_hub_download("themantralab/gol-emergence-dataset", "grids.npy", repo_type="dataset"))
sigs  = np.load(hf_hub_download("themantralab/gol-emergence-dataset", "signatures_norm.npy", repo_type="dataset"))
```

## Diagnostics

| File | Description |
|---|---|
| `diagnostics/canonical_check.txt` | Verification that block→still\_life, blinker→oscillator, glider→glider |
| `diagnostics/generation_log.txt` | Full generation run log with worker progress and timing |
