# GoL Emergence Discovery Dataset — v1

**1.5 million Conway's Game of Life initial conditions** with behavioral class labels,
lifespan metadata, normalized 10-signal trajectories, and full reproducibility seeds.
Released by [Mantra Labs](https://mantra-labs.com) as the foundation dataset for the
GoL Emergence Discovery System — a staged research program building a generative model
that discovers novel emergent structures in cellular automata.

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Seeds | 1,500,000 |
| Grid size | 128 × 128 (16 × 16 seed embedded at center) |
| Timesteps | 257 (T = 0 … 256) |
| Signals per timestep | 10 |
| Behavioral classes | 4 (still\_life, oscillator, dying, glider) |
| Sampling | Density-stratified (4 bands, 0.03–0.30) |
| RNG seed | 3750551643 |
| Generated | 2026-04-30 |
| Rule | Conway's B3/S23, fixed-zero boundary |

### Class Distribution

| Class | Count | Share |
|---|---|---|
| still\_life | 815,485 | 54.4% |
| oscillator | 365,313 | 24.4% |
| dying | 307,915 | 20.5% |
| glider | 11,287 | 0.75% |

---

## Repository Structure

```
gol-emergence-dataset/
│
├── README.md                      ← you are here
│
├── data/                          ← dataset files (see data/README.md)
│   ├── labels.npy                 ← (1.5M,) behavioral class per seed
│   ├── lifespans.npy              ← (1.5M,) lifespan in generations
│   ├── buckets.npy                ← (1.5M,) stratification bucket index
│   ├── sig_mean.npy               ← (10,)   per-signal population mean
│   ├── sig_std.npy                ← (10,)   per-signal population std
│   ├── n_seeds.npy                ← scalar  total seed count
│   ├── seeds.npy                  ← (1.5M, 16, 16) raw 16×16 seed grids
│   ├── seeds.json                 ← RNG seeds for full reproducibility
│   └── diagnostics/               ← generation logs and verification outputs
│
├── figures/                       ← analysis figures (see figures/README.md)
│   ├── signal_samples.png
│   ├── signal_distributions.png
│   ├── tsne_signatures.png
│   ├── lifespan_hist.png
│   ├── normalization_check.png
│   ├── cluster_summary.png
│   ├── temporal_glider.png
│   └── temporal_oscillator.png
│
├── src/                           ← generation and analysis code
│   ├── simulator.py               ← standalone GoL engine (B3/S23)
│   ├── generate_data.py           ← full Stage 1 data pipeline
│   ├── inject_gliders.py          ← glider injection utilities
│   └── analyse_data.py            ← post-hoc analysis and figure generation
│
└── docs/                          ← design documentation
    ├── overarching_design.md      ← project-wide goals and architecture
    ├── s1_design.md               ← Stage 1: data pipeline (complete ✓)
    ├── s2_design.md               ← Stage 2: generative model (upcoming)
    ├── s3_design.md               ← Stage 3: latent space navigation (upcoming)
    └── s4_design.md               ← Stage 4: emergence discovery (upcoming)
```

---

## Large Files (External Hosting)

The following files exceed GitHub's size limits and are hosted separately.
Links will be added when the data repository is published on Hugging Face / Zenodo.

| File | Shape | Size | Description |
|---|---|---|---|
| `grids.npy` | (1.5M, 128, 128) uint8 | 23 GB | Full 128×128 initial condition grids |
| `signatures_norm.npy` | (1.5M, 257, 10) float32 | 15 GB | Normalized 10-signal behavioral trajectories |
| `sig_reference.npy` | (1.5M, 1290) float32 | ~7 GB | FFT-magnitude novelty reference vectors |

> **Reproducibility**: all large files can be regenerated from `seeds.npy` and `seeds.json`
> using `src/generate_data.py` with `--rng-seed 3750551643`.

---

## Quick Start

```python
import numpy as np, json

# Load metadata
labels     = np.load("data/labels.npy")          # (1_500_000,) dtype=U20
lifespans  = np.load("data/lifespans.npy")        # (1_500_000,) int32
buckets    = np.load("data/buckets.npy")          # (1_500_000,) int32
sig_mean   = np.load("data/sig_mean.npy")         # (10,) float32
sig_std    = np.load("data/sig_std.npy")          # (10,) float32

# Load large files (once available externally)
grids      = np.load("data/grids.npy")            # (1_500_000, 128, 128) uint8
sigs       = np.load("data/signatures_norm.npy")  # (1_500_000, 257, 10) float32

# Invert normalization on a single trajectory
raw = sigs[0] * sig_std + sig_mean

# Filter by class
glider_idx = np.where(labels == "glider")[0]
glider_sigs = sigs[glider_idx]  # (11_287, 257, 10)

# Stratified split (use bucket for equal behavioral coverage)
train_mask = (np.arange(len(labels)) % 10) != 0
val_mask   = ~train_mask
```

---

## The 10 Behavioral Signals

Each seed produces a trajectory of shape `(257, 10)` — one row per timestep.

| Index | Signal | Description |
|---|---|---|
| 0 | P(t) | Population — alive cell count |
| 1 | Δcx(t) | Center-of-mass x displacement from t=0 |
| 2 | Δcy(t) | Center-of-mass y displacement from t=0 |
| 3 | V(t) | Spatial variance of alive cells |
| 4 | E(t) | Motion energy — cells that changed state |
| 5 | N\_cc(t) | Connected component count (8-connectivity) |
| 6 | S\_lag\_2(t) | Temporal self-similarity at lag 2 |
| 7 | S\_lag\_4(t) | Temporal self-similarity at lag 4 |
| 8 | S\_lag\_8(t) | Temporal self-similarity at lag 8 |
| 9 | S\_lag\_16(t) | Temporal self-similarity at lag 16 |

Signals are stored normalized (zero mean, unit std across all seeds and timesteps).
Use `sig_mean.npy` and `sig_std.npy` to invert.

---

## Reproducing the Dataset

```bash
git clone https://github.com/themantralab/gol-emergence-dataset
cd gol-emergence-dataset
pip install numpy scipy matplotlib scikit-learn

# Verify simulator
python src/simulator.py

# Regenerate full dataset (requires ~27 GB RAM peak, ~8 CPU cores, ~12 hours)
python src/generate_data.py --n-seeds 1500000 --rng-seed 3750551643 --workers 8
```

---

## Project Roadmap

This dataset is Stage 1 of a four-stage research program.

| Stage | Status | Description |
|---|---|---|
| **1 — Data Pipeline** | ✅ Complete | Generate, simulate, and characterize 1.5M GoL seeds |
| **2 — Generative Model** | 🔜 Upcoming | Train a priorless VAE on behavioral trajectories |
| **3 — Latent Navigation** | 🔜 Upcoming | Diffusion-based sampling over the latent space |
| **4 — Emergence Discovery** | 🔜 Upcoming | LOF-based novelty scoring to find unknown structures |

Each stage will be published as a separate release in this repository.
Design documents for all stages are in `docs/`.

---

## Citation

```bibtex
@dataset{koegler2026gol,
  author    = {Koegler, Maxwell},
  title     = {{GoL Emergence Discovery Dataset v1}},
  year      = {2026},
  publisher = {Mantra Labs},
  url       = {https://github.com/themantralab/gol-emergence-dataset}
}
```

---

## License

Code (`src/`) is released under the **MIT License**.
Data and figures are released under **CC BY 4.0**.

© 2026 Mantra Labs
