# Source Code

## What was used to generate the published dataset

The published dataset was produced by exactly two scripts, run in order:

| Step | Script | Purpose |
|---|---|---|
| 1 | `simulator.py` | Verified first — confirms the GoL engine is correct before any data is generated |
| 2 | `generate_data.py` | Ran once to produce all 1.5M seeds and every file in `data/` |

`inject_gliders.py` and `analyse_data.py` were **not** used in dataset generation.
`analyse_data.py` was used post-hoc to produce the figures in `figures/`.
`inject_gliders.py` was not used at all for the published dataset (see below).

---

## Reproducing the dataset from scratch

```bash
# 1. Install dependencies
pip install numpy scipy matplotlib scikit-learn

# 2. Verify the simulator passes all built-in checks
python src/simulator.py
# Expected: PASS for block, blinker, glider, step_batch, birth rule,
#           overcrowding, and fixed-zero boundary

# 3. Generate the full dataset (~27 GB RAM peak, ~8 cores, ~12 hours)
python src/generate_data.py \
    --n-seeds 1500000 \
    --rng-seed 3750551643 \
    --workers 8
# Writes all files to data/ and data/diagnostics/

# 4. Regenerate figures (optional)
python src/analyse_data.py
# Writes all figures to figures/
```

The RNG seed `3750551643` is the exact seed used for the published dataset.
All other parameters match the defaults in `generate_data.py`.
See `data/seeds.json` for the full seed record.

---

## All source files

| File | Used in published dataset | Description |
|---|---|---|
| `simulator.py` | ✅ Yes — verified before generation | Standalone GoL engine (B3/S23, fixed-zero boundary). No project-internal dependencies — also imported by Stage 4. |
| `generate_data.py` | ✅ Yes — produced all data | Full Stage 1 pipeline: seed generation, simulation, signal extraction, normalization, stratification, clustering, and saving. Imports only `simulator.py`. |
| `analyse_data.py` | ✅ Yes — produced all figures | Post-hoc analysis. Loads saved `.npy` files from `data/` and writes figures to `figures/`. Not part of data generation. |
| `inject_gliders.py` | ❌ Not used | See below. |

---

## inject_gliders.py — not used in the published dataset

`inject_gliders.py` is a standalone utility for injecting known canonical glider
patterns (e.g. the standard period-4 glider) into arbitrary seed grids at specified
positions and orientations.

It was written to support a validation experiment: if you inject a set of known
gliders into the dataset and run t-SNE on the FFT-magnitude signatures
(`sig_reference.npy`), you can verify that the injected gliders cluster together
and land in the expected region of the signature space — providing a ground-truth
check on the FFT fingerprinting approach.

**This experiment was not run for the published dataset.** The canonical check
in `data/diagnostics/canonical_check.txt` (which embeds block, blinker, and glider
patterns individually and verifies their heuristic classification) was sufficient
to validate the pipeline. The injected-glider t-SNE comparison remains a useful
tool for anyone who wants to probe the geometry of the signature space or test
a modified signal definition, but it is not needed to reproduce the published results.

---

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
```

Install with: `pip install numpy scipy matplotlib scikit-learn`

---

## Notes

- `simulator.py` is intentionally standalone with no project-internal imports.
  It is re-used by Stage 4 (`emergence/validity.py`) and must remain self-contained.
- `generate_data.py` is a single-file pipeline. Stages 2–4 load `.npy` files
  directly and never import it.
- Future stages will add new source files to this directory as the project progresses.
