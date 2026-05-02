# Source Code

## Files

| File | Stage | Description |
|---|---|---|
| `simulator.py` | 1 + 4 | Standalone Conway's Game of Life engine (B3/S23, fixed-zero boundary). No external dependencies beyond NumPy. Run directly to execute built-in verifications. |
| `generate_data.py` | 1 | Full data pipeline: seed generation, simulation, signal extraction, normalization, stratification, clustering, and saving. Imports only `simulator.py`. |
| `inject_gliders.py` | 1 | Utilities for injecting known canonical glider patterns into seed grids. Used for dataset validation and canonical checks. |
| `analyse_data.py` | 1 | Post-hoc analysis and figure generation. Loads saved `.npy` files and produces all figures in `figures/`. |

## Usage

```bash
# Verify simulator (run before anything else)
python src/simulator.py

# Generate dataset
python src/generate_data.py --n-seeds 1500000 --rng-seed 3750551643 --workers 8

# Regenerate figures
python src/analyse_data.py
```

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
```

Install with: `pip install numpy scipy matplotlib scikit-learn`

## Notes

- `simulator.py` is intentionally standalone — it is imported by Stage 4 (`emergence/validity.py`) and must have no project-internal dependencies.
- `generate_data.py` is a single-file pipeline. Downstream stages (2–4) load `.npy` files directly and never import this file.
- Future stages will add new source files here as the project progresses.
