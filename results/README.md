# Best Run Results

Curated outputs from the best model runs for each method and PDE benchmark.
These are committed to the repository for reproducibility and paper figure generation.

## Directory Structure

```
results/
├── <method>/          # dmd, opinf, fno
│   └── <pde>/         # ks, hw2d
│       ├── config*.yaml           # Configuration used for the run
│       ├── pipeline_status.yaml   # Step completion timestamps & summary metrics
│       ├── metrics.yaml           # Full evaluation metrics
│       ├── predictions/           # Gamma & state predictions (small .npz files)
│       ├── model/                 # Trained model artifacts (method-specific)
│       └── figures/               # All generated plots
```

## What's Included

| Artifact | DMD | OpInf | FNO |
|----------|-----|-------|-----|
| Configs | `config_step_{1,2,3}.yaml` | `config_step_{1,2,3}.yaml` | `config.yaml` |
| Metrics | `dmd_evaluation_metrics.yaml` | `evaluation_metrics.yaml` | `metrics.yaml` |
| Pipeline status | ✓ | ✓ | ✓ |
| Model files | `dmd_model.npz`, `learning_matrices.npz` | `ensemble_models.npz` | Excluded by default (too large) |
| Predictions | `dmd_predictions.npz` | — (embedded in ensemble, too large) | `predictions.npz` |
| Figures | All PNGs | All PNGs | All PNGs |

## Populating Results

Use the curation script to copy artifacts from a completed run:

```bash
# Curate a best run (auto-detects method and PDE from the directory name)
python scripts/curate_best_run.py local_output/20260316_090304_dmd_ks_temporal_split

# Include FNO model checkpoint (large, use with caution)
python scripts/curate_best_run.py local_output/fno_ks_... --include-model

# Override auto-detected method/PDE
python scripts/curate_best_run.py /path/to/run --method opinf --pde hw2d

# Dry run — see what would be copied without actually copying
python scripts/curate_best_run.py local_output/some_run --dry-run
```

## Current Best Runs

<!-- Update this table after curating new best runs -->

| Method | PDE | Source Run | Test Γ_n Error | Test Γ_c Error |
|--------|-----|-----------|----------------|----------------|
| — | — | — | — | — |
