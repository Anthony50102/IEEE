"""DISCO on HW2D — IEEE/CiSE Special Issue.

This package wraps the upstream DISCO architecture (Morel/Han/Oyallon 2025,
vendored in `upstream/`) with HW2D-specific dataloaders, configs, and
training/eval entry points.

Sub-modules:
    upstream/         — third-party DISCO code, frozen at ddd18f17
    hw2d_dataset.py   — HW2D snippet adapter for the DISCO trainer
    dataset_specs.py  — HW2D analogs of upstream's DATASET_SPECS dict
    train_hw2d.py     — multi-alpha training entry point

The headline claim of the paper this package supports:

    DISCO, when trained on Hasegawa-Wakatani snippets at alpha in {0.1, 1, 5},
    produces stable, statistically faithful rollouts at the held-out
    alpha = 1.5 without retraining.
"""
