"""
DMD (Dynamic Mode Decomposition) ROM Pipeline.

This package implements Optimized DMD (opt-DMD via BOPDMD) for 
reduced-order modeling of the Hasegawa-Wakatani system.

Pipeline Steps:
    step_1_preprocess: Load data, compute POD, project snapshots
    step_2_train: Fit BOPDMD model
    step_3_evaluate: Compute predictions and metrics

Modules:
    utils: Configuration and DMD forecasting utilities
    data: Data loading and POD computation

Training Modes:
    multi_trajectory: Train on full trajectories, test on different ICs
    temporal_split: Train on first n snapshots, predict the rest

Usage:
    # Full pipeline with temporal split mode
    python dmd/step_1_preprocess.py --config config/dmd_temporal_split.yaml
    python dmd/step_2_train.py --config config/dmd_temporal_split.yaml --run-dir <run_dir>
    python dmd/step_3_evaluate.py --config config/dmd_temporal_split.yaml --run-dir <run_dir>
    
    # Multi-trajectory mode (original behavior)
    python dmd/step_1_preprocess.py --config config/dmd_1train_5test.yaml
    python dmd/step_2_train.py --config config/dmd_1train_5test.yaml --run-dir <run_dir>
    python dmd/step_3_evaluate.py --config config/dmd_1train_5test.yaml --run-dir <run_dir>

Physics-Based Gamma:
    Γ_n = -∫d²x ñ ∂φ̃/∂y  (particle flux)
    Γ_c = c1∫d²x (ñ - φ̃)²  (conductive flux)

Author: Anthony Poole
"""

__version__ = "0.1.0"
