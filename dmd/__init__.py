"""
DMD (Dynamic Mode Decomposition) ROM Pipeline.

This package implements Optimized DMD (opt-DMD via BOPDMD) for 
reduced-order modeling of the Hasegawa-Wakatani system.

Modules:
    utils: DMD-specific utility functions
    step_2_fit_dmd: Fit BOPDMD model to training data
    step_3_evaluate_dmd: Evaluate model on test trajectories

Usage:
    # Step 1: Run the standard preprocessing (shared with OpInf)
    python opinf/step_1_parallel_preprocess.py --config config/dmd_1train_5test.yaml
    
    # Step 2: Fit DMD model
    python dmd/step_2_fit_dmd.py --config config/dmd_1train_5test.yaml --run-dir <run_dir>
    
    # Step 3: Evaluate predictions  
    python dmd/step_3_evaluate_dmd.py --config config/dmd_1train_5test.yaml --run-dir <run_dir>

Author: Anthony Poole
"""

__version__ = "0.1.0"
