"""
Step 1: Data Preprocessing and POD Basis Computation.

This script orchestrates:
1. Loading raw simulation data
2. Computing POD basis via method of snapshots
3. Projecting data onto POD basis
4. Saving initial conditions and boundaries
5. Saving full POD basis for later reconstruction

Supports two training modes (set via config):
- "multi_trajectory": Train on full trajectories, test on different ICs
- "temporal_split": Train on first n snapshots of one trajectory, predict rest

Usage:
    python step_1_preprocess.py --config config.yaml

Author: Anthony Poole
"""

import argparse
import os
import time
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmd.utils import (
    load_dmd_config, get_dmd_output_paths, print_dmd_config_summary, save_config,
)
from dmd.data import (
    load_trajectory, compute_pod_basis, project_data, save_basis_and_preprocessing,
)
from opinf.utils import (
    setup_logging, save_step_status, get_run_directory, print_header,
)


def main():
    """Main entry point for Step 1."""
    parser = argparse.ArgumentParser(description="Step 1: Data Preprocessing")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_dmd_config(args.config)
    
    # Get/create run directory
    run_dir = get_run_directory(cfg, args.run_dir)
    cfg.run_dir = run_dir
    
    # Set up logging
    logger = setup_logging("step_1", run_dir, cfg.log_level)
    
    print_header("STEP 1: DATA PREPROCESSING (DMD)")
    print(f"  Run directory: {run_dir}")
    print(f"  Training mode: {cfg.training_mode}")
    print_dmd_config_summary(cfg)
    
    save_step_status(run_dir, "step_1", "running")
    save_config(cfg, run_dir, step_name="step_1")
    
    paths = get_dmd_output_paths(run_dir)
    t_start = time.time()
    
    try:
        # =====================================================================
        # 1. Load training data
        # =====================================================================
        logger.info("Loading training data...")
        
        # Initialize grid dimensions (will be set by load_trajectory)
        n_y, n_x = cfg.n_y, cfg.n_x
        
        if cfg.training_mode == "temporal_split":
            # Single trajectory with explicit train/test ranges
            Q_full, n_y, n_x = load_trajectory(cfg.training_files[0], cfg, logger)
            n_spatial, n_time = Q_full.shape
            
            # Get ranges from config
            train_start, train_end = cfg.train_start, cfg.train_end
            test_start, test_end = cfg.test_start, cfg.test_end
            
            # Validate
            if train_end > n_time or test_end > n_time:
                raise ValueError(f"Range exceeds file length ({n_time} snapshots)")
            
            # Extract ranges
            Q_train = Q_full[:, train_start:train_end]
            Q_test = Q_full[:, test_start:test_end]
            n_train = train_end - train_start
            n_test = test_end - test_start
            
            train_boundaries = np.array([0, n_train])
            test_boundaries = np.array([0, n_test])
            
            logger.info(f"  Temporal split:")
            logger.info(f"    Train: snapshots [{train_start}, {train_end}) = {n_train}")
            logger.info(f"    Test:  snapshots [{test_start}, {test_end}) = {n_test}")
        else:
            # Multi-trajectory mode (original behavior)
            Q_train_list = []
            for fp in cfg.training_files:
                Q, n_y, n_x = load_trajectory(fp, cfg, logger)
                Q_train_list.append(Q)
            
            # Stack training trajectories
            train_timesteps = [Q.shape[1] for Q in Q_train_list]
            Q_train = np.hstack(Q_train_list)
            train_boundaries = np.array([0] + list(np.cumsum(train_timesteps)))
            n_spatial = Q_train.shape[0]
            
            # Load test trajectories
            Q_test_list = []
            for fp in cfg.test_files:
                Q, _, _ = load_trajectory(fp, cfg, logger)
                Q_test_list.append(Q)
            
            test_timesteps = [Q.shape[1] for Q in Q_test_list]
            Q_test = np.hstack(Q_test_list)
            test_boundaries = np.array([0] + list(np.cumsum(test_timesteps)))
        
        logger.info(f"  Q_train shape: {Q_train.shape}")
        logger.info(f"  Q_test shape: {Q_test.shape}")
        
        # =====================================================================
        # 2. Center data
        # =====================================================================
        if cfg.centering_enabled:
            logger.info("Centering data...")
            train_mean = np.mean(Q_train, axis=1, keepdims=True)
            Q_train_centered = Q_train - train_mean
            Q_test_centered = Q_test - train_mean  # Use training mean for test
        else:
            Q_train_centered = Q_train
            Q_test_centered = Q_test
            train_mean = np.zeros((n_spatial, 1))
        
        # =====================================================================
        # 3. Compute POD basis
        # =====================================================================
        U_r, S, V = compute_pod_basis(Q_train_centered, cfg.r, logger)
        
        # Report energy captured
        energy = np.cumsum(S**2) / np.sum(S**2)
        logger.info(f"  POD energy captured (r={cfg.r}): {energy[cfg.r-1]*100:.4f}%")
        
        # =====================================================================
        # 4. Project data
        # =====================================================================
        Xhat_train = project_data(Q_train_centered, U_r, logger, "training")
        Xhat_test = project_data(Q_test_centered, U_r, logger, "test")
        
        # =====================================================================
        # 5. Extract initial conditions
        # =====================================================================
        if cfg.training_mode == "temporal_split":
            # For temporal split: train IC is first snapshot, test IC is split point
            train_ICs = Q_train_centered[:, 0:1]  # (n_spatial, 1)
            test_ICs = Q_test_centered[:, 0:1]    # (n_spatial, 1)
            train_ICs_reduced = Xhat_train[0:1, :]  # (1, r)
            test_ICs_reduced = Xhat_test[0:1, :]    # (1, r)
        else:
            # Multi-trajectory: IC at start of each trajectory
            n_train_traj = len(train_boundaries) - 1
            n_test_traj = len(test_boundaries) - 1
            
            train_ICs = np.zeros((n_spatial, n_train_traj))
            train_ICs_reduced = np.zeros((n_train_traj, cfg.r))
            for i in range(n_train_traj):
                idx = train_boundaries[i]
                train_ICs[:, i] = Q_train_centered[:, idx]
                train_ICs_reduced[i, :] = Xhat_train[idx, :]
            
            test_ICs = np.zeros((n_spatial, n_test_traj))
            test_ICs_reduced = np.zeros((n_test_traj, cfg.r))
            for i in range(n_test_traj):
                idx = test_boundaries[i]
                test_ICs[:, i] = Q_test_centered[:, idx]
                test_ICs_reduced[i, :] = Xhat_test[idx, :]
        
        # =====================================================================
        # 6. Save outputs
        # =====================================================================
        save_basis_and_preprocessing(
            run_dir=run_dir,
            U_r=U_r,
            S=S,
            train_mean=train_mean.squeeze(),
            Xhat_train=Xhat_train,
            Xhat_test=Xhat_test,
            train_boundaries=train_boundaries,
            test_boundaries=test_boundaries,
            train_ICs=train_ICs,
            test_ICs=test_ICs,
            train_ICs_reduced=train_ICs_reduced,
            test_ICs_reduced=test_ICs_reduced,
            n_y=n_y,
            n_x=n_x,
            r_actual=cfg.r,
            training_mode=cfg.training_mode,
            logger=logger,
        )
        
        # Final timing
        t_elapsed = time.time() - t_start
        
        save_step_status(run_dir, "step_1", "completed", {
            "r": cfg.r,
            "n_train_snapshots": Q_train.shape[1],
            "n_test_snapshots": Q_test.shape[1],
            "training_mode": cfg.training_mode,
            "time_seconds": t_elapsed,
        })
        
        print_header("STEP 1 COMPLETE")
        print(f"  Run directory: {run_dir}")
        print(f"  POD modes: {cfg.r}")
        print(f"  Energy captured: {energy[cfg.r-1]*100:.4f}%")
        print(f"  Runtime: {t_elapsed:.1f}s")
        
    except Exception as e:
        logger.error(f"Step 1 failed: {e}", exc_info=True)
        save_step_status(run_dir, "step_1", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
