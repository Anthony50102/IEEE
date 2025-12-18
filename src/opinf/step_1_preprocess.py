"""
Step 1: Data Preprocessing and POD Computation.

This script handles:
1. Loading raw simulation data from HDF5 files
2. Computing POD basis via SVD
3. Projecting training and test data onto POD basis
4. Preparing learning matrices for ROM training
5. Saving all intermediate data

Usage:
    python step_1_preprocess.py --config config.yaml
    python step_1_preprocess.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import gc
import time
import numpy as np
import xarray as xr
import os

from utils import (
    load_config,
    save_config,
    get_run_directory,
    get_data_directory,
    setup_logging,
    save_step_status,
    get_output_paths,
    print_header,
    print_config_summary,
    get_memmap_path,
    cleanup_memmap,
    get_x_sq,
    loader,
    compute_truncation_snapshots,
    PipelineConfig,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_process_snapshots(
    file_path: str,
    index: int,
    engine: str = "h5netcdf",
    max_snapshots: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load and process a single snapshot file into flattened array.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    index : int
        Index of this file (for logging).
    engine : str
        HDF5 engine for xarray.
    max_snapshots : int, optional
        Maximum snapshots to keep (truncation).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    np.ndarray
        Processed array of shape (n_spatial, n_time).
    """
    if verbose:
        print(f"  Loading file {index + 1}: {os.path.basename(file_path)}")
    
    fh = xr.open_dataset(file_path, engine=engine, phony_dims="sort")
    
    density = fh["density"].values
    phi = fh["phi"].values
    fh.close()
    
    # Apply truncation
    original_n_time = density.shape[0]
    if max_snapshots is not None and max_snapshots < original_n_time:
        density = density[:max_snapshots]
        phi = phi[:max_snapshots]
        if verbose:
            print(f"    Truncated: {original_n_time} -> {max_snapshots} snapshots")
    
    # Handle 2D vs 3D input
    if density.ndim == 2:
        n_time = density.shape[0]
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    # Stack fields: (time, y, x) -> (n_spatial, time)
    Q = np.stack([density, phi], axis=0)  # (2, time, y, x)
    del density, phi
    
    Q = Q.transpose(0, 2, 3, 1)  # (2, y, x, time)
    n_field, n_y, n_x, n_time = Q.shape
    Q = Q.reshape(n_field * n_y * n_x, n_time)
    
    if verbose:
        print(f"    Shape: {Q.shape}")
    
    return Q


def load_all_data(
    cfg: PipelineConfig,
    run_dir: str,
    logger,
) -> tuple:
    """
    Load all training and test data into memory-mapped arrays.
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object.
    run_dir : str
        Run directory.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Q_train, Q_test, train_boundaries, test_boundaries, n_spatial)
    """
    logger.info("Loading simulation data...")
    
    # Determine sizes and truncation settings
    train_timesteps = []
    test_timesteps = []
    train_truncations = []
    test_truncations = []
    n_spatial = None
    
    # First pass: determine shapes
    for file_path in cfg.training_files:
        with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
            n_time_original = fh["density"].shape[0]
            if n_spatial is None:
                if fh["density"].ndim == 3:
                    n_spatial = 2 * fh["density"].shape[1] * fh["density"].shape[2]
                else:
                    n_spatial = 2 * fh["density"].shape[1]
        
        if cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            n_time = min(n_time_original, max_snaps) if max_snaps else n_time_original
        else:
            n_time = n_time_original
            max_snaps = None
        
        train_timesteps.append(n_time)
        train_truncations.append(max_snaps)
    
    for file_path in cfg.test_files:
        with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
            n_time_original = fh["density"].shape[0]
        
        if cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            n_time = min(n_time_original, max_snaps) if max_snaps else n_time_original
        else:
            n_time = n_time_original
            max_snaps = None
        
        test_timesteps.append(n_time)
        test_truncations.append(max_snaps)
    
    total_train = sum(train_timesteps)
    total_test = sum(test_timesteps)
    
    logger.info(f"  Spatial DOF: {n_spatial:,}")
    logger.info(f"  Total train snapshots: {total_train:,}")
    logger.info(f"  Total test snapshots: {total_test:,}")
    
    # Create memory-mapped arrays
    cleanup_memmap(run_dir, "Q_train")
    cleanup_memmap(run_dir, "Q_test")
    
    Q_train = np.memmap(
        get_memmap_path(run_dir, "Q_train"),
        dtype='float64',
        mode='w+',
        shape=(n_spatial, total_train)
    )
    Q_test = np.memmap(
        get_memmap_path(run_dir, "Q_test"),
        dtype='float64',
        mode='w+',
        shape=(n_spatial, total_test)
    )
    
    # Compute boundaries
    train_boundaries = [0] + list(np.cumsum(train_timesteps))
    test_boundaries = [0] + list(np.cumsum(test_timesteps))
    
    # Load training data
    logger.info("Loading training trajectories...")
    for i, file_path in enumerate(cfg.training_files):
        Q_ic = load_and_process_snapshots(
            file_path, i, cfg.engine, train_truncations[i], cfg.verbose
        )
        Q_train[:, train_boundaries[i]:train_boundaries[i + 1]] = Q_ic
        del Q_ic
        gc.collect()
    
    # Load test data
    logger.info("Loading test trajectories...")
    for i, file_path in enumerate(cfg.test_files):
        Q_ic = load_and_process_snapshots(
            file_path, i, cfg.engine, test_truncations[i], cfg.verbose
        )
        Q_test[:, test_boundaries[i]:test_boundaries[i + 1]] = Q_ic
        del Q_ic
        gc.collect()
    
    return Q_train, Q_test, train_boundaries, test_boundaries, n_spatial


# =============================================================================
# POD COMPUTATION
# =============================================================================

def compute_pod(Q_train: np.ndarray, logger) -> tuple:
    """
    Compute POD basis from training data.
    
    Parameters
    ----------
    Q_train : np.ndarray
        Training data matrix (n_spatial, n_time).
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (U, S) - POD modes and singular values.
    """
    logger.info("Computing POD basis via SVD...")
    start_time = time.time()
    
    U, S, _ = np.linalg.svd(Q_train, full_matrices=False)
    
    elapsed = time.time() - start_time
    logger.info(f"  SVD completed in {elapsed:.1f} seconds")
    logger.info(f"  U shape: {U.shape}, S shape: {S.shape}")
    
    return U, S


# =============================================================================
# PROJECTION
# =============================================================================

def project_data(
    Q_train: np.ndarray,
    Q_test: np.ndarray,
    U: np.ndarray,
    r: int,
    logger,
) -> tuple:
    """
    Project training and test data onto POD basis.
    
    Parameters
    ----------
    Q_train, Q_test : np.ndarray
        Data matrices.
    U : np.ndarray
        POD modes.
    r : int
        Number of modes to use.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Xhat_train, Xhat_test)
    """
    logger.info(f"Projecting data onto {r} POD modes...")
    
    Ur = U[:, :r]
    
    Xhat_train = Q_train.T @ Ur  # (n_time, r)
    Xhat_test = Q_test.T @ Ur
    
    logger.info(f"  Xhat_train shape: {Xhat_train.shape}")
    logger.info(f"  Xhat_test shape: {Xhat_test.shape}")
    
    return Xhat_train, Xhat_test


# =============================================================================
# LEARNING MATRIX PREPARATION
# =============================================================================

def prepare_learning_matrices(
    Xhat_train: np.ndarray,
    train_boundaries: list,
    cfg: PipelineConfig,
    logger,
) -> dict:
    """
    Prepare matrices for ROM training.
    
    Parameters
    ----------
    Xhat_train : np.ndarray
        Projected training data.
    train_boundaries : list
        Trajectory boundaries.
    cfg : PipelineConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Dictionary of learning matrices.
    """
    logger.info("Preparing learning matrices...")
    r = cfg.r
    
    # STATE LEARNING: Create valid pairs within each trajectory
    n_train_traj = len(train_boundaries) - 1
    X_state_list = []
    Y_state_list = []
    
    for traj_idx in range(n_train_traj):
        start_idx = train_boundaries[traj_idx]
        end_idx = train_boundaries[traj_idx + 1]
        
        Xhat_traj = Xhat_train[start_idx:end_idx, :]
        X_state_list.append(Xhat_traj[:-1, :])
        Y_state_list.append(Xhat_traj[1:, :])
    
    X_state = np.vstack(X_state_list)
    Y_state = np.vstack(Y_state_list)
    
    s = int(r * (r + 1) / 2)
    X_state2 = get_x_sq(X_state)
    D_state = np.concatenate((X_state, X_state2), axis=1)
    D_state_2 = D_state.T @ D_state
    
    logger.info(f"  State pairs: {X_state.shape[0]}")
    
    # OUTPUT LEARNING: Use all timesteps
    X_out = Xhat_train
    K = X_out.shape[0]
    E = np.ones((K, 1))
    
    mean_Xhat = np.mean(X_out, axis=0)
    Xhat_out = X_out - mean_Xhat[np.newaxis, :]
    
    scaling_Xhat = np.maximum(np.abs(np.min(X_out)), np.abs(np.max(X_out)))
    Xhat_out /= scaling_Xhat
    Xhat_out2 = get_x_sq(Xhat_out)
    
    D_out = np.concatenate((Xhat_out, Xhat_out2, E), axis=1)
    D_out_2 = D_out.T @ D_out
    
    logger.info(f"  D_out shape: {D_out.shape}")
    logger.info(f"  D_out_2 condition: {np.linalg.cond(D_out_2):.2e}")
    
    return {
        'X_state': X_state,
        'Y_state': Y_state,
        'D_state': D_state,
        'D_state_2': D_state_2,
        'D_out': D_out,
        'D_out_2': D_out_2,
        'mean_Xhat': mean_Xhat,
        'scaling_Xhat': scaling_Xhat,
    }


def load_reference_gamma(
    cfg: PipelineConfig,
    logger,
) -> dict:
    """
    Load reference Gamma values from training files.
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Dictionary with Y_Gamma and reference statistics.
    """
    logger.info("Loading reference Gamma values...")
    
    Gamma_n_list = []
    Gamma_c_list = []
    
    for file_path in cfg.training_files:
        fh = loader(file_path, ENGINE=cfg.engine)
        
        # Get original gamma data
        gamma_n = fh["gamma_n"].data
        gamma_c = fh["gamma_c"].data
        
        # Apply truncation (same logic as for snapshots)
        if cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            if max_snaps is not None:
                n_time_original = len(gamma_n)
                n_time = min(n_time_original, max_snaps)
                gamma_n = gamma_n[:n_time]
                gamma_c = gamma_c[:n_time]
                logger.info(f"  Truncated gamma for {os.path.basename(file_path)}: "
                          f"{n_time_original} -> {n_time}")
        
        Gamma_n_list.append(gamma_n)
        Gamma_c_list.append(gamma_c)
    
    Gamma_n = np.concatenate(Gamma_n_list)
    Gamma_c = np.concatenate(Gamma_c_list)
    
    Y_Gamma = np.vstack((Gamma_n, Gamma_c))
    
    logger.info(f"  Y_Gamma shape: {Y_Gamma.shape}")
    logger.info(f"  Gamma_n: mean={np.mean(Gamma_n):.4e}, std={np.std(Gamma_n, ddof=1):.4e}")
    logger.info(f"  Gamma_c: mean={np.mean(Gamma_c):.4e}, std={np.std(Gamma_c, ddof=1):.4e}")
    
    return {
        'Y_Gamma': Y_Gamma,
        'mean_Gamma_n': np.mean(Gamma_n),
        'std_Gamma_n': np.std(Gamma_n, ddof=1),
        'mean_Gamma_c': np.mean(Gamma_c),
        'std_Gamma_c': np.std(Gamma_c, ddof=1),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 1."""
    parser = argparse.ArgumentParser(
        description="Step 1: Data Preprocessing and POD Computation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run directory (creates new if not specified)"
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Get/create run directory
    run_dir = get_run_directory(cfg, args.run_dir)
    data_dir = get_data_directory(cfg, args.data_dir)
    
    # Set up logging
    logger = setup_logging("step_1", run_dir, cfg.log_level)
    
    print_header("STEP 1: DATA PREPROCESSING AND POD COMPUTATION")
    print(f"  Run directory: {run_dir}")
    print(f"  Data directory: {data_dir}")
    print_config_summary(cfg)
    
    logger.info(f"Run directory: {run_dir}")
    save_step_status(run_dir, "step_1", "running")
    
    # Save configuration
    save_config(cfg, run_dir)
    logger.info("Configuration saved to run directory")
    
    paths = get_output_paths(run_dir)
    
    try:
        # 1. Load data
        Q_train, Q_test, train_boundaries, test_boundaries, n_spatial = load_all_data(
            cfg, run_dir, logger
        )
        
        # Save boundaries
        np.savez(
            paths["boundaries"],
            train_boundaries=train_boundaries,
            test_boundaries=test_boundaries,
            n_spatial=n_spatial,
        )
        logger.info(f"Saved boundaries to {paths['boundaries']}")
        
        # 2. Compute POD
        U, S = compute_pod(Q_train, logger)
        
        np.savez(paths["pod_file"], U=U, S=S)
        logger.info(f"Saved POD to {paths['pod_file']}")
        
        # 3. Project data
        Xhat_train, Xhat_test = project_data(Q_train, Q_test, U, cfg.r, logger)
        
        np.save(paths["xhat_train"], Xhat_train)
        np.save(paths["xhat_test"], Xhat_test)
        logger.info("Saved projected data")
        
        # 4. Save initial conditions
        train_ICs = np.array([Q_train[:, train_boundaries[i]] 
                             for i in range(len(cfg.training_files))])
        test_ICs = np.array([Q_test[:, test_boundaries[i]] 
                            for i in range(len(cfg.test_files))])
        train_ICs_reduced = np.array([Xhat_train[train_boundaries[i], :] 
                                      for i in range(len(cfg.training_files))])
        test_ICs_reduced = np.array([Xhat_test[test_boundaries[i], :] 
                                     for i in range(len(cfg.test_files))])
        
        np.savez(
            paths["initial_conditions"],
            train_ICs=train_ICs,
            test_ICs=test_ICs,
            train_ICs_reduced=train_ICs_reduced,
            test_ICs_reduced=test_ICs_reduced,
        )
        logger.info("Saved initial conditions")
        
        del train_ICs, test_ICs
        gc.collect()
        
        # 5. Prepare learning matrices
        learning = prepare_learning_matrices(Xhat_train, train_boundaries, cfg, logger)
        
        # 6. Load reference Gamma
        gamma_ref = load_reference_gamma(cfg, logger)
        
        # Save learning matrices
        np.savez(
            paths["learning_matrices"],
            X_state=learning['X_state'],
            Y_state=learning['Y_state'],
            D_state=learning['D_state'],
            D_state_2=learning['D_state_2'],
            D_out=learning['D_out'],
            D_out_2=learning['D_out_2'],
            mean_Xhat=learning['mean_Xhat'],
            scaling_Xhat=learning['scaling_Xhat'],
        )
        logger.info(f"Saved learning matrices to {paths['learning_matrices']}")
        
        # Save gamma reference
        np.savez(
            paths["gamma_ref"],
            Y_Gamma=gamma_ref['Y_Gamma'],
            mean_Gamma_n=gamma_ref['mean_Gamma_n'],
            std_Gamma_n=gamma_ref['std_Gamma_n'],
            mean_Gamma_c=gamma_ref['mean_Gamma_c'],
            std_Gamma_c=gamma_ref['std_Gamma_c'],
        )
        logger.info(f"Saved gamma reference to {paths['gamma_ref']}")
        
        # Cleanup
        del Q_train, Q_test
        gc.collect()
        cleanup_memmap(run_dir, "Q_train")
        cleanup_memmap(run_dir, "Q_test")
        logger.info("Cleaned up temporary files")
        
        # Mark complete
        save_step_status(run_dir, "step_1", "completed", {
            "n_spatial": int(n_spatial),
            "n_train_snapshots": int(train_boundaries[-1]),
            "n_test_snapshots": int(test_boundaries[-1]),
        })
        
        print_header("STEP 1 COMPLETE")
        print(f"  Output directory: {run_dir}")
        logger.info("Step 1 completed successfully")
        
    except Exception as e:
        logger.error(f"Step 1 failed: {e}", exc_info=True)
        save_step_status(run_dir, "step_1", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
