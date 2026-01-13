"""
Step 1: Serial Data Preprocessing and Dimensionality Reduction.

This is a non-MPI serial version of step_1_preprocess.py that produces
identical outputs. Use this when MPI is not available or for debugging.

Supports two reduction methods (set via config):
- "linear": Standard POD via Gram matrix eigendecomposition
- "manifold": Quadratic manifold via greedy mode selection

This script orchestrates:
1. Loading of raw simulation data
2. Computing basis (POD or quadratic manifold)
3. Projecting training and test data onto basis
4. Preparing learning matrices for ROM training

Usage:
    python step_1_preprocess_serial.py --config config.yaml
    python step_1_preprocess_serial.py --config config.yaml --save-pod-energy

Author: Anthony Poole
"""

import argparse
import gc
import os
import time
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config, save_config, get_run_directory, setup_logging,
    save_step_status, get_output_paths, print_header, print_config_summary,
    compute_truncation_snapshots, load_dataset,
)
from pod import (
    compute_manifold_greedy, BasisData, save_basis,
    encode, decode, reconstruction_error,
)
from training import prepare_learning_matrices_serial
from shared.plotting import plot_pod_energy


# =============================================================================
# SERIAL DATA LOADING
# =============================================================================

def get_file_metadata(cfg, file_path: str) -> tuple:
    """Get metadata from a file without loading full data."""
    with load_dataset(file_path, cfg.engine) as fh:
        n_time = fh["density"].shape[0]
        if fh["density"].ndim == 3:
            n_y, n_x = fh["density"].shape[1], fh["density"].shape[2]
        else:
            n_y = n_x = int(np.sqrt(fh["density"].shape[1]))
        n_spatial = cfg.n_fields * n_y * n_x
    
    if cfg.truncation_enabled:
        max_snaps = compute_truncation_snapshots(
            file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
        )
        n_time = min(n_time, max_snaps) if max_snaps else n_time
    else:
        max_snaps = None
    
    return n_spatial, n_time, max_snaps


def load_snapshots(file_path: str, engine: str, max_snapshots=None) -> np.ndarray:
    """Load all snapshots from a file."""
    with load_dataset(file_path, engine) as fh:
        density = fh["density"].values
        phi = fh["phi"].values
    
    if max_snapshots is not None and max_snapshots < density.shape[0]:
        density = density[:max_snapshots]
        phi = phi[:max_snapshots]
    
    n_time = density.shape[0]
    
    # Reshape if needed
    if density.ndim == 2:
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    # Stack fields: (n_fields, n_time, n_y, n_x) -> (n_spatial, n_time)
    Q_full = np.stack([density, phi], axis=0).transpose(0, 2, 3, 1)
    n_field, n_y, n_x, n_time = Q_full.shape
    Q_full = Q_full.reshape(n_field * n_y * n_x, n_time)
    
    return Q_full


def load_all_data_serial(cfg, logger) -> tuple:
    """Load all training and test data (serial version)."""
    
    if cfg.training_mode == "temporal_split":
        return load_temporal_split_serial(cfg, logger)
    
    # Multi-trajectory mode
    logger.info("Loading simulation data (serial)...")
    
    # Get metadata
    train_timesteps, test_timesteps = [], []
    train_truncations, test_truncations = [], []
    n_spatial = None
    
    for fp in cfg.training_files:
        ns, nt, max_snaps = get_file_metadata(cfg, fp)
        n_spatial = n_spatial or ns
        train_timesteps.append(nt)
        train_truncations.append(max_snaps)
    
    for fp in cfg.test_files:
        _, nt, max_snaps = get_file_metadata(cfg, fp)
        test_timesteps.append(nt)
        test_truncations.append(max_snaps)
    
    total_train = sum(train_timesteps)
    total_test = sum(test_timesteps)
    
    logger.info(f"  Spatial DOF: {n_spatial:,}")
    logger.info(f"  Train snapshots: {total_train:,}, Test snapshots: {total_test:,}")
    
    # Allocate arrays
    Q_train = np.zeros((n_spatial, total_train), dtype=np.float64)
    Q_test = np.zeros((n_spatial, total_test), dtype=np.float64)
    
    # Compute boundaries
    train_boundaries = [0] + list(np.cumsum(train_timesteps))
    test_boundaries = [0] + list(np.cumsum(test_timesteps))
    
    # Load data
    for i, fp in enumerate(cfg.training_files):
        logger.info(f"  Loading training file {i+1}/{len(cfg.training_files)}: {os.path.basename(fp)}")
        Q = load_snapshots(fp, cfg.engine, train_truncations[i])
        Q_train[:, train_boundaries[i]:train_boundaries[i + 1]] = Q
        del Q
        gc.collect()
    
    for i, fp in enumerate(cfg.test_files):
        logger.info(f"  Loading test file {i+1}/{len(cfg.test_files)}: {os.path.basename(fp)}")
        Q = load_snapshots(fp, cfg.engine, test_truncations[i])
        Q_test[:, test_boundaries[i]:test_boundaries[i + 1]] = Q
        del Q
        gc.collect()
    
    return Q_train, Q_test, train_boundaries, test_boundaries, n_spatial


def load_temporal_split_serial(cfg, logger) -> tuple:
    """Load data for temporal split mode (serial version)."""
    logger.info("Loading simulation data (temporal split mode, serial)...")
    
    # Get metadata from single training file
    fp = cfg.training_files[0]
    n_spatial, n_time_total, _ = get_file_metadata(cfg, fp)
    
    # Get ranges from config
    train_start, train_end = cfg.train_start, cfg.train_end
    test_start, test_end = cfg.test_start, cfg.test_end
    
    # Validate ranges
    if train_end > n_time_total or test_end > n_time_total:
        raise ValueError(f"Range exceeds file length ({n_time_total} snapshots)")
    if train_start >= train_end:
        raise ValueError(f"Invalid train range: [{train_start}, {train_end})")
    if test_start >= test_end:
        raise ValueError(f"Invalid test range: [{test_start}, {test_end})")
    
    n_train = train_end - train_start
    n_test = test_end - test_start
    
    logger.info(f"  Train: snapshots [{train_start}, {train_end}) = {n_train} snapshots")
    logger.info(f"  Test:  snapshots [{test_start}, {test_end}) = {n_test} snapshots")
    logger.info(f"  Spatial DOF: {n_spatial:,}")
    
    # Load full trajectory
    max_snap_needed = max(train_end, test_end)
    Q_full = load_snapshots(fp, cfg.engine, max_snap_needed)
    
    # Extract train and test ranges
    Q_train = Q_full[:, train_start:train_end].copy()
    Q_test = Q_full[:, test_start:test_end].copy()
    del Q_full
    gc.collect()
    
    # Boundaries for single trajectory each
    train_boundaries = [0, n_train]
    test_boundaries = [0, n_test]
    
    return Q_train, Q_test, train_boundaries, test_boundaries, n_spatial


# =============================================================================
# SERIAL DATA PREPROCESSING
# =============================================================================

def center_data_serial(Q: np.ndarray, logger) -> tuple:
    """Center data by subtracting temporal mean at each spatial location."""
    logger.info("Centering data (subtracting temporal mean)...")
    logger.debug(f"  [DIAG] Raw data range: [{Q.min():.2e}, {Q.max():.2e}]")
    
    temporal_mean = np.mean(Q, axis=1, keepdims=True)
    Q_centered = Q - temporal_mean
    
    logger.debug(f"  [DIAG] Centered data range: [{Q_centered.min():.2e}, {Q_centered.max():.2e}]")
    
    return Q_centered, temporal_mean.squeeze()


def scale_data_serial(Q: np.ndarray, n_fields: int, logger) -> tuple:
    """Scale data so each field's values are in [-1, 1]."""
    logger.info("Scaling data (normalizing each field to [-1, 1])...")
    
    n_spatial = Q.shape[0]
    n_per_field = n_spatial // n_fields
    
    scaling_factors = np.zeros(n_fields)
    Q_scaled = Q.copy()
    
    for j in range(n_fields):
        start, end = j * n_per_field, (j + 1) * n_per_field
        max_val = np.max(np.abs(Q[start:end, :]))
        
        if max_val > 0:
            Q_scaled[start:end, :] /= max_val
            scaling_factors[j] = max_val
        else:
            scaling_factors[j] = 1.0
    
    logger.info(f"  Scaling factors: {scaling_factors}")
    
    return Q_scaled, scaling_factors


# =============================================================================
# SERIAL POD COMPUTATION
# =============================================================================

def compute_pod_serial(Q_train: np.ndarray, logger, target_energy: float = 0.9999) -> tuple:
    """
    Compute POD basis via Gram matrix eigendecomposition (serial version).
    
    Uses the method of snapshots (Sirovich):
    1. Compute Gram matrix D = Q.T @ Q
    2. Eigendecomposition of D
    
    Returns same outputs as compute_pod_distributed:
        eigs, eigv, D_global, r_energy
    """
    logger.info("Computing POD basis via Gram matrix (serial)...")
    
    t0 = time.time()
    
    n_spatial, n_time = Q_train.shape
    logger.debug(f"  [DIAG] Q_train shape: {Q_train.shape}")
    
    # Compute Gram matrix: D = Q.T @ Q
    D_global = Q_train.T @ Q_train
    
    logger.debug(f"  [DIAG] D_global shape: {D_global.shape}")
    logger.debug(f"  [DIAG] D_global trace: {np.trace(D_global):.2e}")
    
    # Eigendecomposition
    try:
        from scipy.linalg import eigh as scipy_eigh
        logger.info("  Using scipy.linalg.eigh...")
        eigs, eigv = scipy_eigh(D_global.copy())
    except ImportError:
        logger.info("  Using numpy.linalg.eigh...")
        eigs, eigv = np.linalg.eigh(D_global.copy())
    
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigs)[::-1]
    eigs, eigv = eigs[idx], eigv[:, idx]
    
    logger.debug(f"  [DIAG] Eigenvalue sum: {np.sum(eigs):.2e}")
    logger.debug(f"  [DIAG] Top 5 eigenvalues: {eigs[:5]}")
    
    # Compute retained energy
    eigs_positive = np.maximum(eigs, 0)
    total_energy = np.sum(eigs_positive)
    if total_energy > 0:
        ret_energy = np.cumsum(eigs_positive) / total_energy
        r_energy = np.argmax(ret_energy >= target_energy) + 1
    else:
        r_energy = 1
    
    elapsed = time.time() - t0
    logger.info(f"  POD computed in {elapsed:.1f}s")
    logger.info(f"  r for {target_energy*100:.2f}% energy: {r_energy}")
    
    return eigs, eigv, D_global, r_energy


def project_data_serial(
    Q_train: np.ndarray, Q_test: np.ndarray,
    eigv: np.ndarray, eigs: np.ndarray, r: int, D_global: np.ndarray, logger
) -> tuple:
    """
    Project data onto POD basis (serial version).
    
    Returns same outputs as project_data_distributed:
        Xhat_train, Xhat_test, Ur_local (=Ur), Ur_full (=Ur)
    """
    logger.info(f"Projecting data onto {r} POD modes...")
    
    # Transformation matrix: Tr = V_r @ diag(1/sqrt(eigs_r))
    eigs_r = eigs[:r]
    eigv_r = eigv[:, :r]
    
    # Handle problematic eigenvalues
    eigs_r_safe = np.where(eigs_r > 1e-14, eigs_r, 1e-14)
    Tr = eigv_r @ np.diag(eigs_r_safe ** (-0.5))
    
    logger.debug(f"  [DIAG] Tr shape: {Tr.shape}")
    logger.debug(f"  [DIAG] Tr has NaN: {np.any(np.isnan(Tr))}")
    
    # Verify Tr.T @ D @ Tr ≈ I
    TrTDTr_err = np.linalg.norm(Tr.T @ D_global @ Tr - np.eye(r))
    logger.debug(f"  [DIAG] ||Tr.T @ D @ Tr - I||: {TrTDTr_err:.2e}")
    
    # Reduced training coordinates: Xhat_train = Tr.T @ D_global
    Xhat_train = (Tr.T @ D_global).T
    
    # POD modes: Ur = Q_train @ Tr
    Ur = Q_train @ Tr
    
    # Verify orthonormality: Ur.T @ Ur ≈ I
    UtU = Ur.T @ Ur
    logger.debug(f"  [DIAG] ||Ur.T @ Ur - I||: {np.linalg.norm(UtU - np.eye(r)):.2e}")
    
    # Project test data: Xhat_test = Q_test.T @ Ur
    Xhat_test = Q_test.T @ Ur
    
    logger.info(f"  Xhat_train shape: {Xhat_train.shape}")
    logger.info(f"  Xhat_test shape: {Xhat_test.shape}")
    
    # Return Ur twice for API compatibility (local = full in serial)
    return Xhat_train, Xhat_test, Ur, Ur


# =============================================================================
# SERIAL REFERENCE DATA & INITIAL CONDITIONS
# =============================================================================

def load_reference_gamma_serial(cfg, logger) -> dict:
    """Load reference Gamma values from training files."""
    logger.info("Loading reference Gamma values...")
    
    Gamma_n_list, Gamma_c_list = [], []
    
    for fp in cfg.training_files:
        fh = load_dataset(fp, cfg.engine)
        gamma_n, gamma_c = fh["gamma_n"].values, fh["gamma_c"].values
        
        # Handle temporal_split mode: use explicit train range
        if cfg.training_mode == "temporal_split":
            train_start, train_end = cfg.train_start, cfg.train_end
            gamma_n = gamma_n[train_start:train_end]
            gamma_c = gamma_c[train_start:train_end]
            logger.info(f"  Temporal split: using gamma[{train_start}:{train_end}]")
        elif cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                fp, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            if max_snaps:
                gamma_n = gamma_n[:max_snaps]
                gamma_c = gamma_c[:max_snaps]
        
        Gamma_n_list.append(gamma_n)
        Gamma_c_list.append(gamma_c)
    
    Gamma_n = np.concatenate(Gamma_n_list)
    Gamma_c = np.concatenate(Gamma_c_list)
    Y_Gamma = np.vstack([Gamma_n, Gamma_c])
    
    logger.info(f"  Y_Gamma shape: {Y_Gamma.shape}")
    
    return {
        'Y_Gamma': Y_Gamma,
        'mean_Gamma_n': np.mean(Gamma_n), 'std_Gamma_n': np.std(Gamma_n, ddof=1),
        'mean_Gamma_c': np.mean(Gamma_c), 'std_Gamma_c': np.std(Gamma_c, ddof=1),
    }


def gather_initial_conditions_serial(
    Q_train, Q_test, Xhat_train, Xhat_test, train_boundaries, test_boundaries
) -> dict:
    """Extract initial conditions (serial version)."""
    n_train = len(train_boundaries) - 1
    n_test = len(test_boundaries) - 1
    
    train_ICs = np.array([Q_train[:, train_boundaries[i]] for i in range(n_train)])
    test_ICs = np.array([Q_test[:, test_boundaries[i]] for i in range(n_test)])
    
    train_ICs_reduced = np.array([Xhat_train[train_boundaries[i]] for i in range(n_train)])
    test_ICs_reduced = np.array([Xhat_test[test_boundaries[i]] for i in range(n_test)])
    
    return {
        'train_ICs': train_ICs,
        'test_ICs': test_ICs,
        'train_ICs_reduced': train_ICs_reduced,
        'test_ICs_reduced': test_ICs_reduced,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for serial Step 1."""
    parser = argparse.ArgumentParser(description="Step 1: Data Preprocessing (Serial)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory")
    parser.add_argument("--save-pod-energy", action="store_true", help="Save POD energy plot")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Get/create run directory
    run_dir = get_run_directory(cfg, args.run_dir)
    
    # Set up logging
    logger = setup_logging("step_1", run_dir, cfg.log_level, rank=0)
    
    method_name = "QUADRATIC MANIFOLD" if cfg.reduction_method == "manifold" else "POD"
    
    print_header(f"STEP 1: DATA PREPROCESSING AND {method_name} (SERIAL)")
    print(f"  Run directory: {run_dir}")
    print(f"  Reduction method: {cfg.reduction_method}")
    print_config_summary(cfg)
    save_step_status(run_dir, "step_1", "running")
    save_config(cfg, run_dir, step_name="step_1")
    
    paths = get_output_paths(run_dir)
    
    t_start = time.time()
    
    try:
        # =====================================================================
        # 1. Load data
        # =====================================================================
        Q_train, Q_test, train_boundaries, test_boundaries, n_spatial = load_all_data_serial(
            cfg, logger
        )
        
        np.savez(paths["boundaries"], train_boundaries=train_boundaries,
                 test_boundaries=test_boundaries, n_spatial=n_spatial)
        
        # =====================================================================
        # 2. Center data
        # =====================================================================
        if cfg.centering_enabled:
            Q_train_centered, train_mean = center_data_serial(Q_train, logger)
            Q_test_centered, test_mean = center_data_serial(Q_test, logger)
        else:
            Q_train_centered, Q_test_centered = Q_train, Q_test
            train_mean = test_mean = np.zeros(n_spatial)
        
        # =====================================================================
        # 3. Scale data (optional)
        # =====================================================================
        scaling_factors = None
        if cfg.scaling_enabled:
            Q_train_centered, scaling_factors = scale_data_serial(
                Q_train_centered, cfg.n_fields, logger
            )
            Q_test_centered, _ = scale_data_serial(
                Q_test_centered, cfg.n_fields, logger
            )
        
        # =====================================================================
        # 4. Dimensionality reduction (method-dependent)
        # =====================================================================
        if cfg.reduction_method == "manifold":
            # Compute quadratic manifold
            basis = compute_manifold_greedy(
                Q_train_centered, cfg.r, cfg.n_vectors_to_check, cfg.reg_magnitude, logger
            )
            
            # Project data
            Xhat_train = encode(Q_train_centered, basis).T  # (n_time, r)
            Xhat_test = encode(Q_test_centered, basis).T
            
            # Reconstruction error
            abs_err, rel_err = reconstruction_error(Q_train_centered, basis)
            logger.info(f"  Training reconstruction error: {rel_err*100:.4f}%")
            
            abs_err_test, rel_err_test = reconstruction_error(Q_test_centered, basis)
            logger.info(f"  Test reconstruction error: {rel_err_test*100:.4f}%")
            
            # Save
            save_basis(basis, paths["pod_basis"].replace(".npy", "_basis.npz"))
            np.save(paths["xhat_train"], Xhat_train)
            np.save(paths["xhat_test"], Xhat_test)
            np.save(paths["pod_basis"], basis.V)
            
            Ur_full = basis.V
            eigs = basis.eigs
            r_actual = basis.r
            r_energy = np.argmax(np.cumsum(eigs)/np.sum(eigs) >= cfg.target_energy) + 1
            selected_modes = basis.selected_indices
            
        else:  # Linear POD (default)
            # Gram matrix eigendecomposition
            eigs, eigv, D_global, r_energy = compute_pod_serial(
                Q_train_centered, logger, cfg.target_energy
            )
            r_actual = min(cfg.r, r_energy)
            selected_modes = np.arange(r_actual)  # Linear POD uses first r modes
            
            logger.info(f"  Using r={r_actual} (config: {cfg.r}, energy-based: {r_energy})")
            np.savez(paths["pod_file"], S=np.sqrt(np.maximum(eigs, 0)), eigs=eigs, eigv=eigv)
            if args.save_pod_energy:
                plot_pod_energy(eigs, r_actual, run_dir, logger)
            
            # Project data
            Xhat_train, Xhat_test, Ur_local, Ur_full = project_data_serial(
                Q_train_centered, Q_test_centered, eigv, eigs, r_actual, D_global, logger
            )
            
            np.save(paths["xhat_train"], Xhat_train)
            np.save(paths["xhat_test"], Xhat_test)
            np.save(paths["pod_basis"], Ur_full)
            
            # Compute reconstruction error
            shift = np.zeros(Q_train_centered.shape[0])  # Already centered
            basis = BasisData("linear", Ur_full, None, shift, r_actual, eigs)
            abs_err, rel_err = reconstruction_error(Q_train_centered, basis)
            logger.info(f"  Training reconstruction error: {rel_err*100:.4f}%")
        
        # Update config with actual r
        cfg.r = r_actual
        
        # =====================================================================
        # 5. Extract initial conditions
        # =====================================================================
        ics = gather_initial_conditions_serial(
            Q_train, Q_test, Xhat_train, Xhat_test, train_boundaries, test_boundaries
        )
        
        np.savez(
            paths["initial_conditions"],
            **ics,
            train_temporal_mean=train_mean,
            test_temporal_mean=test_mean,
        )
        
        # =====================================================================
        # 6. Prepare learning matrices
        # =====================================================================
        learning = prepare_learning_matrices_serial(Xhat_train, train_boundaries, cfg, logger)
        gamma_ref = load_reference_gamma_serial(cfg, logger)
        
        np.savez(paths["learning_matrices"], **learning)
        np.savez(paths["gamma_ref"], **gamma_ref)
        
        preproc = {
            'reduction_method': cfg.reduction_method,
            'centering_applied': cfg.centering_enabled,
            'scaling_applied': cfg.scaling_enabled,
            'r_actual': r_actual, 'r_config': cfg.r, 'r_from_energy': r_energy,
            'n_spatial': n_spatial, 'n_fields': cfg.n_fields,
            'n_x': cfg.n_x, 'n_y': cfg.n_y, 'dt': cfg.dt,
        }
        if scaling_factors is not None:
            preproc['scaling_factors'] = scaling_factors
        np.savez(paths["preprocessing_info"], **preproc)
        
        # =====================================================================
        # Cleanup
        # =====================================================================
        del Q_train, Q_test, Q_train_centered, Q_test_centered
        gc.collect()
        
        total_time = time.time() - t_start
        
        save_step_status(run_dir, "step_1", "completed", {
            "reduction_method": cfg.reduction_method,
            "n_spatial": int(n_spatial),
            "r": r_actual,
            "mpi_ranks": 1,
            "total_time_seconds": total_time,
        })
        print_header("STEP 1 COMPLETE (SERIAL)")
        print(f"  Output: {run_dir}")
        print(f"  Method: {cfg.reduction_method}")
        print(f"  Modes: r={r_actual}")
        print(f"  Runtime: {total_time:.1f}s")
        
        # Log selected modes
        if cfg.reduction_method == "manifold":
            logger.info(f"Selected mode indices (greedy): {selected_modes.tolist()}")
            # Show which POD modes were kept vs skipped
            max_mode = int(np.max(selected_modes))
            skipped = sorted(set(range(max_mode + 1)) - set(selected_modes))
            if skipped:
                logger.info(f"Skipped POD modes (within range 0-{max_mode}): {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        else:
            logger.info(f"Selected mode indices (POD): 0 to {r_actual - 1}")
        
        logger.info(f"Step 1 completed in {total_time:.1f}s")
    
    except Exception as e:
        logger.error(f"Step 1 failed: {e}", exc_info=True)
        save_step_status(run_dir, "step_1", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
