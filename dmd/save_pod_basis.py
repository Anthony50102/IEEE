"""
Save POD Basis for DMD Full-State Reconstruction.

This script computes and saves the POD basis U_r which is needed for
reconstructing full-state predictions from DMD's reduced-state forecasts.

The POD basis is computed using the method of snapshots:
    U = Q @ V @ Σ^{-1}

Since Step 1 only saves the eigendecomposition (eigs, eigv), we need to
reload the training data to compute the actual spatial modes.

Usage:
    python save_pod_basis.py --config config/dmd_1train_5test.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import os
import sys
import time
import argparse
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dmd.utils import (
    load_dmd_config,
    get_dmd_output_paths,
    DMDConfig,
)

# Import shared utilities from opinf
from opinf.utils import (
    setup_logging,
    save_config,
    print_header,
    loader,
)


def load_pod_eigendata(paths: dict, logger) -> tuple:
    """
    Load POD eigenvalues and eigenvectors from Step 1.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (eigs, eigv, S) - eigenvalues, eigenvectors, singular values.
    """
    pod_file = paths['pod_file']
    
    if not os.path.exists(pod_file):
        raise FileNotFoundError(f"POD file not found: {pod_file}")
    
    logger.info(f"Loading POD eigendata from {pod_file}")
    
    pod_data = np.load(pod_file)
    eigs = pod_data['eigs']
    eigv = pod_data['eigv']
    S = pod_data['S']
    
    logger.info(f"  Eigenvalues shape: {eigs.shape}")
    logger.info(f"  Eigenvectors shape: {eigv.shape}")
    logger.info(f"  Singular values shape: {S.shape}")
    
    return eigs, eigv, S


def load_training_data(cfg: DMDConfig, logger) -> np.ndarray:
    """
    Load and stack training data.
    
    Parameters
    ----------
    cfg : DMDConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    np.ndarray
        Training data matrix Q_train of shape (n_spatial, n_time).
    """
    logger.info(f"Loading {len(cfg.training_files)} training file(s)...")
    
    Q_list = []
    
    for i, filename in enumerate(cfg.training_files):
        filepath = os.path.join(cfg.data_dir, filename)
        logger.info(f"  Loading file {i+1}: {filename}")
        
        fh = loader(filepath, ENGINE=cfg.engine)
        
        # Load density and phi fields
        density = fh["density"].data  # (n_time, n_y, n_x)
        phi = fh["phi"].data  # (n_time, n_y, n_x)
        
        n_time_orig = density.shape[0]
        
        # Apply truncation if enabled
        if cfg.truncation_enabled:
            if cfg.truncation_method == "time":
                n_time = int(cfg.truncation_time / cfg.dt)
            else:
                n_time = cfg.truncation_snapshots
            n_time = min(n_time, n_time_orig)
            density = density[:n_time]
            phi = phi[:n_time]
        
        logger.info(f"    Shape: density={density.shape}, phi={phi.shape}")
        
        # Stack fields and flatten: (n_time, n_y, n_x) -> (2*n_y*n_x, n_time)
        n_time, n_y, n_x = density.shape
        density_flat = density.reshape(n_time, -1).T  # (n_y*n_x, n_time)
        phi_flat = phi.reshape(n_time, -1).T  # (n_y*n_x, n_time)
        
        Q_file = np.vstack([density_flat, phi_flat])  # (2*n_y*n_x, n_time)
        Q_list.append(Q_file)
        
        logger.info(f"    Q_file shape: {Q_file.shape}")
    
    # Stack all training data
    Q_train = np.hstack(Q_list)  # (n_spatial, total_n_time)
    logger.info(f"  Total Q_train shape: {Q_train.shape}")
    
    return Q_train


def compute_pod_basis(
    Q_train: np.ndarray,
    eigs: np.ndarray,
    eigv: np.ndarray,
    r: int,
    logger,
) -> tuple:
    """
    Compute the POD basis U_r from training data and eigendecomposition.
    
    In the method of snapshots:
        Gram matrix: G = Q^T @ Q
        Eigendecomposition: G = V @ Λ @ V^T
        POD modes: U_j = (1/σ_j) * Q @ v_j
    
    Where σ_j = sqrt(λ_j) are the singular values.
    
    Parameters
    ----------
    Q_train : np.ndarray, shape (n_spatial, n_time)
        Training data matrix.
    eigs : np.ndarray, shape (n_time,)
        Eigenvalues from Gram matrix (λ_j).
    eigv : np.ndarray, shape (n_time, n_time)
        Eigenvectors from Gram matrix (columns are v_j).
    r : int
        Number of POD modes to compute.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (U_r, temporal_mean) - POD basis and temporal mean.
    """
    logger.info(f"Computing POD basis with r={r} modes...")
    
    n_spatial, n_time = Q_train.shape
    
    # Center the data
    logger.info("  Centering data...")
    temporal_mean = np.mean(Q_train, axis=1)  # (n_spatial,)
    Q_centered = Q_train - temporal_mean[:, np.newaxis]
    
    # Compute transformation matrix: Tr = V_r @ diag(1/sqrt(λ_r))
    eigs_r = eigs[:r]
    eigv_r = eigv[:, :r]
    
    # Check for numerical issues
    n_problematic = np.sum(eigs_r <= 0)
    if n_problematic > 0:
        logger.warning(f"  {n_problematic} eigenvalues <= 0, clipping to small positive")
    
    eigs_r_safe = np.maximum(eigs_r, 1e-14)
    
    # Transformation matrix
    Tr = eigv_r @ np.diag(eigs_r_safe ** (-0.5))  # (n_time, r)
    
    # POD basis: U_r = Q_centered @ Tr
    logger.info("  Computing spatial modes U_r = Q @ Tr...")
    t0 = time.time()
    U_r = Q_centered @ Tr  # (n_spatial, r)
    logger.info(f"    Completed in {time.time() - t0:.1f}s")
    
    logger.info(f"  U_r shape: {U_r.shape}")
    
    # Verify orthonormality
    orth_check = U_r.T @ U_r
    off_diag = orth_check - np.eye(r)
    orth_error = np.max(np.abs(off_diag))
    logger.info(f"  Orthonormality check: max|U^T U - I| = {orth_error:.2e}")
    
    return U_r, temporal_mean


def save_pod_basis(
    U_r: np.ndarray,
    temporal_mean: np.ndarray,
    n_y: int,
    n_x: int,
    paths: dict,
    logger,
):
    """
    Save the POD basis to disk.
    
    Parameters
    ----------
    U_r : np.ndarray, shape (n_spatial, r)
        POD basis.
    temporal_mean : np.ndarray, shape (n_spatial,)
        Temporal mean.
    n_y : int
        Grid points in y.
    n_x : int
        Grid points in x.
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    """
    output_path = paths['pod_basis']
    
    logger.info(f"Saving POD basis to {output_path}")
    
    np.savez(
        output_path,
        U_r=U_r,
        mean=temporal_mean,
        n_y=n_y,
        n_x=n_x,
    )
    
    # Also save as .npy for easy loading
    np.save(output_path.replace('.npz', '_U_r.npy'), U_r)
    np.save(output_path.replace('.npz', '_mean.npy'), temporal_mean)
    
    logger.info(f"  POD basis saved successfully")
    logger.info(f"  U_r shape: {U_r.shape}")
    logger.info(f"  File size: {os.path.getsize(output_path) / 1e6:.2f} MB")


def main():
    """Main entry point for saving POD basis."""
    parser = argparse.ArgumentParser(
        description="Save POD basis for DMD full-state reconstruction"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory from Step 1"
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_dmd_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("save_pod_basis", args.run_dir, cfg.log_level)
    
    print_header("SAVING POD BASIS FOR DMD")
    print(f"  Run directory: {args.run_dir}")
    print(f"  POD modes (r): {cfg.r}")
    
    # Save configuration with step-specific name
    save_config(cfg, args.run_dir, step_name="save_pod_basis")
    logger.info("Configuration saved to run directory")
    
    paths = get_dmd_output_paths(args.run_dir)
    
    start_time = time.time()
    
    try:
        # 1. Load POD eigendata from Step 1
        eigs, eigv, S = load_pod_eigendata(paths, logger)
        
        # 2. Load training data
        Q_train = load_training_data(cfg, logger)
        
        # 3. Compute POD basis
        U_r, temporal_mean = compute_pod_basis(
            Q_train=Q_train,
            eigs=eigs,
            eigv=eigv,
            r=cfg.r,
            logger=logger,
        )
        
        # 4. Save POD basis
        save_pod_basis(
            U_r=U_r,
            temporal_mean=temporal_mean,
            n_y=cfg.n_y,
            n_x=cfg.n_x,
            paths=paths,
            logger=logger,
        )
        
        # Final timing
        total_time = time.time() - start_time
        
        print_header("POD BASIS SAVED SUCCESSFULLY")
        print(f"  Output: {paths['pod_basis']}")
        print(f"  Total runtime: {total_time:.1f} seconds")
        
        logger.info(f"POD basis saved successfully in {total_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Failed to save POD basis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
