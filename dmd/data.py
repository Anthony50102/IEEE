"""
Data loading and preprocessing utilities for DMD.

This module handles:
- Loading simulation data from HDF5 files
- POD basis computation
- Data projection
- Saving preprocessed data

Author: Anthony Poole
"""

import os
import numpy as np

from opinf.utils import load_dataset as loader


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory(file_path: str, cfg, logger) -> tuple:
    """
    Load a single simulation trajectory.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    cfg : DMDConfig
        Configuration object.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Q, n_y, n_x) - State matrix (n_spatial, n_time), grid dimensions.
    """
    logger.info(f"  Loading: {os.path.basename(file_path)}")
    
    fh = loader(file_path, engine=cfg.engine)
    
    # Load fields
    density = fh["density"].data  # (n_time, n_y, n_x)
    phi = fh["phi"].data
    
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
    
    n_time, n_y, n_x = density.shape
    logger.info(f"    Shape: ({n_time}, {n_y}, {n_x})")
    
    # Stack and reshape: (n_time, n_y, n_x) -> (n_spatial, n_time)
    density_flat = density.reshape(n_time, -1).T  # (n_y*n_x, n_time)
    phi_flat = phi.reshape(n_time, -1).T
    
    Q = np.vstack([density_flat, phi_flat])  # (2*n_y*n_x, n_time)
    
    return Q, n_y, n_x


# =============================================================================
# POD COMPUTATION
# =============================================================================

def compute_pod_basis(Q: np.ndarray, r: int, logger) -> tuple:
    """
    Compute POD basis via method of snapshots.
    
    Parameters
    ----------
    Q : np.ndarray, shape (n_spatial, n_time)
        Centered data matrix.
    r : int
        Number of POD modes.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (U_r, S, V) - POD modes (n_spatial, r), singular values, right singular vectors.
    """
    logger.info("Computing POD basis...")
    n_spatial, n_time = Q.shape
    logger.info(f"  Data: {n_spatial} spatial DOF, {n_time} snapshots")
    
    # Method of snapshots: compute Gram matrix Q^T @ Q
    logger.info("  Computing Gram matrix...")
    G = Q.T @ Q  # (n_time, n_time)
    
    # Eigendecomposition
    logger.info("  Computing eigendecomposition...")
    eigs, eigv = np.linalg.eigh(G)
    
    # Sort descending
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    eigv = eigv[:, idx]
    
    # Clip negative eigenvalues
    eigs = np.maximum(eigs, 1e-14)
    S = np.sqrt(eigs)  # Singular values
    
    # Compute POD modes: U = Q @ V @ Σ^{-1}
    logger.info(f"  Computing {r} POD modes...")
    U_r = Q @ eigv[:, :r] @ np.diag(1.0 / S[:r])
    
    # Verify orthonormality
    orth_err = np.max(np.abs(U_r.T @ U_r - np.eye(r)))
    logger.info(f"  Orthonormality error: {orth_err:.2e}")
    
    return U_r, S, eigv


def project_data(Q: np.ndarray, U_r: np.ndarray, logger, name: str = "data") -> np.ndarray:
    """
    Project data onto POD basis.
    
    Parameters
    ----------
    Q : np.ndarray, shape (n_spatial, n_time)
        Data matrix.
    U_r : np.ndarray, shape (n_spatial, r)
        POD basis.
    logger : logging.Logger
        Logger instance.
    name : str
        Name for logging.
    
    Returns
    -------
    np.ndarray, shape (n_time, r)
        Projected data.
    """
    logger.info(f"  Projecting {name} data onto POD basis...")
    Xhat = (U_r.T @ Q).T  # (n_time, r)
    logger.info(f"    Xhat shape: {Xhat.shape}")
    return Xhat


# =============================================================================
# SAVING
# =============================================================================

def save_basis_and_preprocessing(
    run_dir: str,
    U_r: np.ndarray,
    S: np.ndarray,
    train_mean: np.ndarray,
    Xhat_train: np.ndarray,
    Xhat_test: np.ndarray,
    train_boundaries: np.ndarray,
    test_boundaries: np.ndarray,
    train_ICs: np.ndarray,
    test_ICs: np.ndarray,
    train_ICs_reduced: np.ndarray,
    test_ICs_reduced: np.ndarray,
    n_y: int,
    n_x: int,
    r_actual: int,
    training_mode: str,
    logger,
):
    """
    Save all preprocessing outputs.
    
    Parameters
    ----------
    run_dir : str
        Output directory.
    U_r : np.ndarray
        POD basis.
    S : np.ndarray
        Singular values.
    train_mean : np.ndarray
        Training data mean.
    Xhat_train, Xhat_test : np.ndarray
        Projected training/test data.
    train_boundaries, test_boundaries : np.ndarray
        Trajectory boundaries.
    train_ICs, test_ICs : np.ndarray
        Full-space initial conditions.
    train_ICs_reduced, test_ICs_reduced : np.ndarray
        Reduced-space initial conditions.
    n_y, n_x : int
        Grid dimensions.
    r_actual : int
        Number of POD modes.
    training_mode : str
        Training mode used.
    logger : logging.Logger
        Logger instance.
    """
    logger.info("Saving outputs...")
    
    # POD basis (for reconstruction)
    pod_path = os.path.join(run_dir, "pod_basis.npz")
    np.savez(pod_path, U_r=U_r, mean=train_mean, n_y=n_y, n_x=n_x)
    logger.info(f"  Saved POD basis: {pod_path}")
    
    # Singular values
    pod_eig_path = os.path.join(run_dir, "POD.npz")
    np.savez(pod_eig_path, S=S, eigs=S**2, eigv=np.eye(len(S)))  # Compatibility
    logger.info(f"  Saved POD eigendata: {pod_eig_path}")
    
    # Projected data
    np.save(os.path.join(run_dir, "Xhat_train.npy"), Xhat_train)
    np.save(os.path.join(run_dir, "Xhat_test.npy"), Xhat_test)
    logger.info(f"  Saved projected data")
    
    # Boundaries
    np.savez(os.path.join(run_dir, "boundaries.npz"),
             train_boundaries=train_boundaries,
             test_boundaries=test_boundaries,
             n_spatial=U_r.shape[0])
    logger.info(f"  Saved boundaries")
    
    # Initial conditions
    np.savez(os.path.join(run_dir, "initial_conditions.npz"),
             train_ICs=train_ICs,
             test_ICs=test_ICs,
             train_ICs_reduced=train_ICs_reduced,
             test_ICs_reduced=test_ICs_reduced)
    logger.info(f"  Saved initial conditions")
    
    # Preprocessing info
    np.savez(os.path.join(run_dir, "preprocessing_info.npz"),
             r_actual=r_actual,
             n_y=n_y,
             n_x=n_x,
             training_mode=training_mode)
    logger.info(f"  Saved preprocessing info")
    
    # Learning matrices (for compatibility)
    np.savez(os.path.join(run_dir, "learning_matrices.npz"),
             mean_Xhat=np.mean(Xhat_train, axis=0),
             scaling_Xhat=np.max(np.abs(Xhat_train)))
    
    logger.info("All outputs saved successfully")
