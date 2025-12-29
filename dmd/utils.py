"""
DMD Pipeline Utilities.

This module provides shared utilities for the DMD ROM pipeline:
- Configuration loading (extends OpInf config)
- DMD-specific data structures
- Forecasting functions

Author: Anthony Poole
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# Add parent directory to path for importing opinf utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))

from utils import (
    PipelineConfig as OpInfConfig,
    load_config as load_opinf_config,
    save_config,
    get_run_directory,
    setup_logging,
    save_step_status,
    check_step_completed,
    get_output_paths as get_opinf_output_paths,
    print_header,
    print_config_summary as print_opinf_config_summary,
    loader,
)


# =============================================================================
# DMD CONFIGURATION DATACLASS
# =============================================================================

@dataclass 
class DMDConfig(OpInfConfig):
    """
    Configuration container for the DMD pipeline.
    
    Extends OpInfConfig with DMD-specific parameters.
    """
    # DMD-specific parameters
    dmd_rank: Optional[int] = None  # DMD rank (defaults to POD r if None)
    num_trials: int = 0  # Number of bagging trials for BOPDMD (0 = no bagging)
    use_proj: bool = True  # Use POD projection in BOPDMD
    eig_sort: str = "real"  # Eigenvalue sorting method
    
    # Output operator learning (optional, for Gamma prediction)
    learn_output_operator: bool = True
    output_reg: float = 1e-6  # Regularization for output operator


def load_dmd_config(config_path: str) -> DMDConfig:
    """
    Load DMD configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    
    Returns
    -------
    DMDConfig
        Populated configuration object.
    """
    import yaml
    
    # First load base OpInf config
    base_cfg = load_opinf_config(config_path)
    
    # Create DMD config and copy base fields
    cfg = DMDConfig()
    for field_name in vars(base_cfg):
        setattr(cfg, field_name, getattr(base_cfg, field_name))
    
    # Load DMD-specific settings
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    dmd_section = raw.get("dmd", {})
    cfg.dmd_rank = dmd_section.get("rank", None)
    cfg.num_trials = dmd_section.get("num_trials", 0)
    cfg.use_proj = dmd_section.get("use_proj", True)
    cfg.eig_sort = dmd_section.get("eig_sort", "real")
    cfg.learn_output_operator = dmd_section.get("learn_output_operator", True)
    cfg.output_reg = dmd_section.get("output_reg", 1e-6)
    
    return cfg


def get_dmd_output_paths(run_dir: str) -> dict:
    """
    Get standard output file paths for a DMD run.
    
    Extends OpInf paths with DMD-specific outputs.
    
    Parameters
    ----------
    run_dir : str
        Run directory.
    
    Returns
    -------
    dict
        Dictionary of output file paths.
    """
    # Get base paths from OpInf
    paths = get_opinf_output_paths(run_dir)
    
    # Add DMD-specific paths
    paths.update({
        # Step 2 outputs (DMD fitting)
        "dmd_model": os.path.join(run_dir, "dmd_model.npz"),
        "dmd_eigenvalues": os.path.join(run_dir, "dmd_eigenvalues.npy"),
        "dmd_modes": os.path.join(run_dir, "dmd_modes.npy"),
        "dmd_amplitudes": os.path.join(run_dir, "dmd_amplitudes.npy"),
        "dmd_output_operator": os.path.join(run_dir, "dmd_output_operator.npz"),
        
        # Step 3 outputs (forecasts)
        "dmd_forecasts_dir": os.path.join(run_dir, "dmd_forecasts"),
        "dmd_predictions": os.path.join(run_dir, "dmd_predictions.npz"),
        "dmd_metrics": os.path.join(run_dir, "dmd_evaluation_metrics.yaml"),
    })
    
    return paths


def print_dmd_config_summary(cfg: DMDConfig):
    """Print a summary of the DMD configuration."""
    print_header("DMD CONFIGURATION SUMMARY")
    print(f"  Run name: {cfg.run_name or '(auto)'}")
    print(f"  Output base: {cfg.output_base}")
    print(f"  Training files: {len(cfg.training_files)}")
    print(f"  Test files: {len(cfg.test_files)}")
    print(f"  POD modes (r): {cfg.r}")
    print(f"  DMD rank: {cfg.dmd_rank or 'same as POD r'}")
    print(f"  BOPDMD trials: {cfg.num_trials}")
    print(f"  Use projection: {cfg.use_proj}")
    print(f"  Learn output operator: {cfg.learn_output_operator}")
    print(f"  Truncation: {'enabled' if cfg.truncation_enabled else 'disabled'}")
    if cfg.truncation_enabled:
        if cfg.truncation_method == "time":
            print(f"    Time: {cfg.truncation_time} units")
        else:
            print(f"    Snapshots: {cfg.truncation_snapshots}")
    print("=" * 70 + "\n")


# =============================================================================
# DMD FORECASTING
# =============================================================================

def dmd_forecast(
    eigs: np.ndarray,
    modes_reduced: np.ndarray,
    amplitudes: np.ndarray,
    V_global: np.ndarray,
    t: np.ndarray,
) -> tuple:
    """
    Forecast using DMD eigenvalues, reduced modes, and amplitudes.
    
    Computes: x(t) = Σ_j b_j * φ_j * exp(α_j * t)
    
    Where:
    - α_j are continuous-time eigenvalues (eigs)
    - φ_j are full-space modes (V_global @ modes_reduced)
    - b_j are amplitudes
    
    Parameters
    ----------
    eigs : np.ndarray, shape (r,)
        Continuous-time DMD eigenvalues.
    modes_reduced : np.ndarray, shape (r, r)
        Reduced-space DMD modes (columns are modes).
    amplitudes : np.ndarray, shape (r,)
        DMD amplitudes / initial mode coefficients.
    V_global : np.ndarray, shape (n_features, r)
        Global POD basis.
    t : np.ndarray, shape (m,)
        Time vector (assumed to start at 0).
    
    Returns
    -------
    X_pred : np.ndarray, shape (n_features, m)
        Forecasted snapshots in full space.
    modes_full : np.ndarray, shape (n_features, r)
        Full-space DMD modes.
    """
    # Work with copies and explicit dtypes
    W = modes_reduced.astype(np.complex128, copy=True)
    b = np.asarray(amplitudes, dtype=np.complex128).copy()
    
    # Column norms for numerical stability
    col_norms = np.linalg.norm(W, axis=0)
    max_norm = np.max(col_norms) if col_norms.size else 0.0
    
    # Threshold for "tiny" modes
    tiny_thr = 10.0 * np.finfo(float).eps * (max_norm if max_norm > 0 else 1.0)
    tiny = col_norms < tiny_thr
    
    # Safe norms for division
    safe = col_norms.copy()
    safe[tiny] = 1.0
    
    # Normalize columns, adjust amplitudes
    W_unit = W / safe[None, :]
    b_adj = b * safe
    
    # Zero-out genuinely tiny columns
    W_unit[:, tiny] = 0.0
    b_adj[tiny] = 0.0
    
    # Full-space modes
    modes_full = V_global @ W_unit  # (n_features, r)
    
    # Time exponentials: x(t) = Σ_j b_j φ_j e^{α_j t}
    Et = np.exp(np.outer(eigs, t))  # (r, m)
    
    # Forecast
    X_pred = modes_full @ (b_adj[:, None] * Et)  # (n_features, m)
    
    return X_pred, modes_full


def dmd_forecast_reduced(
    eigs: np.ndarray,
    modes_reduced: np.ndarray,
    amplitudes: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Forecast in reduced (POD) space using DMD.
    
    Computes: x_hat(t) = Σ_j b_j * w_j * exp(α_j * t)
    
    This is useful when we only need the reduced state, e.g., for 
    computing output quantities without full state reconstruction.
    
    Parameters
    ----------
    eigs : np.ndarray, shape (r,)
        Continuous-time DMD eigenvalues.
    modes_reduced : np.ndarray, shape (r, r)
        Reduced-space DMD modes.
    amplitudes : np.ndarray, shape (r,)
        DMD amplitudes.
    t : np.ndarray, shape (m,)
        Time vector.
    
    Returns
    -------
    X_hat : np.ndarray, shape (r, m)
        Forecasted snapshots in reduced space.
    """
    b = np.asarray(amplitudes, dtype=np.complex128)
    W = modes_reduced.astype(np.complex128)
    
    # Time exponentials
    Et = np.exp(np.outer(eigs, t))  # (r, m)
    
    # Forecast in reduced space
    X_hat = W @ (b[:, None] * Et)  # (r, m)
    
    return X_hat


def compute_initial_condition_reduced(
    x0_full: np.ndarray,
    V_global: np.ndarray,
    temporal_mean: np.ndarray = None,
) -> np.ndarray:
    """
    Project initial condition from full space to reduced space.
    
    Parameters
    ----------
    x0_full : np.ndarray, shape (n_features,)
        Initial condition in full space.
    V_global : np.ndarray, shape (n_features, r)
        POD basis.
    temporal_mean : np.ndarray, optional, shape (n_features,)
        Temporal mean to subtract before projection.
    
    Returns
    -------
    x0_reduced : np.ndarray, shape (r,)
        Initial condition in reduced space.
    """
    if temporal_mean is not None:
        x0_centered = x0_full - temporal_mean
    else:
        x0_centered = x0_full
    
    x0_reduced = V_global.T @ x0_centered
    return x0_reduced


def solve_output_from_reduced_state(
    X_hat: np.ndarray,
    C: np.ndarray,
    c: np.ndarray = None,
    mean_Xhat: np.ndarray = None,
    scaling_Xhat: float = 1.0,
) -> np.ndarray:
    """
    Compute output quantities from reduced state trajectory.
    
    Uses a linear output operator: y = C @ x_hat_scaled + c
    
    Parameters
    ----------
    X_hat : np.ndarray, shape (r, m) or (m, r)
        Reduced state trajectory.
    C : np.ndarray, shape (n_outputs, r)
        Output operator matrix.
    c : np.ndarray, shape (n_outputs,), optional
        Output bias term.
    mean_Xhat : np.ndarray, shape (r,), optional
        Mean for scaling reduced state.
    scaling_Xhat : float
        Scaling factor for reduced state.
    
    Returns
    -------
    Y : np.ndarray, shape (n_outputs, m)
        Output trajectory.
    """
    # Ensure X_hat is (r, m)
    if X_hat.shape[0] != C.shape[1]:
        X_hat = X_hat.T
    
    # Scale reduced state
    if mean_Xhat is not None:
        X_hat_scaled = (X_hat - mean_Xhat[:, None]) / scaling_Xhat
    else:
        X_hat_scaled = X_hat / scaling_Xhat
    
    # Compute output
    Y = C @ X_hat_scaled
    
    if c is not None:
        Y = Y + c[:, None]
    
    return Y


# =============================================================================
# OUTPUT OPERATOR LEARNING
# =============================================================================

def learn_linear_output_operator(
    X_hat: np.ndarray,
    Y_ref: np.ndarray,
    regularization: float = 1e-6,
    mean_Xhat: np.ndarray = None,
    scaling_Xhat: float = 1.0,
) -> tuple:
    """
    Learn a linear output operator via regularized least squares.
    
    Solves: Y = C @ X_hat_scaled + c
    
    Parameters
    ----------
    X_hat : np.ndarray, shape (r, m) or (m, r)
        Training reduced state trajectory.
    Y_ref : np.ndarray, shape (n_outputs, m) or (m, n_outputs)
        Reference output trajectory.
    regularization : float
        Tikhonov regularization parameter.
    mean_Xhat : np.ndarray, shape (r,), optional
        Mean for scaling.
    scaling_Xhat : float
        Scaling factor.
    
    Returns
    -------
    C : np.ndarray, shape (n_outputs, r)
        Output operator matrix.
    c : np.ndarray, shape (n_outputs,)
        Output bias term.
    """
    # Ensure correct shapes: X_hat = (r, m), Y_ref = (n_outputs, m)
    if X_hat.ndim == 1:
        X_hat = X_hat.reshape(-1, 1)
    if X_hat.shape[0] > X_hat.shape[1]:
        X_hat = X_hat.T
    
    if Y_ref.ndim == 1:
        Y_ref = Y_ref.reshape(1, -1)
    if Y_ref.shape[0] > Y_ref.shape[1]:
        Y_ref = Y_ref.T
    
    r, m = X_hat.shape
    n_outputs = Y_ref.shape[0]
    
    # Scale reduced state
    if mean_Xhat is not None:
        X_hat_scaled = (X_hat - mean_Xhat[:, None]) / scaling_Xhat
    else:
        X_hat_scaled = X_hat / scaling_Xhat
    
    # Build design matrix: [X_hat_scaled; 1] to include bias
    ones = np.ones((1, m))
    D = np.vstack([X_hat_scaled, ones])  # (r+1, m)
    
    # Solve regularized least squares: Y = [C, c] @ D
    # => [C, c]^T = (D @ D^T + reg*I)^{-1} @ D @ Y^T
    DTD = D @ D.T  # (r+1, r+1)
    reg_matrix = regularization * np.eye(r + 1)
    DY = D @ Y_ref.T  # (r+1, n_outputs)
    
    Cc = np.linalg.solve(DTD + reg_matrix, DY).T  # (n_outputs, r+1)
    
    C = Cc[:, :r]
    c = Cc[:, r]
    
    return C, c
