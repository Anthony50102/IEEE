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
    OpInfConfig,
    load_config as load_opinf_config,
    save_config as save_opinf_config,
    get_run_directory,
    setup_logging,
    save_step_status,
    check_step_completed,
    get_output_paths as get_opinf_output_paths,
    print_header,
    print_config_summary as print_opinf_config_summary,
    load_dataset as loader,
)

import yaml


# =============================================================================
# DMD CONFIGURATION DATACLASS
# =============================================================================

@dataclass 
class DMDConfig(OpInfConfig):
    """
    Configuration container for the DMD pipeline.
    
    Extends OpInfConfig with DMD-specific parameters.
    """
    # Training mode
    training_mode: str = "multi_trajectory"  # "multi_trajectory" or "temporal_split"
    train_start: int = 0      # Start snapshot for training (temporal_split only)
    train_end: int = 8000     # End snapshot for training (temporal_split only)
    test_start: int = 8000    # Start snapshot for testing (temporal_split only)
    test_end: int = 16000     # End snapshot for testing (temporal_split only)
    
    # DMD-specific parameters
    dmd_rank: Optional[int] = None  # DMD rank (defaults to POD r if None)
    num_trials: int = 0  # Number of bagging trials for BOPDMD (0 = no bagging)
    use_proj: bool = True  # Use POD projection in BOPDMD
    eig_sort: str = "real"  # Eigenvalue sorting method
    
    # Physics parameters for Gamma computation
    k0: float = 0.15  # Wave number (dx = 2*pi/k0)
    c1: float = 1.0   # Adiabaticity parameter
    
    # State diagnostic plots
    plot_state_error: bool = False  # Plot L2 error over time
    plot_state_snapshots: bool = False  # Plot 2D snapshot comparisons
    n_snapshot_samples: int = 5  # Number of snapshots to compare


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
    cfg.training_mode = dmd_section.get("training_mode", "multi_trajectory")
    cfg.train_start = dmd_section.get("train_start", 0)
    cfg.train_end = dmd_section.get("train_end", 8000)
    cfg.test_start = dmd_section.get("test_start", 8000)
    cfg.test_end = dmd_section.get("test_end", 16000)
    cfg.dmd_rank = dmd_section.get("rank", None)
    cfg.num_trials = dmd_section.get("num_trials", 0)
    cfg.use_proj = dmd_section.get("use_proj", True)
    cfg.eig_sort = dmd_section.get("eig_sort", "real")
    cfg.k0 = dmd_section.get("k0", 0.15)
    cfg.c1 = dmd_section.get("c1", 1.0)
    
    # State diagnostic plot settings
    evaluation = raw.get("evaluation", {})
    cfg.plot_state_error = evaluation.get("plot_state_error", False)
    cfg.plot_state_snapshots = evaluation.get("plot_state_snapshots", False)
    cfg.n_snapshot_samples = evaluation.get("n_snapshot_samples", 5)
    
    return cfg


def save_config(cfg: DMDConfig, output_path: str, step_name: str = None) -> str:
    """
    Save DMD configuration to YAML file.
    
    Extends OpInf save_config with DMD-specific parameters.
    
    Parameters
    ----------
    cfg : DMDConfig
        Configuration object to save.
    output_path : str
        Directory to save the config file.
    step_name : str, optional
        Step name to include in filename (e.g., "step_1").
    
    Returns
    -------
    str
        Path to saved config file.
    """
    config_dict = {
        "run_name": cfg.run_name,
        "run_dir": cfg.run_dir,
        "paths": {
            "output_base": cfg.output_base,
            "data_dir": cfg.data_dir,
            "training_files": cfg.training_files,
            "test_files": cfg.test_files,
        },
        "physics": {
            "dt": cfg.dt, 
            "n_fields": cfg.n_fields, 
            "n_x": cfg.n_x, 
            "n_y": cfg.n_y,
            "k0": cfg.k0,
            "c1": cfg.c1,
        },
        "training_mode": {
            "mode": cfg.training_mode,
            "train_start": cfg.train_start,
            "train_end": cfg.train_end,
            "test_start": cfg.test_start,
            "test_end": cfg.test_end,
        },
        "reduction": {
            "method": cfg.reduction_method,
            "r": cfg.r,
            "target_energy": cfg.target_energy,
        },
        "dmd": {
            "rank": cfg.dmd_rank,
            "num_trials": cfg.num_trials,
            "use_proj": cfg.use_proj,
            "eig_sort": cfg.eig_sort,
        },
        "truncation": {
            "enabled": cfg.truncation_enabled,
            "method": cfg.truncation_method,
            "snapshots": cfg.truncation_snapshots,
            "time": cfg.truncation_time,
        },
        "preprocessing": {
            "centering": cfg.centering_enabled, 
            "scaling": cfg.scaling_enabled,
        },
        "training": {
            "training_end": cfg.training_end, 
            "n_steps": cfg.n_steps,
        },
        "evaluation": {
            "save_predictions": cfg.save_predictions, 
            "generate_plots": cfg.generate_plots,
            "plot_state_error": cfg.plot_state_error,
            "plot_state_snapshots": cfg.plot_state_snapshots,
            "n_snapshot_samples": cfg.n_snapshot_samples,
        },
        "execution": {
            "verbose": cfg.verbose, 
            "log_level": cfg.log_level, 
            "engine": cfg.engine,
        },
    }
    
    filename = f"config_{step_name}.yaml" if step_name else "config.yaml"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    return filepath


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
    
    # Override paths to match DMD naming convention
    paths.update({
        # Step 1 outputs (override with DMD-specific names)
        "xhat_train": os.path.join(run_dir, "Xhat_train.npy"),
        "xhat_test": os.path.join(run_dir, "Xhat_test.npy"),
        "boundaries": os.path.join(run_dir, "boundaries.npz"),
        "initial_conditions": os.path.join(run_dir, "initial_conditions.npz"),
        "preprocessing_info": os.path.join(run_dir, "preprocessing_info.npz"),
        
        # POD basis for reconstruction
        "pod_basis": os.path.join(run_dir, "pod_basis.npz"),
        "pod_file": os.path.join(run_dir, "POD.npz"),
        
        # Step 2 outputs (DMD fitting)
        "dmd_model": os.path.join(run_dir, "dmd_model.npz"),
        "dmd_eigenvalues": os.path.join(run_dir, "dmd_eigenvalues.npy"),
        "dmd_modes": os.path.join(run_dir, "dmd_modes.npy"),
        "dmd_amplitudes": os.path.join(run_dir, "dmd_amplitudes.npy"),
        
        # Step 3 outputs (forecasts)
        "dmd_forecasts_dir": os.path.join(run_dir, "dmd_forecasts"),
        "dmd_predictions": os.path.join(run_dir, "dmd_predictions.npz"),
        "dmd_metrics": os.path.join(run_dir, "dmd_evaluation_metrics.yaml"),
        "figures_dir": os.path.join(run_dir, "figures"),
    })
    
    return paths


def print_dmd_config_summary(cfg: DMDConfig):
    """Print a summary of the DMD configuration."""
    print_header("DMD CONFIGURATION SUMMARY")
    print(f"  Run name: {cfg.run_name or '(auto)'}")
    print(f"  Output base: {cfg.output_base}")
    print(f"  Training mode: {cfg.training_mode}")
    if cfg.training_mode == "temporal_split":
        print(f"    Train snapshots: [{cfg.train_start}, {cfg.train_end})")
        print(f"    Test snapshots: [{cfg.test_start}, {cfg.test_end})")
    else:
        print(f"  Training files: {len(cfg.training_files)}")
        print(f"  Test files: {len(cfg.test_files)}")
    print(f"  POD modes (r): {cfg.r}")
    print(f"  DMD rank: {cfg.dmd_rank or 'same as POD r'}")
    print(f"  BOPDMD trials: {cfg.num_trials}")
    print(f"  Use projection: {cfg.use_proj}")
    print(f"  Physics: k0={cfg.k0}, c1={cfg.c1} (dx={2*np.pi/cfg.k0:.4f})")
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





# =============================================================================
# GAMMA COMPUTATION FROM FIELDS
# =============================================================================

def periodic_gradient(input_field: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions.

    Parameters
    ----------
    input_field : np.ndarray
        Input array (can be 2D, 3D with time, etc.).
    dx : float
        The spacing between grid points.
    axis : int
        Axis along which the gradient is taken.

    Returns
    -------
    np.ndarray
        Gradient with periodic boundary conditions.
    """
    # Handle negative axis
    if axis < 0:
        axis = input_field.ndim + axis
    
    # Pad with periodic boundary conditions
    pad_width = [(0, 0)] * input_field.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(input_field, pad_width=pad_width, mode="wrap")
    
    # Compute central difference
    slices_plus = [slice(None)] * padded.ndim
    slices_minus = [slice(None)] * padded.ndim
    slices_plus[axis] = slice(2, None)
    slices_minus[axis] = slice(None, -2)
    
    gradient = (padded[tuple(slices_plus)] - padded[tuple(slices_minus)]) / (2 * dx)
    
    return gradient


def get_gamma_n(n: np.ndarray, p: np.ndarray, dx: float, dy_p: np.ndarray = None) -> np.ndarray:
    """
    Compute the average particle flux (Γ_n) using the formula:
    
        Γ_n = - ∫ d²x  ñ ∂φ̃/∂y
    
    Parameters
    ----------
    n : np.ndarray
        Density field. Shape (..., n_y, n_x) where ... can be time dimension.
    p : np.ndarray
        Potential field. Shape (..., n_y, n_x).
    dx : float
        Grid spacing.
    dy_p : np.ndarray, optional
        Pre-computed gradient of potential in y-direction.

    Returns
    -------
    np.ndarray
        Computed average particle flux value(s). Scalar if no time dim, else (n_time,).
    """
    if dy_p is None:
        dy_p = periodic_gradient(p, dx=dx, axis=-2)  # gradient in y
    gamma_n = -np.mean((n * dy_p), axis=(-1, -2))  # mean over y & x
    return gamma_n


def get_gamma_c(n: np.ndarray, p: np.ndarray, c1: float, dx: float) -> np.ndarray:
    """
    Compute the sink Γ_c using the formula:
    
        Γ_c = c1 ∫ d²x  (ñ - φ̃)²
    
    Parameters
    ----------
    n : np.ndarray
        Density field. Shape (..., n_y, n_x).
    p : np.ndarray
        Potential field. Shape (..., n_y, n_x).
    c1 : float
        Adiabaticity parameter (typically 1.0 for HW2D).
    dx : float
        Grid spacing.

    Returns
    -------
    np.ndarray
        Computed conductive flux value(s). Scalar if no time dim, else (n_time,).
    """
    gamma_c = c1 * np.mean((n - p) ** 2, axis=(-1, -2))  # mean over y & x
    return gamma_c


def compute_gamma_from_state(
    Q: np.ndarray,
    n_fields: int,
    n_y: int,
    n_x: int,
    k0: float = 0.15,
    c1: float = 1.0,
) -> tuple:
    """
    Compute Gamma_n and Gamma_c from full state vector.
    
    Parameters
    ----------
    Q : np.ndarray, shape (n_spatial, n_time) or (n_spatial,)
        Full state vector with density and potential stacked.
        Layout: [density.flatten(), phi.flatten()]
    n_fields : int
        Number of fields (should be 2).
    n_y : int
        Number of grid points in y.
    n_x : int
        Number of grid points in x.
    k0 : float
        Wave number for grid spacing (dx = 2π/k0).
    c1 : float
        Adiabaticity parameter.
    
    Returns
    -------
    tuple
        (Gamma_n, Gamma_c) arrays.
    """
    dx = 2 * np.pi / k0
    
    # Handle 1D vs 2D input
    if Q.ndim == 1:
        Q = Q[:, np.newaxis]
    
    n_spatial, n_time = Q.shape
    spatial_per_field = n_y * n_x
    
    # Extract density and potential
    # Assuming layout: [density, phi] stacked
    n_flat = Q[:spatial_per_field, :]  # (n_y*n_x, n_time)
    p_flat = Q[spatial_per_field:2*spatial_per_field, :]  # (n_y*n_x, n_time)
    
    # Reshape to (n_time, n_y, n_x)
    n_field = n_flat.T.reshape(n_time, n_y, n_x)  # (n_time, n_y, n_x)
    p_field = p_flat.T.reshape(n_time, n_y, n_x)  # (n_time, n_y, n_x)
    
    # Compute Gamma
    gamma_n = get_gamma_n(n_field, p_field, dx)  # (n_time,)
    gamma_c = get_gamma_c(n_field, p_field, c1, dx)  # (n_time,)
    
    return gamma_n, gamma_c


def reconstruct_full_state(
    X_hat: np.ndarray,
    pod_basis: np.ndarray,
    temporal_mean=None,
) -> np.ndarray:
    """
    Reconstruct full state from reduced state using POD basis.
    
    Q_reconstructed = U @ X_hat + mean
    
    Parameters
    ----------
    X_hat : np.ndarray, shape (r, n_time) or (n_time, r)
        Reduced state trajectory.
    pod_basis : np.ndarray, shape (n_spatial, r)
        POD basis matrix U.
    temporal_mean : np.ndarray, shape (n_spatial,), optional
        Temporal mean to add back.
    
    Returns
    -------
    Q : np.ndarray, shape (n_spatial, n_time)
        Reconstructed full state.
    """
    # Ensure X_hat is (r, n_time)
    if X_hat.shape[0] != pod_basis.shape[1]:
        X_hat = X_hat.T
    
    # Reconstruct
    Q = pod_basis @ X_hat  # (n_spatial, n_time)
    
    # Add back mean if provided
    if temporal_mean is not None:
        Q = Q + temporal_mean[:, np.newaxis]
    
    return Q
