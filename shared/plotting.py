"""
Plotting utilities for ROM methods.

This module provides standardized plotting functions that can be used
across all three methods (OpInf, Manifold OpInf, DMD).

Author: Anthony Poole
"""

import os
import numpy as np
from typing import Optional, List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# POD ENERGY PLOTS
# =============================================================================

def plot_pod_energy(eigs, r, output_dir, logger, filename="pod_energy.png"):
    """
    Generate POD energy diagnostic plots.
    
    Creates a 3-panel figure showing:
    1. Singular values (log scale)
    2. Cumulative retained energy
    3. Energy contribution per mode (log scale)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    eigs_pos = np.maximum(eigs, 0)
    ret_energy = np.cumsum(eigs_pos) / np.sum(eigs_pos)
    singular_values = np.sqrt(eigs_pos)
    mode_energy = eigs_pos / np.sum(eigs_pos)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Singular values
    ax = axes[0]
    ax.semilogy(singular_values, 'b-', linewidth=1.5)
    if 0 < r <= len(singular_values):
        ax.axvline(r, color='r', linestyle='--', label=f'r={r}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Singular value')
    ax.set_title('Singular Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Retained energy
    ax = axes[1]
    ax.plot(ret_energy * 100, 'b-', linewidth=1.5)
    if 0 < r <= len(ret_energy):
        ax.axvline(r, color='r', linestyle='--', label=f'r={r}: {ret_energy[r-1]*100:.4f}%')
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Retained energy (%)')
    ax.set_title('Cumulative Retained Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy per mode
    ax = axes[2]
    ax.semilogy(mode_energy[mode_energy > 0] * 100, 'b-', linewidth=1.5)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Energy contribution (%)')
    ax.set_title('Energy per Mode')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Also save data
    np.savez(
        os.path.join(output_dir, "pod_energy_data.npz"),
        eigenvalues=eigs, singular_values=singular_values,
        retained_energy=ret_energy, truncation_rank=r,
    )
    
    logger.info(f"Saved POD energy plot to {plot_path}")


# =============================================================================
# GAMMA PREDICTION PLOTS
# =============================================================================

def plot_gamma_timeseries(pred_n: np.ndarray, pred_c: np.ndarray,
                          ref_n: np.ndarray, ref_c: np.ndarray,
                          dt: float, output_path: str, logger,
                          title_prefix: str = "", method_name: str = "ROM"):
    """
    Generate a Gamma comparison plot (reference vs prediction).
    
    This is the standard plotting function for all ROM methods.
    Handles both single predictions and ensemble predictions (with uncertainty).
    
    Parameters
    ----------
    pred_n : np.ndarray, shape (n_steps,) or (n_ensemble, n_steps)
        Predicted particle flux. If 2D, ensemble mean and ±2σ are plotted.
    pred_c : np.ndarray, shape (n_steps,) or (n_ensemble, n_steps)
        Predicted conductive flux.
    ref_n : np.ndarray, shape (n_steps,)
        Reference particle flux.
    ref_c : np.ndarray, shape (n_steps,)
        Reference conductive flux.
    dt : float
        Time step size.
    output_path : str
        Full path to save the figure.
    logger : logging.Logger
        Logger instance.
    title_prefix : str
        Prefix for the plot title (e.g., "Train Trajectory 1: ").
    method_name : str
        Name of the method for legend (e.g., "DMD", "OpInf").
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    n_steps = len(ref_n)
    t = np.arange(n_steps) * dt
    
    # Check if ensemble (2D) or single prediction (1D)
    is_ensemble = pred_n.ndim == 2
    
    if is_ensemble:
        mean_n, std_n = np.mean(pred_n, axis=0), np.std(pred_n, axis=0)
        mean_c, std_c = np.mean(pred_c, axis=0), np.std(pred_c, axis=0)
    else:
        mean_n, std_n = pred_n, None
        mean_c, std_c = pred_c, None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Gamma_n plot
    ax = axes[0]
    ax.plot(t, ref_n, 'k-', label='Reference', linewidth=1, alpha=0.8)
    ax.plot(t, mean_n, 'b--', label=f'{method_name} Prediction', linewidth=1)
    if is_ensemble and std_n is not None:
        ax.fill_between(t, mean_n - 2*std_n, mean_n + 2*std_n,
                        alpha=0.3, color='blue', label='±2σ')
    ax.set_ylabel(r'$\Gamma_n$ (Particle Flux)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_title(f'{title_prefix}Transport Coefficients', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add error annotation
    err_n = np.abs(np.mean(ref_n) - np.mean(mean_n)) / np.abs(np.mean(ref_n))
    ax.text(0.02, 0.95, f'Mean Error: {err_n:.2%}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gamma_c plot
    ax = axes[1]
    ax.plot(t, ref_c, 'k-', label='Reference', linewidth=1, alpha=0.8)
    ax.plot(t, mean_c, 'r--', label=f'{method_name} Prediction', linewidth=1)
    if is_ensemble and std_c is not None:
        ax.fill_between(t, mean_c - 2*std_c, mean_c + 2*std_c,
                        alpha=0.3, color='red', label='±2σ')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\Gamma_c$ (Conductive Flux)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add error annotation
    err_c = np.abs(np.mean(ref_c) - np.mean(mean_c)) / np.abs(np.mean(ref_c))
    ax.text(0.02, 0.95, f'Mean Error: {err_c:.2%}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved {output_path}")


# Alias for backwards compatibility
plot_gamma_comparison = plot_gamma_timeseries


# =============================================================================
# ERROR DISTRIBUTION PLOTS
# =============================================================================

def plot_error_distribution(results: list, output_dir: str, logger, 
                            filename="error_distribution.png"):
    """
    Plot error distribution from hyperparameter sweep.
    
    Creates histograms and box plots of error metrics.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    if not results:
        logger.warning("No results to plot")
        return
    
    errors = {
        r'$E_{\mu,n}$': np.array([r['mean_err_Gamma_n'] for r in results]),
        r'$E_{\sigma,n}$': np.array([r['std_err_Gamma_n'] for r in results]),
        r'$E_{\mu,c}$': np.array([r['mean_err_Gamma_c'] for r in results]),
        r'$E_{\sigma,c}$': np.array([r['std_err_Gamma_c'] for r in results]),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, (name, arr) in zip(axes, errors.items()):
        # Histogram
        ax.hist(arr, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.median(arr), color='r', linestyle='--', label=f'Median: {np.median(arr):.4f}')
        ax.axvline(np.mean(arr), color='g', linestyle=':', label=f'Mean: {np.mean(arr):.4f}')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved error distribution plot to {plot_path}")


# =============================================================================
# RECONSTRUCTION ERROR PLOTS
# =============================================================================

def plot_reconstruction_error(time: np.ndarray, ref: np.ndarray, pred: np.ndarray,
                               output_dir: str, logger, label: str = "reconstruction",
                               filename: str = None):
    """
    Plot reconstruction error over time.
    
    Creates a 2-panel figure showing:
    1. Reference vs prediction time series
    2. Absolute and relative error
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Time series comparison
    ax = axes[0]
    ax.plot(time, ref, 'k-', label='Reference', linewidth=1)
    ax.plot(time, pred, 'b-', label='Prediction', linewidth=1, alpha=0.7)
    ax.set_ylabel(label)
    ax.legend()
    ax.set_title(f'{label}: Reference vs Prediction')
    ax.grid(True, alpha=0.3)
    
    # Error
    ax = axes[1]
    abs_err = np.abs(ref - pred)
    ax.semilogy(time, abs_err, 'r-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Reconstruction Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename is None:
        filename = f'{label.lower().replace(" ", "_")}_error.png'
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved reconstruction error plot to {plot_path}")


# =============================================================================
# EIGENVALUE/SINGULAR VALUE PLOTS
# =============================================================================

def plot_singular_values(singular_values: np.ndarray, output_dir: str, logger,
                          r: int = None, filename: str = "singular_values.png"):
    """Plot singular value spectrum."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogy(np.arange(1, len(singular_values) + 1), singular_values, 'b.-', markersize=4)
    
    if r is not None and 0 < r <= len(singular_values):
        ax.axvline(r, color='r', linestyle='--', label=f'Truncation: r={r}')
        ax.legend()
    
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Spectrum')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved singular values plot to {plot_path}")


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# =============================================================================
# STATE PREDICTION DIAGNOSTICS
# =============================================================================

def plot_state_error_timeseries(
    pred_states: np.ndarray,
    ref_states: np.ndarray,
    dt: float,
    output_path: str,
    logger,
    n_y: Optional[int] = None,
    n_x: Optional[int] = None,
    title_prefix: str = "",
    method_name: str = "ROM",
):
    """
    Plot relative L2 error of state predictions over time.
    
    Creates a 2-panel figure showing:
    1. Total relative L2 error over time
    2. Per-field relative L2 error (density and potential) if n_y, n_x provided
    
    Parameters
    ----------
    pred_states : np.ndarray, shape (n_spatial, n_time)
        Predicted full states.
    ref_states : np.ndarray, shape (n_spatial, n_time)
        Reference full states.
    dt : float
        Time step size.
    output_path : str
        Full path to save the figure.
    logger : logging.Logger
        Logger instance.
    n_y : int, optional
        Grid size in y (for per-field error).
    n_x : int, optional
        Grid size in x (for per-field error).
    title_prefix : str
        Prefix for the plot title.
    method_name : str
        Name of the method for legend.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping state error plot")
        return
    
    n_spatial, n_time = pred_states.shape
    t = np.arange(n_time) * dt
    
    # Compute relative L2 error at each timestep
    diff = pred_states - ref_states
    l2_error = np.linalg.norm(diff, axis=0) / np.linalg.norm(ref_states, axis=0)
    
    # Check if we can compute per-field errors
    has_fields = (n_y is not None and n_x is not None and n_spatial == 2 * n_y * n_x)
    
    if has_fields:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        axes = [axes]
    
    # Total L2 error
    ax = axes[0]
    ax.semilogy(t, l2_error, 'b-', linewidth=1, label=f'{method_name}')
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(f'{title_prefix}State Prediction Error', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    mean_err = np.mean(l2_error)
    max_err = np.max(l2_error)
    ax.text(0.02, 0.95, f'Mean: {mean_err:.2e}, Max: {max_err:.2e}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Per-field errors if available
    if has_fields:
        spatial_per_field = n_y * n_x
        
        # Density error
        density_pred = pred_states[:spatial_per_field, :]
        density_ref = ref_states[:spatial_per_field, :]
        density_err = np.linalg.norm(density_pred - density_ref, axis=0) / np.linalg.norm(density_ref, axis=0)
        
        # Potential error
        phi_pred = pred_states[spatial_per_field:, :]
        phi_ref = ref_states[spatial_per_field:, :]
        phi_err = np.linalg.norm(phi_pred - phi_ref, axis=0) / np.linalg.norm(phi_ref, axis=0)
        
        ax = axes[1]
        ax.semilogy(t, density_err, 'b-', linewidth=1, label='Density (n)', alpha=0.8)
        ax.semilogy(t, phi_err, 'r-', linewidth=1, label='Potential (φ)', alpha=0.8)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title('Per-Field Error', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved state error plot to {output_path}")


def plot_state_snapshots(
    pred_states: np.ndarray,
    ref_states: np.ndarray,
    n_y: int,
    n_x: int,
    dt: float,
    output_path: str,
    logger,
    snapshot_indices: Optional[List[int]] = None,
    n_snapshots: int = 5,
    title_prefix: str = "",
    method_name: str = "ROM",
):
    """
    Plot 2D field comparisons at selected timesteps.
    
    Creates a multi-panel figure with rows for each snapshot and columns:
    [Reference Density | Predicted Density | Error | Reference φ | Predicted φ | Error]
    
    Parameters
    ----------
    pred_states : np.ndarray, shape (n_spatial, n_time)
        Predicted full states.
    ref_states : np.ndarray, shape (n_spatial, n_time)
        Reference full states.
    n_y : int
        Grid size in y.
    n_x : int
        Grid size in x.
    dt : float
        Time step size.
    output_path : str
        Full path to save the figure.
    logger : logging.Logger
        Logger instance.
    snapshot_indices : list, optional
        Specific timestep indices to plot. If None, evenly spaced.
    n_snapshots : int
        Number of snapshots if snapshot_indices not specified.
    title_prefix : str
        Prefix for the plot title.
    method_name : str
        Name of the method for title.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping snapshot plots")
        return
    
    n_spatial, n_time = pred_states.shape
    spatial_per_field = n_y * n_x
    
    # Validate dimensions
    if n_spatial != 2 * spatial_per_field:
        logger.warning(f"State dimension mismatch: {n_spatial} != 2*{spatial_per_field}")
        return
    
    # Select snapshot indices
    if snapshot_indices is None:
        # Evenly spaced, including start and end
        snapshot_indices = np.linspace(0, n_time - 1, n_snapshots, dtype=int)
    
    n_snaps = len(snapshot_indices)
    
    # Create figure: 6 columns (ref_n, pred_n, err_n, ref_phi, pred_phi, err_phi)
    fig, axes = plt.subplots(n_snaps, 6, figsize=(18, 3 * n_snaps))
    if n_snaps == 1:
        axes = axes[np.newaxis, :]
    
    for row, idx in enumerate(snapshot_indices):
        t_val = idx * dt
        
        # Extract and reshape fields
        ref_n = ref_states[:spatial_per_field, idx].reshape(n_y, n_x)
        ref_phi = ref_states[spatial_per_field:, idx].reshape(n_y, n_x)
        pred_n = pred_states[:spatial_per_field, idx].reshape(n_y, n_x)
        pred_phi = pred_states[spatial_per_field:, idx].reshape(n_y, n_x)
        
        err_n = np.abs(pred_n - ref_n)
        err_phi = np.abs(pred_phi - ref_phi)
        
        # Determine color limits for density
        vmin_n, vmax_n = min(ref_n.min(), pred_n.min()), max(ref_n.max(), pred_n.max())
        vmin_phi, vmax_phi = min(ref_phi.min(), pred_phi.min()), max(ref_phi.max(), pred_phi.max())
        
        # Row label
        axes[row, 0].set_ylabel(f't = {t_val:.2f}', fontsize=11, fontweight='bold')
        
        # Density panels
        im0 = axes[row, 0].imshow(ref_n, cmap='RdBu_r', aspect='equal', vmin=vmin_n, vmax=vmax_n)
        im1 = axes[row, 1].imshow(pred_n, cmap='RdBu_r', aspect='equal', vmin=vmin_n, vmax=vmax_n)
        im2 = axes[row, 2].imshow(err_n, cmap='hot', aspect='equal')
        
        # Potential panels
        im3 = axes[row, 3].imshow(ref_phi, cmap='RdBu_r', aspect='equal', vmin=vmin_phi, vmax=vmax_phi)
        im4 = axes[row, 4].imshow(pred_phi, cmap='RdBu_r', aspect='equal', vmin=vmin_phi, vmax=vmax_phi)
        im5 = axes[row, 5].imshow(err_phi, cmap='hot', aspect='equal')
        
        # Add colorbars on last row
        if row == n_snaps - 1:
            for col, im in enumerate([im0, im1, im2, im3, im4, im5]):
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Turn off ticks for cleaner look
        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    # Column titles
    col_titles = ['Ref Density', f'{method_name} Density', 'Density |Error|',
                  'Ref Potential', f'{method_name} Potential', 'Potential |Error|']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)
    
    fig.suptitle(f'{title_prefix}State Snapshots', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved snapshot comparison to {output_path}")


# =============================================================================
# STATE DIAGNOSTIC PLOT GENERATION
# =============================================================================

def generate_state_diagnostic_plots(
    reduced_states: List[np.ndarray],
    ref_files: List[str],
    boundaries: np.ndarray,
    pod_basis: Optional[np.ndarray],
    temporal_mean: Optional[np.ndarray],
    n_y: int,
    n_x: int,
    engine: str,
    dt: float,
    output_dir: str,
    logger,
    method_name: str = "ROM",
    prefix: str = "",
    ref_offset: int = 0,
    plot_error: bool = True,
    plot_snapshots: bool = True,
    n_snapshots: int = 5,
    is_nan_flags: Optional[List[bool]] = None,
    reduction_method: str = "linear",
    manifold_W: Optional[np.ndarray] = None,
    manifold_shift: Optional[np.ndarray] = None,
    pde: str = "hw2d",
    dx: float = None,
):
    """
    Generate state-based diagnostic plots (L2 error and snapshots).
    
    This is a unified function for both DMD and OpInf pipelines.
    
    Parameters
    ----------
    reduced_states : list of np.ndarray
        List of reduced-space predictions, one per trajectory.
        Each array should be shape (r, n_time) or (n_time, r).
        For ensemble methods, pass the ensemble mean.
    ref_files : list of str
        Reference data files, one per trajectory.
    boundaries : np.ndarray
        Trajectory boundaries array.
    pod_basis : np.ndarray, shape (n_spatial, r)
        POD basis matrix.
    temporal_mean : np.ndarray or None, shape (n_spatial,)
        Temporal mean for reconstruction.
    n_y : int
        Grid size in y.
    n_x : int
        Grid size in x.
    engine : str
        xarray engine for loading data.
    dt : float
        Time step.
    output_dir : str
        Directory to save plots.
    logger : logging.Logger
        Logger instance.
    method_name : str
        Name of the method for plot labels (e.g., "DMD", "OpInf").
    prefix : str
        Prefix for filenames (e.g., "train_", "test_").
    ref_offset : int
        Offset into reference file for temporal_split mode.
    plot_error : bool
        Whether to generate L2 error timeseries plots.
    plot_snapshots : bool
        Whether to generate snapshot comparison plots.
    n_snapshots : int
        Number of snapshots for comparison plots.
    is_nan_flags : list of bool, optional
        Flags indicating which trajectories have NaN predictions.
    reduction_method : str
        Reduction method used ("linear" for POD or "manifold" for quadratic).
    manifold_W : np.ndarray, optional
        Quadratic coefficient matrix for manifold reconstruction.
        Required if reduction_method == "manifold".
    manifold_shift : np.ndarray, optional
        Mean shift vector for manifold reconstruction.
        Required if reduction_method == "manifold".
    """
    # Import here to avoid circular imports
    from shared.data_io import load_reference_states, load_ks_reference_states, reconstruct_full_state, reconstruct_full_state_manifold
    
    if pod_basis is None:
        logger.warning("POD basis not available, skipping state diagnostic plots")
        return
    
    # Validate manifold parameters
    if reduction_method == "manifold":
        if manifold_W is None or manifold_shift is None:
            logger.warning("Manifold reconstruction requires W and shift matrices. Skipping plots.")
            return
    
    if not plot_error and not plot_snapshots:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    n_traj = len(boundaries) - 1
    
    logger.info(f"Generating state diagnostic plots for {n_traj} {prefix.strip('_') or 'trajectory'}(ies)...")
    logger.info(f"  Using {reduction_method} reconstruction")
    
    for i in range(n_traj):
        # Check for NaN/missing predictions
        if is_nan_flags is not None and is_nan_flags[i]:
            logger.warning(f"  Skipping trajectory {i+1} (NaN)")
            continue
        
        X_hat = reduced_states[i]
        if X_hat is None:
            logger.warning(f"  Skipping trajectory {i+1} (missing)")
            continue
        
        n_steps = boundaries[i + 1] - boundaries[i]
        
        # Load reference states (PDE-specific)
        if pde == "ks":
            ref_Q, ref_N = load_ks_reference_states(
                ref_files[i], n_steps, ref_offset
            )
        else:
            ref_Q, ref_ny, ref_nx = load_reference_states(
                ref_files[i], engine, n_steps, ref_offset
            )
        
        # Truncate POD basis if needed
        r = X_hat.shape[0] if X_hat.shape[0] < X_hat.shape[1] else X_hat.shape[1]
        U_r = pod_basis[:, :r] if pod_basis.shape[1] >= r else pod_basis
        
        # Reconstruct predicted full state based on reduction method
        if reduction_method == "manifold":
            pred_Q = reconstruct_full_state_manifold(X_hat, U_r, manifold_W, manifold_shift)
        else:
            pred_Q = reconstruct_full_state(X_hat, U_r, temporal_mean)
        
        title_prefix = f'{prefix.replace("_", " ").title()}Trajectory {i+1}: '
        
        # L2 error timeseries
        if plot_error:
            error_path = os.path.join(output_dir, f'{prefix}traj_{i+1}_state_error.png')
            if pde == "ks":
                plot_ks_state_error_timeseries(
                    pred_states=pred_Q,
                    ref_states=ref_Q,
                    dt=dt,
                    output_path=error_path,
                    logger=logger,
                    title_prefix=title_prefix,
                    method_name=method_name
                )
            else:
                plot_state_error_timeseries(
                    pred_states=pred_Q,
                    ref_states=ref_Q,
                    dt=dt,
                    output_path=error_path,
                    logger=logger,
                    n_y=n_y,
                    n_x=n_x,
                    title_prefix=title_prefix,
                    method_name=method_name
                )
        
        # Snapshot comparisons
        if plot_snapshots:
            snapshot_path = os.path.join(output_dir, f'{prefix}traj_{i+1}_snapshots.png')
            if pde == "ks":
                ks_dx = dx if dx is not None else (n_x * 1.0 / n_x if n_x > 0 else 0.5)
                plot_ks_state_snapshots(
                    pred_states=pred_Q,
                    ref_states=ref_Q,
                    N=ref_N,
                    dt=dt,
                    dx=ks_dx,
                    output_path=snapshot_path,
                    logger=logger,
                    n_snapshots=n_snapshots,
                    title_prefix=title_prefix,
                    method_name=method_name
                )
            else:
                plot_state_snapshots(
                    pred_states=pred_Q,
                    ref_states=ref_Q,
                    n_y=n_y,
                    n_x=n_x,
                    dt=dt,
                    output_path=snapshot_path,
                    logger=logger,
                    n_snapshots=n_snapshots,
                    title_prefix=title_prefix,
                    method_name=method_name
                )
    
    logger.info(f"State diagnostic plots saved to {output_dir}")


# =============================================================================
# KS / GENERIC QOI PLOTS
# =============================================================================

def plot_qoi_timeseries(
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    ref_1: np.ndarray,
    ref_2: np.ndarray,
    dt: float,
    output_path: str,
    logger,
    label_1: str = "Energy",
    label_2: str = "Enstrophy",
    symbol_1: str = r"$E$",
    symbol_2: str = r"$P$",
    title_prefix: str = "",
    method_name: str = "ROM",
):
    """
    Generate a generic QoI comparison plot (reference vs prediction).

    Works for both HW2D (Gamma_n, Gamma_c) and KS (energy, enstrophy).
    Handles single predictions and ensemble predictions (with uncertainty).

    Parameters
    ----------
    pred_1, pred_2 : np.ndarray, shape (n_steps,) or (n_ensemble, n_steps)
        Predicted QoIs. If 2D, ensemble mean and +/-2sigma are plotted.
    ref_1, ref_2 : np.ndarray, shape (n_steps,)
        Reference QoIs.
    dt : float
        Time step size.
    output_path : str
        Full path to save the figure.
    logger : logging.Logger
        Logger instance.
    label_1, label_2 : str
        Human-readable names for the QoIs.
    symbol_1, symbol_2 : str
        LaTeX symbols for axis labels.
    title_prefix : str
        Prefix for the plot title.
    method_name : str
        Name of the method for legend.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return

    n_steps = len(ref_1)
    t = np.arange(n_steps) * dt

    is_ensemble = pred_1.ndim == 2

    if is_ensemble:
        mean_1, std_1 = np.mean(pred_1, axis=0), np.std(pred_1, axis=0)
        mean_2, std_2 = np.mean(pred_2, axis=0), np.std(pred_2, axis=0)
    else:
        mean_1, std_1 = pred_1, None
        mean_2, std_2 = pred_2, None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # QoI 1 plot
    ax = axes[0]
    ax.plot(t, ref_1, 'k-', label='Reference', linewidth=1, alpha=0.8)
    ax.plot(t, mean_1, 'b--', label=f'{method_name} Prediction', linewidth=1)
    if is_ensemble and std_1 is not None:
        ax.fill_between(t, mean_1 - 2 * std_1, mean_1 + 2 * std_1,
                        alpha=0.3, color='blue', label='±2σ')
    ax.set_ylabel(f'{symbol_1} ({label_1})', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_title(f'{title_prefix}Quantities of Interest', fontsize=14)
    ax.grid(True, alpha=0.3)

    ref_mean_1 = np.mean(ref_1)
    if abs(ref_mean_1) > 1e-12:
        err_1 = np.abs(np.mean(ref_1) - np.mean(mean_1)) / np.abs(ref_mean_1)
        ax.text(0.02, 0.95, f'Mean Error: {err_1:.2%}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # QoI 2 plot
    ax = axes[1]
    ax.plot(t, ref_2, 'k-', label='Reference', linewidth=1, alpha=0.8)
    ax.plot(t, mean_2, 'r--', label=f'{method_name} Prediction', linewidth=1)
    if is_ensemble and std_2 is not None:
        ax.fill_between(t, mean_2 - 2 * std_2, mean_2 + 2 * std_2,
                        alpha=0.3, color='red', label='±2σ')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(f'{symbol_2} ({label_2})', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ref_mean_2 = np.mean(ref_2)
    if abs(ref_mean_2) > 1e-12:
        err_2 = np.abs(np.mean(ref_2) - np.mean(mean_2)) / np.abs(ref_mean_2)
        ax.text(0.02, 0.95, f'Mean Error: {err_2:.2%}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved {output_path}")


def plot_ks_state_snapshots(
    pred_states: np.ndarray,
    ref_states: np.ndarray,
    N: int,
    dt: float,
    dx: float,
    output_path: str,
    logger,
    snapshot_indices: Optional[List[int]] = None,
    n_snapshots: int = 5,
    title_prefix: str = "",
    method_name: str = "ROM",
):
    """
    Plot 1D KS field comparisons at selected timesteps.

    Creates a multi-row figure with rows:
    - Top: space–time heatmap comparison (reference, predicted, error)
    - Below: line plots at selected timesteps

    Parameters
    ----------
    pred_states : np.ndarray, shape (N, n_time)
        Predicted states.
    ref_states : np.ndarray, shape (N, n_time)
        Reference states.
    N : int
        Number of spatial points.
    dt : float
        Time step.
    dx : float
        Spatial grid spacing.
    output_path : str
        Full path to save the figure.
    logger : logging.Logger
        Logger instance.
    snapshot_indices : list, optional
        Specific timestep indices to plot lines for.
    n_snapshots : int
        Number of snapshots if indices not specified.
    title_prefix : str
        Prefix for the plot title.
    method_name : str
        Method name for labels.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping KS snapshot plots")
        return

    n_spatial, n_time = pred_states.shape
    L = N * dx
    t_max = n_time * dt
    x = np.linspace(0, L, N, endpoint=False)

    if snapshot_indices is None:
        snapshot_indices = np.linspace(0, n_time - 1, n_snapshots, dtype=int)

    n_snaps = len(snapshot_indices)

    fig, axes = plt.subplots(1 + n_snaps, 3, figsize=(16, 3 * (1 + n_snaps)),
                             gridspec_kw={'height_ratios': [2] + [1] * n_snaps})

    # Determine consistent color limits
    vmin = min(ref_states.min(), pred_states.min())
    vmax = max(ref_states.max(), pred_states.max())

    # Top row: space-time heatmaps
    extent = (0, t_max, 0, L)

    im0 = axes[0, 0].imshow(ref_states, cmap='RdBu', aspect='auto', origin='lower',
                              extent=extent, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Reference u(x,t)')
    axes[0, 0].set_ylabel('x')

    im1 = axes[0, 1].imshow(pred_states, cmap='RdBu', aspect='auto', origin='lower',
                              extent=extent, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'{method_name} u(x,t)')

    err = np.abs(pred_states - ref_states)
    im2 = axes[0, 2].imshow(err, cmap='hot', aspect='auto', origin='lower', extent=extent)
    axes[0, 2].set_title('|Error|')
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    for col in range(2):
        fig.colorbar([im0, im1][col], ax=axes[0, col], fraction=0.046, pad=0.04)

    # Line plots at selected timesteps
    for row, idx in enumerate(snapshot_indices, start=1):
        t_val = idx * dt
        ref_line = ref_states[:, idx]
        pred_line = pred_states[:, idx]

        axes[row, 0].plot(x, ref_line, 'k-', linewidth=1, label='Reference')
        axes[row, 0].plot(x, pred_line, 'b--', linewidth=1, label=method_name)
        axes[row, 0].set_ylabel(f't={t_val:.1f}')
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(x, pred_line - ref_line, 'r-', linewidth=1)
        axes[row, 1].set_title('Error' if row == 1 else '')
        axes[row, 1].grid(True, alpha=0.3)

        rel_l2 = np.linalg.norm(pred_line - ref_line) / max(np.linalg.norm(ref_line), 1e-12)
        axes[row, 2].text(0.5, 0.5, f'Rel L2: {rel_l2:.2e}',
                          transform=axes[row, 2].transAxes,
                          ha='center', va='center', fontsize=12)
        axes[row, 2].axis('off')

    fig.suptitle(f'{title_prefix}KS State Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved KS snapshot comparison to {output_path}")


def plot_ks_state_error_timeseries(
    pred_states: np.ndarray,
    ref_states: np.ndarray,
    dt: float,
    output_path: str,
    logger,
    title_prefix: str = "",
    method_name: str = "ROM",
):
    """
    Plot relative L2 error of KS state predictions over time.

    Parameters
    ----------
    pred_states : np.ndarray, shape (N, n_time)
        Predicted states.
    ref_states : np.ndarray, shape (N, n_time)
        Reference states.
    dt : float
        Time step.
    output_path : str
        Full path to save the figure.
    logger : logging.Logger
        Logger instance.
    title_prefix : str
        Prefix for the plot title.
    method_name : str
        Method name for legend.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping state error plot")
        return

    n_spatial, n_time = pred_states.shape
    t = np.arange(n_time) * dt

    diff = pred_states - ref_states
    ref_norms = np.linalg.norm(ref_states, axis=0)
    ref_norms = np.where(ref_norms < 1e-12, 1e-12, ref_norms)
    l2_error = np.linalg.norm(diff, axis=0) / ref_norms

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.semilogy(t, l2_error, 'b-', linewidth=1, label=method_name)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(f'{title_prefix}KS State Prediction Error', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    mean_err = np.mean(l2_error)
    max_err = np.max(l2_error)
    ax.text(0.02, 0.95, f'Mean: {mean_err:.2e}, Max: {max_err:.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved KS state error plot to {output_path}")


# =============================================================================
# KS FULL TRAJECTORY PLOTS
# =============================================================================

def plot_ks_full_trajectory_reconstruction(
    pred_states,
    ref_states,
    dt, dx,
    train_n_steps,
    output_path,
    logger,
    method_name="ROM",
    vmin=None, vmax=None,
    err_vmax=None,
    t_start=0.0,
):
    """
    Full trajectory space-time reconstruction for KS equation.

    Three-panel figure showing reference, prediction, and absolute error
    as space-time heatmaps. A horizontal dashed line marks the train/test
    boundary. Colormap limits default to reference data range for
    cross-method consistency.

    Parameters
    ----------
    pred_states : np.ndarray, shape (n_time, N)
        Concatenated train+test predicted u field.
    ref_states : np.ndarray, shape (n_time, N)
        Concatenated train+test reference u field.
    dt : float
        Time step size.
    dx : float
        Spatial grid spacing.
    train_n_steps : int
        Number of training timesteps (for boundary marker).
    output_path : str
        Path to save figure.
    logger : logging.Logger
        Logger instance.
    method_name : str
        Name for titles (e.g., "FNO", "OpInf", "DMD").
    vmin, vmax : float, optional
        Colormap limits for field values. Defaults to reference min/max.
    err_vmax : float, optional
        Upper limit for error colormap. Defaults to 99th percentile.
    t_start : float
        Time offset for first snapshot.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping trajectory reconstruction plot")
        return

    n_time, N = ref_states.shape

    if vmin is None:
        vmin = ref_states.min()
    if vmax is None:
        vmax = ref_states.max()

    abs_error = np.abs(pred_states - ref_states)
    if err_vmax is None:
        err_vmax = np.percentile(abs_error, 99)

    # Build pcolormesh edge arrays
    x_edges = np.arange(N + 1) * dx
    t_edges = t_start + np.arange(n_time + 1) * dt
    t_boundary = t_start + train_n_steps * dt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Panel 1 — Reference
    im0 = axes[0].pcolormesh(x_edges, t_edges, ref_states,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='flat')
    axes[0].axhline(t_boundary, color='k', linestyle='--', linewidth=1.5)
    axes[0].set_title('Reference')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Time')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2 — Prediction
    im1 = axes[1].pcolormesh(x_edges, t_edges, pred_states,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='flat')
    axes[1].axhline(t_boundary, color='k', linestyle='--', linewidth=1.5)
    axes[1].set_title(method_name)
    axes[1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — |Error|
    im2 = axes[2].pcolormesh(x_edges, t_edges, abs_error,
                             cmap='hot', vmin=0, vmax=err_vmax, shading='flat')
    axes[2].axhline(t_boundary, color='w', linestyle='--', linewidth=1.5)
    axes[2].set_title('|Error|')
    axes[2].set_xlabel('x')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f'{method_name} — Full Trajectory u(x,t)', fontsize=14)
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved KS full trajectory reconstruction to {output_path}")


def plot_ks_full_trajectory_qoi(
    pred_qoi_1, pred_qoi_2,
    ref_qoi_1, ref_qoi_2,
    dt,
    train_n_steps,
    output_path,
    logger,
    method_name="ROM",
    label_1="Energy", label_2="Enstrophy",
    symbol_1=r"$E$", symbol_2=r"$P$",
    ylim_1=None, ylim_2=None,
    t_start=0.0,
):
    """
    Full trajectory derived quantity comparison for KS equation.

    Two-panel figure showing Energy and Enstrophy timeseries with
    reference vs prediction. A vertical dashed line marks the
    train/test boundary. Axis limits can be specified for cross-method
    consistency.

    Parameters
    ----------
    pred_qoi_1, pred_qoi_2 : np.ndarray, shape (n_time,)
        Predicted QoI arrays (Energy, Enstrophy).
    ref_qoi_1, ref_qoi_2 : np.ndarray, shape (n_time,)
        Reference QoI arrays.
    dt : float
        Time step size.
    train_n_steps : int
        Number of training timesteps (for boundary marker).
    output_path : str
        Path to save figure.
    logger : logging.Logger
        Logger instance.
    method_name : str
        Name for legend entries.
    label_1, label_2 : str
        Labels for the two QoIs.
    symbol_1, symbol_2 : str
        Symbols for axis labels.
    ylim_1, ylim_2 : tuple, optional
        (ymin, ymax) for each panel. Defaults to reference range with 10% padding.
    t_start : float
        Time offset for first snapshot.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping trajectory QoI plot")
        return

    n_time = len(ref_qoi_1)
    t = t_start + np.arange(n_time) * dt
    t_boundary = t_start + train_n_steps * dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Panel 1: QoI 1 (Energy) ---
    axes[0].plot(t, ref_qoi_1, 'k-', linewidth=1.5, alpha=0.8, label='Reference')
    axes[0].plot(t, pred_qoi_1, 'b-', linewidth=1.5, alpha=0.8, label=method_name)
    axes[0].axvline(t_boundary, color='gray', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='Train/Test')
    axes[0].set_ylabel(f'{label_1} ({symbol_1})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if ylim_1 is not None:
        axes[0].set_ylim(ylim_1)
    else:
        ref_range = ref_qoi_1.max() - ref_qoi_1.min()
        pad = 0.1 * ref_range if ref_range > 0 else 1.0
        axes[0].set_ylim(ref_qoi_1.min() - pad, ref_qoi_1.max() + pad)

    ref_mean_1 = np.mean(ref_qoi_1)
    rel_err_1 = np.abs(np.mean(pred_qoi_1) - ref_mean_1) / max(np.abs(ref_mean_1), 1e-12)
    axes[0].text(0.02, 0.95, f'Rel. mean error: {rel_err_1:.2e}',
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 2: QoI 2 (Enstrophy) ---
    axes[1].plot(t, ref_qoi_2, 'k-', linewidth=1.5, alpha=0.8, label='Reference')
    axes[1].plot(t, pred_qoi_2, 'r-', linewidth=1.5, alpha=0.8, label=method_name)
    axes[1].axvline(t_boundary, color='gray', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='Train/Test')
    axes[1].set_ylabel(f'{label_2} ({symbol_2})')
    axes[1].set_xlabel('Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if ylim_2 is not None:
        axes[1].set_ylim(ylim_2)
    else:
        ref_range = ref_qoi_2.max() - ref_qoi_2.min()
        pad = 0.1 * ref_range if ref_range > 0 else 1.0
        axes[1].set_ylim(ref_qoi_2.min() - pad, ref_qoi_2.max() + pad)

    ref_mean_2 = np.mean(ref_qoi_2)
    rel_err_2 = np.abs(np.mean(pred_qoi_2) - ref_mean_2) / max(np.abs(ref_mean_2), 1e-12)
    axes[1].text(0.02, 0.95, f'Rel. mean error: {rel_err_2:.2e}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{method_name} — Full Trajectory Derived Quantities', fontsize=14)
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved KS full trajectory QoI plot to {output_path}")


# =============================================================================
# KS PHYSICS-PRESERVATION CROSS-METHOD PLOTS
# =============================================================================

def plot_ks_cross_method_psd(
    methods: list,
    ref_u: np.ndarray,
    grid: dict,
    output_path: str,
    logger,
    color_ref: str = "black",
):
    """
    Log-log power spectral density comparison across methods.

    Parameters
    ----------
    methods : list[dict]
        Method dicts with keys ``name``, ``color``, ``u_test``.
    ref_u : np.ndarray, shape (n_time, N)
        Reference field (test region).
    grid : dict
        KS grid parameters (``dx``, ``N``).
    output_path : str
        Full path for saved figure.
    logger : logging.Logger
    color_ref : str
        Colour for reference line.
    """
    if not HAS_MATPLOTLIB:
        return

    from shared.physics import compute_ks_psd

    k_ref, psd_ref = compute_ks_psd(ref_u, grid["dx"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.loglog(k_ref[1:], psd_ref[1:], color=color_ref, linewidth=1.5,
              label="Reference DNS")

    for m in methods:
        k, psd = compute_ks_psd(m["u_test"], grid["dx"])
        ax.loglog(k[1:], psd[1:], color=m["color"], linewidth=1.2,
                  label=m["name"], alpha=0.85)

    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Power Spectral Density $|\hat{u}|^2$")
    ax.set_title("Time-Averaged Spatial Power Spectrum — KS (Test Region)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


def plot_ks_cross_method_pdf(
    methods: list,
    ref_u: np.ndarray,
    output_path: str,
    logger,
    n_bins: int = 100,
    color_ref: str = "black",
):
    """
    Overlaid probability density functions of u(x,t).

    Parameters
    ----------
    methods : list[dict]
        Method dicts with keys ``name``, ``color``, ``u_test``.
    ref_u : np.ndarray, shape (n_time, N)
        Reference field (test region).
    output_path : str
    logger : logging.Logger
    n_bins : int
        Number of histogram bins.
    color_ref : str
    """
    if not HAS_MATPLOTLIB:
        return

    from shared.physics import compute_ks_field_pdf

    centres_ref, density_ref = compute_ks_field_pdf(ref_u, n_bins)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(centres_ref, density_ref, color=color_ref, linewidth=1.5,
            label="Reference DNS")

    for m in methods:
        centres, density = compute_ks_field_pdf(m["u_test"], n_bins)
        ax.plot(centres, density, color=m["color"], linewidth=1.2,
                label=m["name"], alpha=0.85)

    ax.set_xlabel(r"$u$")
    ax.set_ylabel("Probability Density")
    ax.set_title("Field Value Distribution — KS (Test Region)")
    ax.legend(loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


def plot_ks_cross_method_autocorrelation(
    methods: list,
    ref_u: np.ndarray,
    grid: dict,
    output_path: str,
    logger,
    color_ref: str = "black",
):
    """
    Spatial autocorrelation C(Δx) comparison.

    Parameters
    ----------
    methods : list[dict]
        Method dicts with keys ``name``, ``color``, ``u_test``.
    ref_u : np.ndarray, shape (n_time, N)
        Reference field (test region).
    grid : dict
        KS grid parameters.
    output_path : str
    logger : logging.Logger
    color_ref : str
    """
    if not HAS_MATPLOTLIB:
        return

    from shared.physics import compute_ks_spatial_autocorrelation

    lags_ref, C_ref = compute_ks_spatial_autocorrelation(ref_u, grid["dx"])
    half = len(C_ref) // 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lags_ref[:half], C_ref[:half], color=color_ref, linewidth=1.5,
            label="Reference DNS")

    for m in methods:
        lags, C = compute_ks_spatial_autocorrelation(m["u_test"], grid["dx"])
        ax.plot(lags[:half], C[:half], color=m["color"], linewidth=1.2,
                label=m["name"], alpha=0.85)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Spatial Lag $\Delta x$")
    ax.set_ylabel(r"Autocorrelation $C(\Delta x)$")
    ax.set_title("Spatial Autocorrelation — KS (Test Region)")
    ax.legend(loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


def plot_ks_cross_method_energy_rate(
    methods: list,
    ref_u: np.ndarray,
    grid: dict,
    dt: float,
    output_path: str,
    logger,
    color_ref: str = "black",
):
    """
    Energy rate balance dE/dt comparison (finite-difference vs PDE terms).

    Top panel: dE/dt timeseries for each method.
    Bottom panel: energy budget residual (how far the method violates the
    KS energy equation).

    Parameters
    ----------
    methods : list[dict]
        Method dicts with keys ``name``, ``color``, ``u_test``.
    ref_u : np.ndarray, shape (n_time, N)
        Reference field (test region).
    grid : dict
        KS grid parameters.
    dt : float
        Time step.
    output_path : str
    logger : logging.Logger
    color_ref : str
    """
    if not HAS_MATPLOTLIB:
        return

    from shared.physics import compute_ks_energy_rate

    ref_rates = compute_ks_energy_rate(ref_u, grid["dx"], dt)
    n_inner = len(ref_rates["dEdt_fd"])
    t = np.arange(n_inner) * dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- dE/dt comparison ---
    ax = axes[0]
    ax.plot(t, ref_rates["dEdt_fd"], color=color_ref, linewidth=1.0,
            label="Reference (FD)", alpha=0.8)
    ax.plot(t, ref_rates["dEdt_pde"][1:-1], color=color_ref, linewidth=1.0,
            linestyle="--", label="Reference (PDE)", alpha=0.6)

    for m in methods:
        rates = compute_ks_energy_rate(m["u_test"], grid["dx"], dt)
        n_m = len(rates["dEdt_fd"])
        t_m = np.arange(n_m) * dt
        ax.plot(t_m, rates["dEdt_fd"], color=m["color"], linewidth=1.0,
                label=f"{m['name']} (FD)", alpha=0.85)

    ax.set_ylabel(r"$dE/dt$")
    ax.set_title("Energy Rate — KS (Test Region)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # --- Residual (energy budget violation) ---
    ax = axes[1]
    ax.plot(t, ref_rates["residual"], color=color_ref, linewidth=1.0,
            label="Reference", alpha=0.8)

    for m in methods:
        rates = compute_ks_energy_rate(m["u_test"], grid["dx"], dt)
        n_m = len(rates["residual"])
        t_m = np.arange(n_m) * dt
        ax.plot(t_m, rates["residual"], color=m["color"], linewidth=1.0,
                label=m["name"], alpha=0.85)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Energy Budget Residual")
    ax.legend(loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


def plot_ks_cross_method_moments(
    methods: list,
    ref_u: np.ndarray,
    dt: float,
    output_path: str,
    logger,
    color_ref: str = "black",
):
    """
    Four-panel plot of spatial statistical moments vs time.

    Panels: mean, variance, skewness, excess kurtosis.

    Parameters
    ----------
    methods : list[dict]
        Method dicts with keys ``name``, ``color``, ``u_test``.
    ref_u : np.ndarray, shape (n_time, N)
        Reference field (test region).
    dt : float
        Time step.
    output_path : str
    logger : logging.Logger
    color_ref : str
    """
    if not HAS_MATPLOTLIB:
        return

    from shared.physics import compute_ks_statistical_moments

    ref_moments = compute_ks_statistical_moments(ref_u)
    n_time = ref_u.shape[0]
    t = np.arange(n_time) * dt

    labels = ["Mean", "Variance", "Skewness", "Excess Kurtosis"]
    keys = ["mean", "variance", "skewness", "kurtosis"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for i, (key, label) in enumerate(zip(keys, labels)):
        ax = axes[i]
        ax.plot(t, ref_moments[key], color=color_ref, linewidth=1.2,
                label="Reference DNS")

        for m in methods:
            mom = compute_ks_statistical_moments(m["u_test"])
            n_m = len(mom[key])
            t_m = np.arange(n_m) * dt
            ax.plot(t_m, mom[key], color=m["color"], linewidth=1.0,
                    label=m["name"], alpha=0.85)

        ax.set_ylabel(label)
        if i >= 2:
            ax.set_xlabel("Test Time")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Spatial Statistical Moments — KS (Test Region)", fontsize=14)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"  Saved {output_path}")
