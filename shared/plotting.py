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
    """
    # Import here to avoid circular imports
    from shared.data_io import load_reference_states, reconstruct_full_state
    
    if pod_basis is None:
        logger.warning("POD basis not available, skipping state diagnostic plots")
        return
    
    if not plot_error and not plot_snapshots:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    n_traj = len(boundaries) - 1
    
    logger.info(f"Generating state diagnostic plots for {n_traj} {prefix.strip('_') or 'trajectory'}(ies)...")
    
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
        
        # Load reference states
        ref_Q, ref_ny, ref_nx = load_reference_states(
            ref_files[i], engine, n_steps, ref_offset
        )
        
        # Truncate POD basis if needed
        r = X_hat.shape[0] if X_hat.shape[0] < X_hat.shape[1] else X_hat.shape[1]
        U_r = pod_basis[:, :r] if pod_basis.shape[1] >= r else pod_basis
        
        # Reconstruct predicted full state
        pred_Q = reconstruct_full_state(X_hat, U_r, temporal_mean)
        
        title_prefix = f'{prefix.replace("_", " ").title()}Trajectory {i+1}: '
        
        # L2 error timeseries
        if plot_error:
            error_path = os.path.join(output_dir, f'{prefix}traj_{i+1}_state_error.png')
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
