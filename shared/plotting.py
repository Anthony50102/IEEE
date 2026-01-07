"""
Plotting utilities for ROM methods.

This module provides standardized plotting functions that can be used
across all three methods (OpInf, Manifold OpInf, DMD).

Author: Anthony Poole
"""

import os
import numpy as np

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
