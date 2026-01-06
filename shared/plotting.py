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

def plot_gamma_predictions(predictions: dict, ref_files: list, boundaries: np.ndarray,
                           dt: float, engine: str, output_dir: str, logger,
                           start_offset: int = 0):
    """
    Generate Gamma time series comparison plots.
    
    Creates one figure per trajectory showing:
    - Reference vs ensemble mean
    - ±2σ uncertainty band
    
    Args:
        start_offset: For temporal_split mode, the starting snapshot index
                      for loading reference data.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    # Import here to avoid circular imports
    from utils import load_dataset
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating Gamma plots in {output_dir}")
    
    n_traj = len(predictions['Gamma_n'])
    
    for traj_idx in range(n_traj):
        fh = load_dataset(ref_files[traj_idx], engine)
        traj_len = boundaries[traj_idx + 1] - boundaries[traj_idx]
        
        # Apply offset for temporal_split mode
        ref_n = fh["gamma_n"].values[start_offset:start_offset + traj_len]
        ref_c = fh["gamma_c"].values[start_offset:start_offset + traj_len]
        
        pred_n = predictions['Gamma_n'][traj_idx]
        pred_c = predictions['Gamma_c'][traj_idx]
        
        mean_n, std_n = np.mean(pred_n, axis=0), np.std(pred_n, axis=0)
        mean_c, std_c = np.mean(pred_c, axis=0), np.std(pred_c, axis=0)
        
        time = np.arange(traj_len) * dt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gamma_n
        ax = axes[0]
        ax.plot(time, ref_n, 'k-', label='Reference', linewidth=1)
        ax.plot(time, mean_n, 'b-', label='Ensemble Mean', linewidth=1)
        ax.fill_between(time, mean_n - 2*std_n, mean_n + 2*std_n,
                       alpha=0.3, color='blue', label='±2σ')
        ax.set_ylabel(r'$\Gamma_n$')
        ax.legend(loc='upper right')
        ax.set_title(f'Trajectory {traj_idx + 1}: Particle Flux')
        ax.grid(True, alpha=0.3)
        
        # Gamma_c
        ax = axes[1]
        ax.plot(time, ref_c, 'k-', label='Reference', linewidth=1)
        ax.plot(time, mean_c, 'r-', label='Ensemble Mean', linewidth=1)
        ax.fill_between(time, mean_c - 2*std_c, mean_c + 2*std_c,
                       alpha=0.3, color='red', label='±2σ')
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\Gamma_c$')
        ax.legend(loc='upper right')
        ax.set_title(f'Trajectory {traj_idx + 1}: Conductive Flux')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'traj_{traj_idx + 1}_gamma.png')
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        
        logger.info(f"  Saved {fig_path}")


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
