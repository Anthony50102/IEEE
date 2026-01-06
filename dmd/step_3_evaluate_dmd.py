"""
Step 3: DMD Evaluation and Prediction.

This script handles:
1. Loading fitted DMD model from Step 2
2. Computing forecasts for training and test trajectories
3. Computing evaluation metrics
4. Generating diagnostic plots (optional)
5. Saving results

Usage:
    python step_3_evaluate_dmd.py --config config/dmd_1train_5test.yaml --run-dir /path/to/run

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
    print_dmd_config_summary,
    dmd_forecast_reduced,
    compute_gamma_from_state,
    reconstruct_full_state,
    DMDConfig,
)

# Import shared utilities from opinf
from opinf.utils import (
    setup_logging,
    save_config,
    save_step_status,
    check_step_completed,
    print_header,
    load_dataset as loader,
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_dmd_model(paths: dict, logger) -> dict:
    """
    Load DMD model from Step 2.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        DMD model dictionary.
    """
    logger.info(f"Loading DMD model from {paths['dmd_model']}")
    
    model_data = np.load(paths['dmd_model'])
    
    model = {
        'eigs': model_data['eigs'],
        'modes_reduced': model_data['modes_reduced'],
        'amplitudes': model_data['amplitudes'],
        'dt': float(model_data['dt']),
        'dmd_rank': int(model_data['dmd_rank']),
    }
    
    logger.info(f"  Eigenvalues shape: {model['eigs'].shape}")
    logger.info(f"  Modes shape: {model['modes_reduced'].shape}")
    logger.info(f"  DMD rank: {model['dmd_rank']}")
    logger.info(f"  dt: {model['dt']:.4f}")
    
    return model


def load_pod_basis(paths: dict, logger) -> dict:
    """
    Load POD basis from Step 1 for full-state reconstruction.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        POD basis dictionary with U_r, mean, n_y, n_x.
    """
    pod_basis_path = paths['pod_basis']
    
    if not os.path.exists(pod_basis_path):
        logger.warning(f"POD basis not found at {pod_basis_path}")
        logger.warning("Will not be able to compute physics-based Gamma")
        return None
    
    logger.info(f"Loading POD basis from {pod_basis_path}")
    
    basis_data = np.load(pod_basis_path)
    
    pod_basis = {
        'U_r': basis_data['U_r'],  # (n_spatial, r)
        'mean': basis_data['mean'],  # (n_spatial,)
        'n_y': int(basis_data['n_y']),
        'n_x': int(basis_data['n_x']),
    }
    
    logger.info(f"  U_r shape: {pod_basis['U_r'].shape}")
    logger.info(f"  Grid: {pod_basis['n_y']} x {pod_basis['n_x']}")
    
    return pod_basis


def load_initial_conditions(paths: dict, logger) -> dict:
    """
    Load initial conditions from Step 1.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Initial conditions dictionary.
    """
    logger.info(f"Loading initial conditions from {paths['initial_conditions']}")
    
    ic_data = np.load(paths['initial_conditions'])
    
    ics = {
        'train_ICs': ic_data['train_ICs'],
        'test_ICs': ic_data['test_ICs'],
        'train_ICs_reduced': ic_data['train_ICs_reduced'],
        'test_ICs_reduced': ic_data['test_ICs_reduced'],
    }
    
    logger.info(f"  Train ICs reduced shape: {ics['train_ICs_reduced'].shape}")
    logger.info(f"  Test ICs reduced shape: {ics['test_ICs_reduced'].shape}")
    
    return ics


def load_boundaries(paths: dict, logger) -> dict:
    """
    Load data boundaries from Step 1.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Boundaries dictionary.
    """
    logger.info(f"Loading boundaries from {paths['boundaries']}")
    
    bounds_data = np.load(paths['boundaries'])
    
    bounds = {
        'train_boundaries': bounds_data['train_boundaries'],
        'test_boundaries': bounds_data['test_boundaries'],
        'n_spatial': int(bounds_data['n_spatial']),
    }
    
    logger.info(f"  Train boundaries: {bounds['train_boundaries']}")
    logger.info(f"  Test boundaries: {bounds['test_boundaries']}")
    
    return bounds


# =============================================================================
# FORECASTING
# =============================================================================

def compute_dmd_amplitudes_from_ic(
    x0_reduced: np.ndarray,
    modes_reduced: np.ndarray,
) -> np.ndarray:
    """
    Compute DMD amplitudes from initial condition.
    
    Solves: x0 = W @ b  =>  b = W^+ @ x0
    
    Parameters
    ----------
    x0_reduced : np.ndarray, shape (r,)
        Initial condition in reduced space.
    modes_reduced : np.ndarray, shape (r, r)
        Reduced DMD modes.
    
    Returns
    -------
    np.ndarray, shape (r,)
        DMD amplitudes for this initial condition.
    """
    # Solve least squares: modes_reduced @ b = x0
    # Using pseudoinverse: b = pinv(modes_reduced) @ x0
    b = np.linalg.lstsq(modes_reduced, x0_reduced, rcond=None)[0]
    return b


def forecast_trajectory(
    dmd_model: dict,
    x0_reduced: np.ndarray,
    n_steps: int,
    dt: float,
    pod_basis: dict = None,
    k0: float = 0.15,
    c1: float = 1.0,
    logger = None,
) -> dict:
    """
    Forecast a single trajectory using DMD.
    
    Parameters
    ----------
    dmd_model : dict
        DMD model dictionary.
    x0_reduced : np.ndarray, shape (r,)
        Initial condition in reduced space.
    n_steps : int
        Number of time steps.
    dt : float
        Time step.
    pod_basis : dict, optional
        POD basis for full-state reconstruction and Gamma computation.
    k0 : float, optional
        Wavenumber for grid spacing (default 0.15).
    c1 : float, optional
        Adiabaticity parameter (default 1.0).
    logger : logging.Logger, optional
        Logger instance.
    
    Returns
    -------
    dict
        Forecast results with X_hat, Gamma_n, Gamma_c.
    """
    eigs = dmd_model['eigs']
    modes_reduced = dmd_model['modes_reduced']
    
    # Compute amplitudes from this IC
    # We need to re-compute amplitudes for the new IC
    # b = pinv(W) @ x0
    r = len(eigs)
    
    # Truncate IC to match DMD rank
    x0_r = x0_reduced[:r]
    
    # Compute amplitudes
    amplitudes = compute_dmd_amplitudes_from_ic(x0_r, modes_reduced)
    
    # Create time vector (starting from 0)
    t = np.arange(n_steps) * dt
    
    # Forecast in reduced space
    X_hat = dmd_forecast_reduced(eigs, modes_reduced, amplitudes, t)
    
    # Take real part (DMD can produce complex values)
    X_hat_real = np.real(X_hat)  # (r, n_steps)
    
    # Check for numerical issues
    is_nan = np.any(np.isnan(X_hat_real)) or np.any(np.isinf(X_hat_real))
    
    if is_nan:
        if logger:
            logger.warning("  NaN/Inf detected in forecast")
        return {'is_nan': True}
    
    result = {
        'is_nan': False,
        'X_hat': X_hat_real,  # (r, n_steps)
    }
    
    # Compute Gamma from physics if POD basis available
    if pod_basis is not None:
        # Reconstruct full state
        Q_full = reconstruct_full_state(
            X_hat=X_hat_real,
            pod_basis=pod_basis['U_r'],
            temporal_mean=pod_basis.get('mean', None),
        )  # (n_spatial, n_steps)
        
        # Compute Gamma from reconstructed state
        Gamma_n, Gamma_c = compute_gamma_from_state(
            Q=Q_full,
            n_fields=2,  # density and phi
            n_y=pod_basis['n_y'],
            n_x=pod_basis['n_x'],
            k0=k0,
            c1=c1,
        )
        
        result['Gamma_n'] = Gamma_n
        result['Gamma_c'] = Gamma_c
    
    return result


def compute_all_forecasts(
    dmd_model: dict,
    ics_reduced: np.ndarray,
    boundaries: np.ndarray,
    dt: float,
    pod_basis: dict = None,
    k0: float = 0.15,
    c1: float = 1.0,
    logger = None,
    dataset_name: str = "trajectory",
) -> dict:
    """
    Compute forecasts for all trajectories.
    
    Parameters
    ----------
    dmd_model : dict
        DMD model.
    ics_reduced : np.ndarray, shape (n_traj, r)
        Initial conditions in reduced space.
    boundaries : np.ndarray
        Trajectory boundaries.
    dt : float
        Time step.
    pod_basis : dict, optional
        POD basis for full-state reconstruction.
    k0 : float, optional
        Wavenumber for grid spacing.
    c1 : float, optional
        Adiabaticity parameter.
    logger : logging.Logger, optional
        Logger instance.
    dataset_name : str
        Name for logging.
    
    Returns
    -------
    dict
        Forecasts organized by trajectory.
    """
    n_traj = len(boundaries) - 1
    
    forecasts = {
        'X_hat': [],
        'Gamma_n': [],
        'Gamma_c': [],
        'is_nan': [],
    }
    
    if logger:
        logger.info(f"Computing forecasts for {n_traj} {dataset_name}(s)...")
    
    for traj_idx in range(n_traj):
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        x0 = ics_reduced[traj_idx, :]
        
        if logger:
            logger.info(f"  Trajectory {traj_idx + 1}/{n_traj} ({traj_length} steps)")
        
        result = forecast_trajectory(
            dmd_model=dmd_model,
            x0_reduced=x0,
            n_steps=traj_length,
            dt=dt,
            pod_basis=pod_basis,
            k0=k0,
            c1=c1,
            logger=logger,
        )
        
        forecasts['is_nan'].append(result['is_nan'])
        
        if result['is_nan']:
            forecasts['X_hat'].append(None)
            forecasts['Gamma_n'].append(None)
            forecasts['Gamma_c'].append(None)
        else:
            forecasts['X_hat'].append(result['X_hat'])
            if 'Gamma_n' in result:
                forecasts['Gamma_n'].append(result['Gamma_n'])
                forecasts['Gamma_c'].append(result['Gamma_c'])
            else:
                forecasts['Gamma_n'].append(None)
                forecasts['Gamma_c'].append(None)
    
    return forecasts


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    forecasts: dict,
    reference_files: list,
    boundaries: np.ndarray,
    engine: str,
    logger,
    dataset_name: str = "trajectory",
    ref_offset: int = 0,
) -> dict:
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    forecasts : dict
        DMD forecasts.
    reference_files : list
        Reference data file paths.
    boundaries : np.ndarray
        Trajectory boundaries.
    engine : str
        HDF5 engine.
    logger : logging.Logger
        Logger instance.
    dataset_name : str
        Name for logging.
    ref_offset : int
        Offset into reference file (for temporal_split mode where test
        data starts at a later index in the same file).
    
    Returns
    -------
    dict
        Metrics dictionary.
    """
    logger.info(f"Computing metrics for {dataset_name} data...")
    
    metrics = {
        'trajectories': [],
        'summary': {},
    }
    
    n_traj = len(boundaries) - 1
    
    all_mean_errors_n = []
    all_std_errors_n = []
    all_mean_errors_c = []
    all_std_errors_c = []
    
    for traj_idx in range(n_traj):
        if forecasts['is_nan'][traj_idx]:
            logger.warning(f"  Trajectory {traj_idx + 1}: NaN forecast, skipping metrics")
            metrics['trajectories'].append({'is_nan': True})
            continue
        
        # Load reference data
        fh = loader(reference_files[traj_idx], engine=engine)
        
        # Get trajectory length and apply offset
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        ref_Gamma_n = fh["gamma_n"].data[ref_offset:ref_offset + traj_length]
        ref_Gamma_c = fh["gamma_c"].data[ref_offset:ref_offset + traj_length]
        
        # Get predictions
        pred_Gamma_n = forecasts['Gamma_n'][traj_idx]
        pred_Gamma_c = forecasts['Gamma_c'][traj_idx]
        
        if pred_Gamma_n is None:
            logger.warning(f"  Trajectory {traj_idx + 1}: No Gamma predictions")
            metrics['trajectories'].append({'has_gamma': False})
            continue
        
        # Reference statistics
        ref_mean_n = np.mean(ref_Gamma_n)
        ref_std_n = np.std(ref_Gamma_n, ddof=1)
        ref_mean_c = np.mean(ref_Gamma_c)
        ref_std_c = np.std(ref_Gamma_c, ddof=1)
        
        # Prediction statistics
        pred_mean_n = np.mean(pred_Gamma_n)
        pred_std_n = np.std(pred_Gamma_n, ddof=1)
        pred_mean_c = np.mean(pred_Gamma_c)
        pred_std_c = np.std(pred_Gamma_c, ddof=1)
        
        # Relative errors
        err_mean_n = np.abs(ref_mean_n - pred_mean_n) / np.abs(ref_mean_n)
        err_std_n = np.abs(ref_std_n - pred_std_n) / ref_std_n
        err_mean_c = np.abs(ref_mean_c - pred_mean_c) / np.abs(ref_mean_c)
        err_std_c = np.abs(ref_std_c - pred_std_c) / ref_std_c
        
        all_mean_errors_n.append(err_mean_n)
        all_std_errors_n.append(err_std_n)
        all_mean_errors_c.append(err_mean_c)
        all_std_errors_c.append(err_std_c)
        
        # RMSE
        rmse_n = np.sqrt(np.mean((pred_Gamma_n - ref_Gamma_n)**2))
        rmse_c = np.sqrt(np.mean((pred_Gamma_c - ref_Gamma_c)**2))
        
        # Normalized RMSE
        nrmse_n = rmse_n / ref_std_n
        nrmse_c = rmse_c / ref_std_c
        
        traj_metrics = {
            'is_nan': False,
            'has_gamma': True,
            'trajectory': traj_idx,
            'n_steps': traj_length,
            'ref_mean_Gamma_n': float(ref_mean_n),
            'ref_std_Gamma_n': float(ref_std_n),
            'pred_mean_Gamma_n': float(pred_mean_n),
            'pred_std_Gamma_n': float(pred_std_n),
            'err_mean_Gamma_n': float(err_mean_n),
            'err_std_Gamma_n': float(err_std_n),
            'rmse_Gamma_n': float(rmse_n),
            'nrmse_Gamma_n': float(nrmse_n),
            'ref_mean_Gamma_c': float(ref_mean_c),
            'ref_std_Gamma_c': float(ref_std_c),
            'pred_mean_Gamma_c': float(pred_mean_c),
            'pred_std_Gamma_c': float(pred_std_c),
            'err_mean_Gamma_c': float(err_mean_c),
            'err_std_Gamma_c': float(err_std_c),
            'rmse_Gamma_c': float(rmse_c),
            'nrmse_Gamma_c': float(nrmse_c),
        }
        
        metrics['trajectories'].append(traj_metrics)
        
        logger.info(f"  Trajectory {traj_idx + 1}:")
        logger.info(f"    Γn: mean_err={err_mean_n:.4f}, std_err={err_std_n:.4f}, RMSE={rmse_n:.4e}")
        logger.info(f"    Γc: mean_err={err_mean_c:.4f}, std_err={err_std_c:.4f}, RMSE={rmse_c:.4e}")
    
    # Summary statistics
    if all_mean_errors_n:
        metrics['summary'] = {
            'mean_err_Gamma_n': float(np.mean(all_mean_errors_n)),
            'std_err_Gamma_n': float(np.mean(all_std_errors_n)),
            'mean_err_Gamma_c': float(np.mean(all_mean_errors_c)),
            'std_err_Gamma_c': float(np.mean(all_std_errors_c)),
            'n_valid_trajectories': len(all_mean_errors_n),
        }
        
        logger.info(f"  {dataset_name.capitalize()} Summary:")
        logger.info(f"    Avg Γn mean_err: {metrics['summary']['mean_err_Gamma_n']:.4f}")
        logger.info(f"    Avg Γc mean_err: {metrics['summary']['mean_err_Gamma_c']:.4f}")
    
    return metrics


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(
    forecasts: dict,
    reference_files: list,
    boundaries: np.ndarray,
    cfg: DMDConfig,
    output_dir: str,
    logger,
    prefix: str = "",
    ref_offset: int = 0,
):
    """
    Generate diagnostic plots.
    
    Parameters
    ----------
    forecasts : dict
        DMD forecasts.
    reference_files : list
        Reference data files.
    boundaries : np.ndarray
        Trajectory boundaries.
    cfg : DMDConfig
        Configuration.
    output_dir : str
        Output directory for figures.
    logger : logging.Logger
        Logger instance.
    prefix : str
        Prefix for filenames (e.g., "train_" or "test_").
    ref_offset : int
        Offset into reference file (for temporal_split mode).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating plots in {output_dir}")
    
    n_traj = len(boundaries) - 1
    
    for traj_idx in range(n_traj):
        if forecasts['is_nan'][traj_idx]:
            continue
        
        # Load reference
        fh = loader(reference_files[traj_idx], engine=cfg.engine)
        
        # Get trajectory length and apply offset
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        ref_Gamma_n = fh["gamma_n"].data[ref_offset:ref_offset + traj_length]
        ref_Gamma_c = fh["gamma_c"].data[ref_offset:ref_offset + traj_length]
        
        # Get predictions
        pred_Gamma_n = forecasts['Gamma_n'][traj_idx]
        pred_Gamma_c = forecasts['Gamma_c'][traj_idx]
        
        if pred_Gamma_n is None:
            continue
        
        time = np.arange(traj_length) * cfg.dt
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gamma_n
        ax = axes[0]
        ax.plot(time, ref_Gamma_n, 'k-', label='Reference', linewidth=1)
        ax.plot(time, pred_Gamma_n, 'b--', label='DMD Prediction', linewidth=1)
        ax.set_ylabel(r'$\Gamma_n$')
        ax.legend(loc='upper right')
        ax.set_title(f'{prefix.replace("_", " ").title()}Trajectory {traj_idx + 1}: Particle Flux')
        ax.grid(True, alpha=0.3)
        
        # Gamma_c
        ax = axes[1]
        ax.plot(time, ref_Gamma_c, 'k-', label='Reference', linewidth=1)
        ax.plot(time, pred_Gamma_c, 'r--', label='DMD Prediction', linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\Gamma_c$')
        ax.legend(loc='upper right')
        ax.set_title(f'{prefix.replace("_", " ").title()}Trajectory {traj_idx + 1}: Conductive Flux')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'{prefix}traj_{traj_idx + 1}_gamma.png')
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        
        logger.info(f"  Saved {fig_path}")


def plot_eigenvalue_spectrum(dmd_model: dict, output_dir: str, logger):
    """
    Plot DMD eigenvalue spectrum.
    
    Parameters
    ----------
    dmd_model : dict
        DMD model.
    output_dir : str
        Output directory.
    logger : logging.Logger
        Logger instance.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    eigs = dmd_model['eigs']
    dt = dmd_model['dt']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Continuous-time eigenvalues
    ax = axes[0]
    ax.scatter(eigs.real, eigs.imag, c='blue', alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Real(λ)')
    ax.set_ylabel('Imag(λ)')
    ax.set_title('DMD Eigenvalues (Continuous-time)')
    ax.grid(True, alpha=0.3)
    
    # Discrete-time eigenvalues (on unit circle)
    ax = axes[1]
    discrete_eigs = np.exp(eigs * dt)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    ax.scatter(discrete_eigs.real, discrete_eigs.imag, c='red', alpha=0.7, 
               edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Real(μ)')
    ax.set_ylabel('Imag(μ)')
    ax.set_title('DMD Eigenvalues (Discrete-time)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'dmd_eigenvalues.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    
    logger.info(f"  Saved eigenvalue plot to {fig_path}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_predictions(
    train_forecasts: dict,
    test_forecasts: dict,
    paths: dict,
    logger,
):
    """
    Save forecast predictions.
    
    Parameters
    ----------
    train_forecasts : dict
        Training forecasts.
    test_forecasts : dict
        Test forecasts.
    paths : dict
        Output paths.
    logger : logging.Logger
        Logger instance.
    """
    # Create forecasts directory
    os.makedirs(paths['dmd_forecasts_dir'], exist_ok=True)
    
    # Save individual trajectory forecasts
    for traj_idx, X_hat in enumerate(train_forecasts['X_hat']):
        if X_hat is not None:
            np.save(
                os.path.join(paths['dmd_forecasts_dir'], f'train_traj_{traj_idx}_Xhat.npy'),
                X_hat
            )
            if train_forecasts['Gamma_n'][traj_idx] is not None:
                np.savez(
                    os.path.join(paths['dmd_forecasts_dir'], f'train_traj_{traj_idx}_gamma.npz'),
                    Gamma_n=train_forecasts['Gamma_n'][traj_idx],
                    Gamma_c=train_forecasts['Gamma_c'][traj_idx],
                )
    
    for traj_idx, X_hat in enumerate(test_forecasts['X_hat']):
        if X_hat is not None:
            np.save(
                os.path.join(paths['dmd_forecasts_dir'], f'test_traj_{traj_idx}_Xhat.npy'),
                X_hat
            )
            if test_forecasts['Gamma_n'][traj_idx] is not None:
                np.savez(
                    os.path.join(paths['dmd_forecasts_dir'], f'test_traj_{traj_idx}_gamma.npz'),
                    Gamma_n=test_forecasts['Gamma_n'][traj_idx],
                    Gamma_c=test_forecasts['Gamma_c'][traj_idx],
                )
    
    logger.info(f"Saved forecasts to {paths['dmd_forecasts_dir']}")


def save_metrics(
    train_metrics: dict,
    test_metrics: dict,
    paths: dict,
    logger,
):
    """
    Save evaluation metrics to YAML.
    
    Parameters
    ----------
    train_metrics : dict
        Training metrics.
    test_metrics : dict
        Test metrics.
    paths : dict
        Output paths.
    logger : logging.Logger
        Logger instance.
    """
    metrics = {
        'training': train_metrics,
        'test': test_metrics,
    }
    
    with open(paths['dmd_metrics'], 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved metrics to {paths['dmd_metrics']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 3: DMD Evaluation."""
    parser = argparse.ArgumentParser(
        description="Step 3: DMD Evaluation and Prediction"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory from previous steps"
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_dmd_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("step_3_dmd", args.run_dir, cfg.log_level)
    
    print_header("STEP 3: DMD EVALUATION AND PREDICTION")
    print(f"  Run directory: {args.run_dir}")
    
    # Check previous steps
    if not check_step_completed(args.run_dir, "step_2_dmd"):
        logger.error("Step 2 (DMD) has not completed. Run step_2_fit_dmd.py first.")
        return
    
    save_step_status(args.run_dir, "step_3_dmd", "running")
    
    # Save configuration with step-specific name
    save_config(cfg, args.run_dir, step_name="step_3_dmd")
    logger.info("Configuration saved to run directory")
    
    paths = get_dmd_output_paths(args.run_dir)
    
    start_time = time.time()
    
    try:
        # 1. Load DMD model
        dmd_model = load_dmd_model(paths, logger)
        
        # 2. Load POD basis for full-state reconstruction
        pod_basis = load_pod_basis(paths, logger)
        
        # 3. Load initial conditions and boundaries
        ics = load_initial_conditions(paths, logger)
        bounds = load_boundaries(paths, logger)
        
        # 4. Compute training forecasts
        logger.info("\n" + "="*50)
        logger.info("TRAINING TRAJECTORY FORECASTS")
        logger.info("="*50)
        
        train_forecasts = compute_all_forecasts(
            dmd_model=dmd_model,
            ics_reduced=ics['train_ICs_reduced'],
            boundaries=bounds['train_boundaries'],
            dt=cfg.dt,
            pod_basis=pod_basis,
            k0=cfg.k0,
            c1=cfg.c1,
            logger=logger,
            dataset_name="training trajectory",
        )
        
        # 5. Compute test forecasts
        logger.info("\n" + "="*50)
        logger.info("TEST TRAJECTORY FORECASTS")
        logger.info("="*50)
        
        test_forecasts = compute_all_forecasts(
            dmd_model=dmd_model,
            ics_reduced=ics['test_ICs_reduced'],
            boundaries=bounds['test_boundaries'],
            dt=cfg.dt,
            pod_basis=pod_basis,
            k0=cfg.k0,
            c1=cfg.c1,
            logger=logger,
            dataset_name="test trajectory",
        )
        
        # 5. Compute metrics
        logger.info("\n" + "="*50)
        logger.info("COMPUTING METRICS")
        logger.info("="*50)
        
        # Determine reference files and offsets for metrics
        if cfg.training_mode == "temporal_split":
            train_ref_files = cfg.training_files
            test_ref_files = cfg.training_files  # Same file, different portion
            train_ref_offset = cfg.train_start
            test_ref_offset = cfg.test_start
        else:
            train_ref_files = cfg.training_files
            test_ref_files = cfg.test_files
            train_ref_offset = 0
            test_ref_offset = 0
        
        train_metrics = compute_metrics(
            forecasts=train_forecasts,
            reference_files=train_ref_files,
            boundaries=bounds['train_boundaries'],
            engine=cfg.engine,
            logger=logger,
            dataset_name="training",
            ref_offset=train_ref_offset,
        )
        
        test_metrics = compute_metrics(
            forecasts=test_forecasts,
            reference_files=test_ref_files,
            boundaries=bounds['test_boundaries'],
            engine=cfg.engine,
            logger=logger,
            dataset_name="test",
            ref_offset=test_ref_offset,
        )
        
        # 6. Save predictions
        if cfg.save_predictions:
            save_predictions(train_forecasts, test_forecasts, paths, logger)
        
        # 7. Save metrics
        save_metrics(train_metrics, test_metrics, paths, logger)
        
        # 8. Generate plots
        if cfg.generate_plots:
            logger.info("\n" + "="*50)
            logger.info("GENERATING PLOTS")
            logger.info("="*50)
            
            figures_dir = paths['figures_dir']
            os.makedirs(figures_dir, exist_ok=True)
            
            # Eigenvalue spectrum
            plot_eigenvalue_spectrum(dmd_model, figures_dir, logger)
            
            # Training trajectory plots
            generate_plots(
                forecasts=train_forecasts,
                reference_files=train_ref_files,
                boundaries=bounds['train_boundaries'],
                cfg=cfg,
                output_dir=figures_dir,
                logger=logger,
                prefix="train_",
                ref_offset=train_ref_offset,
            )
            
            # Test trajectory plots
            generate_plots(
                forecasts=test_forecasts,
                reference_files=test_ref_files,
                boundaries=bounds['test_boundaries'],
                cfg=cfg,
                output_dir=figures_dir,
                logger=logger,
                prefix="test_",
                ref_offset=test_ref_offset,
            )
        
        # Final timing
        total_time = time.time() - start_time
        
        save_step_status(args.run_dir, "step_3_dmd", "completed", {
            "n_train_trajectories": len(bounds['train_boundaries']) - 1,
            "n_test_trajectories": len(bounds['test_boundaries']) - 1,
            "total_time_seconds": total_time,
        })
        
        # Print summary
        print_header("STEP 3 (DMD) COMPLETE")
        print(f"  Output directory: {args.run_dir}")
        print(f"  Total runtime: {total_time:.1f} seconds")
        
        if test_metrics.get('summary'):
            print(f"\n  TEST SUMMARY:")
            print(f"    Avg Γn mean error: {test_metrics['summary']['mean_err_Gamma_n']:.4f}")
            print(f"    Avg Γc mean error: {test_metrics['summary']['mean_err_Gamma_c']:.4f}")
        
        logger.info(f"Step 3 (DMD) completed successfully in {total_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Step 3 (DMD) failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_3_dmd", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
