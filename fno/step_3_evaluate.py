"""
Step 3: FNO Evaluation and Prediction.

This script orchestrates:
1. Loading trained FNO model from Step 2
2. Computing full autoregressive rollout predictions
3. Computing physics-based Gamma quantities from predicted states
4. Computing evaluation metrics
5. Generating diagnostic plots (consistent with OpInf and DMD)

Usage:
    python step_3_evaluate.py --config config/fno_temporal_split.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml
import h5py
import torch
import torch.nn as nn

# Reuse functions from step_1 and step_2
from step_1_train import (
    load_config, get_config_value, create_model, setup_logging,
    load_checkpoint, load_trajectory_slice,
)

# Parent directory for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.physics import compute_gamma_n, compute_gamma_c, get_hw2d_grid_params
from shared.plotting import plot_gamma_timeseries, generate_state_diagnostic_plots


from typing import Optional


# =============================================================================
# Helper Functions
# =============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_step_completed(run_dir: str, step: str) -> bool:
    """Check if a step has completed based on status file."""
    status_path = os.path.join(run_dir, "status.txt")
    if os.path.exists(status_path):
        with open(status_path) as f:
            return f"{step}_completed" in f.read()
    return False


def save_step_status(run_dir: str, step: str, status: str, info: Optional[dict] = None):
    """Save step status to file."""
    status_path = os.path.join(run_dir, "status.txt")
    with open(status_path, 'a') as f:
        f.write(f"{step}_{status}\n")
        if info:
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")


# =============================================================================
# Data Loading
# =============================================================================

def load_reference_data(file_path: str, start: int, end: int, engine: str = "h5py"):
    """
    Load reference data including density, potential, and gamma values.
    
    Returns:
        dict with 'density', 'phi', 'gamma_n', 'gamma_c' as numpy arrays
    """
    with h5py.File(file_path, 'r') as f:
        data = {
            'density': np.array(f['density'][start:end]),  # (T, H, W)
            'phi': np.array(f['phi'][start:end]),          # (T, H, W)
            'gamma_n': np.array(f['gamma_n'][start:end]),  # (T,)
            'gamma_c': np.array(f['gamma_c'][start:end]),  # (T,)
        }
    return data


# =============================================================================
# FNO Prediction
# =============================================================================

@torch.no_grad()
def compute_fno_rollout(
    model: nn.Module,
    initial_state: np.ndarray,
    n_steps: int,
    device: str,
) -> np.ndarray:
    """
    Compute full autoregressive rollout from FNO model.
    
    Parameters
    ----------
    model : nn.Module
        Trained FNO model.
    initial_state : np.ndarray, shape (2, H, W)
        Initial state [density, potential].
    n_steps : int
        Number of prediction steps.
    device : str
        'cuda' or 'cpu'.
    
    Returns
    -------
    predictions : np.ndarray, shape (n_steps, 2, H, W)
        Predicted states at each timestep.
    """
    model.eval()
    
    predictions = []
    x = torch.from_numpy(initial_state[np.newaxis]).float().to(device)
    
    for _ in range(n_steps):
        y_pred = model(x)
        predictions.append(y_pred[0].cpu().numpy())
        x = y_pred
    
    return np.array(predictions)  # (n_steps, 2, H, W)


def compute_gamma_from_predictions(
    predictions: np.ndarray,
    dx: float,
    c1: float = 1.0,
) -> tuple:
    """
    Compute Gamma_n and Gamma_c from FNO predicted states.
    
    Parameters
    ----------
    predictions : np.ndarray, shape (n_steps, 2, H, W)
        Predicted states [density, potential].
    dx : float
        Grid spacing.
    c1 : float
        Adiabaticity parameter.
    
    Returns
    -------
    gamma_n : np.ndarray, shape (n_steps,)
        Particle flux.
    gamma_c : np.ndarray, shape (n_steps,)
        Conductive flux.
    """
    n_steps = len(predictions)
    gamma_n = np.zeros(n_steps)
    gamma_c = np.zeros(n_steps)
    
    for t in range(n_steps):
        density = predictions[t, 0]   # (H, W)
        phi = predictions[t, 1]       # (H, W)
        gamma_n[t] = compute_gamma_n(density, phi, dx)
        gamma_c[t] = compute_gamma_c(density, phi, c1)
    
    return gamma_n, gamma_c


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    pred_gamma_n: np.ndarray,
    pred_gamma_c: np.ndarray,
    ref_gamma_n: np.ndarray,
    ref_gamma_c: np.ndarray,
) -> dict:
    """
    Compute evaluation metrics for gamma predictions.
    
    Returns dictionary with:
    - err_mean_Gamma_n: Relative error in mean of Gamma_n
    - err_std_Gamma_n: Relative error in std of Gamma_n
    - err_mean_Gamma_c: Relative error in mean of Gamma_c
    - err_std_Gamma_c: Relative error in std of Gamma_c
    - mse_Gamma_n: MSE for Gamma_n
    - mse_Gamma_c: MSE for Gamma_c
    """
    # Mean and std
    ref_mean_n, ref_std_n = np.mean(ref_gamma_n), np.std(ref_gamma_n, ddof=1)
    ref_mean_c, ref_std_c = np.mean(ref_gamma_c), np.std(ref_gamma_c, ddof=1)
    pred_mean_n, pred_std_n = np.mean(pred_gamma_n), np.std(pred_gamma_n, ddof=1)
    pred_mean_c, pred_std_c = np.mean(pred_gamma_c), np.std(pred_gamma_c, ddof=1)
    
    # Relative errors
    err_mean_n = np.abs(ref_mean_n - pred_mean_n) / np.abs(ref_mean_n) if ref_mean_n != 0 else 0.0
    err_std_n = np.abs(ref_std_n - pred_std_n) / ref_std_n if ref_std_n > 0 else 0.0
    err_mean_c = np.abs(ref_mean_c - pred_mean_c) / np.abs(ref_mean_c) if ref_mean_c != 0 else 0.0
    err_std_c = np.abs(ref_std_c - pred_std_c) / ref_std_c if ref_std_c > 0 else 0.0
    
    # MSE
    mse_n = np.mean((pred_gamma_n - ref_gamma_n) ** 2)
    mse_c = np.mean((pred_gamma_c - ref_gamma_c) ** 2)
    
    return {
        'err_mean_Gamma_n': float(err_mean_n),
        'err_std_Gamma_n': float(err_std_n),
        'err_mean_Gamma_c': float(err_mean_c),
        'err_std_Gamma_c': float(err_std_c),
        'mse_Gamma_n': float(mse_n),
        'mse_Gamma_c': float(mse_c),
        'ref_mean_Gamma_n': float(ref_mean_n),
        'ref_std_Gamma_n': float(ref_std_n),
        'ref_mean_Gamma_c': float(ref_mean_c),
        'ref_std_Gamma_c': float(ref_std_c),
        'pred_mean_Gamma_n': float(pred_mean_n),
        'pred_std_Gamma_n': float(pred_std_n),
        'pred_mean_Gamma_c': float(pred_mean_c),
        'pred_std_Gamma_c': float(pred_std_c),
    }


def compute_state_metrics(
    predictions: np.ndarray,
    reference: np.ndarray,
) -> dict:
    """
    Compute state-level metrics.
    
    Parameters
    ----------
    predictions : np.ndarray, shape (n_steps, 2, H, W)
    reference : np.ndarray, shape (n_steps, 2, H, W)
    
    Returns
    -------
    dict with MSE, relative L2 error per field and total
    """
    n_steps = len(predictions)
    
    # Total MSE
    mse_total = np.mean((predictions - reference) ** 2)
    
    # Per-field MSE
    mse_density = np.mean((predictions[:, 0] - reference[:, 0]) ** 2)
    mse_phi = np.mean((predictions[:, 1] - reference[:, 1]) ** 2)
    
    # Relative L2 error over time
    pred_flat = predictions.reshape(n_steps, -1)
    ref_flat = reference.reshape(n_steps, -1)
    l2_error = np.linalg.norm(pred_flat - ref_flat, axis=1) / np.linalg.norm(ref_flat, axis=1)
    
    return {
        'mse_total': float(mse_total),
        'mse_density': float(mse_density),
        'mse_potential': float(mse_phi),
        'mean_rel_l2_error': float(np.mean(l2_error)),
        'max_rel_l2_error': float(np.max(l2_error)),
        'final_rel_l2_error': float(l2_error[-1]),
    }


# =============================================================================
# Plotting
# =============================================================================

def generate_gamma_plots(
    pred_gamma_n: np.ndarray,
    pred_gamma_c: np.ndarray,
    ref_gamma_n: np.ndarray,
    ref_gamma_c: np.ndarray,
    dt: float,
    output_path: str,
    logger,
    title_prefix: str = "",
):
    """Generate Gamma comparison plot using shared plotting function."""
    plot_gamma_timeseries(
        pred_n=pred_gamma_n,
        pred_c=pred_gamma_c,
        ref_n=ref_gamma_n,
        ref_c=ref_gamma_c,
        dt=dt,
        output_path=output_path,
        logger=logger,
        title_prefix=title_prefix,
        method_name="FNO"
    )


def generate_state_snapshots(
    predictions: np.ndarray,
    reference: np.ndarray,
    dt: float,
    output_dir: str,
    logger,
    n_snapshots: int = 5,
    prefix: str = "",
):
    """
    Generate state comparison snapshots.
    
    Creates comparison figures for density and potential at selected timesteps.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_steps = len(predictions)
    snapshot_indices = np.linspace(0, n_steps - 1, n_snapshots, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for snap_idx, t_idx in enumerate(snapshot_indices):
        t_val = t_idx * dt
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        field_names = ['Density', 'Potential']
        
        for row, (name, pred_field, ref_field) in enumerate(zip(
            field_names, predictions[t_idx], reference[t_idx]
        )):
            error = pred_field - ref_field
            
            # Reference
            vmin, vmax = ref_field.min(), ref_field.max()
            im0 = axes[row, 0].imshow(ref_field, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{name} - Reference')
            axes[row, 0].axis('off')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
            
            # Prediction
            im1 = axes[row, 1].imshow(pred_field, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{name} - FNO Prediction')
            axes[row, 1].axis('off')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)
            
            # Error
            err_max = max(abs(error.min()), abs(error.max()))
            if err_max == 0:
                err_max = 1.0
            im2 = axes[row, 2].imshow(error, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
            axes[row, 2].set_title(f'{name} - Error (MSE={np.mean(error**2):.2e})')
            axes[row, 2].axis('off')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        
        plt.suptitle(f'{prefix}t = {t_val:.2f} (step {t_idx})', fontsize=14)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'{prefix}snapshot_{snap_idx:02d}_t{t_idx:04d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"  Saved: {fig_path}")
    
    logger.info(f"  Generated {n_snapshots} snapshot figures in {output_dir}")


def generate_rollout_error_plot(
    predictions: np.ndarray,
    reference: np.ndarray,
    dt: float,
    output_path: str,
    logger,
    title_prefix: str = "",
):
    """
    Generate rollout error over time plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_steps = len(predictions)
    t = np.arange(n_steps) * dt
    
    # Compute errors over time
    mse_per_step = np.mean((predictions - reference) ** 2, axis=(1, 2, 3))
    
    # Per-field MSE
    mse_density = np.mean((predictions[:, 0] - reference[:, 0]) ** 2, axis=(1, 2))
    mse_phi = np.mean((predictions[:, 1] - reference[:, 1]) ** 2, axis=(1, 2))
    
    # Relative L2 error
    pred_flat = predictions.reshape(n_steps, -1)
    ref_flat = reference.reshape(n_steps, -1)
    rel_l2 = np.linalg.norm(pred_flat - ref_flat, axis=1) / np.linalg.norm(ref_flat, axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # MSE plot
    ax = axes[0]
    ax.semilogy(t, mse_per_step, 'k-', linewidth=1.5, label='Total MSE')
    ax.semilogy(t, mse_density, 'b--', linewidth=1, label='Density MSE', alpha=0.7)
    ax.semilogy(t, mse_phi, 'r--', linewidth=1, label='Potential MSE', alpha=0.7)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title(f'{title_prefix}FNO Rollout Error', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Relative L2 error
    ax = axes[1]
    ax.semilogy(t, rel_l2, 'b-', linewidth=1.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Relative L2 Error Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.95, f'Mean: {np.mean(rel_l2):.2e}, Max: {np.max(rel_l2):.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved rollout error plot to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for Step 3."""
    parser = argparse.ArgumentParser(description="Step 3: FNO Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    run_dir = args.run_dir
    
    # Setup logging
    logger = setup_logging("step_3", run_dir, cfg.get('log_level', 'INFO'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print_header("STEP 3: FNO EVALUATION")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    
    # Check that step 2 completed
    if not check_step_completed(run_dir, "step2"):
        logger.warning("Step 2 may not have completed. Proceeding anyway...")
    
    save_step_status(run_dir, "step_3", "running")
    t_start = time.time()
    
    try:
        # Get data paths
        data_dir = get_config_value(cfg, 'paths', 'data_dir')
        train_file = get_config_value(cfg, 'paths', 'training_files')[0]
        file_path = os.path.join(data_dir, train_file)
        
        # Get temporal split parameters
        train_start = cfg.get('train_start', 0)
        train_end = cfg.get('train_end', 400)
        test_start = cfg.get('test_start', 400)
        test_end = cfg.get('test_end', 500)
        
        # Get physics parameters
        physics_cfg = cfg.get('physics', {})
        dt = physics_cfg.get('dt', 0.025)
        n_x = physics_cfg.get('n_x', 512)
        n_y = physics_cfg.get('n_y', 512)
        c1 = physics_cfg.get('c1', 1.0)
        k0 = physics_cfg.get('k0', 0.15)
        
        # Compute grid spacing
        grid_params = get_hw2d_grid_params(k0=k0, nx=n_x)
        dx = grid_params['dx']
        
        logger.info(f"Loading data from {file_path}")
        logger.info(f"  Train: [{train_start}, {train_end})")
        logger.info(f"  Test:  [{test_start}, {test_end})")
        logger.info(f"  Physics: dt={dt}, dx={dx:.4f}, c1={c1}, k0={k0}")
        
        # Load reference data
        train_ref = load_reference_data(file_path, train_start, train_end)
        test_ref = load_reference_data(file_path, test_start, test_end)
        
        # Load state data for predictions
        train_states = load_trajectory_slice(file_path, train_start, train_end)
        test_states = load_trajectory_slice(file_path, test_start, test_end)
        
        logger.info(f"  Train states: {train_states.shape}")
        logger.info(f"  Test states: {test_states.shape}")
        
        # Load model (try rollout checkpoint first, then best single-step)
        model = create_model(cfg, device)
        
        rollout_ckpt = os.path.join(run_dir, "checkpoint_rollout_final.pt")
        best_ckpt = os.path.join(run_dir, "checkpoint_best.pt")
        
        if os.path.exists(rollout_ckpt):
            epoch, metrics = load_checkpoint(rollout_ckpt, model, device=device)
            logger.info(f"Loaded rollout checkpoint from epoch {epoch}")
        elif os.path.exists(best_ckpt):
            epoch, metrics = load_checkpoint(best_ckpt, model, device=device)
            logger.info(f"Loaded single-step checkpoint from epoch {epoch}")
        else:
            raise FileNotFoundError("No model checkpoint found!")
        
        # Create output directories
        figures_dir = os.path.join(run_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        results = {}
        
        # =====================================================================
        # TRAINING SET EVALUATION
        # =====================================================================
        logger.info("Evaluating on TRAINING set...")
        
        n_train_steps = len(train_states) - 1  # Predict from first state
        train_predictions = compute_fno_rollout(
            model, train_states[0], n_train_steps, device
        )
        
        # Compute gamma from predictions
        train_pred_gamma_n, train_pred_gamma_c = compute_gamma_from_predictions(
            train_predictions, dx, c1
        )
        
        # Reference gamma (skip first timestep to align with predictions)
        train_ref_gamma_n = train_ref['gamma_n'][1:]
        train_ref_gamma_c = train_ref['gamma_c'][1:]
        
        # Compute metrics
        train_gamma_metrics = compute_metrics(
            train_pred_gamma_n, train_pred_gamma_c,
            train_ref_gamma_n, train_ref_gamma_c
        )
        train_state_metrics = compute_state_metrics(
            train_predictions, train_states[1:]
        )
        
        results['train'] = {
            'gamma_metrics': train_gamma_metrics,
            'state_metrics': train_state_metrics,
            'n_steps': n_train_steps,
        }
        
        logger.info(f"  Train Γn error: mean={train_gamma_metrics['err_mean_Gamma_n']:.4f}, "
                    f"std={train_gamma_metrics['err_std_Gamma_n']:.4f}")
        logger.info(f"  Train state MSE: {train_state_metrics['mse_total']:.6f}")
        
        # Generate training plots
        train_fig_dir = os.path.join(figures_dir, "train")
        os.makedirs(train_fig_dir, exist_ok=True)
        
        generate_gamma_plots(
            train_pred_gamma_n, train_pred_gamma_c,
            train_ref_gamma_n, train_ref_gamma_c,
            dt, os.path.join(train_fig_dir, "gamma_comparison.png"),
            logger, title_prefix="Training: "
        )
        
        generate_rollout_error_plot(
            train_predictions, train_states[1:],
            dt, os.path.join(train_fig_dir, "rollout_error.png"),
            logger, title_prefix="Training: "
        )
        
        generate_state_snapshots(
            train_predictions, train_states[1:],
            dt, train_fig_dir, logger, n_snapshots=5, prefix="train_"
        )
        
        # =====================================================================
        # TEST SET EVALUATION
        # =====================================================================
        logger.info("Evaluating on TEST set...")
        
        n_test_steps = len(test_states) - 1
        test_predictions = compute_fno_rollout(
            model, test_states[0], n_test_steps, device
        )
        
        # Compute gamma from predictions
        test_pred_gamma_n, test_pred_gamma_c = compute_gamma_from_predictions(
            test_predictions, dx, c1
        )
        
        # Reference gamma
        test_ref_gamma_n = test_ref['gamma_n'][1:]
        test_ref_gamma_c = test_ref['gamma_c'][1:]
        
        # Compute metrics
        test_gamma_metrics = compute_metrics(
            test_pred_gamma_n, test_pred_gamma_c,
            test_ref_gamma_n, test_ref_gamma_c
        )
        test_state_metrics = compute_state_metrics(
            test_predictions, test_states[1:]
        )
        
        results['test'] = {
            'gamma_metrics': test_gamma_metrics,
            'state_metrics': test_state_metrics,
            'n_steps': n_test_steps,
        }
        
        logger.info(f"  Test Γn error: mean={test_gamma_metrics['err_mean_Gamma_n']:.4f}, "
                    f"std={test_gamma_metrics['err_std_Gamma_n']:.4f}")
        logger.info(f"  Test state MSE: {test_state_metrics['mse_total']:.6f}")
        
        # Generate test plots
        test_fig_dir = os.path.join(figures_dir, "test")
        os.makedirs(test_fig_dir, exist_ok=True)
        
        generate_gamma_plots(
            test_pred_gamma_n, test_pred_gamma_c,
            test_ref_gamma_n, test_ref_gamma_c,
            dt, os.path.join(test_fig_dir, "gamma_comparison.png"),
            logger, title_prefix="Test: "
        )
        
        generate_rollout_error_plot(
            test_predictions, test_states[1:],
            dt, os.path.join(test_fig_dir, "rollout_error.png"),
            logger, title_prefix="Test: "
        )
        
        generate_state_snapshots(
            test_predictions, test_states[1:],
            dt, test_fig_dir, logger, n_snapshots=5, prefix="test_"
        )
        
        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        
        # Save metrics to YAML
        metrics_path = os.path.join(run_dir, "metrics.yaml")
        with open(metrics_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save predictions
        predictions_path = os.path.join(run_dir, "predictions.npz")
        np.savez(
            predictions_path,
            train_Gamma_n=train_pred_gamma_n,
            train_Gamma_c=train_pred_gamma_c,
            train_ref_Gamma_n=train_ref_gamma_n,
            train_ref_Gamma_c=train_ref_gamma_c,
            test_Gamma_n=test_pred_gamma_n,
            test_Gamma_c=test_pred_gamma_c,
            test_ref_Gamma_n=test_ref_gamma_n,
            test_ref_Gamma_c=test_ref_gamma_c,
        )
        logger.info(f"Saved predictions to {predictions_path}")
        
        # Save full state predictions if not too large
        state_size_mb = (train_predictions.nbytes + test_predictions.nbytes) / 1e6
        if state_size_mb < 500:  # Only save if < 500MB
            state_pred_path = os.path.join(run_dir, "state_predictions.npz")
            np.savez_compressed(
                state_pred_path,
                train_predictions=train_predictions,
                test_predictions=test_predictions,
            )
            logger.info(f"Saved state predictions to {state_pred_path}")
        else:
            logger.info(f"Skipping state predictions save (size: {state_size_mb:.1f}MB)")
        
        t_elapsed = time.time() - t_start
        
        save_step_status(run_dir, "step_3", "completed", {"time_seconds": t_elapsed})
        
        # Print summary
        logger.info("=" * 60)
        logger.info("  STEP 3 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Runtime: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
        logger.info("TEST SUMMARY:")
        logger.info(f"  Γn mean error: {test_gamma_metrics['err_mean_Gamma_n']:.4f}")
        logger.info(f"  Γn std error:  {test_gamma_metrics['err_std_Gamma_n']:.4f}")
        logger.info(f"  Γc mean error: {test_gamma_metrics['err_mean_Gamma_c']:.4f}")
        logger.info(f"  Γc std error:  {test_gamma_metrics['err_std_Gamma_c']:.4f}")
        logger.info(f"  State MSE:     {test_state_metrics['mse_total']:.6f}")
        logger.info(f"  Rel L2 (mean): {test_state_metrics['mean_rel_l2_error']:.4f}")
        logger.info(f"Results saved to: {run_dir}")
        
    except Exception as e:
        logger.error(f"Step 3 failed: {e}", exc_info=True)
        save_step_status(run_dir, "step_3", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
