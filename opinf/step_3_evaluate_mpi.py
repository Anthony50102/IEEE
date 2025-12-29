"""
Step 3: Evaluation and Prediction (MPI Parallel Version).

This script handles:
1. Loading trained ensemble models
2. Computing ensemble predictions on training and test trajectories (distributed)
3. Computing evaluation metrics
4. Generating diagnostic plots (optional)
5. Saving results

Parallelization strategy:
- Trajectories are distributed across MPI ranks
- Each rank processes its assigned trajectories independently
- Results are gathered to rank 0 for saving and summary

Memory optimization:
- X_OpInf storage is optional (--save-reduced-states flag)
- Without X_OpInf: ~435 MB base + ~13 MB per trajectory
- With X_OpInf: ~435 MB base + ~650 MB per trajectory (r=100, 100 models, 8000 steps)

Usage:
    mpirun -np 4 python step_3_evaluate_mpi.py --config config.yaml --run-dir /path/to/run
    mpirun -np 4 python step_3_evaluate_mpi.py --config config.yaml --run-dir /path/to/run --save-reduced-states

Author: Anthony Poole
"""

import argparse
import os
import time
import gc
import numpy as np
import yaml

from mpi4py import MPI

from utils import (
    load_config,
    save_config,
    setup_logging,
    save_step_status,
    check_step_completed,
    get_output_paths,
    print_header,
    get_x_sq,
    loader,
    solve_opinf_difference_model,
    PipelineConfig,
)


# =============================================================================
# MPI UTILITIES
# =============================================================================

def distribute_trajectories(n_traj: int, rank: int, size: int) -> list:
    """
    Distribute trajectories across MPI ranks.
    
    Parameters
    ----------
    n_traj : int
        Total number of trajectories.
    rank : int
        MPI rank.
    size : int
        Number of MPI ranks.
    
    Returns
    -------
    list
        List of trajectory indices assigned to this rank.
    """
    # Round-robin distribution for better load balancing
    return [i for i in range(n_traj) if i % size == rank]


class DummyLogger:
    """Dummy logger for non-root ranks."""
    def info(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


# =============================================================================
# PREPROCESSING VERIFICATION
# =============================================================================

def load_preprocessing_info(filepath: str, logger) -> dict:
    """
    Load preprocessing information from Step 1.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Preprocessing info file not found: {filepath}")
        return {
            'centering_applied': True,
            'scaling_applied': False,
            'scaling_factors': None,
            'r_actual': None,
        }
    
    data = np.load(filepath, allow_pickle=True)
    info = {
        'centering_applied': bool(data['centering_applied']),
        'scaling_applied': bool(data.get('scaling_applied', False)),
        'r_actual': int(data['r_actual']),
        'r_config': int(data['r_config']),
        'r_from_energy': int(data['r_from_energy']),
        'n_spatial': int(data['n_spatial']),
        'n_fields': int(data['n_fields']),
        'n_x': int(data['n_x']),
        'n_y': int(data['n_y']),
        'dt': float(data['dt']),
    }
    
    # Load scaling factors if present
    if 'scaling_factors' in data:
        info['scaling_factors'] = data['scaling_factors']
    else:
        info['scaling_factors'] = None
    
    logger.info("Preprocessing info:")
    logger.info(f"  Centering applied: {info['centering_applied']}")
    logger.info(f"  Scaling applied: {info['scaling_applied']}")
    if info['scaling_factors'] is not None:
        logger.info(f"  Scaling factors: {info['scaling_factors']}")
    logger.info(f"  POD modes (r): {info['r_actual']}")
    
    return info


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_file(filepath: str) -> dict:
    """Load a single model from an individual NPZ file."""
    data = np.load(filepath, allow_pickle=True)
    model = {
        'A': data['A'],
        'F': data['F'],
        'C': data['C'],
        'G': data['G'],
        'c': data['c'],
        'total_error': float(data['total_error']),
        'mean_err_Gamma_n': float(data['mean_err_Gamma_n']),
        'std_err_Gamma_n': float(data['std_err_Gamma_n']),
        'mean_err_Gamma_c': float(data['mean_err_Gamma_c']),
        'std_err_Gamma_c': float(data['std_err_Gamma_c']),
        'alpha_state_lin': float(data['alpha_state_lin']),
        'alpha_state_quad': float(data['alpha_state_quad']),
        'alpha_out_lin': float(data['alpha_out_lin']),
        'alpha_out_quad': float(data['alpha_out_quad']),
    }
    return model


def load_ensemble_from_directory(operators_dir: str, logger) -> list:
    """Load ensemble models from individual files in a directory."""
    if not os.path.exists(operators_dir):
        logger.warning(f"Operators directory not found: {operators_dir}")
        return []
    
    model_files = sorted([
        f for f in os.listdir(operators_dir) 
        if f.startswith('model_') and f.endswith('.npz')
    ])
    
    if not model_files:
        logger.warning(f"No model files found in {operators_dir}")
        return []
    
    logger.info(f"Loading {len(model_files)} models from {operators_dir}")
    
    models = []
    for filename in model_files:
        filepath = os.path.join(operators_dir, filename)
        try:
            model = load_model_from_file(filepath)
            models.append((model['total_error'], model))
        except Exception as e:
            logger.warning(f"  Failed to load {filename}: {e}")
    
    models.sort(key=lambda x: x[0])
    logger.info(f"  Loaded {len(models)} models successfully")
    return models


def load_ensemble(filepath: str, logger, operators_dir: str = None) -> list:
    """Load ensemble models from NPZ file or directory."""
    if operators_dir and os.path.exists(operators_dir):
        models = load_ensemble_from_directory(operators_dir, logger)
        if models:
            return models
        logger.info("  Falling back to single ensemble file...")
    
    logger.info(f"Loading ensemble from {filepath}")
    data = np.load(filepath, allow_pickle=True)
    num_models = int(data['num_models'])
    
    models = []
    for i in range(num_models):
        prefix = f'model_{i}_'
        model = {
            'A': data[prefix + 'A'],
            'F': data[prefix + 'F'],
            'C': data[prefix + 'C'],
            'G': data[prefix + 'G'],
            'c': data[prefix + 'c'],
            'total_error': float(data[prefix + 'total_error']),
            'mean_err_Gamma_n': float(data[prefix + 'mean_err_Gamma_n']),
            'std_err_Gamma_n': float(data[prefix + 'std_err_Gamma_n']),
            'mean_err_Gamma_c': float(data[prefix + 'mean_err_Gamma_c']),
            'std_err_Gamma_c': float(data[prefix + 'std_err_Gamma_c']),
            'alpha_state_lin': float(data[prefix + 'alpha_state_lin']),
            'alpha_state_quad': float(data[prefix + 'alpha_state_quad']),
            'alpha_out_lin': float(data[prefix + 'alpha_out_lin']),
            'alpha_out_quad': float(data[prefix + 'alpha_out_quad']),
        }
        models.append((model['total_error'], model))
    
    logger.info(f"  Loaded {len(models)} models")
    return models


def estimate_memory_usage(n_models: int, r: int, n_steps: int, n_traj_per_rank: int, 
                          save_reduced: bool, logger) -> None:
    """Estimate and log memory usage per rank."""
    s = r * (r + 1) // 2
    
    # Model memory
    model_bytes = (r * r + r * s + 2 * r + 2 * s + 2) * 8  # A, F, C, G, c
    total_model_mb = (model_bytes * n_models) / 1e6
    
    # Per trajectory
    gamma_mb = (2 * n_models * n_steps * 8) / 1e6
    x_opinf_mb = (n_models * n_steps * r * 8) / 1e6 if save_reduced else 0
    per_traj_mb = gamma_mb + x_opinf_mb
    
    # Total
    total_mb = total_model_mb + per_traj_mb * n_traj_per_rank
    
    logger.info(f"Memory estimate per rank:")
    logger.info(f"  Models ({n_models}): {total_model_mb:.1f} MB")
    logger.info(f"  Per trajectory: {per_traj_mb:.1f} MB (Gamma: {gamma_mb:.1f} MB" + 
                (f", X_OpInf: {x_opinf_mb:.1f} MB)" if save_reduced else ")"))
    logger.info(f"  Trajectories assigned: {n_traj_per_rank}")
    logger.info(f"  Total estimated: {total_mb:.1f} MB")


# =============================================================================
# PREDICTION
# =============================================================================

def predict_trajectory(
    u0: np.ndarray,
    n_steps: int,
    model: dict,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    save_reduced: bool = False,
) -> dict:
    """
    Run prediction for a single trajectory with one model.
    
    Parameters
    ----------
    u0 : np.ndarray
        Initial condition (reduced coordinates).
    n_steps : int
        Number of time steps.
    model : dict
        Model with operators (A, F, C, G, c).
    mean_Xhat : np.ndarray
        Mean for scaling.
    scaling_Xhat : float
        Scale factor.
    save_reduced : bool
        Whether to return X_OpInf (memory intensive).
    
    Returns
    -------
    dict
        Predictions with keys 'Gamma_n', 'Gamma_c', and optionally 'X_OpInf'.
    """
    A = model['A']
    F = model['F']
    C = model['C']
    G = model['G']
    c = model['c']
    
    f = lambda x: np.dot(A, x) + np.dot(F, get_x_sq(x))
    is_nan, Xhat_pred = solve_opinf_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True}
    
    X_OpInf = Xhat_pred.T  # (n_steps, r)
    
    # Apply output operators
    Xhat_scaled = (X_OpInf - mean_Xhat[np.newaxis, :]) / scaling_Xhat
    Xhat_2 = get_x_sq(Xhat_scaled)
    
    Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
    
    result = {
        'is_nan': False,
        'Gamma_n': Y_OpInf[0, :],
        'Gamma_c': Y_OpInf[1, :],
    }
    
    if save_reduced:
        result['X_OpInf'] = X_OpInf
    
    return result


def compute_trajectory_predictions(
    traj_idx: int,
    models: list,
    IC_reduced: np.ndarray,
    traj_length: int,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    save_reduced: bool,
    logger,
) -> dict:
    """
    Compute predictions for a single trajectory with all models.
    
    Returns
    -------
    dict
        Dictionary with Gamma_n, Gamma_c arrays of shape (n_valid_models, n_steps).
    """
    traj_Gamma_n = []
    traj_Gamma_c = []
    traj_X_OpInf = [] if save_reduced else None
    n_nan = 0
    
    for model_idx, (score, model) in enumerate(models):
        result = predict_trajectory(
            IC_reduced, traj_length, model, mean_Xhat, scaling_Xhat, save_reduced
        )
        
        if result['is_nan']:
            n_nan += 1
            continue
        
        traj_Gamma_n.append(result['Gamma_n'])
        traj_Gamma_c.append(result['Gamma_c'])
        if save_reduced:
            traj_X_OpInf.append(result['X_OpInf'])
    
    if n_nan > 0:
        logger.warning(f"    Trajectory {traj_idx + 1}: {n_nan} models produced NaN")
    
    predictions = {
        'traj_idx': traj_idx,
        'Gamma_n': np.array(traj_Gamma_n),
        'Gamma_c': np.array(traj_Gamma_c),
        'n_valid_models': len(traj_Gamma_n),
    }
    
    if save_reduced:
        predictions['X_OpInf'] = np.array(traj_X_OpInf)
    
    return predictions


def compute_distributed_predictions(
    models: list,
    ICs_reduced: np.ndarray,
    boundaries: np.ndarray,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    traj_indices: list,
    save_reduced: bool,
    logger,
    dataset_name: str = "trajectory",
) -> list:
    """
    Compute predictions for trajectories assigned to this rank.
    
    Returns
    -------
    list
        List of prediction dictionaries, one per assigned trajectory.
    """
    logger.info(f"Processing {len(traj_indices)} {dataset_name}(s)...")
    
    local_predictions = []
    
    for local_idx, traj_idx in enumerate(traj_indices):
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        u0 = ICs_reduced[traj_idx, :]
        
        logger.info(f"  Trajectory {traj_idx + 1} ({traj_length} steps) [{local_idx + 1}/{len(traj_indices)}]")
        
        preds = compute_trajectory_predictions(
            traj_idx, models, u0, traj_length,
            mean_Xhat, scaling_Xhat, save_reduced, logger
        )
        
        local_predictions.append(preds)
        
        # Periodic garbage collection
        if (local_idx + 1) % 5 == 0:
            gc.collect()
    
    return local_predictions


# =============================================================================
# METRICS
# =============================================================================

def compute_trajectory_metrics(
    predictions: dict,
    reference_file: str,
    traj_length: int,
    engine: str,
    logger,
) -> dict:
    """Compute metrics for a single trajectory."""
    fh = loader(reference_file, ENGINE=engine)
    ref_Gamma_n = fh["gamma_n"].data[:traj_length]
    ref_Gamma_c = fh["gamma_c"].data[:traj_length]
    
    pred_Gamma_n = predictions['Gamma_n']
    pred_Gamma_c = predictions['Gamma_c']
    
    mean_pred_n = np.mean(pred_Gamma_n, axis=0)
    mean_pred_c = np.mean(pred_Gamma_c, axis=0)
    
    # Reference statistics
    ref_mean_n, ref_std_n = np.mean(ref_Gamma_n), np.std(ref_Gamma_n, ddof=1)
    ref_mean_c, ref_std_c = np.mean(ref_Gamma_c), np.std(ref_Gamma_c, ddof=1)
    
    # Ensemble mean statistics
    ens_mean_n, ens_std_n = np.mean(mean_pred_n), np.std(mean_pred_n, ddof=1)
    ens_mean_c, ens_std_c = np.mean(mean_pred_c), np.std(mean_pred_c, ddof=1)
    
    # Relative errors
    err_mean_n = np.abs(ref_mean_n - ens_mean_n) / np.abs(ref_mean_n)
    err_std_n = np.abs(ref_std_n - ens_std_n) / ref_std_n
    err_mean_c = np.abs(ref_mean_c - ens_mean_c) / np.abs(ref_mean_c)
    err_std_c = np.abs(ref_std_c - ens_std_c) / ref_std_c
    
    # Pointwise RMSE
    rmse_n = np.sqrt(np.mean((mean_pred_n - ref_Gamma_n)**2))
    rmse_c = np.sqrt(np.mean((mean_pred_c - ref_Gamma_c)**2))
    
    return {
        'trajectory': predictions['traj_idx'],
        'n_steps': traj_length,
        'n_valid_models': predictions['n_valid_models'],
        'ref_mean_Gamma_n': float(ref_mean_n),
        'ref_std_Gamma_n': float(ref_std_n),
        'pred_mean_Gamma_n': float(ens_mean_n),
        'pred_std_Gamma_n': float(ens_std_n),
        'err_mean_Gamma_n': float(err_mean_n),
        'err_std_Gamma_n': float(err_std_n),
        'rmse_Gamma_n': float(rmse_n),
        'ref_mean_Gamma_c': float(ref_mean_c),
        'ref_std_Gamma_c': float(ref_std_c),
        'pred_mean_Gamma_c': float(ens_mean_c),
        'pred_std_Gamma_c': float(ens_std_c),
        'err_mean_Gamma_c': float(err_mean_c),
        'err_std_Gamma_c': float(err_std_c),
        'rmse_Gamma_c': float(rmse_c),
    }


def compute_distributed_metrics(
    local_predictions: list,
    reference_files: list,
    boundaries: np.ndarray,
    engine: str,
    logger,
) -> list:
    """Compute metrics for all locally assigned trajectories."""
    local_metrics = []
    
    for preds in local_predictions:
        traj_idx = preds['traj_idx']
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        
        metrics = compute_trajectory_metrics(
            preds, reference_files[traj_idx], traj_length, engine, logger
        )
        local_metrics.append(metrics)
        
        logger.info(f"  Trajectory {traj_idx + 1}: "
                   f"Γn err={metrics['err_mean_Gamma_n']:.4f}, "
                   f"Γc err={metrics['err_mean_Gamma_c']:.4f}")
    
    return local_metrics


# =============================================================================
# PLOTTING (Rank 0 only)
# =============================================================================

def generate_plots(
    all_predictions: dict,
    reference_files: list,
    boundaries: np.ndarray,
    cfg: PipelineConfig,
    output_dir: str,
    logger,
):
    """Generate diagnostic plots (called by rank 0 only)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating plots in {output_dir}")
    
    for traj_idx in sorted(all_predictions.keys()):
        preds = all_predictions[traj_idx]
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        
        fh = loader(reference_files[traj_idx], ENGINE=cfg.engine)
        ref_Gamma_n = fh["gamma_n"].data[:traj_length]
        ref_Gamma_c = fh["gamma_c"].data[:traj_length]
        
        pred_Gamma_n = preds['Gamma_n']
        pred_Gamma_c = preds['Gamma_c']
        
        mean_pred_n = np.mean(pred_Gamma_n, axis=0)
        std_pred_n = np.std(pred_Gamma_n, axis=0)
        mean_pred_c = np.mean(pred_Gamma_c, axis=0)
        std_pred_c = np.std(pred_Gamma_c, axis=0)
        
        time = np.arange(traj_length) * cfg.dt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax = axes[0]
        ax.plot(time, ref_Gamma_n, 'k-', label='Reference', linewidth=1)
        ax.plot(time, mean_pred_n, 'b-', label='Ensemble Mean', linewidth=1)
        ax.fill_between(time, mean_pred_n - 2*std_pred_n, mean_pred_n + 2*std_pred_n,
                       alpha=0.3, color='blue', label='±2σ')
        ax.set_ylabel(r'$\Gamma_n$')
        ax.legend(loc='upper right')
        ax.set_title(f'Trajectory {traj_idx + 1}: Particle Flux')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(time, ref_Gamma_c, 'k-', label='Reference', linewidth=1)
        ax.plot(time, mean_pred_c, 'r-', label='Ensemble Mean', linewidth=1)
        ax.fill_between(time, mean_pred_c - 2*std_pred_c, mean_pred_c + 2*std_pred_c,
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
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 3 (MPI parallel)."""
    parser = argparse.ArgumentParser(
        description="Step 3: Evaluation and Prediction (MPI Parallel)"
    )
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--run-dir", type=str, required=True,
                       help="Run directory from previous steps")
    parser.add_argument("--save-reduced-states", action="store_true",
                       help="Save reduced state trajectories X_OpInf (memory intensive)")
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Load configuration (all ranks)
    cfg = load_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Setup logging (rank 0 gets real logger, others get dummy)
    if rank == 0:
        logger = setup_logging("step_3_mpi", args.run_dir, cfg.log_level)
        
        print_header("STEP 3: EVALUATION AND PREDICTION (MPI PARALLEL)")
        print(f"  Run directory: {args.run_dir}")
        print(f"  MPI ranks: {size}")
        print(f"  Save reduced states: {args.save_reduced_states}")
        
        if not check_step_completed(args.run_dir, "step_2"):
            logger.error("Step 2 has not completed. Run step_2_train_rom.py first.")
            comm.Abort(1)
            return
        
        save_step_status(args.run_dir, "step_3", "running")
        save_config(cfg, args.run_dir, step_name="step_3")
        logger.info("Configuration saved to run directory")
    else:
        logger = DummyLogger()
    
    comm.Barrier()
    
    paths = get_output_paths(args.run_dir)
    
    try:
        # Load preprocessing info (rank 0 logs)
        if rank == 0:
            preproc_info = load_preprocessing_info(paths["preprocessing_info"], logger)
        
        # All ranks load models (needed for predictions)
        if rank == 0:
            logger.info("Loading ensemble models...")
        
        models = load_ensemble(
            paths["ensemble_models"], 
            logger if rank == 0 else DummyLogger(),
            operators_dir=paths["operators_dir"]
        )
        
        if len(models) == 0:
            if rank == 0:
                logger.error("No models loaded!")
                save_step_status(args.run_dir, "step_3", "failed", {"error": "No models"})
            return
        
        # Load supporting data (all ranks need this)
        learning = np.load(paths["learning_matrices"])
        mean_Xhat = learning['mean_Xhat']
        scaling_Xhat = float(learning['scaling_Xhat'])
        
        ICs = np.load(paths["initial_conditions"])
        train_ICs_reduced = ICs['train_ICs_reduced']
        test_ICs_reduced = ICs['test_ICs_reduced']
        
        boundaries_data = np.load(paths["boundaries"])
        train_boundaries = boundaries_data['train_boundaries']
        test_boundaries = boundaries_data['test_boundaries']
        
        n_train = len(train_boundaries) - 1
        n_test = len(test_boundaries) - 1
        
        # Distribute trajectories
        train_traj_indices = distribute_trajectories(n_train, rank, size)
        test_traj_indices = distribute_trajectories(n_test, rank, size)
        
        if rank == 0:
            logger.info(f"Trajectory distribution:")
            logger.info(f"  Training: {n_train} total, {len(train_traj_indices)} on rank 0")
            logger.info(f"  Test: {n_test} total, {len(test_traj_indices)} on rank 0")
            
            # Memory estimate
            r = mean_Xhat.shape[0]
            max_steps = max(
                max((train_boundaries[i+1] - train_boundaries[i]) for i in range(n_train)),
                max((test_boundaries[i+1] - test_boundaries[i]) for i in range(n_test))
            )
            max_traj_per_rank = max(len(train_traj_indices), len(test_traj_indices))
            estimate_memory_usage(len(models), r, max_steps, max_traj_per_rank,
                                 args.save_reduced_states, logger)
        
        comm.Barrier()
        start_time = time.time()
        
        # Compute training predictions (distributed)
        if rank == 0:
            logger.info("\n" + "="*50)
            logger.info("TRAINING TRAJECTORY PREDICTIONS")
            logger.info("="*50)
        
        local_train_preds = compute_distributed_predictions(
            models, train_ICs_reduced, train_boundaries,
            mean_Xhat, scaling_Xhat, train_traj_indices,
            args.save_reduced_states, logger, "training trajectory"
        )
        
        # Compute test predictions (distributed)
        if rank == 0:
            logger.info("\n" + "="*50)
            logger.info("TEST TRAJECTORY PREDICTIONS")
            logger.info("="*50)
        
        local_test_preds = compute_distributed_predictions(
            models, test_ICs_reduced, test_boundaries,
            mean_Xhat, scaling_Xhat, test_traj_indices,
            args.save_reduced_states, logger, "test trajectory"
        )
        
        comm.Barrier()
        pred_time = time.time() - start_time
        
        if rank == 0:
            logger.info(f"\nPredictions completed in {pred_time:.1f}s")
        
        # Compute metrics locally
        if rank == 0:
            logger.info("\n" + "="*50)
            logger.info("COMPUTING METRICS")
            logger.info("="*50)
        
        local_train_metrics = compute_distributed_metrics(
            local_train_preds, cfg.training_files, train_boundaries, cfg.engine, logger
        )
        local_test_metrics = compute_distributed_metrics(
            local_test_preds, cfg.test_files, test_boundaries, cfg.engine, logger
        )
        
        # Gather all predictions and metrics to rank 0
        all_train_preds = comm.gather(local_train_preds, root=0)
        all_test_preds = comm.gather(local_test_preds, root=0)
        all_train_metrics = comm.gather(local_train_metrics, root=0)
        all_test_metrics = comm.gather(local_test_metrics, root=0)
        
        # Rank 0 handles saving and summary
        if rank == 0:
            # Flatten gathered lists
            train_preds_flat = [p for sublist in all_train_preds for p in sublist]
            test_preds_flat = [p for sublist in all_test_preds for p in sublist]
            train_metrics_flat = [m for sublist in all_train_metrics for m in sublist]
            test_metrics_flat = [m for sublist in all_test_metrics for m in sublist]
            
            # Sort by trajectory index
            train_preds_flat.sort(key=lambda x: x['traj_idx'])
            test_preds_flat.sort(key=lambda x: x['traj_idx'])
            train_metrics_flat.sort(key=lambda x: x['trajectory'])
            test_metrics_flat.sort(key=lambda x: x['trajectory'])
            
            # Convert to dict keyed by traj_idx for plotting
            train_preds_dict = {p['traj_idx']: p for p in train_preds_flat}
            test_preds_dict = {p['traj_idx']: p for p in test_preds_flat}
            
            # Save predictions
            if cfg.save_predictions:
                save_dict = {
                    'n_train_traj': n_train,
                    'n_test_traj': n_test,
                    'num_models': len(models),
                    'train_boundaries': train_boundaries,
                    'test_boundaries': test_boundaries,
                }
                
                for p in train_preds_flat:
                    i = p['traj_idx']
                    save_dict[f'train_traj_{i}_Gamma_n'] = p['Gamma_n']
                    save_dict[f'train_traj_{i}_Gamma_c'] = p['Gamma_c']
                    if args.save_reduced_states and 'X_OpInf' in p:
                        save_dict[f'train_traj_{i}_X_OpInf'] = p['X_OpInf']
                
                for p in test_preds_flat:
                    i = p['traj_idx']
                    save_dict[f'test_traj_{i}_Gamma_n'] = p['Gamma_n']
                    save_dict[f'test_traj_{i}_Gamma_c'] = p['Gamma_c']
                    if args.save_reduced_states and 'X_OpInf' in p:
                        save_dict[f'test_traj_{i}_X_OpInf'] = p['X_OpInf']
                
                np.savez(paths["predictions"], **save_dict)
                logger.info(f"Saved predictions to {paths['predictions']}")
            
            # Compile metrics
            train_metrics = {
                'trajectories': train_metrics_flat,
                'ensemble': {
                    'mean_err_Gamma_n': float(np.mean([m['err_mean_Gamma_n'] for m in train_metrics_flat])),
                    'std_err_Gamma_n': float(np.mean([m['err_std_Gamma_n'] for m in train_metrics_flat])),
                    'mean_err_Gamma_c': float(np.mean([m['err_mean_Gamma_c'] for m in train_metrics_flat])),
                    'std_err_Gamma_c': float(np.mean([m['err_std_Gamma_c'] for m in train_metrics_flat])),
                }
            }
            test_metrics = {
                'trajectories': test_metrics_flat,
                'ensemble': {
                    'mean_err_Gamma_n': float(np.mean([m['err_mean_Gamma_n'] for m in test_metrics_flat])),
                    'std_err_Gamma_n': float(np.mean([m['err_std_Gamma_n'] for m in test_metrics_flat])),
                    'mean_err_Gamma_c': float(np.mean([m['err_mean_Gamma_c'] for m in test_metrics_flat])),
                    'std_err_Gamma_c': float(np.mean([m['err_std_Gamma_c'] for m in test_metrics_flat])),
                }
            }
            
            # Save metrics
            all_metrics = {'train': train_metrics, 'test': test_metrics}
            with open(paths["metrics"], 'w') as f:
                yaml.dump(all_metrics, f, default_flow_style=False)
            logger.info(f"Saved metrics to {paths['metrics']}")
            
            # Generate plots
            if cfg.generate_plots:
                generate_plots(
                    train_preds_dict, cfg.training_files, train_boundaries,
                    cfg, os.path.join(paths["figures_dir"], "train"), logger
                )
                generate_plots(
                    test_preds_dict, cfg.test_files, test_boundaries,
                    cfg, os.path.join(paths["figures_dir"], "test"), logger
                )
            
            # Print summary
            print_header("EVALUATION SUMMARY")
            print("\n  Training Data:")
            for traj in train_metrics_flat:
                print(f"    Trajectory {traj['trajectory'] + 1}:")
                print(f"      Γn: mean_err={traj['err_mean_Gamma_n']:.4f}, std_err={traj['err_std_Gamma_n']:.4f}")
                print(f"      Γc: mean_err={traj['err_mean_Gamma_c']:.4f}, std_err={traj['err_std_Gamma_c']:.4f}")
            
            print("\n  Test Data:")
            for traj in test_metrics_flat:
                print(f"    Trajectory {traj['trajectory'] + 1}:")
                print(f"      Γn: mean_err={traj['err_mean_Gamma_n']:.4f}, std_err={traj['err_std_Gamma_n']:.4f}")
                print(f"      Γc: mean_err={traj['err_mean_Gamma_c']:.4f}, std_err={traj['err_std_Gamma_c']:.4f}")
            
            total_time = time.time() - start_time
            
            save_step_status(args.run_dir, "step_3", "completed", {
                "train_mean_err_Gamma_n": train_metrics['ensemble']['mean_err_Gamma_n'],
                "test_mean_err_Gamma_n": test_metrics['ensemble']['mean_err_Gamma_n'],
                "total_time_seconds": total_time,
                "mpi_ranks": size,
            })
            
            print_header("STEP 3 COMPLETE")
            print(f"  Total runtime: {total_time:.1f}s with {size} MPI ranks")
            logger.info(f"Step 3 completed successfully in {total_time:.1f}s")
    
    except Exception as e:
        if rank == 0:
            logger.error(f"Step 3 failed: {e}", exc_info=True)
            save_step_status(args.run_dir, "step_3", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
