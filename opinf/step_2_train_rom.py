"""
Step 2: ROM Training via Hyperparameter Sweep.

This script handles:
1. Loading pre-computed POD basis and learning matrices
2. Parallel hyperparameter sweep over regularization parameters
3. Model selection (top-k or threshold based)
4. Saving ensemble of best models

Supports both serial and MPI-parallel execution.

Usage:
    # Serial execution (for testing)
    python step_2_train_rom.py --config config.yaml --run-dir /path/to/run
    
    # Parallel execution (for HPC)
    mpirun -np 56 python step_2_train_rom.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import gc
import time
import numpy as np
import os
from itertools import product

from utils import (
    load_config,
    save_config,
    setup_logging,
    save_step_status,
    check_step_completed,
    get_output_paths,
    print_header,
    print_config_summary,
    get_x_sq,
    solve_opinf_difference_model,
    PipelineConfig,
)

# Try to import MPI - fall back to serial execution if not available
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


# =============================================================================
# SINGLE HYPERPARAMETER EVALUATION
# =============================================================================

def evaluate_hyperparameters(
    alpha_state_lin: float,
    alpha_state_quad: float,
    alpha_out_lin: float,
    alpha_out_quad: float,
    D_state: np.ndarray,
    D_state_2: np.ndarray,
    Y_state: np.ndarray,
    D_out: np.ndarray,
    D_out_2: np.ndarray,
    Y_Gamma: np.ndarray,
    X_state: np.ndarray,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    mean_Gamma_n_ref: float,
    std_Gamma_n_ref: float,
    mean_Gamma_c_ref: float,
    std_Gamma_c_ref: float,
    r: int,
    n_steps: int,
    training_end: int,
) -> dict:
    """
    Evaluate a single hyperparameter combination.
    
    Returns
    -------
    dict
        Results dictionary with error metrics and hyperparameters.
    """
    s = int(r * (r + 1) / 2)
    d_state = r + s
    d_out = r + s + 1
    
    # Solve state operator learning
    regg = np.zeros(d_state)
    regg[:r] = alpha_state_lin
    regg[r:r + s] = alpha_state_quad
    regularizer = np.diag(regg)
    D_state_reg = D_state_2 + regularizer
    
    O = np.linalg.solve(D_state_reg, np.dot(D_state.T, Y_state)).T
    del D_state_reg, regularizer  # Free memory
    A = O[:, :r]
    F = O[:, r:r + s]
    del O  # Free memory
    
    # Integrate model
    f = lambda x: np.dot(A, x) + np.dot(F, get_x_sq(x))
    u0 = X_state[0, :]
    is_nan, Xhat_pred = solve_opinf_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {
            'is_nan': True,
            'alpha_state_lin': alpha_state_lin,
            'alpha_state_quad': alpha_state_quad,
            'alpha_out_lin': alpha_out_lin,
            'alpha_out_quad': alpha_out_quad,
        }
    
    X_OpInf = Xhat_pred.T
    del Xhat_pred  # Free memory immediately
    
    # Prepare for output operator
    Xhat_OpInf_scaled = (X_OpInf - mean_Xhat[np.newaxis, :]) / scaling_Xhat
    del X_OpInf  # Free memory
    Xhat_2_OpInf = get_x_sq(Xhat_OpInf_scaled)
    
    # Solve output operator learning
    regg_out = np.zeros(d_out)
    regg_out[:r] = alpha_out_lin
    regg_out[r:r + s] = alpha_out_quad
    regg_out[r + s:] = alpha_out_lin
    regularizer_out = np.diag(regg_out)
    D_out_reg = D_out_2 + regularizer_out
    
    O_out = np.linalg.solve(D_out_reg, np.dot(D_out.T, Y_Gamma.T)).T
    C = O_out[:, :r]
    G = O_out[:, r:r + s]
    c = O_out[:, r + s]
    
    # Compute predictions
    Y_OpInf = (
        C @ Xhat_OpInf_scaled.T
        + G @ Xhat_2_OpInf.T
        + c[:, np.newaxis]
    )
    
    # Free large temporary arrays
    del Xhat_OpInf_scaled, Xhat_2_OpInf, O_out, D_out_reg
    
    ts_Gamma_n = Y_OpInf[0, :]
    ts_Gamma_c = Y_OpInf[1, :]
    
    # Compute error metrics
    mean_Gamma_n_pred = np.mean(ts_Gamma_n[:training_end])
    std_Gamma_n_pred = np.std(ts_Gamma_n[:training_end], ddof=1)
    mean_Gamma_c_pred = np.mean(ts_Gamma_c[:training_end])
    std_Gamma_c_pred = np.std(ts_Gamma_c[:training_end], ddof=1)
    
    mean_err_Gamma_n = np.abs(mean_Gamma_n_ref - mean_Gamma_n_pred) / np.abs(mean_Gamma_n_ref)
    std_err_Gamma_n = np.abs(std_Gamma_n_ref - std_Gamma_n_pred) / std_Gamma_n_ref
    mean_err_Gamma_c = np.abs(mean_Gamma_c_ref - mean_Gamma_c_pred) / np.abs(mean_Gamma_c_ref)
    std_err_Gamma_c = np.abs(std_Gamma_c_ref - std_Gamma_c_pred) / std_Gamma_c_ref
    
    total_error = mean_err_Gamma_n + std_err_Gamma_n + mean_err_Gamma_c + std_err_Gamma_c
    
    return {
        'is_nan': False,
        'total_error': total_error,
        'mean_err_Gamma_n': mean_err_Gamma_n,
        'std_err_Gamma_n': std_err_Gamma_n,
        'mean_err_Gamma_c': mean_err_Gamma_c,
        'std_err_Gamma_c': std_err_Gamma_c,
        'alpha_state_lin': alpha_state_lin,
        'alpha_state_quad': alpha_state_quad,
        'alpha_out_lin': alpha_out_lin,
        'alpha_out_quad': alpha_out_quad,
    }


# =============================================================================
# SERIAL SWEEP
# =============================================================================

def serial_hyperparameter_sweep(
    cfg: PipelineConfig,
    data: dict,
    logger,
) -> list:
    """
    Run hyperparameter sweep in serial (single process).
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration.
    data : dict
        Pre-loaded data matrices.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    list
        List of result dictionaries.
    """
    # Build parameter grid
    param_grid = list(product(
        cfg.state_lin,
        cfg.state_quad,
        cfg.output_lin,
        cfg.output_quad,
    ))
    
    n_total = len(param_grid)
    logger.info(f"Serial sweep: {n_total:,} combinations")
    
    results = []
    n_nan = 0
    best_error = float('inf')
    
    for i, (asl, asq, aol, aoq) in enumerate(param_grid):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{n_total} (best: {best_error:.4e}, NaN: {n_nan})")
        
        result = evaluate_hyperparameters(
            asl, asq, aol, aoq,
            data['D_state'], data['D_state_2'], data['Y_state'],
            data['D_out'], data['D_out_2'], data['Y_Gamma'],
            data['X_state'], data['mean_Xhat'], data['scaling_Xhat'],
            data['mean_Gamma_n'], data['std_Gamma_n'],
            data['mean_Gamma_c'], data['std_Gamma_c'],
            cfg.r, cfg.n_steps, cfg.training_end,
        )
        
        if result['is_nan']:
            n_nan += 1
        else:
            results.append(result)
            if result['total_error'] < best_error:
                best_error = result['total_error']
    
    logger.info(f"Sweep complete: {len(results)} valid, {n_nan} NaN")
    return results


# =============================================================================
# PARALLEL SWEEP (MPI) WITH SHARED MEMORY
# =============================================================================

def create_shared_array(node_comm, shape, dtype=np.float64):
    """
    Create a numpy array backed by MPI shared memory within a node.
    
    Only node_rank 0 allocates the memory; other ranks attach to it.
    """
    from mpi4py import MPI
    
    node_rank = node_comm.Get_rank()
    
    # Calculate size in bytes
    itemsize = np.dtype(dtype).itemsize
    nbytes = int(np.prod(shape)) * itemsize
    
    # Only node rank 0 allocates
    if node_rank == 0:
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    else:
        win = MPI.Win.Allocate_shared(0, itemsize, comm=node_comm)
    
    # Get buffer from rank 0's allocation
    buf, itemsize = win.Shared_query(0)
    arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    
    return arr, win


def parallel_hyperparameter_sweep(
    cfg: PipelineConfig,
    data: dict,
    logger,
    comm,
) -> list:
    """
    Run hyperparameter sweep in parallel using MPI.
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration.
    data : dict
        Pre-loaded data matrices.
    logger : logging.Logger
        Logger instance.
    comm : MPI.Comm
        MPI communicator.
    
    Returns
    -------
    list
        On rank 0: list of result dictionaries.
        On other ranks: empty list.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Build parameter grid on all ranks
    param_grid = list(product(
        cfg.state_lin,
        cfg.state_quad,
        cfg.output_lin,
        cfg.output_quad,
    ))
    
    n_total = len(param_grid)
    
    if rank == 0:
        logger.info(f"Parallel sweep: {n_total:,} combinations across {size} ranks")
        # Memory warning for HPC users
        if size > 14:
            logger.warning(
                f"⚠️  Running {size} MPI ranks may cause memory issues. "
                f"Consider using 8-14 ranks per node for r={cfg.r}. "
                f"Memory per rank estimate: ~{2 * cfg.r * cfg.n_steps * 8 / 1e6:.0f} MB"
            )
    
    # Distribute work
    params_per_rank = n_total // size
    remainder = n_total % size
    
    if rank < remainder:
        start = rank * (params_per_rank + 1)
        end = start + params_per_rank + 1
    else:
        start = rank * params_per_rank + remainder
        end = start + params_per_rank
    
    my_params = param_grid[start:end]
    
    if rank == 0:
        logger.info(f"  Each rank processes ~{len(my_params)} combinations")
    
    # Process assigned parameters
    local_results = []
    n_nan = 0
    gc_interval = 50  # Run garbage collection every N iterations
    
    for i, (asl, asq, aol, aoq) in enumerate(my_params):
        result = evaluate_hyperparameters(
            asl, asq, aol, aoq,
            data['D_state'], data['D_state_2'], data['Y_state'],
            data['D_out'], data['D_out_2'], data['Y_Gamma'],
            data['X_state'], data['mean_Xhat'], data['scaling_Xhat'],
            data['mean_Gamma_n'], data['std_Gamma_n'],
            data['mean_Gamma_c'], data['std_Gamma_c'],
            cfg.r, cfg.n_steps, cfg.training_end,
        )
        
        if result['is_nan']:
            n_nan += 1
        else:
            local_results.append(result)
        
        # Periodic garbage collection to prevent memory buildup
        if (i + 1) % gc_interval == 0:
            gc.collect()
    
    # Gather results to rank 0
    all_results = comm.gather(local_results, root=0)
    all_nan_counts = comm.gather(n_nan, root=0)
    
    if rank == 0:
        # Flatten results
        combined = []
        for rank_results in all_results:
            combined.extend(rank_results)
        
        total_nan = sum(all_nan_counts)
        logger.info(f"Sweep complete: {len(combined)} valid, {total_nan} NaN")
        return combined
    else:
        return []


# =============================================================================
# MODEL SELECTION
# =============================================================================

def log_error_statistics(results: list, logger) -> dict:
    """
    Log detailed statistics about sweep results to help with threshold tuning.
    
    Parameters
    ----------
    results : list
        List of result dictionaries from sweep.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Statistics dictionary.
    """
    if not results:
        logger.warning("No results to analyze!")
        return {}
    
    # Extract error arrays
    mean_err_n = np.array([r['mean_err_Gamma_n'] for r in results])
    std_err_n = np.array([r['std_err_Gamma_n'] for r in results])
    mean_err_c = np.array([r['mean_err_Gamma_c'] for r in results])
    std_err_c = np.array([r['std_err_Gamma_c'] for r in results])
    total_err = np.array([r['total_error'] for r in results])
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("ERROR DISTRIBUTION STATISTICS (for threshold tuning)")
    logger.info("=" * 70)
    
    # Per-metric statistics
    metrics = {
        'mean_err_Gamma_n': mean_err_n,
        'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c,
        'std_err_Gamma_c': std_err_c,
        'total_error': total_err,
    }
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    logger.info("")
    logger.info(f"{'Metric':<20} | {'Min':>10} | {'P10':>10} | {'P25':>10} | {'P50':>10} | {'P75':>10} | {'P90':>10} | {'Max':>10}")
    logger.info("-" * 105)
    
    stats = {}
    for name, arr in metrics.items():
        pcts = np.percentile(arr, percentiles)
        stats[name] = {
            'min': np.min(arr),
            'max': np.max(arr),
            'mean': np.mean(arr),
            'std': np.std(arr),
            'percentiles': dict(zip(percentiles, pcts)),
        }
        logger.info(f"{name:<20} | {np.min(arr):>10.4f} | {pcts[2]:>10.4f} | {pcts[3]:>10.4f} | {pcts[4]:>10.4f} | {pcts[5]:>10.4f} | {pcts[6]:>10.4f} | {np.max(arr):>10.4f}")
    
    # Threshold sensitivity analysis
    logger.info("")
    logger.info("THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("(Models passing at different threshold combinations)")
    logger.info("")
    
    mean_thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    std_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
    
    # Header
    header = "{:<10}".format("mean/std")
    for st in std_thresholds:
        header += f" | {st:>6.2f}"
    logger.info(header)
    logger.info("-" * (12 + 9 * len(std_thresholds)))
    
    # Count models for each threshold combo
    for mt in mean_thresholds:
        row = f"{mt:<10.2f}"
        for st in std_thresholds:
            count = sum(1 for r in results 
                       if r['mean_err_Gamma_n'] < mt 
                       and r['std_err_Gamma_n'] < st
                       and r['mean_err_Gamma_c'] < mt 
                       and r['std_err_Gamma_c'] < st)
            row += f" | {count:>6}"
        logger.info(row)
    
    # Best models summary
    logger.info("")
    logger.info("TOP 10 MODELS BY TOTAL ERROR:")
    logger.info(f"{'Rank':<6} | {'Total':>10} | {'Mean Γn':>10} | {'Std Γn':>10} | {'Mean Γc':>10} | {'Std Γc':>10}")
    logger.info("-" * 75)
    
    sorted_results = sorted(results, key=lambda x: x['total_error'])
    for i, r in enumerate(sorted_results[:10]):
        logger.info(f"{i+1:<6} | {r['total_error']:>10.4f} | {r['mean_err_Gamma_n']:>10.4f} | "
                   f"{r['std_err_Gamma_n']:>10.4f} | {r['mean_err_Gamma_c']:>10.4f} | {r['std_err_Gamma_c']:>10.4f}")
    
    # Suggested thresholds
    logger.info("")
    logger.info("SUGGESTED THRESHOLDS (based on percentiles):")
    logger.info(f"  Conservative (P10): mean_threshold={stats['mean_err_Gamma_n']['percentiles'][10]:.3f}, "
               f"std_threshold={stats['std_err_Gamma_n']['percentiles'][10]:.3f}")
    logger.info(f"  Moderate (P25):     mean_threshold={stats['mean_err_Gamma_n']['percentiles'][25]:.3f}, "
               f"std_threshold={stats['std_err_Gamma_n']['percentiles'][25]:.3f}")
    logger.info(f"  Relaxed (P50):      mean_threshold={stats['mean_err_Gamma_n']['percentiles'][50]:.3f}, "
               f"std_threshold={stats['std_err_Gamma_n']['percentiles'][50]:.3f}")
    logger.info("=" * 70)
    logger.info("")
    
    return stats


def select_models(
    results: list,
    method: str,
    num_top: int = 20,
    threshold_mean: float = 0.05,
    threshold_std: float = 0.30,
    logger = None,
) -> list:
    """
    Select best models from sweep results.
    
    Parameters
    ----------
    results : list
        List of result dictionaries from sweep.
    method : str
        Selection method ("top_k" or "threshold").
    num_top : int
        Number of top models (for top_k).
    threshold_mean, threshold_std : float
        Error thresholds (for threshold method).
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    list
        Selected models sorted by total_error.
    """
    if method == "top_k":
        # Sort by total error and take top k
        sorted_results = sorted(results, key=lambda x: x['total_error'])
        selected = sorted_results[:num_top]
        if logger:
            logger.info(f"Selected top {len(selected)} models")
    
    elif method == "threshold":
        # Filter by threshold criteria
        selected = [
            r for r in results
            if (r['mean_err_Gamma_n'] < threshold_mean and
                r['std_err_Gamma_n'] < threshold_std and
                r['mean_err_Gamma_c'] < threshold_mean and
                r['std_err_Gamma_c'] < threshold_std)
        ]
        selected = sorted(selected, key=lambda x: x['total_error'])
        if logger:
            logger.info(f"Selected {len(selected)} models meeting threshold criteria")
    
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    return selected


# =============================================================================
# OPERATOR RECOMPUTATION
# =============================================================================

def recompute_operators(
    selected: list,
    data: dict,
    r: int,
    logger = None,
) -> list:
    """
    Recompute operator matrices for selected models.
    
    During the sweep, we don't store operators to save memory.
    This function recomputes them for the selected models.
    
    Parameters
    ----------
    selected : list
        Selected model parameter dictionaries.
    data : dict
        Pre-loaded data matrices.
    r : int
        Number of POD modes.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    list
        Models with operator matrices (A, F, C, G, c).
    """
    if logger:
        logger.info(f"Recomputing operators for {len(selected)} models...")
    
    s = int(r * (r + 1) / 2)
    d_state = r + s
    d_out = r + s + 1
    
    models_with_ops = []
    
    for params in selected:
        alpha_state_lin = params['alpha_state_lin']
        alpha_state_quad = params['alpha_state_quad']
        alpha_out_lin = params['alpha_out_lin']
        alpha_out_quad = params['alpha_out_quad']
        
        # State operators
        regg = np.zeros(d_state)
        regg[:r] = alpha_state_lin
        regg[r:r + s] = alpha_state_quad
        regularizer = np.diag(regg)
        D_state_reg = data['D_state_2'] + regularizer
        
        O = np.linalg.solve(D_state_reg, np.dot(data['D_state'].T, data['Y_state'])).T
        A = O[:, :r]
        F = O[:, r:r + s]
        
        # Output operators
        regg_out = np.zeros(d_out)
        regg_out[:r] = alpha_out_lin
        regg_out[r:r + s] = alpha_out_quad
        regg_out[r + s:] = alpha_out_lin
        regularizer_out = np.diag(regg_out)
        D_out_reg = data['D_out_2'] + regularizer_out
        
        O_out = np.linalg.solve(D_out_reg, np.dot(data['D_out'].T, data['Y_Gamma'].T)).T
        C = O_out[:, :r]
        G = O_out[:, r:r + s]
        c = O_out[:, r + s]
        
        # Build model with operators
        model = {
            'A': A.copy(),
            'F': F.copy(),
            'C': C.copy(),
            'G': G.copy(),
            'c': c.copy(),
            'alpha_state_lin': alpha_state_lin,
            'alpha_state_quad': alpha_state_quad,
            'alpha_out_lin': alpha_out_lin,
            'alpha_out_quad': alpha_out_quad,
            'total_error': params['total_error'],
            'mean_err_Gamma_n': params['mean_err_Gamma_n'],
            'std_err_Gamma_n': params['std_err_Gamma_n'],
            'mean_err_Gamma_c': params['mean_err_Gamma_c'],
            'std_err_Gamma_c': params['std_err_Gamma_c'],
        }
        models_with_ops.append((params['total_error'], model))
    
    return models_with_ops


# =============================================================================
# SAVE ENSEMBLE
# =============================================================================

def save_ensemble(
    models: list,
    output_path: str,
    cfg: PipelineConfig,
    logger = None,
) -> str:
    """
    Save ensemble models to NPZ file.
    
    Parameters
    ----------
    models : list
        List of (score, model) tuples.
    output_path : str
        Path for output file.
    cfg : PipelineConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    str
        Path to saved file.
    """
    ensemble_data = {
        'num_models': len(models),
        'selection_method': cfg.selection_method,
        'r': cfg.r,
        'threshold_mean': cfg.threshold_mean,
        'threshold_std': cfg.threshold_std,
        'num_top_models': cfg.num_top_models,
    }
    
    for i, (score, model) in enumerate(models):
        prefix = f'model_{i}_'
        for key in ['A', 'F', 'C', 'G', 'c',
                    'alpha_state_lin', 'alpha_state_quad',
                    'alpha_out_lin', 'alpha_out_quad',
                    'total_error', 'mean_err_Gamma_n', 'std_err_Gamma_n',
                    'mean_err_Gamma_c', 'std_err_Gamma_c']:
            ensemble_data[prefix + key] = model[key]
    
    np.savez(output_path, **ensemble_data)
    
    if logger:
        logger.info(f"Saved {len(models)} models to {output_path}")
    
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 2."""
    parser = argparse.ArgumentParser(
        description="Step 2: ROM Training via Hyperparameter Sweep"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory from Step 1"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only compute and display error statistics (no model saving). "
             "Useful for tuning thresholds."
    )
    args = parser.parse_args()
    
    # Initialize MPI if available
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        parallel = size > 1
    else:
        comm = None
        rank = 0
        size = 1
        parallel = False
    
    # Load configuration
    cfg = load_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("step_2", args.run_dir, cfg.log_level, rank)
    
    if rank == 0:
        print_header("STEP 2: ROM TRAINING VIA HYPERPARAMETER SWEEP")
        print(f"  Run directory: {args.run_dir}")
        print(f"  Parallel: {parallel} ({size} processes)")
        print_config_summary(cfg)
        
        # Check Step 1 completed
        if not check_step_completed(args.run_dir, "step_1"):
            logger.error("Step 1 has not completed. Run step_1_preprocess.py first.")
            if HAS_MPI:
                comm.Abort(1)
            return
        
        save_step_status(args.run_dir, "step_2", "running")
    
    if HAS_MPI:
        comm.Barrier()
    
    paths = get_output_paths(args.run_dir)
    
    # Track shared memory windows for cleanup
    shared_windows = []
    
    try:
        # =====================================================================
        # SHARED MEMORY DATA LOADING (like parallel_sweep.py)
        # Only rank 0 on each node loads data; other ranks share that memory
        # =====================================================================
        
        if parallel:
            # Create node-local communicator for shared memory
            node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            node_rank = node_comm.Get_rank()
            node_size = node_comm.Get_size()
            
            if rank == 0:
                logger.info(f"Memory optimization: {node_size} ranks sharing memory per node")
            
            # Step 1: Rank 0 loads data and determines shapes
            if rank == 0:
                logger.info("Loading pre-computed data (rank 0)...")
                learning = np.load(paths["learning_matrices"])
                gamma_ref = np.load(paths["gamma_ref"])
                
                # Store local copies temporarily
                data_local = {
                    'X_state': learning['X_state'].copy(),
                    'Y_state': learning['Y_state'].copy(),
                    'D_state': learning['D_state'].copy(),
                    'D_state_2': learning['D_state_2'].copy(),
                    'D_out': learning['D_out'].copy(),
                    'D_out_2': learning['D_out_2'].copy(),
                    'mean_Xhat': learning['mean_Xhat'].copy(),
                    'Y_Gamma': gamma_ref['Y_Gamma'].copy(),
                }
                scalars = {
                    'scaling_Xhat': float(learning['scaling_Xhat']),
                    'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
                    'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
                    'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
                    'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
                }
                shapes = {k: v.shape for k, v in data_local.items()}
                
                logger.info(f"  X_state shape: {shapes['X_state']}")
                logger.info(f"  D_out shape: {shapes['D_out']}")
                
                # Close file handles
                learning.close()
                gamma_ref.close()
            else:
                data_local = None
                shapes = None
                scalars = None
            
            # Step 2: Broadcast shapes and scalars to all ranks
            shapes = comm.bcast(shapes, root=0)
            scalars = comm.bcast(scalars, root=0)
            
            # Step 3: Create shared memory arrays on each node
            X_state_shared, win1 = create_shared_array(node_comm, shapes['X_state'])
            Y_state_shared, win2 = create_shared_array(node_comm, shapes['Y_state'])
            D_state_shared, win3 = create_shared_array(node_comm, shapes['D_state'])
            D_state_2_shared, win4 = create_shared_array(node_comm, shapes['D_state_2'])
            D_out_shared, win5 = create_shared_array(node_comm, shapes['D_out'])
            D_out_2_shared, win6 = create_shared_array(node_comm, shapes['D_out_2'])
            mean_Xhat_shared, win7 = create_shared_array(node_comm, shapes['mean_Xhat'])
            Y_Gamma_shared, win8 = create_shared_array(node_comm, shapes['Y_Gamma'])
            
            shared_windows = [win1, win2, win3, win4, win5, win6, win7, win8]
            
            # Step 4: Fill shared memory from rank 0 on each node
            # Create communicator of just node-rank-0s
            node_root_comm = comm.Split(color=0 if node_rank == 0 else MPI.UNDEFINED, key=rank)
            
            if node_rank == 0:
                # Broadcast data from global rank 0 to all node-rank-0s
                if node_root_comm != MPI.COMM_NULL:
                    X_state_shared[:] = node_root_comm.bcast(data_local['X_state'] if rank == 0 else None, root=0)
                    Y_state_shared[:] = node_root_comm.bcast(data_local['Y_state'] if rank == 0 else None, root=0)
                    D_state_shared[:] = node_root_comm.bcast(data_local['D_state'] if rank == 0 else None, root=0)
                    D_state_2_shared[:] = node_root_comm.bcast(data_local['D_state_2'] if rank == 0 else None, root=0)
                    D_out_shared[:] = node_root_comm.bcast(data_local['D_out'] if rank == 0 else None, root=0)
                    D_out_2_shared[:] = node_root_comm.bcast(data_local['D_out_2'] if rank == 0 else None, root=0)
                    mean_Xhat_shared[:] = node_root_comm.bcast(data_local['mean_Xhat'] if rank == 0 else None, root=0)
                    Y_Gamma_shared[:] = node_root_comm.bcast(data_local['Y_Gamma'] if rank == 0 else None, root=0)
            
            # Synchronize within node so all ranks see the data
            node_comm.Barrier()
            
            # Clean up
            if node_root_comm != MPI.COMM_NULL:
                node_root_comm.Free()
            if rank == 0:
                del data_local
                gc.collect()
            
            # Build data dict pointing to shared memory
            data = {
                'X_state': X_state_shared,
                'Y_state': Y_state_shared,
                'D_state': D_state_shared,
                'D_state_2': D_state_2_shared,
                'D_out': D_out_shared,
                'D_out_2': D_out_2_shared,
                'mean_Xhat': mean_Xhat_shared,
                'Y_Gamma': Y_Gamma_shared,
                'scaling_Xhat': scalars['scaling_Xhat'],
                'mean_Gamma_n': scalars['mean_Gamma_n'],
                'std_Gamma_n': scalars['std_Gamma_n'],
                'mean_Gamma_c': scalars['mean_Gamma_c'],
                'std_Gamma_c': scalars['std_Gamma_c'],
            }
            
            if rank == 0:
                logger.info("Shared memory setup complete")
            
            comm.Barrier()
        
        else:
            # Serial mode - load data directly
            if rank == 0:
                logger.info("Loading pre-computed data...")
            
            learning = np.load(paths["learning_matrices"])
            gamma_ref = np.load(paths["gamma_ref"])
            
            data = {
                'X_state': learning['X_state'],
                'Y_state': learning['Y_state'],
                'D_state': learning['D_state'],
                'D_state_2': learning['D_state_2'],
                'D_out': learning['D_out'],
                'D_out_2': learning['D_out_2'],
                'mean_Xhat': learning['mean_Xhat'],
                'scaling_Xhat': float(learning['scaling_Xhat']),
                'Y_Gamma': gamma_ref['Y_Gamma'],
                'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
                'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
                'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
                'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
            }
            
            if rank == 0:
                logger.info(f"  X_state shape: {data['X_state'].shape}")
                logger.info(f"  D_out shape: {data['D_out'].shape}")
        
        # Run sweep
        if rank == 0:
            logger.info("Starting hyperparameter sweep...")
        
        start_time = time.time()
        
        if parallel:
            results = parallel_hyperparameter_sweep(cfg, data, logger, comm)
        else:
            results = serial_hyperparameter_sweep(cfg, data, logger)
        
        elapsed = time.time() - start_time
        
        # Rank 0 handles selection and saving
        if rank == 0:
            logger.info(f"Sweep completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            
            if len(results) == 0:
                logger.error("No valid models found!")
                save_step_status(args.run_dir, "step_2", "failed", 
                               {"error": "No valid models"})
                return
            
            # Log detailed statistics to help with threshold tuning
            error_stats = log_error_statistics(results, logger)
            
            # Stats-only mode: just show statistics and exit
            if args.stats_only:
                logger.info("")
                logger.info("=" * 70)
                logger.info("STATS-ONLY MODE: Exiting without saving models.")
                logger.info("Review the statistics above and adjust threshold_mean/threshold_std in your config.")
                logger.info("=" * 70)
                print_header("STATS-ONLY MODE COMPLETE")
                return
            
            # Show current threshold settings
            logger.info(f"Current threshold settings: mean={cfg.threshold_mean}, std={cfg.threshold_std}")
            logger.info(f"Selection method: {cfg.selection_method}")
            
            # Select best models
            selected = select_models(
                results,
                cfg.selection_method,
                cfg.num_top_models,
                cfg.threshold_mean,
                cfg.threshold_std,
                logger,
            )
            
            if len(selected) == 0:
                logger.error("No models met selection criteria!")
                logger.error("Tip: Run with --stats-only to see error distributions and tune thresholds")
                save_step_status(args.run_dir, "step_2", "failed",
                               {"error": "No models met criteria"})
                return
            
            # Recompute operators
            models_with_ops = recompute_operators(selected, data, cfg.r, logger)
            
            # Save ensemble
            save_ensemble(models_with_ops, paths["ensemble_models"], cfg, logger)
            
            # Save sweep summary
            np.savez(
                paths["sweep_results"],
                n_total=len(results),
                n_selected=len(selected),
                best_error=selected[0]['total_error'] if selected else np.nan,
                worst_selected=selected[-1]['total_error'] if selected else np.nan,
            )
            
            # Print summary
            print_header("MODEL SELECTION SUMMARY")
            print(f"  Total valid models: {len(results)}")
            print(f"  Selected models: {len(selected)}")
            print(f"  Best total error: {selected[0]['total_error']:.6e}")
            print(f"  Worst selected: {selected[-1]['total_error']:.6e}")
            
            print(f"\n  {'Model':>6} | {'Total Err':>10} | {'Mean Γn':>8} | {'Std Γn':>8}")
            print(f"  {'-'*50}")
            for i, params in enumerate(selected[:10]):
                print(f"  {i+1:>6} | {params['total_error']:>10.4e} | "
                      f"{params['mean_err_Gamma_n']:>8.4f} | {params['std_err_Gamma_n']:>8.4f}")
            
            save_step_status(args.run_dir, "step_2", "completed", {
                "n_models": len(selected),
                "best_error": float(selected[0]['total_error']),
                "sweep_time_seconds": elapsed,
            })
            
            print_header("STEP 2 COMPLETE")
            logger.info("Step 2 completed successfully")
    
    except Exception as e:
        if rank == 0:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            save_step_status(args.run_dir, "step_2", "failed", {"error": str(e)})
        raise
    
    finally:
        # Cleanup shared memory windows
        if parallel and shared_windows:
            comm.Barrier()
            for win in shared_windows:
                win.Free()
            if rank == 0:
                logger.info("Shared memory windows freed")


if __name__ == "__main__":
    main()
