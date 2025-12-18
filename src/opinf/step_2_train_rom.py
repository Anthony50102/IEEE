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
    A = O[:, :r]
    F = O[:, r:r + s]
    
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
    
    # Prepare for output operator
    Xhat_OpInf_scaled = (X_OpInf - mean_Xhat[np.newaxis, :]) / scaling_Xhat
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
# PARALLEL SWEEP (MPI)
# =============================================================================

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
    
    for asl, asq, aol, aoq in my_params:
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
    
    try:
        # Load pre-computed data
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


if __name__ == "__main__":
    main()
