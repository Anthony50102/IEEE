"""
ROM training and hyperparameter sweep utilities.

This module handles:
- Hyperparameter evaluation
- Parallel hyperparameter sweep
- Model selection
- Operator computation

Author: Anthony Poole
"""

import gc
import os
import numpy as np
from itertools import product
from mpi4py import MPI

from core import get_quadratic_terms, solve_difference_model


# =============================================================================
# LEARNING MATRIX PREPARATION
# =============================================================================

def prepare_learning_matrices(Xhat_train, train_boundaries, cfg, rank, logger) -> dict:
    """Prepare matrices for ROM training (rank 0 only)."""
    if rank != 0:
        return None
    
    return _prepare_learning_matrices_impl(Xhat_train, train_boundaries, cfg.r, logger)


def prepare_learning_matrices_serial(Xhat_train, train_boundaries, cfg, logger) -> dict:
    """Prepare matrices for ROM training (serial version)."""
    return _prepare_learning_matrices_impl(Xhat_train, train_boundaries, cfg.r, logger)


def _prepare_learning_matrices_impl(Xhat_train, train_boundaries, r, logger) -> dict:
    """Implementation of learning matrix preparation."""
    logger.info("Preparing learning matrices...")
    
    # STATE LEARNING: Create valid pairs within each trajectory
    n_traj = len(train_boundaries) - 1
    X_state_list, Y_state_list = [], []
    
    for traj_idx in range(n_traj):
        start, end = train_boundaries[traj_idx], train_boundaries[traj_idx + 1]
        Xhat_traj = Xhat_train[start:end, :]
        X_state_list.append(Xhat_traj[:-1, :])
        Y_state_list.append(Xhat_traj[1:, :])
    
    X_state = np.vstack(X_state_list)
    Y_state = np.vstack(Y_state_list)
    
    X_state2 = get_quadratic_terms(X_state)
    D_state = np.concatenate([X_state, X_state2], axis=1)
    D_state_2 = D_state.T @ D_state
    
    logger.info(f"  State pairs: {X_state.shape[0]}")
    
    # OUTPUT LEARNING: Use all timesteps
    X_out = Xhat_train
    K = X_out.shape[0]
    
    mean_Xhat = np.mean(X_out, axis=0)
    Xhat_out = X_out - mean_Xhat
    
    scaling_Xhat = max(np.abs(X_out.min()), np.abs(X_out.max()))
    scaling_Xhat = max(scaling_Xhat, 1e-14)
    Xhat_out /= scaling_Xhat
    
    Xhat_out2 = get_quadratic_terms(Xhat_out)
    D_out = np.concatenate([Xhat_out, Xhat_out2, np.ones((K, 1))], axis=1)
    D_out_2 = D_out.T @ D_out
    
    logger.info(f"  D_out shape: {D_out.shape}")
    
    return {
        'X_state': X_state, 'Y_state': Y_state,
        'D_state': D_state, 'D_state_2': D_state_2,
        'D_out': D_out, 'D_out_2': D_out_2,
        'mean_Xhat': mean_Xhat, 'scaling_Xhat': scaling_Xhat,
    }


# =============================================================================
# HYPERPARAMETER EVALUATION
# =============================================================================

def evaluate_hyperparameters(
    alpha_state_lin: float, alpha_state_quad: float,
    alpha_out_lin: float, alpha_out_quad: float,
    data: dict, r: int, n_steps: int, training_end: int,
) -> dict:
    """
    Evaluate a single hyperparameter combination.
    
    Trains the ROM with given regularization and evaluates on training data.
    """
    s = r * (r + 1) // 2
    d_state = r + s
    d_out = r + s + 1
    
    # Solve state operator learning
    reg_state = np.zeros(d_state)
    reg_state[:r] = alpha_state_lin
    reg_state[r:] = alpha_state_quad
    
    DtD_state = data['D_state_2'] + np.diag(reg_state)
    O = np.linalg.solve(DtD_state, data['D_state'].T @ data['Y_state']).T
    A, F = O[:, :r], O[:, r:]
    del DtD_state
    
    # Integrate model
    f = lambda x: A @ x + F @ get_quadratic_terms(x)
    u0 = data['X_state'][0, :]
    is_nan, Xhat_pred = solve_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True, 'alpha_state_lin': alpha_state_lin,
                'alpha_state_quad': alpha_state_quad, 'alpha_out_lin': alpha_out_lin,
                'alpha_out_quad': alpha_out_quad}
    
    X_OpInf = Xhat_pred.T
    del Xhat_pred
    
    # Prepare for output operator
    Xhat_scaled = (X_OpInf - data['mean_Xhat']) / data['scaling_Xhat']
    Xhat_2 = get_quadratic_terms(Xhat_scaled)
    del X_OpInf
    
    # Solve output operator learning
    reg_out = np.zeros(d_out)
    reg_out[:r] = alpha_out_lin
    reg_out[r:r+s] = alpha_out_quad
    reg_out[r+s:] = alpha_out_lin
    
    DtD_out = data['D_out_2'] + np.diag(reg_out)
    O_out = np.linalg.solve(DtD_out, data['D_out'].T @ data['Y_Gamma'].T).T
    C, G, c = O_out[:, :r], O_out[:, r:r+s], O_out[:, r+s]
    
    # Compute output predictions
    Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
    del Xhat_scaled, Xhat_2
    
    ts_Gamma_n = Y_OpInf[0, :training_end]
    ts_Gamma_c = Y_OpInf[1, :training_end]
    
    # Compute error metrics (relative errors in statistics)
    mean_err_n = abs(data['mean_Gamma_n'] - np.mean(ts_Gamma_n)) / abs(data['mean_Gamma_n'])
    std_err_n = abs(data['std_Gamma_n'] - np.std(ts_Gamma_n, ddof=1)) / data['std_Gamma_n']
    mean_err_c = abs(data['mean_Gamma_c'] - np.mean(ts_Gamma_c)) / abs(data['mean_Gamma_c'])
    std_err_c = abs(data['std_Gamma_c'] - np.std(ts_Gamma_c, ddof=1)) / data['std_Gamma_c']
    
    total_error = mean_err_n + std_err_n + mean_err_c + std_err_c
    
    return {
        'is_nan': False,
        'total_error': total_error,
        'mean_err_Gamma_n': mean_err_n, 'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c, 'std_err_Gamma_c': std_err_c,
        'alpha_state_lin': alpha_state_lin, 'alpha_state_quad': alpha_state_quad,
        'alpha_out_lin': alpha_out_lin, 'alpha_out_quad': alpha_out_quad,
    }


def parallel_hyperparameter_sweep(cfg, data: dict, logger, comm) -> list:
    """Run hyperparameter sweep in parallel using MPI."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Build parameter grid
    param_grid = list(product(cfg.state_lin, cfg.state_quad, cfg.output_lin, cfg.output_quad))
    n_total = len(param_grid)
    
    if rank == 0:
        logger.info(f"Parallel sweep: {n_total:,} combinations across {size} ranks")
    
    # Distribute work
    per_rank = n_total // size
    remainder = n_total % size
    
    if rank < remainder:
        start = rank * (per_rank + 1)
        end = start + per_rank + 1
    else:
        start = rank * per_rank + remainder
        end = start + per_rank
    
    my_params = param_grid[start:end]
    
    # Process assigned parameters
    local_results = []
    n_nan = 0
    
    for i, (asl, asq, aol, aoq) in enumerate(my_params):
        result = evaluate_hyperparameters(
            asl, asq, aol, aoq, data, cfg.r, cfg.n_steps, cfg.training_end
        )
        
        if result['is_nan']:
            n_nan += 1
        else:
            local_results.append(result)
        
        if (i + 1) % 50 == 0:
            gc.collect()
    
    # Gather results to rank 0
    all_results = comm.gather(local_results, root=0)
    all_nan = comm.gather(n_nan, root=0)
    
    if rank == 0:
        combined = [r for rank_results in all_results for r in rank_results]
        logger.info(f"Sweep complete: {len(combined)} valid, {sum(all_nan)} NaN")
        return combined
    return []


# =============================================================================
# MODEL SELECTION
# =============================================================================

def log_error_statistics(results: list, logger) -> dict:
    """Log detailed statistics about sweep results."""
    if not results:
        logger.warning("No results to analyze!")
        return {}
    
    errors = {
        'mean_n': np.array([r['mean_err_Gamma_n'] for r in results]),
        'std_n': np.array([r['std_err_Gamma_n'] for r in results]),
        'mean_c': np.array([r['mean_err_Gamma_c'] for r in results]),
        'std_c': np.array([r['std_err_Gamma_c'] for r in results]),
        'total': np.array([r['total_error'] for r in results]),
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("ERROR DISTRIBUTION STATISTICS")
    logger.info("=" * 70)
    
    for name, arr in errors.items():
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        logger.info(f"  {name}: min={arr.min():.4f}, P10={p10:.4f}, "
                   f"P50={p50:.4f}, P90={p90:.4f}, max={arr.max():.4f}")
    
    logger.info("")
    logger.info("TOP 10 MODELS BY TOTAL ERROR:")
    sorted_results = sorted(results, key=lambda x: x['total_error'])[:10]
    for i, r in enumerate(sorted_results):
        logger.info(f"  {i+1}: total={r['total_error']:.4f}, "
                   f"mean_n={r['mean_err_Gamma_n']:.4f}, std_n={r['std_err_Gamma_n']:.4f}")
    
    logger.info("=" * 70)
    return errors


def select_models(results: list, thresh_mean: float, thresh_std: float, logger) -> list:
    """
    Select models meeting threshold criteria.
    
    Models must have all four error metrics below their respective thresholds.
    """
    selected = [
        r for r in results
        if (r['mean_err_Gamma_n'] < thresh_mean and r['std_err_Gamma_n'] < thresh_std
            and r['mean_err_Gamma_c'] < thresh_mean and r['std_err_Gamma_c'] < thresh_std)
    ]
    selected = sorted(selected, key=lambda x: x['total_error'])
    logger.info(f"Selected {len(selected)} models meeting threshold criteria")
    
    return selected


# =============================================================================
# OPERATOR COMPUTATION
# =============================================================================

def compute_operators(params: dict, data: dict, r: int) -> dict:
    """Compute operator matrices for a single hyperparameter set."""
    s = r * (r + 1) // 2
    d_state = r + s
    d_out = r + s + 1
    
    # State operators
    reg_state = np.zeros(d_state)
    reg_state[:r] = params['alpha_state_lin']
    reg_state[r:] = params['alpha_state_quad']
    
    O = np.linalg.solve(
        data['D_state_2'] + np.diag(reg_state),
        data['D_state'].T @ data['Y_state']
    ).T
    A, F = O[:, :r].copy(), O[:, r:].copy()
    
    # Output operators
    reg_out = np.zeros(d_out)
    reg_out[:r] = params['alpha_out_lin']
    reg_out[r:r+s] = params['alpha_out_quad']
    reg_out[r+s:] = params['alpha_out_lin']
    
    O_out = np.linalg.solve(
        data['D_out_2'] + np.diag(reg_out),
        data['D_out'].T @ data['Y_Gamma'].T
    ).T
    C, G, c = O_out[:, :r].copy(), O_out[:, r:r+s].copy(), O_out[:, r+s].copy()
    
    return {
        'A': A, 'F': F, 'C': C, 'G': G, 'c': c,
        'alpha_state_lin': params['alpha_state_lin'],
        'alpha_state_quad': params['alpha_state_quad'],
        'alpha_out_lin': params['alpha_out_lin'],
        'alpha_out_quad': params['alpha_out_quad'],
        'total_error': params['total_error'],
        'mean_err_Gamma_n': params['mean_err_Gamma_n'],
        'std_err_Gamma_n': params['std_err_Gamma_n'],
        'mean_err_Gamma_c': params['mean_err_Gamma_c'],
        'std_err_Gamma_c': params['std_err_Gamma_c'],
    }


def recompute_operators_parallel(selected: list, data: dict, r: int, 
                                  operators_dir: str, comm, logger) -> list:
    """Recompute operator matrices for selected models in parallel."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_models = len(selected)
    
    if rank == 0:
        logger.info(f"Recomputing operators for {n_models} models...")
        os.makedirs(operators_dir, exist_ok=True)
    
    comm.Barrier()
    
    # Distribute work
    per_rank = n_models // size
    remainder = n_models % size
    
    if rank < remainder:
        start = rank * (per_rank + 1)
        end = start + per_rank + 1
    else:
        start = rank * per_rank + remainder
        end = start + per_rank
    
    # Compute and save models
    local_results = []
    for idx in range(start, end):
        model = compute_operators(selected[idx], data, r)
        filepath = os.path.join(operators_dir, f"model_{idx:04d}.npz")
        np.savez(filepath, **model)
        local_results.append((model['total_error'], model))
    
    # Gather results
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        combined = sorted(
            [r for rank_results in all_results for r in rank_results],
            key=lambda x: x[0]
        )
        logger.info(f"  Saved {len(combined)} models to {operators_dir}")
        return combined
    return []
