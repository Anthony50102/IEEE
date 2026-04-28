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

# Lazy MPI import - only import when actually needed for parallel functions
MPI = None
def _get_mpi():
    global MPI
    if MPI is None:
        from mpi4py import MPI as _MPI
        MPI = _MPI
    return MPI

from core import get_quadratic_terms, solve_difference_model, project_to_stable


# =============================================================================
# LEARNING MATRIX PREPARATION
# =============================================================================

def prepare_learning_matrices(Xhat_train, train_boundaries, cfg, rank, logger) -> dict:
    """Prepare matrices for ROM training (rank 0 only)."""
    if rank != 0:
        return None
    
    return _prepare_learning_matrices_impl(
        Xhat_train, train_boundaries, cfg.r, logger,
        closure_enabled=getattr(cfg, 'closure_enabled', False),
        closure_cubic=getattr(cfg, 'closure_cubic', True),
        closure_constant=getattr(cfg, 'closure_constant', True),
    )


def prepare_learning_matrices_serial(Xhat_train, train_boundaries, cfg, logger) -> dict:
    """Prepare matrices for ROM training (serial version)."""
    return _prepare_learning_matrices_impl(
        Xhat_train, train_boundaries, cfg.r, logger,
        closure_enabled=getattr(cfg, 'closure_enabled', False),
        closure_cubic=getattr(cfg, 'closure_cubic', True),
        closure_constant=getattr(cfg, 'closure_constant', True),
    )


def _prepare_learning_matrices_impl(Xhat_train, train_boundaries, r, logger,
                                     closure_enabled=False, closure_cubic=True,
                                     closure_constant=True) -> dict:
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
    
    # Build state data matrix: [linear | quadratic | cubic_diag? | constant?]
    include_cubic = closure_enabled and closure_cubic
    include_constant = closure_enabled and closure_constant
    
    X_state2 = get_quadratic_terms(X_state)
    parts = [X_state, X_state2]
    if include_cubic:
        from core import get_cubic_diagonal_terms
        parts.append(get_cubic_diagonal_terms(X_state))
    if include_constant:
        parts.append(np.ones((X_state.shape[0], 1)))
    
    D_state = np.concatenate(parts, axis=1)
    D_state_2 = D_state.T @ D_state
    
    logger.info(f"  State pairs: {X_state.shape[0]}")
    logger.info(f"  D_state columns: {D_state.shape[1]} (closure={closure_enabled})")
    
    # OUTPUT LEARNING: Use all timesteps (unchanged)
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
        'closure_enabled': closure_enabled,
        'include_cubic': include_cubic,
        'include_constant': include_constant,
    }


# =============================================================================
# HYPERPARAMETER EVALUATION
# =============================================================================

def evaluate_hyperparameters(
    alpha_state_lin: float, alpha_state_quad: float,
    alpha_out_lin: float, alpha_out_quad: float,
    data: dict, r: int, n_steps: int, training_end: int,
    alpha_state_cubic: float = 0.0,
    stability_projection: bool = False,
    stability_max_rho: float = 0.999,
    test_IC: np.ndarray = None,
    n_test_steps: int = 0,
    test_ref_energy: np.ndarray = None,
    test_ref_reduced: np.ndarray = None,
) -> dict:
    """
    Evaluate a single hyperparameter combination.
    
    Trains the ROM with given regularization and evaluates on training data.
    Optionally rolls out past training onto a test window to assess stability.
    
    If data contains 'physics_energy' precomputes (energy_a, energy_b, energy_N,
    ref_energy_mean, ref_energy_std), energy is computed analytically from POD
    coefficients instead of via the learned output model. This avoids the output
    model masking dissipation in the state dynamics.
    
    Parameters (test-time rollout, all optional):
        test_IC : initial condition in reduced space for the test window.
        n_test_steps : number of steps to roll out past training.
        test_ref_energy : reference energy array for the test period.
    """
    include_cubic = data.get('include_cubic', False)
    include_constant = data.get('include_constant', False)
    use_physics_energy = 'energy_a' in data
    
    s = r * (r + 1) // 2
    d_state = r + s
    if include_cubic:
        d_state += r
    if include_constant:
        d_state += 1
    d_out = r + s + 1
    
    # Build regularization vector for state operator
    reg_state = np.zeros(d_state)
    col = 0
    reg_state[col:col + r] = alpha_state_lin
    col += r
    reg_state[col:col + s] = alpha_state_quad
    col += s
    if include_cubic:
        reg_state[col:col + r] = alpha_state_cubic
        col += r
    if include_constant:
        reg_state[col:col + 1] = alpha_state_lin
        col += 1
    
    DtD_state = data['D_state_2'] + np.diag(reg_state)
    O = np.linalg.solve(DtD_state, data['D_state'].T @ data['Y_state']).T
    
    # Extract operators
    col = 0
    A = O[:, col:col + r]
    col += r
    F = O[:, col:col + s]
    col += s
    H = O[:, col:col + r] if include_cubic else None
    if include_cubic:
        col += r
    c_state = O[:, col] if include_constant else None
    del DtD_state
    
    # Apply stability projection to A if enabled
    if stability_projection:
        A = project_to_stable(A, stability_max_rho)
    
    # Build state transition function with closure terms
    def f(x):
        result = A @ x + F @ get_quadratic_terms(x)
        if H is not None:
            from core import get_cubic_diagonal_terms
            result += H @ get_cubic_diagonal_terms(x)
        if c_state is not None:
            result += c_state
        return result
    
    u0 = data['X_state'][0, :]
    is_nan, Xhat_pred = solve_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True, 'alpha_state_lin': alpha_state_lin,
                'alpha_state_quad': alpha_state_quad, 'alpha_out_lin': alpha_out_lin,
                'alpha_out_quad': alpha_out_quad, 'alpha_state_cubic': alpha_state_cubic}
    
    X_OpInf = Xhat_pred.T  # (n_steps, r)
    del Xhat_pred
    
    if use_physics_energy:
        # Compute energy analytically from POD coefficients:
        # E_k = 0.5/N * (||x_hat_k||^2 + 2*b^T*x_hat_k + a)
        # where a=||u_mean||^2, b=Ur^T*u_mean, N=spatial points
        energy_a = data['energy_a']
        energy_b = data['energy_b']
        energy_N = data['energy_N']
        
        X_eval = X_OpInf[:training_end, :]
        norms_sq = np.sum(X_eval ** 2, axis=1)
        cross = X_eval @ energy_b
        ts_energy = 0.5 / energy_N * (norms_sq + 2.0 * cross + energy_a)
        
        mean_err_n = abs(data['ref_energy_mean'] - np.mean(ts_energy)) / abs(data['ref_energy_mean'])
        std_err_n = abs(data['ref_energy_std'] - np.std(ts_energy, ddof=1)) / data['ref_energy_std']
        # Enstrophy not computed in physics mode — set to 0 (disabled via thresholds)
        mean_err_c = 0.0
        std_err_c = 0.0
        total_error = mean_err_n + std_err_n
    else:
        # Use learned output model (original path)
        Xhat_scaled = (X_OpInf - data['mean_Xhat']) / data['scaling_Xhat']
        Xhat_2 = get_quadratic_terms(Xhat_scaled)
        
        reg_out = np.zeros(d_out)
        reg_out[:r] = alpha_out_lin
        reg_out[r:r+s] = alpha_out_quad
        reg_out[r+s:] = alpha_out_lin
        
        DtD_out = data['D_out_2'] + np.diag(reg_out)
        O_out = np.linalg.solve(DtD_out, data['D_out'].T @ data['Y_Gamma'].T).T
        C, G, c = O_out[:, :r], O_out[:, r:r+s], O_out[:, r+s]
        
        Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
        del Xhat_scaled, Xhat_2
        
        ts_Gamma_n = Y_OpInf[0, :training_end]
        ts_Gamma_c = Y_OpInf[1, :training_end]
        
        mean_err_n = abs(data['mean_Gamma_n'] - np.mean(ts_Gamma_n)) / abs(data['mean_Gamma_n'])
        std_err_n = abs(data['std_Gamma_n'] - np.std(ts_Gamma_n, ddof=1)) / data['std_Gamma_n']
        mean_err_c = abs(data['mean_Gamma_c'] - np.mean(ts_Gamma_c)) / abs(data['mean_Gamma_c'])
        std_err_c = abs(data['std_Gamma_c'] - np.std(ts_Gamma_c, ddof=1)) / data['std_Gamma_c']
        total_error = mean_err_n + std_err_n + mean_err_c + std_err_c
    
    # Compute training trajectory MSE in reduced space
    X_ref_train = data['X_state'][:training_end, :]   # (training_end, r)
    X_pred_train = X_OpInf[:training_end, :]
    n_compare_train = min(len(X_ref_train), len(X_pred_train))
    train_mse = float(np.mean((X_pred_train[:n_compare_train] - X_ref_train[:n_compare_train]) ** 2))
    total_error = train_mse
    
    del X_OpInf
    
    # --- Test-time rollout stability check ---
    # Roll out past training to detect instability. Unstable models (NaN)
    # get total_error = inf. Stable models keep training MSE as total_error.
    test_stable = True
    test_mse = 0.0
    test_energy_err = 0.0

    if test_IC is not None and n_test_steps > 0:
        is_nan_test, Xhat_test_pred = solve_difference_model(test_IC, n_test_steps, f)
        if is_nan_test:
            test_stable = False
            test_mse = float('inf')
            test_energy_err = float('inf')
            total_error = float('inf')
        else:
            X_pred = Xhat_test_pred.T  # (n_test_steps, r)
            del Xhat_test_pred

            if test_ref_reduced is not None:
                n_compare = min(len(X_pred), len(test_ref_reduced))
                test_mse = float(np.mean((X_pred[:n_compare] - test_ref_reduced[:n_compare]) ** 2))

            # Compute test energy error for diagnostics (not used for selection)
            if use_physics_energy and test_ref_energy is not None:
                energy_a = data['energy_a']
                energy_b = data['energy_b']
                energy_N = data['energy_N']
                norms_sq = np.sum(X_pred ** 2, axis=1)
                cross = X_pred @ energy_b
                test_energy = 0.5 / energy_N * (norms_sq + 2.0 * cross + energy_a)
                ref_mean = float(np.mean(test_ref_energy))
                pred_mean = float(np.mean(test_energy))
                test_energy_err = abs(pred_mean - ref_mean) / abs(ref_mean) if ref_mean != 0 else 0.0

            del X_pred

    return {
        'is_nan': False,
        'total_error': total_error,
        'train_mse': train_mse,
        'mean_err_Gamma_n': mean_err_n, 'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c, 'std_err_Gamma_c': std_err_c,
        'alpha_state_lin': alpha_state_lin, 'alpha_state_quad': alpha_state_quad,
        'alpha_out_lin': alpha_out_lin, 'alpha_out_quad': alpha_out_quad,
        'alpha_state_cubic': alpha_state_cubic,
        'test_stable': test_stable,
        'test_mse': test_mse,
        'test_energy_err': test_energy_err,
    }


def parallel_hyperparameter_sweep(cfg, data: dict, logger, comm) -> list:
    """Run hyperparameter sweep in parallel using MPI."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Build parameter grid — include cubic if closure is enabled
    include_cubic = data.get('include_cubic', False)
    if include_cubic and len(cfg.state_cubic) > 0:
        param_grid = list(product(cfg.state_lin, cfg.state_quad, cfg.output_lin, cfg.output_quad, cfg.state_cubic))
    else:
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
    
    for i, params in enumerate(my_params):
        if len(params) == 5:
            asl, asq, aol, aoq, asc = params
        else:
            asl, asq, aol, aoq = params
            asc = 0.0
        
        result = evaluate_hyperparameters(
            asl, asq, aol, aoq, data, cfg.r, cfg.n_steps, cfg.training_end,
            alpha_state_cubic=asc,
            stability_projection=getattr(cfg, 'stability_projection', False),
            stability_max_rho=getattr(cfg, 'stability_max_rho', 0.999),
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


def select_models(results: list, thresh_mean: float, thresh_std: float, logger,
                  thresh_mean_c: float = 0.0, thresh_std_c: float = 0.0) -> list:
    """
    Select models meeting threshold criteria.
    
    Models must have error metrics below their respective thresholds.
    If thresh_mean_c/thresh_std_c are 0, they default to thresh_mean/thresh_std.
    Set them to large values (e.g., 100.0) to disable Gamma_c filtering.
    """
    tm_c = thresh_mean_c if thresh_mean_c > 0 else thresh_mean
    ts_c = thresh_std_c if thresh_std_c > 0 else thresh_std
    
    selected = [
        r for r in results
        if (r['mean_err_Gamma_n'] < thresh_mean and r['std_err_Gamma_n'] < thresh_std
            and r['mean_err_Gamma_c'] < tm_c and r['std_err_Gamma_c'] < ts_c)
    ]
    selected = sorted(selected, key=lambda x: x['total_error'])
    logger.info(f"Selected {len(selected)} models meeting threshold criteria "
                f"(mean_n<{thresh_mean}, std_n<{thresh_std}, mean_c<{tm_c}, std_c<{ts_c})")
    
    return selected


# =============================================================================
# OPERATOR COMPUTATION
# =============================================================================

def compute_operators(params: dict, data: dict, r: int,
                      stability_projection: bool = False,
                      stability_max_rho: float = 0.999) -> dict:
    """Compute operator matrices for a single hyperparameter set."""
    include_cubic = data.get('include_cubic', False)
    include_constant = data.get('include_constant', False)
    
    s = r * (r + 1) // 2
    d_state = r + s
    if include_cubic:
        d_state += r
    if include_constant:
        d_state += 1
    d_out = r + s + 1
    
    # State operators
    reg_state = np.zeros(d_state)
    col = 0
    reg_state[col:col + r] = params['alpha_state_lin']
    col += r
    reg_state[col:col + s] = params['alpha_state_quad']
    col += s
    if include_cubic:
        reg_state[col:col + r] = params.get('alpha_state_cubic', 0.0)
        col += r
    if include_constant:
        reg_state[col:col + 1] = params['alpha_state_lin']
        col += 1
    
    O = np.linalg.solve(
        data['D_state_2'] + np.diag(reg_state),
        data['D_state'].T @ data['Y_state']
    ).T
    
    col = 0
    A = O[:, col:col + r].copy()
    col += r
    F = O[:, col:col + s].copy()
    col += s
    H = O[:, col:col + r].copy() if include_cubic else None
    if include_cubic:
        col += r
    c_state = O[:, col].copy() if include_constant else None
    
    # Apply stability projection to A if enabled
    if stability_projection:
        A = project_to_stable(A, stability_max_rho)
    
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
    
    result = {
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
        'closure_enabled': include_cubic or include_constant,
    }
    if H is not None:
        result['H'] = H
        result['alpha_state_cubic'] = params.get('alpha_state_cubic', 0.0)
    if c_state is not None:
        result['c_state'] = c_state
    
    return result


def recompute_operators_parallel(selected: list, data: dict, r: int, 
                                  operators_dir: str, comm, logger,
                                  stability_projection: bool = False,
                                  stability_max_rho: float = 0.999) -> list:
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
        model = compute_operators(selected[idx], data, r,
                                  stability_projection=stability_projection,
                                  stability_max_rho=stability_max_rho)
        filepath = os.path.join(operators_dir, f"model_{idx:04d}.npz")
        np.savez(filepath, **model)
        # Only store metadata (error and index) to avoid MPI size overflow
        local_results.append((model['total_error'], idx))
    
    # Gather only lightweight metadata (avoids MPI 2GB limit)
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Sort by error and reload models from disk
        combined_meta = sorted(
            [r for rank_results in all_results for r in rank_results],
            key=lambda x: x[0]
        )
        # Reload full models from saved files
        combined = []
        for total_error, idx in combined_meta:
            filepath = os.path.join(operators_dir, f"model_{idx:04d}.npz")
            model = dict(np.load(filepath))
            combined.append((total_error, model))
        logger.info(f"  Saved {len(combined)} models to {operators_dir}")
        return combined
    return []
