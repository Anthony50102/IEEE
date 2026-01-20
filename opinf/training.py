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
import sys
import numpy as np
from itertools import product

# Add parent directory to path for importing shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Lazy MPI import - only import when actually needed for parallel functions
MPI = None
def _get_mpi():
    global MPI
    if MPI is None:
        from mpi4py import MPI as _MPI
        MPI = _MPI
    return MPI

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
# HYPERPARAMETER EVALUATION (STATE ONLY - NO OUTPUT OPERATORS)
# =============================================================================

def evaluate_hyperparameters_state_only(
    alpha_state_lin: float, alpha_state_quad: float,
    data: dict, r: int, n_steps: int, training_end: int,
) -> dict:
    """
    Evaluate a single hyperparameter combination (state operators only).
    
    Uses state reconstruction error instead of Gamma error for model selection.
    This is useful when:
    - You don't want to learn output operators (Gamma will be computed from physics)
    - You want faster training (fewer hyperparameters to sweep)
    
    Error metric: relative Frobenius norm of state prediction error over training region.
    """
    s = r * (r + 1) // 2
    d_state = r + s
    
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
                'alpha_state_quad': alpha_state_quad, 'alpha_out_lin': 0.0,
                'alpha_out_quad': 0.0}
    
    X_pred = Xhat_pred.T  # (n_steps, r)
    
    # Compute state prediction error on training portion
    # Compare against the reference reduced states
    # Note: X_state may be smaller than n_steps due to trajectory pair formation
    n_ref = min(training_end, data['X_state'].shape[0])
    X_ref = data['X_state'][:n_ref, :]  # (n_ref, r)
    X_pred_train = X_pred[:n_ref, :]
    
    # Relative Frobenius norm error
    state_error = np.linalg.norm(X_pred_train - X_ref, 'fro') / np.linalg.norm(X_ref, 'fro')
    
    # Also check for stability (is the solution bounded?)
    max_norm = np.max(np.linalg.norm(X_pred, axis=1))
    ref_max_norm = np.max(np.linalg.norm(X_ref, axis=1))
    stability_ratio = max_norm / ref_max_norm if ref_max_norm > 0 else np.inf
    
    # Penalize unstable solutions heavily
    if stability_ratio > 10.0:
        state_error = state_error * stability_ratio
    
    del X_pred, Xhat_pred
    
    return {
        'is_nan': False,
        'total_error': state_error,
        'state_error': state_error,
        'stability_ratio': stability_ratio,
        # Dummy Gamma errors for compatibility with model selection
        'mean_err_Gamma_n': 0.0, 'std_err_Gamma_n': 0.0,
        'mean_err_Gamma_c': 0.0, 'std_err_Gamma_c': 0.0,
        'alpha_state_lin': alpha_state_lin, 'alpha_state_quad': alpha_state_quad,
        'alpha_out_lin': 0.0, 'alpha_out_quad': 0.0,
    }


# =============================================================================
# HYPERPARAMETER EVALUATION (PHYSICS-BASED GAMMA - NO OUTPUT OPERATORS)
# =============================================================================

def evaluate_hyperparameters_physics_gamma(
    alpha_state_lin: float, alpha_state_quad: float,
    data: dict, physics_data: dict, r: int, n_steps: int, training_end: int,
) -> dict:
    """
    Evaluate hyperparameters using physics-based Gamma computation.
    
    This evaluates state operators only, but computes Gamma from physics
    for model selection (using the same mean/std error metrics as learned output).
    
    This is slower than state-only evaluation because it requires:
    1. Reconstructing full state from reduced coordinates
    2. Computing Gamma from the physics (gradients, fluxes)
    
    Parameters
    ----------
    alpha_state_lin, alpha_state_quad : float
        State operator regularization.
    data : dict
        Training data (D_state, D_state_2, Y_state, X_state, etc.).
    physics_data : dict
        Physics reconstruction data:
        - 'pod_basis': (n_spatial, r) POD basis matrix
        - 'temporal_mean': (n_spatial,) temporal mean (or None if not centered)
        - 'n_y', 'n_x': grid dimensions
        - 'k0', 'c1': physics parameters
        - 'ref_Gamma_n', 'ref_Gamma_c': reference Gamma time series
        - 'mean_Gamma_n', 'std_Gamma_n', 'mean_Gamma_c', 'std_Gamma_c': reference statistics
    r : int
        Number of modes.
    n_steps : int
        Number of integration steps.
    training_end : int
        End of training period for metrics.
    
    Returns
    -------
    dict
        Evaluation results with Gamma errors computed from physics.
    """
    from shared.physics import compute_gamma_from_state_vector
    
    s = r * (r + 1) // 2
    d_state = r + s
    
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
                'alpha_state_quad': alpha_state_quad, 'alpha_out_lin': 0.0,
                'alpha_out_quad': 0.0}
    
    X_pred = Xhat_pred.T  # (n_steps, r)
    
    # Extract physics data
    pod_basis = physics_data['pod_basis']  # (n_spatial, r)
    temporal_mean = physics_data.get('temporal_mean')  # (n_spatial,) or None
    n_y, n_x = physics_data['n_y'], physics_data['n_x']
    k0, c1 = physics_data['k0'], physics_data['c1']
    dx = 2 * np.pi / k0
    
    # Reconstruct full state and compute Gamma for training region
    # Process one timestep at a time to minimize memory footprint
    # (full-space arrays are large: 512×512×2 = 524,288 elements = 4 MB float64)
    Gamma_n_list = []
    Gamma_c_list = []
    
    # Precompute the reconstruction matrix slice (small: r columns)
    V_r = pod_basis[:, :r]  # (n_spatial, r)
    
    # Clamp training_end to available predictions
    n_eval = min(training_end, X_pred.shape[0])
    
    for t in range(n_eval):
        z_t = X_pred[t, :]  # (r,)
        
        # Reconstruct full state: x = V @ z + mean (creates single 524k vector)
        x_full = V_r @ z_t  # (n_spatial,)
        if temporal_mean is not None:
            x_full += temporal_mean
        
        g_n, g_c = compute_gamma_from_state_vector(x_full, n_y, n_x, dx, c1)
        Gamma_n_list.append(g_n)
        Gamma_c_list.append(g_c)
        
        # x_full is freed automatically each loop iteration
    
    del X_pred, Xhat_pred
    
    Gamma_n_pred = np.array(Gamma_n_list)
    Gamma_c_pred = np.array(Gamma_c_list)
    
    # Compute error metrics (relative errors in statistics)
    mean_err_n = abs(physics_data['mean_Gamma_n'] - np.mean(Gamma_n_pred)) / abs(physics_data['mean_Gamma_n'])
    std_err_n = abs(physics_data['std_Gamma_n'] - np.std(Gamma_n_pred, ddof=1)) / physics_data['std_Gamma_n']
    mean_err_c = abs(physics_data['mean_Gamma_c'] - np.mean(Gamma_c_pred)) / abs(physics_data['mean_Gamma_c'])
    std_err_c = abs(physics_data['std_Gamma_c'] - np.std(Gamma_c_pred, ddof=1)) / physics_data['std_Gamma_c']
    
    total_error = mean_err_n + std_err_n + mean_err_c + std_err_c
    
    return {
        'is_nan': False,
        'total_error': total_error,
        'mean_err_Gamma_n': mean_err_n, 'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c, 'std_err_Gamma_c': std_err_c,
        'alpha_state_lin': alpha_state_lin, 'alpha_state_quad': alpha_state_quad,
        'alpha_out_lin': 0.0, 'alpha_out_quad': 0.0,
    }


# =============================================================================
# HYPERPARAMETER EVALUATION (WITH OUTPUT OPERATORS)
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


# =============================================================================
# MANIFOLD-AWARE HYPERPARAMETER EVALUATION
# =============================================================================

def _quadratic_features_batch(z: np.ndarray) -> np.ndarray:
    """
    Compute quadratic features for a batch of vectors.
    
    Parameters
    ----------
    z : np.ndarray, shape (r, n_time) or (n_time, r)
        Reduced coordinates.
    
    Returns
    -------
    np.ndarray, shape (s, n_time) or (n_time, s)
        Quadratic features where s = r*(r+1)/2.
    """
    if z.ndim == 1:
        r = z.shape[0]
        return np.concatenate([z[i:i+1] * z[:i+1] for i in range(r)], axis=0)
    
    # Batch case: assume shape (r, n_time)
    r = z.shape[0]
    features = [z[i:i+1, :] * z[:i+1, :] for i in range(r)]
    return np.concatenate(features, axis=0)


def _manifold_decode(z: np.ndarray, V: np.ndarray, W: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Decode reduced coordinates to full state using quadratic manifold.
    
    x = V @ z + W @ h(z) + shift
    
    Parameters
    ----------
    z : np.ndarray, shape (r,) or (r, n_time)
        Reduced coordinates.
    V : np.ndarray, shape (n_spatial, r)
        Linear basis.
    W : np.ndarray, shape (n_spatial, s)
        Quadratic coefficient matrix.
    shift : np.ndarray, shape (n_spatial,)
        Mean shift.
    
    Returns
    -------
    np.ndarray, shape (n_spatial,) or (n_spatial, n_time)
        Reconstructed full state.
    """
    h_z = _quadratic_features_batch(z)
    if z.ndim == 1:
        return V @ z + W @ h_z + shift
    else:
        return V @ z + W @ h_z + shift[:, None]


def _manifold_encode_decode_chunked(
    z: np.ndarray, V: np.ndarray, W: np.ndarray, shift: np.ndarray,
    chunk_size: int = 100
) -> np.ndarray:
    """
    Encode-then-decode reduced coordinates using chunked processing.
    
    This function decodes z to full space and re-encodes to get z_corrected,
    but processes in chunks to avoid memory issues with large spatial dimensions.
    
    z_corrected = V.T @ (V @ z + W @ h(z))  (shift cancels out)
    
    Parameters
    ----------
    z : np.ndarray, shape (r, n_time)
        Reduced coordinates.
    V : np.ndarray, shape (n_spatial, r)
        Linear basis.
    W : np.ndarray, shape (n_spatial, s)
        Quadratic coefficient matrix.
    shift : np.ndarray, shape (n_spatial,)
        Mean shift (unused, kept for API consistency).
    chunk_size : int
        Number of time steps to process at once.
    
    Returns
    -------
    np.ndarray, shape (r, n_time)
        Re-encoded coordinates.
    """
    r, n_time = z.shape
    z_reencoded = np.zeros_like(z)
    
    # Precompute V.T @ V (r x r, small) and V.T @ W (r x s, small)
    VtV = V.T @ V  # (r, r)
    VtW = V.T @ W  # (r, s)
    
    for start in range(0, n_time, chunk_size):
        end = min(start + chunk_size, n_time)
        z_chunk = z[:, start:end]
        
        # Compute h(z) for this chunk
        h_z_chunk = _quadratic_features_batch(z_chunk)  # (s, chunk)
        
        # z_reencoded = V.T @ (V @ z + W @ h(z)) = VtV @ z + VtW @ h(z)
        # This avoids materializing the full (n_spatial, chunk) array
        z_reencoded[:, start:end] = VtV @ z_chunk + VtW @ h_z_chunk
    
    return z_reencoded


def _manifold_consistency_error_chunked(
    z: np.ndarray, V: np.ndarray, W: np.ndarray, shift: np.ndarray,
    chunk_size: int = 100
) -> float:
    """
    Compute manifold consistency error using chunked processing.
    
    Measures ||z - encode(decode(z))|| / ||z|| without materializing full-space arrays.
    
    Parameters
    ----------
    z : np.ndarray, shape (r, n_time)
        Reduced coordinates to check.
    V : np.ndarray, shape (n_spatial, r)
        Linear basis.
    W : np.ndarray, shape (n_spatial, s)
        Quadratic coefficient matrix.
    shift : np.ndarray, shape (n_spatial,)
        Mean shift.
    chunk_size : int
        Number of time steps to process at once.
    
    Returns
    -------
    float
        Relative consistency error.
    """
    z_reencoded = _manifold_encode_decode_chunked(z, V, W, shift, chunk_size)
    return np.linalg.norm(z_reencoded - z, 'fro') / np.linalg.norm(z, 'fro')


def _manifold_encode(x: np.ndarray, V: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Encode full state to reduced coordinates (linear projection only).
    
    z = V.T @ (x - shift)
    
    Parameters
    ----------
    x : np.ndarray, shape (n_spatial,) or (n_spatial, n_time)
        Full state.
    V : np.ndarray, shape (n_spatial, r)
        Linear basis.
    shift : np.ndarray, shape (n_spatial,)
        Mean shift.
    
    Returns
    -------
    np.ndarray, shape (r,) or (r, n_time)
        Reduced coordinates.
    """
    if x.ndim == 1:
        return V.T @ (x - shift)
    else:
        return V.T @ (x - shift[:, None])


def evaluate_hyperparameters_manifold(
    alpha_state_lin: float, alpha_state_quad: float,
    alpha_out_lin: float, alpha_out_quad: float,
    data: dict, manifold_data: dict, r: int, n_steps: int, training_end: int,
    consistency_weight: float = 1.0, reencode_output: bool = True,
) -> dict:
    """
    Evaluate hyperparameters with manifold-aware training.
    
    This extends the standard evaluation with:
    1. Consistency loss: penalizes predictions that don't decode well in full space
    2. Re-encode for output: decodes prediction, re-encodes, then computes output
    
    The consistency loss measures how well predicted reduced states 
    can be decoded to full space and re-encoded:
        ||z_pred - encode(decode(z_pred))||
    
    For a perfect manifold approximation, this should be small.
    For predictions that stray from the training manifold, this grows.
    
    Parameters
    ----------
    alpha_state_lin, alpha_state_quad : float
        State operator regularization.
    alpha_out_lin, alpha_out_quad : float
        Output operator regularization.
    data : dict
        Training data (same as standard evaluate_hyperparameters).
    manifold_data : dict
        Manifold basis: {'V': array, 'W': array, 'shift': array, 'r': int}.
    r : int
        Number of modes.
    n_steps : int
        Number of integration steps.
    training_end : int
        End of training period for metrics.
    consistency_weight : float
        Weight for manifold consistency error in total error.
    reencode_output : bool
        If True, decode→re-encode before computing output operator prediction.
    
    Returns
    -------
    dict
        Evaluation results including manifold consistency error.
    """
    s = r * (r + 1) // 2
    d_state = r + s
    d_out = r + s + 1
    
    # Extract manifold basis
    V = manifold_data['V']
    W = manifold_data['W']
    shift = manifold_data['shift']
    
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
    
    # Xhat_pred is (r, n_steps)
    
    # =========================================================================
    # MANIFOLD CONSISTENCY ERROR
    # =========================================================================
    # Decode to full space, then re-encode
    # This measures how well predictions stay on the learned manifold
    # Use chunked processing to avoid memory issues with large spatial dimensions
    
    # Sample subset for efficiency (full consistency check is expensive)
    n_check = min(500, n_steps)
    check_indices = np.linspace(0, n_steps - 1, n_check, dtype=int)
    z_subset = Xhat_pred[:, check_indices]
    
    # Compute consistency error using chunked processing (avoids full-space allocation)
    consistency_err = _manifold_consistency_error_chunked(z_subset, V, W, shift, chunk_size=50)
    
    # =========================================================================
    # OUTPUT PREDICTION (with optional re-encoding)
    # =========================================================================
    X_OpInf = Xhat_pred.T  # (n_steps, r)
    
    if reencode_output:
        # Use re-encoded coordinates for output prediction
        # This makes the quadratic manifold structure influence the output
        # Use chunked processing to avoid memory issues with large spatial dimensions
        z_corrected = _manifold_encode_decode_chunked(Xhat_pred, V, W, shift, chunk_size=100)
        X_for_output = z_corrected.T  # (n_steps, r)
    else:
        X_for_output = X_OpInf
    
    del Xhat_pred
    
    # Prepare for output operator
    Xhat_scaled = (X_for_output - data['mean_Xhat']) / data['scaling_Xhat']
    Xhat_2 = get_quadratic_terms(Xhat_scaled)
    del X_for_output, X_OpInf
    
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
    
    # Total error includes weighted manifold consistency
    gamma_error = mean_err_n + std_err_n + mean_err_c + std_err_c
    total_error = gamma_error + consistency_weight * consistency_err
    
    return {
        'is_nan': False,
        'total_error': total_error,
        'gamma_error': gamma_error,
        'manifold_consistency_err': consistency_err,
        'mean_err_Gamma_n': mean_err_n, 'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c, 'std_err_Gamma_c': std_err_c,
        'alpha_state_lin': alpha_state_lin, 'alpha_state_quad': alpha_state_quad,
        'alpha_out_lin': alpha_out_lin, 'alpha_out_quad': alpha_out_quad,
    }


def parallel_hyperparameter_sweep(cfg, data: dict, logger, comm, 
                                   manifold_data: dict = None,
                                   physics_data: dict = None) -> list:
    """
    Run hyperparameter sweep in parallel using MPI.
    
    If manifold_data is provided and cfg.manifold_aware_training is True,
    uses manifold-aware evaluation that includes:
    - Consistency loss for staying on manifold
    - Optional decode→re-encode for output prediction
    
    Selection metric (cfg.selection_metric):
    - "gamma_learned": Use learned output operators (default, requires sweeping output params)
    - "gamma_physics": Compute Gamma from physics (requires physics_data, no output params)
    - "state_error": Use state reconstruction error (fast, no output params)
    
    Parameters
    ----------
    cfg : OpInfConfig
        Configuration object.
    data : dict
        Training data matrices.
    logger : Logger
        Logger instance.
    comm : MPI.Comm
        MPI communicator.
    manifold_data : dict, optional
        Manifold basis for manifold-aware training.
    physics_data : dict, optional
        Physics reconstruction data for gamma_physics mode:
        - 'pod_basis': (n_spatial, r) POD basis
        - 'temporal_mean': (n_spatial,) or None
        - 'n_y', 'n_x', 'k0', 'c1': grid and physics params
        - 'mean_Gamma_n', 'std_Gamma_n', 'mean_Gamma_c', 'std_Gamma_c': reference stats
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Determine evaluation mode from selection_metric
    selection_metric = getattr(cfg, 'selection_metric', 'gamma_learned')
    use_learned_output = getattr(cfg, 'use_learned_output', True)
    
    # Determine if we should use manifold-aware training
    use_manifold = (
        manifold_data is not None 
        and cfg.reduction_method == "manifold"
        and getattr(cfg, 'manifold_aware_training', True)
        and selection_metric == "gamma_learned"  # Only with learned output
    )
    
    # Determine evaluation function to use
    if selection_metric == "state_error":
        eval_mode = "state_only"
    elif selection_metric == "gamma_physics":
        if physics_data is None:
            raise ValueError("selection_metric='gamma_physics' requires physics_data")
        eval_mode = "physics_gamma"
    elif selection_metric == "gamma_learned":
        eval_mode = "manifold" if use_manifold else "learned_output"
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    
    # Build parameter grid - only need output params for learned_output mode
    if eval_mode in ("state_only", "physics_gamma"):
        # Only sweep over state hyperparameters (much smaller grid)
        param_grid = [(asl, asq, 0.0, 0.0) for asl, asq in product(cfg.state_lin, cfg.state_quad)]
    else:
        param_grid = list(product(cfg.state_lin, cfg.state_quad, cfg.output_lin, cfg.output_quad))
    n_total = len(param_grid)
    
    if rank == 0:
        if eval_mode == "state_only":
            logger.info(f"Parallel sweep (STATE ERROR selection): {n_total:,} combinations across {size} ranks")
        elif eval_mode == "physics_gamma":
            logger.info(f"Parallel sweep (PHYSICS GAMMA selection): {n_total:,} combinations across {size} ranks")
            logger.info(f"  Note: Gamma computed from physics (slower, but no output operators)")
        elif eval_mode == "manifold":
            logger.info(f"Parallel sweep (MANIFOLD-AWARE): {n_total:,} combinations across {size} ranks")
            logger.info(f"  Consistency weight: {cfg.manifold_consistency_weight}")
            logger.info(f"  Re-encode for output: {cfg.manifold_reencode_output}")
        else:
            logger.info(f"Parallel sweep (LEARNED OUTPUT): {n_total:,} combinations across {size} ranks")
    
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
        if eval_mode == "state_only":
            result = evaluate_hyperparameters_state_only(
                asl, asq, data, cfg.r, cfg.n_steps, cfg.training_end
            )
        elif eval_mode == "physics_gamma":
            result = evaluate_hyperparameters_physics_gamma(
                asl, asq, data, physics_data, cfg.r, cfg.n_steps, cfg.training_end
            )
        elif eval_mode == "manifold":
            result = evaluate_hyperparameters_manifold(
                asl, asq, aol, aoq, data, manifold_data, cfg.r, cfg.n_steps, cfg.training_end,
                consistency_weight=cfg.manifold_consistency_weight,
                reencode_output=cfg.manifold_reencode_output,
            )
        else:
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
        
        # Log manifold-specific statistics if applicable
        if use_manifold and combined:
            consistency_errs = [r.get('manifold_consistency_err', 0) for r in combined]
            logger.info(f"Manifold consistency error: min={min(consistency_errs):.4f}, "
                       f"max={max(consistency_errs):.4f}, mean={np.mean(consistency_errs):.4f}")
        
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
                  selection_metric: str = "gamma_learned") -> list:
    """
    Select models meeting threshold criteria.
    
    Parameters
    ----------
    results : list
        List of evaluation results from hyperparameter sweep.
    thresh_mean : float
        Threshold for mean Gamma errors.
    thresh_std : float
        Threshold for std Gamma errors.
    logger : Logger
        Logger instance.
    selection_metric : str
        One of "gamma_learned", "gamma_physics", "state_error".
        - gamma_learned/gamma_physics: Use Gamma thresholds
        - state_error: Sort by state error (no thresholds)
    
    Returns
    -------
    list
        Selected models sorted by total error.
    """
    if selection_metric == "state_error":
        # For state-only mode, just select top models by total_error (which is state_error)
        selected = sorted(results, key=lambda x: x['total_error'])
        logger.info(f"State-only mode: selected top {len(selected)} models by state error")
    else:
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

def compute_operators(params: dict, data: dict, r: int, state_only: bool = False) -> dict:
    """Compute operator matrices for a single hyperparameter set."""
    s = r * (r + 1) // 2
    d_state = r + s
    
    # State operators
    reg_state = np.zeros(d_state)
    reg_state[:r] = params['alpha_state_lin']
    reg_state[r:] = params['alpha_state_quad']
    
    O = np.linalg.solve(
        data['D_state_2'] + np.diag(reg_state),
        data['D_state'].T @ data['Y_state']
    ).T
    A, F = O[:, :r].copy(), O[:, r:].copy()
    
    result = {
        'A': A, 'F': F,
        'alpha_state_lin': params['alpha_state_lin'],
        'alpha_state_quad': params['alpha_state_quad'],
        'total_error': params['total_error'],
        'use_learned_output': not state_only,
    }
    
    if state_only:
        # No output operators
        result['alpha_out_lin'] = 0.0
        result['alpha_out_quad'] = 0.0
        result['mean_err_Gamma_n'] = 0.0
        result['std_err_Gamma_n'] = 0.0
        result['mean_err_Gamma_c'] = 0.0
        result['std_err_Gamma_c'] = 0.0
        if 'state_error' in params:
            result['state_error'] = params['state_error']
    else:
        # Output operators
        d_out = r + s + 1
        reg_out = np.zeros(d_out)
        reg_out[:r] = params['alpha_out_lin']
        reg_out[r:r+s] = params['alpha_out_quad']
        reg_out[r+s:] = params['alpha_out_lin']
        
        O_out = np.linalg.solve(
            data['D_out_2'] + np.diag(reg_out),
            data['D_out'].T @ data['Y_Gamma'].T
        ).T
        C, G, c = O_out[:, :r].copy(), O_out[:, r:r+s].copy(), O_out[:, r+s].copy()
        
        result['C'] = C
        result['G'] = G
        result['c'] = c
        result['alpha_out_lin'] = params['alpha_out_lin']
        result['alpha_out_quad'] = params['alpha_out_quad']
        result['mean_err_Gamma_n'] = params['mean_err_Gamma_n']
        result['std_err_Gamma_n'] = params['std_err_Gamma_n']
        result['mean_err_Gamma_c'] = params['mean_err_Gamma_c']
        result['std_err_Gamma_c'] = params['std_err_Gamma_c']
    
    return result


def recompute_operators_parallel(selected: list, data: dict, r: int, 
                                  operators_dir: str, comm, logger,
                                  state_only: bool = False) -> list:
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
        model = compute_operators(selected[idx], data, r, state_only=state_only)
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
