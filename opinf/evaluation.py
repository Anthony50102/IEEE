"""
Prediction and evaluation utilities.

This module handles:
- Trajectory prediction with trained models
- Ensemble predictions
- Metric computation (RMSE, relative errors)

Author: Anthony Poole
"""

import numpy as np

from core import get_quadratic_terms, solve_difference_model
from utils import load_dataset


# =============================================================================
# PREDICTION
# =============================================================================

def predict_trajectory(u0: np.ndarray, n_steps: int, model: dict,
                       mean_Xhat: np.ndarray, scaling_Xhat: float,
                       physics_reconstruction: dict = None) -> dict:
    """
    Run prediction for a single trajectory with one model.
    
    Parameters
    ----------
    u0 : np.ndarray
        Initial condition in reduced space.
    n_steps : int
        Number of time steps.
    model : dict
        Model containing A, F operators, and optionally C, G, c output operators.
    mean_Xhat, scaling_Xhat : array, float
        Statistics for output operator scaling.
    physics_reconstruction : dict, optional
        If model doesn't have output operators, use this to compute Gamma from physics.
        Should contain: pod_basis, temporal_mean, n_y, n_x, k0, c1.
    
    Returns
    -------
    dict
        Prediction results with 'X_OpInf', and optionally 'Gamma_n', 'Gamma_c'.
    """
    from shared.physics import compute_gamma_from_state_vector
    
    A, F = model['A'], model['F']
    
    # State evolution
    f = lambda x: A @ x + F @ get_quadratic_terms(x)
    is_nan, Xhat_pred = solve_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True}
    
    X_OpInf = Xhat_pred.T  # (n_steps, r)
    
    result = {
        'is_nan': False,
        'X_OpInf': X_OpInf,
    }
    
    # Compute output using learned operators if available
    has_output = model.get('has_output_operators', 'C' in model)
    if has_output and 'C' in model:
        C, G, c = model['C'], model['G'], model['c']
        Xhat_scaled = (X_OpInf - mean_Xhat) / scaling_Xhat
        Xhat_2 = get_quadratic_terms(Xhat_scaled)
        Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
        result['Gamma_n'] = Y_OpInf[0, :]
        result['Gamma_c'] = Y_OpInf[1, :]
    
    # Or compute from physics if reconstruction data is provided
    elif physics_reconstruction is not None:
        pod_basis = physics_reconstruction['pod_basis']
        temporal_mean = physics_reconstruction.get('temporal_mean')
        manifold_W = physics_reconstruction.get('manifold_W')
        manifold_shift = physics_reconstruction.get('manifold_shift')
        reduction_method = physics_reconstruction.get('reduction_method', 'linear')
        n_y, n_x = physics_reconstruction['n_y'], physics_reconstruction['n_x']
        k0, c1 = physics_reconstruction['k0'], physics_reconstruction['c1']
        dx = 2 * np.pi / k0
        r = X_OpInf.shape[1]
        
        # Import once outside loop for efficiency
        if reduction_method == "manifold" and manifold_W is not None:
            from training import _quadratic_features_batch
        
        Gamma_n_list, Gamma_c_list = [], []
        for t in range(n_steps):
            z = X_OpInf[t, :]  # reduced state
            
            # Reconstruct full state based on reduction method
            if reduction_method == "manifold" and manifold_W is not None:
                # Manifold: x_full = V @ z + W @ h(z) + shift
                h_z = _quadratic_features_batch(z)
                x_full = pod_basis[:, :r] @ z + manifold_W @ h_z
                if manifold_shift is not None:
                    x_full += manifold_shift
            else:
                # Linear POD: x_full = V @ z + mean
                x_full = pod_basis[:, :r] @ z
                if temporal_mean is not None:
                    x_full += temporal_mean
            
            g_n, g_c = compute_gamma_from_state_vector(x_full, n_y, n_x, dx, c1)
            Gamma_n_list.append(g_n)
            Gamma_c_list.append(g_c)
        
        result['Gamma_n'] = np.array(Gamma_n_list)
        result['Gamma_c'] = np.array(Gamma_c_list)
    
    return result


def compute_ensemble_predictions(models: list, ICs: np.ndarray, boundaries: np.ndarray,
                                  mean_Xhat: np.ndarray, scaling_Xhat: float,
                                  logger, name: str = "trajectory",
                                  physics_reconstruction: dict = None) -> dict:
    """
    Compute ensemble predictions for multiple trajectories.
    
    Parameters
    ----------
    models : list
        List of (score, model_dict) tuples.
    ICs : np.ndarray
        Initial conditions in reduced space, shape (n_traj, r).
    boundaries : np.ndarray
        Trajectory boundary indices.
    mean_Xhat, scaling_Xhat : array, float
        Statistics for output operator scaling.
    logger : Logger
        Logger instance.
    name : str
        Name for logging.
    physics_reconstruction : dict, optional
        If models don't have output operators, use this to compute Gamma from physics.
    
    Returns
    -------
    dict
        Predictions with 'Gamma_n', 'Gamma_c', 'X_OpInf' lists.
    """
    n_traj = len(boundaries) - 1
    predictions = {'Gamma_n': [], 'Gamma_c': [], 'X_OpInf': []}
    
    # Ensure ICs is 2D (n_traj, r)
    if ICs.ndim == 1:
        ICs = ICs.reshape(1, -1)
    
    # Validate ICs shape matches number of trajectories
    if ICs.shape[0] != n_traj:
        raise ValueError(
            f"ICs shape {ICs.shape} doesn't match n_traj={n_traj}. "
            f"Expected ({n_traj}, r). This may indicate a bug in step_1 preprocessing."
        )
    
    # Check if any model has output operators
    has_output = any(m[1].get('has_output_operators', 'C' in m[1]) for m in models)
    if not has_output and physics_reconstruction is None:
        logger.warning("Models have no output operators and no physics_reconstruction provided. "
                      "Gamma predictions will not be computed.")
    
    logger.info(f"Processing {n_traj} {name}(s)...")
    
    for traj_idx in range(n_traj):
        traj_len = boundaries[traj_idx + 1] - boundaries[traj_idx]
        u0 = ICs[traj_idx, :]
        
        logger.info(f"  Trajectory {traj_idx + 1}/{n_traj} ({traj_len} steps)")
        
        traj_Gamma_n, traj_Gamma_c, traj_X = [], [], []
        
        for model_idx, (score, model) in enumerate(models):
            result = predict_trajectory(u0, traj_len, model, mean_Xhat, scaling_Xhat,
                                        physics_reconstruction=physics_reconstruction)
            
            if result['is_nan']:
                logger.warning(f"    Model {model_idx + 1}: NaN")
                continue
            
            # Append Gamma only if they were computed
            if 'Gamma_n' in result:
                traj_Gamma_n.append(result['Gamma_n'])
                traj_Gamma_c.append(result['Gamma_c'])
            traj_X.append(result['X_OpInf'])
        
        # Store results (Gamma arrays may be empty if not computed)
        predictions['Gamma_n'].append(np.array(traj_Gamma_n) if traj_Gamma_n else np.array([]))
        predictions['Gamma_c'].append(np.array(traj_Gamma_c) if traj_Gamma_c else np.array([]))
        predictions['X_OpInf'].append(np.array(traj_X))
    
    return predictions


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(predictions: dict, ref_files: list, boundaries: np.ndarray,
                    engine: str, logger, start_offset: int = 0) -> dict:
    """Compute evaluation metrics comparing predictions to reference.
    
    Args:
        start_offset: For temporal_split mode, the starting snapshot index
                      (e.g., train_start or test_start) for loading reference data.
    """
    logger.info("Computing evaluation metrics...")
    
    metrics = {'trajectories': [], 'ensemble': {}}
    n_traj = len(predictions['Gamma_n'])
    
    # Check if we have any Gamma predictions
    has_gamma = n_traj > 0 and len(predictions['Gamma_n'][0]) > 0
    
    if not has_gamma:
        logger.warning("No Gamma predictions available - skipping Gamma metrics")
        metrics['ensemble'] = {
            'mean_err_Gamma_n': float('nan'),
            'std_err_Gamma_n': float('nan'),
            'mean_err_Gamma_c': float('nan'),
            'std_err_Gamma_c': float('nan'),
        }
        return metrics
    
    all_errs = {'mean_n': [], 'std_n': [], 'mean_c': [], 'std_c': []}
    
    for traj_idx in range(n_traj):
        fh = load_dataset(ref_files[traj_idx], engine)
        traj_len = boundaries[traj_idx + 1] - boundaries[traj_idx]
        
        # Apply offset for temporal_split mode
        ref_n = fh["gamma_n"].values[start_offset:start_offset + traj_len]
        ref_c = fh["gamma_c"].values[start_offset:start_offset + traj_len]
        
        # Ensemble mean predictions
        pred_n = predictions['Gamma_n'][traj_idx]
        pred_c = predictions['Gamma_c'][traj_idx]
        mean_pred_n = np.mean(pred_n, axis=0)
        mean_pred_c = np.mean(pred_c, axis=0)
        
        # Reference and prediction statistics
        ref_mean_n, ref_std_n = np.mean(ref_n), np.std(ref_n, ddof=1)
        ref_mean_c, ref_std_c = np.mean(ref_c), np.std(ref_c, ddof=1)
        
        ens_mean_n, ens_std_n = np.mean(mean_pred_n), np.std(mean_pred_n, ddof=1)
        ens_mean_c, ens_std_c = np.mean(mean_pred_c), np.std(mean_pred_c, ddof=1)
        
        # Relative errors
        err_mean_n = abs(ref_mean_n - ens_mean_n) / abs(ref_mean_n)
        err_std_n = abs(ref_std_n - ens_std_n) / ref_std_n
        err_mean_c = abs(ref_mean_c - ens_mean_c) / abs(ref_mean_c)
        err_std_c = abs(ref_std_c - ens_std_c) / ref_std_c
        
        all_errs['mean_n'].append(err_mean_n)
        all_errs['std_n'].append(err_std_n)
        all_errs['mean_c'].append(err_mean_c)
        all_errs['std_c'].append(err_std_c)
        
        # RMSE
        rmse_n = np.sqrt(np.mean((mean_pred_n - ref_n)**2))
        rmse_c = np.sqrt(np.mean((mean_pred_c - ref_c)**2))
        
        traj_metrics = {
            'trajectory': traj_idx,
            'n_steps': traj_len,
            'ref_mean_Gamma_n': float(ref_mean_n), 'ref_std_Gamma_n': float(ref_std_n),
            'pred_mean_Gamma_n': float(ens_mean_n), 'pred_std_Gamma_n': float(ens_std_n),
            'err_mean_Gamma_n': float(err_mean_n), 'err_std_Gamma_n': float(err_std_n),
            'rmse_Gamma_n': float(rmse_n),
            'ref_mean_Gamma_c': float(ref_mean_c), 'ref_std_Gamma_c': float(ref_std_c),
            'pred_mean_Gamma_c': float(ens_mean_c), 'pred_std_Gamma_c': float(ens_std_c),
            'err_mean_Gamma_c': float(err_mean_c), 'err_std_Gamma_c': float(err_std_c),
            'rmse_Gamma_c': float(rmse_c),
        }
        metrics['trajectories'].append(traj_metrics)
        
        logger.info(f"  Traj {traj_idx + 1}: Γn err=[{err_mean_n:.4f}, {err_std_n:.4f}], "
                   f"Γc err=[{err_mean_c:.4f}, {err_std_c:.4f}]")
    
    metrics['ensemble'] = {
        'mean_err_Gamma_n': float(np.mean(all_errs['mean_n'])),
        'std_err_Gamma_n': float(np.mean(all_errs['std_n'])),
        'mean_err_Gamma_c': float(np.mean(all_errs['mean_c'])),
        'std_err_Gamma_c': float(np.mean(all_errs['std_c'])),
    }
    
    return metrics
