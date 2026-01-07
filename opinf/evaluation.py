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
                       mean_Xhat: np.ndarray, scaling_Xhat: float) -> dict:
    """Run prediction for a single trajectory with one model."""
    A, F = model['A'], model['F']
    C, G, c = model['C'], model['G'], model['c']
    
    # State evolution
    f = lambda x: A @ x + F @ get_quadratic_terms(x)
    is_nan, Xhat_pred = solve_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True}
    
    X_OpInf = Xhat_pred.T
    
    # Output computation
    Xhat_scaled = (X_OpInf - mean_Xhat) / scaling_Xhat
    Xhat_2 = get_quadratic_terms(Xhat_scaled)
    Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
    
    return {
        'is_nan': False,
        'X_OpInf': X_OpInf,
        'Gamma_n': Y_OpInf[0, :],
        'Gamma_c': Y_OpInf[1, :],
    }


def compute_ensemble_predictions(models: list, ICs: np.ndarray, boundaries: np.ndarray,
                                  mean_Xhat: np.ndarray, scaling_Xhat: float,
                                  logger, name: str = "trajectory") -> dict:
    """Compute ensemble predictions for multiple trajectories."""
    n_traj = len(boundaries) - 1
    predictions = {'Gamma_n': [], 'Gamma_c': [], 'X_OpInf': []}
    
    # Ensure ICs is 2D (n_traj, r)
    if ICs.ndim == 1:
        ICs = ICs.reshape(1, -1)
    
    logger.info(f"Processing {n_traj} {name}(s)...")
    
    for traj_idx in range(n_traj):
        traj_len = boundaries[traj_idx + 1] - boundaries[traj_idx]
        u0 = ICs[traj_idx, :]
        
        logger.info(f"  Trajectory {traj_idx + 1}/{n_traj} ({traj_len} steps)")
        
        traj_Gamma_n, traj_Gamma_c, traj_X = [], [], []
        
        for model_idx, (score, model) in enumerate(models):
            result = predict_trajectory(u0, traj_len, model, mean_Xhat, scaling_Xhat)
            
            if result['is_nan']:
                logger.warning(f"    Model {model_idx + 1}: NaN")
                continue
            
            traj_Gamma_n.append(result['Gamma_n'])
            traj_Gamma_c.append(result['Gamma_c'])
            traj_X.append(result['X_OpInf'])
        
        predictions['Gamma_n'].append(np.array(traj_Gamma_n))
        predictions['Gamma_c'].append(np.array(traj_Gamma_c))
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
