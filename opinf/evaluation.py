"""
Prediction and evaluation utilities.

This module handles:
- Trajectory prediction with trained models
- Ensemble predictions
- Metric computation (RMSE, relative errors)

Author: Anthony Poole
"""

import numpy as np
import h5py

from core import get_quadratic_terms, solve_difference_model
from utils import load_dataset


# =============================================================================
# PREDICTION
# =============================================================================

def predict_trajectory(u0: np.ndarray, n_steps: int, model: dict,
                       mean_Xhat: np.ndarray, scaling_Xhat: float) -> dict:
    """Run prediction for a single trajectory with one model."""
    A, F = model['A'], model['F']
    C = model.get('C', None)
    G = model.get('G', None)
    c = model.get('c', None)
    H = model.get('H', None)
    c_state = model.get('c_state', None)
    
    # State evolution with optional closure terms
    def f(x):
        result = A @ x + F @ get_quadratic_terms(x)
        if H is not None:
            result += H @ (x ** 3)
        if c_state is not None:
            result += c_state
        return result
    
    is_nan, Xhat_pred = solve_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True}
    
    X_OpInf = Xhat_pred.T
    
    # Output computation (only if output operators exist)
    if C is not None and G is not None and c is not None:
        Xhat_scaled = (X_OpInf - mean_Xhat) / scaling_Xhat
        Xhat_2 = get_quadratic_terms(Xhat_scaled)
        Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
        Gamma_n = Y_OpInf[0, :]
        Gamma_c = Y_OpInf[1, :]
    else:
        # No output operators — QoI will be recomputed from physics
        Gamma_n = np.zeros(X_OpInf.shape[0])
        Gamma_c = np.zeros(X_OpInf.shape[0])
    
    return {
        'is_nan': False,
        'X_OpInf': X_OpInf,
        'Gamma_n': Gamma_n,
        'Gamma_c': Gamma_c,
    }


def compute_ensemble_predictions(models: list, ICs: np.ndarray, boundaries: np.ndarray,
                                  mean_Xhat: np.ndarray, scaling_Xhat: float,
                                  logger, name: str = "trajectory",
                                  max_ensemble: int = 0,
                                  max_steps: int = 0) -> dict:
    """Compute ensemble predictions for multiple trajectories.
    
    If max_ensemble > 0, only use the first max_ensemble non-divergent models.
    If max_steps > 0, cap rollout length to max_steps (for consistency with sweep).
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
    
    logger.info(f"Processing {n_traj} {name}(s)...")
    
    for traj_idx in range(n_traj):
        traj_len = boundaries[traj_idx + 1] - boundaries[traj_idx]
        if max_steps > 0 and traj_len > max_steps:
            traj_len = max_steps
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
            
            if max_ensemble > 0 and len(traj_Gamma_n) >= max_ensemble:
                logger.info(f"    Reached max_ensemble={max_ensemble} valid models")
                break
        
        predictions['Gamma_n'].append(np.array(traj_Gamma_n))
        predictions['Gamma_c'].append(np.array(traj_Gamma_c))
        predictions['X_OpInf'].append(np.array(traj_X))
    
    return predictions


def recompute_qoi_from_physics(predictions: dict, pod_basis: np.ndarray,
                                temporal_mean: np.ndarray,
                                pde: str, ks_L: float = 100.0, ks_N: int = 200,
                                ns_Lx: float = 6.283185307179586,
                                ns_Ly: float = None,
                                ns_ny: int = 256, ns_nx: int = 256,
                                logger=None) -> dict:
    """
    Replace learned-operator QoI with physics-based QoI.
    
    For each model in the ensemble, reconstruct the full state from the
    reduced state using the POD basis, then compute energy/enstrophy
    directly from the physics formulas.
    
    Parameters
    ----------
    predictions : dict
        Output from compute_ensemble_predictions(). Must contain 'X_OpInf'.
    pod_basis : np.ndarray, shape (n_spatial, r)
        POD basis matrix.
    temporal_mean : np.ndarray, shape (n_spatial,)
        Temporal mean for de-centering.
    pde : str
        PDE type ("hw2d", "ks", or "ns").
    ks_L : float
        KS domain length.
    ks_N : int
        KS spatial grid points.
    ns_Lx, ns_Ly : float
        NS domain lengths.
    ns_ny, ns_nx : int
        NS grid dimensions.
    logger : optional
        Logger instance.
    """
    from shared.data_io import reconstruct_full_state
    
    if pde == "ks":
        from shared.physics import compute_ks_qoi_from_state_vector
        dx = ks_L / ks_N
    elif pde == "ns":
        from shared.physics import compute_ns_qoi_from_state_vector
    else:
        from shared.physics import compute_gamma_n, compute_gamma_c
    
    for traj_idx in range(len(predictions['X_OpInf'])):
        X_ensemble = predictions['X_OpInf'][traj_idx]  # (n_models, n_steps, r)
        
        if X_ensemble.size == 0:
            continue
        
        new_gamma_n = []
        new_gamma_c = []
        
        n_models = X_ensemble.shape[0]
        if logger:
            logger.info(f"  Traj {traj_idx + 1}: recomputing QoI from physics for {n_models} models...")
        
        for model_idx in range(n_models):
            X_model = X_ensemble[model_idx]  # (n_steps, r)
            
            # Reconstruct full state: X_model.T is (r, n_steps)
            Q_full = reconstruct_full_state(X_model.T, pod_basis, temporal_mean)  # (N, n_steps)
            
            if pde == "ks":
                energy, enstrophy = compute_ks_qoi_from_state_vector(Q_full, ks_N, dx)
                new_gamma_n.append(energy)
                new_gamma_c.append(enstrophy)
            elif pde == "ns":
                energy, enstrophy = compute_ns_qoi_from_state_vector(
                    Q_full, ns_ny, ns_nx, ns_Lx, ns_Ly
                )
                new_gamma_n.append(energy)
                new_gamma_c.append(enstrophy)
            else:
                raise NotImplementedError("Physics-based QoI for hw2d not yet implemented in OpInf")
        
        predictions['Gamma_n'][traj_idx] = np.array(new_gamma_n)
        predictions['Gamma_c'][traj_idx] = np.array(new_gamma_c)
    
    if logger:
        logger.info("  QoI recomputed from physics.")
    
    return predictions


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(predictions: dict, ref_files: list, boundaries: np.ndarray,
                    engine: str, logger, start_offset: int = 0,
                    pde: str = "hw2d") -> dict:
    """Compute evaluation metrics comparing predictions to reference.
    
    Args:
        start_offset: For temporal_split mode, the starting snapshot index
                      (e.g., train_start or test_start) for loading reference data.
        pde: PDE type ("hw2d" or "ks").
    """
    logger.info("Computing evaluation metrics...")
    
    metrics = {'trajectories': [], 'ensemble': {}}
    n_traj = len(predictions['Gamma_n'])
    
    all_errs = {'mean_n': [], 'std_n': [], 'mean_c': [], 'std_c': []}
    
    for traj_idx in range(n_traj):
        traj_len = boundaries[traj_idx + 1] - boundaries[traj_idx]
        
        # Use actual prediction length (may be capped by max_steps)
        pred_n = predictions['Gamma_n'][traj_idx]
        pred_c = predictions['Gamma_c'][traj_idx]
        if pred_n.size > 0:
            actual_len = pred_n.shape[-1]
            if actual_len < traj_len:
                traj_len = actual_len
        
        # Load reference QoIs based on PDE type
        if pde == "ks":
            import h5py
            with h5py.File(ref_files[traj_idx], 'r') as f:
                ref_n = np.array(f['energy'][start_offset:start_offset + traj_len])
                ref_c = np.array(f['enstrophy'][start_offset:start_offset + traj_len])
        elif pde == "ns":
            import h5py
            with h5py.File(ref_files[traj_idx], 'r') as f:
                ref_n = np.array(f['energy'][start_offset:start_offset + traj_len])
                ref_c = np.array(f['enstrophy'][start_offset:start_offset + traj_len])
        else:
            with h5py.File(ref_files[traj_idx], 'r') as f:
                ref_n = np.asarray(f['gamma_n'][start_offset:start_offset + traj_len])
                ref_c = np.asarray(f['gamma_c'][start_offset:start_offset + traj_len])
        
        # Handle case where all models produced NaN (empty predictions)
        if pred_n.size == 0 or pred_c.size == 0:
            logger.warning(f"  Traj {traj_idx + 1}: all models NaN, reporting NaN metrics")
            nan_metrics = {
                'trajectory': traj_idx, 'n_steps': traj_len,
                'ref_mean_Gamma_n': float(np.mean(ref_n)), 'ref_std_Gamma_n': float(np.std(ref_n, ddof=1)),
                'pred_mean_Gamma_n': float('nan'), 'pred_std_Gamma_n': float('nan'),
                'err_mean_Gamma_n': float('nan'), 'err_std_Gamma_n': float('nan'),
                'rmse_Gamma_n': float('nan'),
                'ref_mean_Gamma_c': float(np.mean(ref_c)), 'ref_std_Gamma_c': float(np.std(ref_c, ddof=1)),
                'pred_mean_Gamma_c': float('nan'), 'pred_std_Gamma_c': float('nan'),
                'err_mean_Gamma_c': float('nan'), 'err_std_Gamma_c': float('nan'),
                'rmse_Gamma_c': float('nan'),
            }
            metrics['trajectories'].append(nan_metrics)
            all_errs['mean_n'].append(float('nan'))
            all_errs['std_n'].append(float('nan'))
            all_errs['mean_c'].append(float('nan'))
            all_errs['std_c'].append(float('nan'))
            continue
        
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
        'mean_err_Gamma_n': float(np.nanmean(all_errs['mean_n'])),
        'std_err_Gamma_n': float(np.nanmean(all_errs['std_n'])),
        'mean_err_Gamma_c': float(np.nanmean(all_errs['mean_c'])),
        'std_err_Gamma_c': float(np.nanmean(all_errs['std_c'])),
    }
    
    return metrics
