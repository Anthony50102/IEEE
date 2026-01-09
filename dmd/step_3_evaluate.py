"""
Step 3: DMD Evaluation and Prediction.

This script orchestrates:
1. Loading fitted DMD model from Step 2
2. Computing forecasts for training and test trajectories
3. Reconstructing full state and computing physics-based Gamma
4. Computing evaluation metrics
5. Generating diagnostic plots

Usage:
    python step_3_evaluate.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmd.utils import (
    load_dmd_config, get_dmd_output_paths, print_dmd_config_summary, save_config,
    dmd_forecast_reduced, compute_gamma_from_state, reconstruct_full_state,
    predict_gamma_learned,
)
from opinf.utils import (
    setup_logging, save_step_status, check_step_completed,
    print_header, load_dataset as loader,
)
from shared.plotting import plot_gamma_comparison, generate_state_diagnostic_plots


# =============================================================================
# MODEL & DATA LOADING
# =============================================================================

def load_dmd_model(paths: dict, logger) -> dict:
    """Load DMD model from Step 2."""
    logger.info(f"Loading DMD model from {paths['dmd_model']}")
    data = np.load(paths['dmd_model'])
    model = {
        'eigs': data['eigs'],
        'modes_reduced': data['modes_reduced'],
        'amplitudes': data['amplitudes'],
        'dt': float(data['dt']),
        'dmd_rank': int(data['dmd_rank']),
        'use_pod': bool(data.get('use_pod', True)),  # Backward compatible
        'use_learned_output': bool(data.get('use_learned_output', False)),
    }
    
    # Load output model if present
    if model['use_learned_output'] and 'output_C' in data:
        model['output_model'] = {
            'C': data['output_C'],
            'G': data['output_G'],
            'c': data['output_c'],
            'mean_X': data['output_mean_X'],
            'scaling_X': float(data['output_scaling_X']),
        }
        logger.info(f"  Loaded learned output model (C: {model['output_model']['C'].shape})")
    
    logger.info(f"  Rank: {model['dmd_rank']}, dt: {model['dt']}, use_pod: {model['use_pod']}, learned_output: {model['use_learned_output']}")
    return model


def load_pod_basis(paths: dict, logger, use_pod: bool = True):
    """Load POD basis for full-state reconstruction (or mean for raw mode)."""
    pod_path = paths['pod_basis']
    if not os.path.exists(pod_path):
        logger.warning("POD basis/mean not found, will not compute physics-based Gamma")
        return None
    
    logger.info(f"Loading reconstruction data from {pod_path}")
    data = np.load(pod_path, allow_pickle=True)
    
    result = {
        'mean': data['mean'],
        'n_y': int(data['n_y']),
        'n_x': int(data['n_x']),
        'use_pod': bool(data.get('use_pod', True)),
    }
    
    if use_pod and data['U_r'] is not None:
        result['U_r'] = data['U_r']
    else:
        result['U_r'] = None
    
    return result


def load_supporting_data(paths: dict, logger) -> tuple:
    """Load ICs and boundaries."""
    ics = np.load(paths['initial_conditions'])
    bounds = np.load(paths['boundaries'])
    return (
        ics['train_ICs_reduced'], ics['test_ICs_reduced'],
        bounds['train_boundaries'], bounds['test_boundaries'],
    )


# =============================================================================
# FORECASTING
# =============================================================================

def compute_amplitudes_from_ic(x0: np.ndarray, modes_reduced: np.ndarray) -> np.ndarray:
    """Compute DMD amplitudes from initial condition via least squares."""
    return np.linalg.lstsq(modes_reduced, x0, rcond=None)[0]


def forecast_trajectory(model: dict, x0: np.ndarray, n_steps: int, pod_basis: dict, 
                        k0: float, c1: float, use_learned_output: bool = False) -> dict:
    """
    Forecast a single trajectory.
    
    Handles both POD mode (reconstruct via U_r) and raw mode (DMD modes are full-space).
    Can use either physics-based or learned output model for Gamma.
    
    Parameters
    ----------
    model : dict
        DMD model with eigs, modes_reduced, and optionally output_model.
    x0 : np.ndarray
        Initial condition in reduced space.
    n_steps : int
        Number of time steps to forecast.
    pod_basis : dict
        POD basis for full-state reconstruction (only needed for physics Gamma).
    k0, c1 : float
        Physics parameters for Gamma computation.
    use_learned_output : bool
        If True, use learned C, G, c operators instead of physics-based Gamma.
    """
    r = model['dmd_rank']
    use_pod = model.get('use_pod', True)
    
    if use_pod:
        # POD mode: x0 is in reduced space, truncate to DMD rank
        x0_r = x0[:r]
    else:
        # Raw mode: x0 is full-space IC, need to use all features for amplitude computation
        # modes_reduced is actually modes in full space
        x0_r = x0
    
    # Compute amplitudes for this IC
    amplitudes = compute_amplitudes_from_ic(x0_r, model['modes_reduced'])
    
    # Time vector
    t = np.arange(n_steps) * model['dt']
    
    # Forecast
    X_hat = dmd_forecast_reduced(model['eigs'], model['modes_reduced'], amplitudes, t)
    X_hat = np.real(X_hat)  # (n_features, n_steps)
    
    if np.any(np.isnan(X_hat)) or np.any(np.isinf(X_hat)):
        return {'is_nan': True}
    
    result = {'is_nan': False, 'X_hat': X_hat}
    
    # Compute Gamma using learned output model
    if use_learned_output and 'output_model' in model:
        Gamma_n, Gamma_c = predict_gamma_learned(X_hat, model['output_model'])
        result['Gamma_n'] = Gamma_n
        result['Gamma_c'] = Gamma_c
    # Compute Gamma from physics (fallback or default)
    elif pod_basis is not None:
        mean_data = pod_basis.get('mean')
        
        if use_pod and pod_basis.get('U_r') is not None:
            # POD mode: reconstruct full state via U_r @ X_hat + mean
            U_r_truncated = pod_basis['U_r'][:, :r]
            Q_full = reconstruct_full_state(X_hat, U_r_truncated, mean_data)
        else:
            # Raw mode: DMD output is already in full state space, just add mean
            # X_hat is (n_spatial, n_steps), need to add mean
            if mean_data is not None:
                Q_full = X_hat + mean_data.reshape(-1, 1)
            else:
                Q_full = X_hat
        
        Gamma_n, Gamma_c = compute_gamma_from_state(
            Q_full, 2, pod_basis['n_y'], pod_basis['n_x'], k0, c1
        )
        result['Gamma_n'] = Gamma_n
        result['Gamma_c'] = Gamma_c
    
    return result


def compute_forecasts(model: dict, ICs: np.ndarray, boundaries: np.ndarray,
                      pod_basis, k0: float, c1: float, logger, name: str,
                      use_learned_output: bool = False) -> dict:
    """Compute forecasts for all trajectories."""
    n_traj = len(boundaries) - 1
    forecasts = {'X_hat': [], 'Gamma_n': [], 'Gamma_c': [], 'is_nan': []}
    
    # Ensure ICs is 2D (n_traj, r)
    if ICs.ndim == 1:
        ICs = ICs.reshape(1, -1)
    
    # Validate ICs shape
    if ICs.shape[0] != n_traj:
        raise ValueError(
            f"ICs shape {ICs.shape} doesn't match n_traj={n_traj}. "
            f"Expected ({n_traj}, r)."
        )
    
    logger.info(f"Computing {n_traj} {name} forecast(s)...")
    if use_learned_output:
        logger.info("  Using learned output model for Gamma")
    
    for i in range(n_traj):
        n_steps = boundaries[i + 1] - boundaries[i]
        x0 = ICs[i, :]
        
        result = forecast_trajectory(model, x0, n_steps, pod_basis, k0, c1,
                                      use_learned_output=use_learned_output)
        
        forecasts['is_nan'].append(result['is_nan'])
        if result['is_nan']:
            forecasts['X_hat'].append(None)
            forecasts['Gamma_n'].append(None)
            forecasts['Gamma_c'].append(None)
            logger.warning(f"  Trajectory {i+1}: NaN detected")
        else:
            forecasts['X_hat'].append(result['X_hat'])
            forecasts['Gamma_n'].append(result.get('Gamma_n'))
            forecasts['Gamma_c'].append(result.get('Gamma_c'))
            logger.info(f"  Trajectory {i+1}: {n_steps} steps")
    
    return forecasts


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(forecasts: dict, ref_files: list, boundaries: np.ndarray,
                    engine: str, logger, name: str, ref_offset: int = 0) -> dict:
    """Compute evaluation metrics.
    
    Parameters
    ----------
    ref_offset : int
        Offset into reference file (for temporal_split mode where test
        data starts at a later index in the same file).
    """
    logger.info(f"Computing {name} metrics...")
    
    metrics = {'trajectories': [], 'summary': {}}
    all_errs = {'mean_n': [], 'std_n': [], 'mean_c': [], 'std_c': []}
    
    n_traj = len(boundaries) - 1
    
    for i in range(n_traj):
        if forecasts['is_nan'][i] or forecasts['Gamma_n'][i] is None:
            metrics['trajectories'].append({'valid': False})
            continue
        
        # Load reference
        fh = loader(ref_files[i], engine=engine)
        n_steps = boundaries[i + 1] - boundaries[i]
        # Apply offset for temporal_split mode
        ref_n = fh["gamma_n"].data[ref_offset:ref_offset + n_steps]
        ref_c = fh["gamma_c"].data[ref_offset:ref_offset + n_steps]
        
        pred_n = forecasts['Gamma_n'][i]
        pred_c = forecasts['Gamma_c'][i]
        
        # Stats and errors
        ref_mean_n, ref_std_n = np.mean(ref_n), np.std(ref_n, ddof=1)
        ref_mean_c, ref_std_c = np.mean(ref_c), np.std(ref_c, ddof=1)
        pred_mean_n, pred_std_n = np.mean(pred_n), np.std(pred_n, ddof=1)
        pred_mean_c, pred_std_c = np.mean(pred_c), np.std(pred_c, ddof=1)
        
        err_mean_n = np.abs(ref_mean_n - pred_mean_n) / np.abs(ref_mean_n)
        err_std_n = np.abs(ref_std_n - pred_std_n) / ref_std_n
        err_mean_c = np.abs(ref_mean_c - pred_mean_c) / np.abs(ref_mean_c)
        err_std_c = np.abs(ref_std_c - pred_std_c) / ref_std_c
        
        all_errs['mean_n'].append(err_mean_n)
        all_errs['std_n'].append(err_std_n)
        all_errs['mean_c'].append(err_mean_c)
        all_errs['std_c'].append(err_std_c)
        
        metrics['trajectories'].append({
            'valid': True,
            'trajectory': i,
            'err_mean_Gamma_n': float(err_mean_n),
            'err_std_Gamma_n': float(err_std_n),
            'err_mean_Gamma_c': float(err_mean_c),
            'err_std_Gamma_c': float(err_std_c),
        })
        
        logger.info(f"  Traj {i+1}: Γn=[{err_mean_n:.4f}, {err_std_n:.4f}], "
                    f"Γc=[{err_mean_c:.4f}, {err_std_c:.4f}]")
    
    if all_errs['mean_n']:
        metrics['summary'] = {
            'mean_err_Gamma_n': float(np.mean(all_errs['mean_n'])),
            'std_err_Gamma_n': float(np.mean(all_errs['std_n'])),
            'mean_err_Gamma_c': float(np.mean(all_errs['mean_c'])),
            'std_err_Gamma_c': float(np.mean(all_errs['std_c'])),
        }
    
    return metrics


# =============================================================================
# PLOTTING
# =============================================================================

def generate_gamma_plots(forecasts: dict, ref_files: list, boundaries: np.ndarray,
                         engine: str, dt: float, output_dir: str, logger, 
                         prefix: str = "", ref_offset: int = 0):
    """
    Generate plots comparing predicted vs reference Gamma values.
    
    Uses the shared plotting module for consistent styling across methods.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_traj = len(boundaries) - 1
    
    for i in range(n_traj):
        if forecasts['is_nan'][i] or forecasts['Gamma_n'][i] is None:
            continue
        
        # Load reference
        fh = loader(ref_files[i], engine=engine)
        n_steps = boundaries[i + 1] - boundaries[i]
        ref_n = fh["gamma_n"].data[ref_offset:ref_offset + n_steps]
        ref_c = fh["gamma_c"].data[ref_offset:ref_offset + n_steps]
        
        pred_n = forecasts['Gamma_n'][i]
        pred_c = forecasts['Gamma_c'][i]
        
        # Use shared plotting function
        output_path = os.path.join(output_dir, f'{prefix}traj_{i+1}_gamma.png')
        plot_gamma_comparison(
            pred_n=pred_n, pred_c=pred_c,
            ref_n=ref_n, ref_c=ref_c,
            dt=dt, output_path=output_path, logger=logger,
            title_prefix=f'{prefix.replace("_", " ").title()}Trajectory {i+1}: ',
            method_name="DMD"
        )
    
    logger.info(f"Plots saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 3."""
    parser = argparse.ArgumentParser(description="Step 3: DMD Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    args = parser.parse_args()
    
    cfg = load_dmd_config(args.config)
    cfg.run_dir = args.run_dir
    
    logger = setup_logging("step_3", args.run_dir, cfg.log_level)
    
    print_header("STEP 3: DMD EVALUATION")
    print(f"  Run directory: {args.run_dir}")
    print_dmd_config_summary(cfg)
    
    if not check_step_completed(args.run_dir, "step_2"):
        logger.error("Step 2 has not completed!")
        return
    
    save_step_status(args.run_dir, "step_3", "running")
    save_config(cfg, args.run_dir, step_name="step_3")
    
    paths = get_dmd_output_paths(args.run_dir)
    t_start = time.time()
    
    try:
        # Load model and data
        model = load_dmd_model(paths, logger)
        use_pod = model.get('use_pod', True)
        use_learned_output = model.get('use_learned_output', False)
        pod_basis = load_pod_basis(paths, logger, use_pod=use_pod)
        train_ICs, test_ICs, train_bounds, test_bounds = load_supporting_data(paths, logger)
        
        logger.info(f"  Use POD: {use_pod}")
        logger.info(f"  Use learned output: {use_learned_output}")
        
        # Compute forecasts
        train_pred = compute_forecasts(model, train_ICs, train_bounds, pod_basis,
                                        cfg.k0, cfg.c1, logger, "training",
                                        use_learned_output=use_learned_output)
        test_pred = compute_forecasts(model, test_ICs, test_bounds, pod_basis,
                                       cfg.k0, cfg.c1, logger, "test",
                                       use_learned_output=use_learned_output)
        
        # Determine reference files and offsets for metrics
        if cfg.training_mode == "temporal_split":
            # For temporal split, use the same file for both train and test
            train_ref_files = cfg.training_files
            test_ref_files = cfg.training_files  # Same file, different portion
            train_ref_offset = cfg.train_start
            test_ref_offset = cfg.test_start
        else:
            train_ref_files = cfg.training_files
            test_ref_files = cfg.test_files
            train_ref_offset = 0
            test_ref_offset = 0
        
        # Compute metrics
        train_metrics = compute_metrics(train_pred, train_ref_files, train_bounds,
                                         cfg.engine, logger, "training", 
                                         ref_offset=train_ref_offset)
        test_metrics = compute_metrics(test_pred, test_ref_files, test_bounds,
                                        cfg.engine, logger, "test",
                                        ref_offset=test_ref_offset)
        
        # Save results
        all_metrics = {
            'train': train_metrics, 
            'test': test_metrics, 
            'use_pod': use_pod,
            'use_learned_output': use_learned_output,
        }
        with open(paths['dmd_metrics'], 'w') as f:
            yaml.dump(all_metrics, f, default_flow_style=False)
        logger.info(f"Saved metrics to {paths['dmd_metrics']}")
        
        # Save predictions
        if cfg.save_predictions:
            save_dict = {}
            save_dict['n_train_traj'] = np.array(len(train_bounds) - 1)
            save_dict['n_test_traj'] = np.array(len(test_bounds) - 1)
            for i, gn in enumerate(train_pred['Gamma_n']):
                if gn is not None:
                    save_dict[f'train_traj_{i}_Gamma_n'] = gn
            for i, gn in enumerate(test_pred['Gamma_n']):
                if gn is not None:
                    save_dict[f'test_traj_{i}_Gamma_n'] = gn
            np.savez(paths['dmd_predictions'], **save_dict)
            logger.info(f"Saved predictions to {paths['dmd_predictions']}")
        
        # Generate plots
        if cfg.generate_plots:
            figures_dir = os.path.join(args.run_dir, "figures")
            logger.info("Generating Gamma comparison plots...")
            
            generate_gamma_plots(
                train_pred, train_ref_files, train_bounds,
                cfg.engine, cfg.dt, figures_dir, logger,
                prefix="train_", ref_offset=train_ref_offset
            )
            generate_gamma_plots(
                test_pred, test_ref_files, test_bounds,
                cfg.engine, cfg.dt, figures_dir, logger,
                prefix="test_", ref_offset=test_ref_offset
            )
        
        # Generate state diagnostic plots (optional)
        if cfg.plot_state_error or cfg.plot_state_snapshots:
            figures_dir = os.path.join(args.run_dir, "figures")
            
            # Extract POD basis info for shared function
            if use_pod and pod_basis and pod_basis.get('U_r') is not None:
                U_r = pod_basis['U_r'][:, :model['dmd_rank']]
            else:
                U_r = None  # Raw mode: predictions are already in full space
            mean = pod_basis.get('mean') if pod_basis else None
            n_y = pod_basis['n_y'] if pod_basis else cfg.n_y
            n_x = pod_basis['n_x'] if pod_basis else cfg.n_x
            
            generate_state_diagnostic_plots(
                reduced_states=train_pred['X_hat'],
                ref_files=train_ref_files,
                boundaries=train_bounds,
                pod_basis=U_r,
                temporal_mean=mean,
                n_y=n_y, n_x=n_x,
                engine=cfg.engine, dt=cfg.dt,
                output_dir=figures_dir, logger=logger,
                method_name="DMD" + (" (no POD)" if not use_pod else ""),
                prefix="train_",
                ref_offset=train_ref_offset,
                plot_error=cfg.plot_state_error,
                plot_snapshots=cfg.plot_state_snapshots,
                n_snapshots=cfg.n_snapshot_samples,
                is_nan_flags=train_pred['is_nan']
            )
            generate_state_diagnostic_plots(
                reduced_states=test_pred['X_hat'],
                ref_files=test_ref_files,
                boundaries=test_bounds,
                pod_basis=U_r,
                temporal_mean=mean,
                n_y=n_y, n_x=n_x,
                engine=cfg.engine, dt=cfg.dt,
                output_dir=figures_dir, logger=logger,
                method_name="DMD" + (" (no POD)" if not use_pod else ""),
                prefix="test_",
                ref_offset=test_ref_offset,
                plot_error=cfg.plot_state_error,
                plot_snapshots=cfg.plot_state_snapshots,
                n_snapshots=cfg.n_snapshot_samples,
                is_nan_flags=test_pred['is_nan']
            )
        
        # Final timing
        t_elapsed = time.time() - t_start
        
        save_step_status(args.run_dir, "step_3", "completed", {
            "time_seconds": t_elapsed,
        })
        
        # Print summary
        print_header("STEP 3 COMPLETE")
        print(f"  Runtime: {t_elapsed:.1f}s")
        
        if test_metrics.get('summary'):
            print(f"\n  TEST SUMMARY:")
            print(f"    Avg Γn mean error: {test_metrics['summary']['mean_err_Gamma_n']:.4f}")
            print(f"    Avg Γc mean error: {test_metrics['summary']['mean_err_Gamma_c']:.4f}")
        
    except Exception as e:
        logger.error(f"Step 3 failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_3", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
