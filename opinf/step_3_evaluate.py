"""
Step 3: Evaluation and Prediction.

This script handles:
1. Loading trained ensemble models
2. Computing ensemble predictions on training and test trajectories
3. Computing evaluation metrics
4. Generating diagnostic plots (optional)
5. Saving results

Usage:
    python step_3_evaluate.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import os
import time
import numpy as np
import yaml

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
# PREPROCESSING VERIFICATION
# =============================================================================

def load_preprocessing_info(filepath: str, logger) -> dict:
    """
    Load preprocessing information from Step 1.
    
    Parameters
    ----------
    filepath : str
        Path to preprocessing_info.npz file.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Preprocessing settings dictionary.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Preprocessing info file not found: {filepath}")
        logger.warning("  Assuming default settings (centering applied, no scaling)")
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
    logger.info(f"  POD modes (r): {info['r_actual']} (config: {info['r_config']}, energy-based: {info['r_from_energy']})")
    logger.info(f"  Spatial DOF: {info['n_spatial']} ({info['n_fields']} fields, {info['n_x']}x{info['n_y']} grid)")
    
    return info


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_file(filepath: str) -> dict:
    """
    Load a single model from an individual NPZ file.
    
    Parameters
    ----------
    filepath : str
        Path to model NPZ file.
    
    Returns
    -------
    dict
        Model dictionary with operators and metadata.
    """
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
    """
    Load ensemble models from individual files in a directory.
    
    Parameters
    ----------
    operators_dir : str
        Path to directory containing model files.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    list
        List of (score, model) tuples, sorted by score.
    """
    if not os.path.exists(operators_dir):
        logger.warning(f"Operators directory not found: {operators_dir}")
        return []
    
    # Find all model files
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
    
    # Sort by total_error
    models.sort(key=lambda x: x[0])
    
    logger.info(f"  Loaded {len(models)} models successfully")
    return models


def load_ensemble(filepath: str, logger, operators_dir: str = None) -> list:
    """
    Load ensemble models from NPZ file or directory.
    
    First tries to load from individual files in operators_dir if provided,
    falls back to the single NPZ file.
    
    Parameters
    ----------
    filepath : str
        Path to ensemble NPZ file.
    logger : logging.Logger
        Logger instance.
    operators_dir : str, optional
        Path to directory with individual model files.
    
    Returns
    -------
    list
        List of (score, model) tuples.
    """
    # Try loading from individual files first (new format)
    if operators_dir and os.path.exists(operators_dir):
        models = load_ensemble_from_directory(operators_dir, logger)
        if models:
            return models
        logger.info("  Falling back to single ensemble file...")
    
    # Fall back to single NPZ file (backward compatibility)
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
        score = model['total_error']
        models.append((score, model))
    
    logger.info(f"  Loaded {len(models)} models")
    return models


# =============================================================================
# PREDICTION
# =============================================================================

def predict_trajectory(
    u0: np.ndarray,
    n_steps: int,
    model: dict,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
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
    
    Returns
    -------
    dict
        Predictions with keys 'X_OpInf', 'Gamma_n', 'Gamma_c', 'is_nan'.
    """
    A = model['A']
    F = model['F']
    C = model['C']
    G = model['G']
    c = model['c']
    
    # State evolution function
    f = lambda x: np.dot(A, x) + np.dot(F, get_x_sq(x))
    
    # Integrate
    is_nan, Xhat_pred = solve_opinf_difference_model(u0, n_steps, f)
    
    if is_nan:
        return {'is_nan': True}
    
    X_OpInf = Xhat_pred.T  # (n_steps, r)
    
    # Apply output operators
    Xhat_scaled = (X_OpInf - mean_Xhat[np.newaxis, :]) / scaling_Xhat
    Xhat_2 = get_x_sq(Xhat_scaled)
    
    Y_OpInf = (
        C @ Xhat_scaled.T
        + G @ Xhat_2.T
        + c[:, np.newaxis]
    )
    
    return {
        'is_nan': False,
        'X_OpInf': X_OpInf,
        'Gamma_n': Y_OpInf[0, :],
        'Gamma_c': Y_OpInf[1, :],
    }


def compute_ensemble_predictions(
    models: list,
    ICs_reduced: np.ndarray,
    boundaries: np.ndarray,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    logger,
    dataset_name: str = "trajectory",
) -> dict:
    """
    Compute ensemble predictions for multiple trajectories.
    
    Parameters
    ----------
    models : list
        List of (score, model) tuples.
    ICs_reduced : np.ndarray
        Initial conditions in reduced space.
    boundaries : np.ndarray
        Trajectory boundaries.
    mean_Xhat : np.ndarray
        Mean for scaling.
    scaling_Xhat : float
        Scale factor.
    logger : logging.Logger
        Logger instance.
    dataset_name : str
        Name for logging.
    
    Returns
    -------
    dict
        Predictions organized by trajectory.
    """
    n_traj = len(boundaries) - 1
    predictions = {
        'Gamma_n': [],
        'Gamma_c': [],
        'X_OpInf': [],
    }
    
    logger.info(f"Processing {n_traj} {dataset_name}(s)...")
    
    for traj_idx in range(n_traj):
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        u0 = ICs_reduced[traj_idx, :]
        
        logger.info(f"  Trajectory {traj_idx + 1}/{n_traj} ({traj_length} steps)")
        
        traj_Gamma_n = []
        traj_Gamma_c = []
        traj_X_OpInf = []
        
        for model_idx, (score, model) in enumerate(models):
            result = predict_trajectory(
                u0, traj_length, model, mean_Xhat, scaling_Xhat
            )
            
            if result['is_nan']:
                logger.warning(f"    Model {model_idx + 1}: NaN encountered")
                continue
            
            traj_Gamma_n.append(result['Gamma_n'])
            traj_Gamma_c.append(result['Gamma_c'])
            traj_X_OpInf.append(result['X_OpInf'])
        
        predictions['Gamma_n'].append(np.array(traj_Gamma_n))
        predictions['Gamma_c'].append(np.array(traj_Gamma_c))
        predictions['X_OpInf'].append(np.array(traj_X_OpInf))
    
    return predictions


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    predictions: dict,
    reference_files: list,
    boundaries: np.ndarray,
    engine: str,
    logger,
) -> dict:
    """
    Compute evaluation metrics comparing predictions to reference.
    
    Parameters
    ----------
    predictions : dict
        Ensemble predictions.
    reference_files : list
        Paths to reference data files.
    boundaries : np.ndarray
        Trajectory boundaries.
    engine : str
        HDF5 engine.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Metrics dictionary.
    """
    logger.info("Computing evaluation metrics...")
    
    metrics = {
        'trajectories': [],
        'ensemble': {},
    }
    
    n_traj = len(predictions['Gamma_n'])
    
    all_mean_errors_n = []
    all_std_errors_n = []
    all_mean_errors_c = []
    all_std_errors_c = []
    
    for traj_idx in range(n_traj):
        # Load reference
        fh = loader(reference_files[traj_idx], ENGINE=engine)
        ref_Gamma_n = fh["gamma_n"].data
        ref_Gamma_c = fh["gamma_c"].data
        
        # Get trajectory slice
        start = boundaries[traj_idx]
        end = boundaries[traj_idx + 1]
        traj_length = end - start
        
        ref_Gamma_n = ref_Gamma_n[:traj_length]
        ref_Gamma_c = ref_Gamma_c[:traj_length]
        
        # Ensemble mean prediction
        pred_Gamma_n = predictions['Gamma_n'][traj_idx]  # (n_models, n_steps)
        pred_Gamma_c = predictions['Gamma_c'][traj_idx]
        
        mean_pred_n = np.mean(pred_Gamma_n, axis=0)
        std_pred_n = np.std(pred_Gamma_n, axis=0)
        mean_pred_c = np.mean(pred_Gamma_c, axis=0)
        std_pred_c = np.std(pred_Gamma_c, axis=0)
        
        # Reference statistics
        ref_mean_n = np.mean(ref_Gamma_n)
        ref_std_n = np.std(ref_Gamma_n, ddof=1)
        ref_mean_c = np.mean(ref_Gamma_c)
        ref_std_c = np.std(ref_Gamma_c, ddof=1)
        
        # Ensemble mean statistics
        ens_mean_n = np.mean(mean_pred_n)
        ens_std_n = np.std(mean_pred_n, ddof=1)
        ens_mean_c = np.mean(mean_pred_c)
        ens_std_c = np.std(mean_pred_c, ddof=1)
        
        # Relative errors
        err_mean_n = np.abs(ref_mean_n - ens_mean_n) / np.abs(ref_mean_n)
        err_std_n = np.abs(ref_std_n - ens_std_n) / ref_std_n
        err_mean_c = np.abs(ref_mean_c - ens_mean_c) / np.abs(ref_mean_c)
        err_std_c = np.abs(ref_std_c - ens_std_c) / ref_std_c
        
        all_mean_errors_n.append(err_mean_n)
        all_std_errors_n.append(err_std_n)
        all_mean_errors_c.append(err_mean_c)
        all_std_errors_c.append(err_std_c)
        
        # Pointwise RMSE
        rmse_n = np.sqrt(np.mean((mean_pred_n - ref_Gamma_n)**2))
        rmse_c = np.sqrt(np.mean((mean_pred_c - ref_Gamma_c)**2))
        
        traj_metrics = {
            'trajectory': traj_idx,
            'n_steps': traj_length,
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
        metrics['trajectories'].append(traj_metrics)
        
        logger.info(f"  Trajectory {traj_idx + 1}:")
        logger.info(f"    Γn mean err: {err_mean_n:.4f}, std err: {err_std_n:.4f}, RMSE: {rmse_n:.4e}")
        logger.info(f"    Γc mean err: {err_mean_c:.4f}, std err: {err_std_c:.4f}, RMSE: {rmse_c:.4e}")
    
    # Ensemble-level metrics
    metrics['ensemble'] = {
        'mean_err_Gamma_n': float(np.mean(all_mean_errors_n)),
        'std_err_Gamma_n': float(np.mean(all_std_errors_n)),
        'mean_err_Gamma_c': float(np.mean(all_mean_errors_c)),
        'std_err_Gamma_c': float(np.mean(all_std_errors_c)),
    }
    
    return metrics


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(
    predictions: dict,
    reference_files: list,
    boundaries: np.ndarray,
    cfg: PipelineConfig,
    output_dir: str,
    logger,
):
    """
    Generate diagnostic plots.
    
    Parameters
    ----------
    predictions : dict
        Ensemble predictions.
    reference_files : list
        Paths to reference data files.
    boundaries : np.ndarray
        Trajectory boundaries.
    cfg : PipelineConfig
        Configuration.
    output_dir : str
        Directory for figures.
    logger : logging.Logger
        Logger instance.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating plots in {output_dir}")
    
    n_traj = len(predictions['Gamma_n'])
    
    for traj_idx in range(n_traj):
        # Load reference
        fh = loader(reference_files[traj_idx], ENGINE=cfg.engine)
        ref_Gamma_n = fh["gamma_n"].data
        ref_Gamma_c = fh["gamma_c"].data
        
        # Get trajectory length
        traj_length = boundaries[traj_idx + 1] - boundaries[traj_idx]
        ref_Gamma_n = ref_Gamma_n[:traj_length]
        ref_Gamma_c = ref_Gamma_c[:traj_length]
        
        # Ensemble predictions
        pred_Gamma_n = predictions['Gamma_n'][traj_idx]
        pred_Gamma_c = predictions['Gamma_c'][traj_idx]
        
        mean_pred_n = np.mean(pred_Gamma_n, axis=0)
        std_pred_n = np.std(pred_Gamma_n, axis=0)
        mean_pred_c = np.mean(pred_Gamma_c, axis=0)
        std_pred_c = np.std(pred_Gamma_c, axis=0)
        
        time = np.arange(traj_length) * cfg.dt
        
        # Plot Gamma_n
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gamma_n
        ax = axes[0]
        ax.plot(time, ref_Gamma_n, 'k-', label='Reference', linewidth=1)
        ax.plot(time, mean_pred_n, 'b-', label='Ensemble Mean', linewidth=1)
        ax.fill_between(
            time,
            mean_pred_n - 2*std_pred_n,
            mean_pred_n + 2*std_pred_n,
            alpha=0.3, color='blue', label='±2σ'
        )
        ax.set_ylabel(r'$\Gamma_n$')
        ax.legend(loc='upper right')
        ax.set_title(f'Trajectory {traj_idx + 1}: Particle Flux')
        ax.grid(True, alpha=0.3)
        
        # Gamma_c
        ax = axes[1]
        ax.plot(time, ref_Gamma_c, 'k-', label='Reference', linewidth=1)
        ax.plot(time, mean_pred_c, 'r-', label='Ensemble Mean', linewidth=1)
        ax.fill_between(
            time,
            mean_pred_c - 2*std_pred_c,
            mean_pred_c + 2*std_pred_c,
            alpha=0.3, color='red', label='±2σ'
        )
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
    """Main entry point for Step 3."""
    parser = argparse.ArgumentParser(
        description="Step 3: Evaluation and Prediction"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory from previous steps"
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("step_3", args.run_dir, cfg.log_level)
    
    print_header("STEP 3: EVALUATION AND PREDICTION")
    print(f"  Run directory: {args.run_dir}")
    
    # Check previous steps
    if not check_step_completed(args.run_dir, "step_2"):
        logger.error("Step 2 has not completed. Run step_2_train_rom.py first.")
        return
    
    save_step_status(args.run_dir, "step_3", "running")
    
    # Save configuration with step-specific name
    save_config(cfg, args.run_dir, step_name="step_3")
    logger.info("Configuration saved to run directory")
    
    paths = get_output_paths(args.run_dir)
    
    try:
        # Load and verify preprocessing settings
        preproc_info = load_preprocessing_info(paths["preprocessing_info"], logger)
        
        # Load models (try individual files first, fall back to ensemble file)
        models = load_ensemble(
            paths["ensemble_models"], 
            logger, 
            operators_dir=paths["operators_dir"]
        )
        
        if len(models) == 0:
            logger.error("No models loaded!")
            save_step_status(args.run_dir, "step_3", "failed", {"error": "No models"})
            return
        
        # Load supporting data
        logger.info("Loading supporting data...")
        learning = np.load(paths["learning_matrices"])
        mean_Xhat = learning['mean_Xhat']
        scaling_Xhat = float(learning['scaling_Xhat'])
        
        ICs = np.load(paths["initial_conditions"])
        train_ICs_reduced = ICs['train_ICs_reduced']
        test_ICs_reduced = ICs['test_ICs_reduced']
        
        boundaries_data = np.load(paths["boundaries"])
        train_boundaries = boundaries_data['train_boundaries']
        test_boundaries = boundaries_data['test_boundaries']
        
        # Compute predictions
        start_time = time.time()
        
        logger.info("Computing training trajectory predictions...")
        train_predictions = compute_ensemble_predictions(
            models, train_ICs_reduced, train_boundaries,
            mean_Xhat, scaling_Xhat, logger, "training trajectory"
        )
        
        logger.info("Computing test trajectory predictions...")
        test_predictions = compute_ensemble_predictions(
            models, test_ICs_reduced, test_boundaries,
            mean_Xhat, scaling_Xhat, logger, "test trajectory"
        )
        
        pred_time = time.time() - start_time
        logger.info(f"Predictions completed in {pred_time:.1f}s")
        
        # Save predictions
        if cfg.save_predictions:
            save_dict = {
                'n_train_traj': len(train_boundaries) - 1,
                'n_test_traj': len(test_boundaries) - 1,
                'num_models': len(models),
                'train_boundaries': train_boundaries,
                'test_boundaries': test_boundaries,
            }
            
            for i in range(len(train_boundaries) - 1):
                save_dict[f'train_traj_{i}_Gamma_n'] = train_predictions['Gamma_n'][i]
                save_dict[f'train_traj_{i}_Gamma_c'] = train_predictions['Gamma_c'][i]
                save_dict[f'train_traj_{i}_X_OpInf'] = train_predictions['X_OpInf'][i]
            
            for i in range(len(test_boundaries) - 1):
                save_dict[f'test_traj_{i}_Gamma_n'] = test_predictions['Gamma_n'][i]
                save_dict[f'test_traj_{i}_Gamma_c'] = test_predictions['Gamma_c'][i]
                save_dict[f'test_traj_{i}_X_OpInf'] = test_predictions['X_OpInf'][i]
            
            np.savez(paths["predictions"], **save_dict)
            logger.info(f"Saved predictions to {paths['predictions']}")
        
        # Compute metrics
        logger.info("Computing metrics on training data...")
        train_metrics = compute_metrics(
            train_predictions, cfg.training_files, train_boundaries, cfg.engine, logger
        )
        
        logger.info("Computing metrics on test data...")
        test_metrics = compute_metrics(
            test_predictions, cfg.test_files, test_boundaries, cfg.engine, logger
        )
        
        # Save metrics
        all_metrics = {
            'train': train_metrics,
            'test': test_metrics,
        }
        with open(paths["metrics"], 'w') as f:
            yaml.dump(all_metrics, f, default_flow_style=False)
        logger.info(f"Saved metrics to {paths['metrics']}")
        
        # Generate plots
        if cfg.generate_plots:
            generate_plots(
                train_predictions, cfg.training_files, train_boundaries,
                cfg, os.path.join(paths["figures_dir"], "train"), logger
            )
            generate_plots(
                test_predictions, cfg.test_files, test_boundaries,
                cfg, os.path.join(paths["figures_dir"], "test"), logger
            )
        
        # Print summary
        print_header("EVALUATION SUMMARY")
        print("\n  Training Data:")
        for traj in train_metrics['trajectories']:
            print(f"    Trajectory {traj['trajectory'] + 1}:")
            print(f"      Γn: mean_err={traj['err_mean_Gamma_n']:.4f}, std_err={traj['err_std_Gamma_n']:.4f}")
            print(f"      Γc: mean_err={traj['err_mean_Gamma_c']:.4f}, std_err={traj['err_std_Gamma_c']:.4f}")
        
        print("\n  Test Data:")
        for traj in test_metrics['trajectories']:
            print(f"    Trajectory {traj['trajectory'] + 1}:")
            print(f"      Γn: mean_err={traj['err_mean_Gamma_n']:.4f}, std_err={traj['err_std_Gamma_n']:.4f}")
            print(f"      Γc: mean_err={traj['err_mean_Gamma_c']:.4f}, std_err={traj['err_std_Gamma_c']:.4f}")
        
        save_step_status(args.run_dir, "step_3", "completed", {
            "train_mean_err_Gamma_n": train_metrics['ensemble']['mean_err_Gamma_n'],
            "test_mean_err_Gamma_n": test_metrics['ensemble']['mean_err_Gamma_n'],
        })
        
        print_header("STEP 3 COMPLETE")
        logger.info("Step 3 completed successfully")
    
    except Exception as e:
        logger.error(f"Step 3 failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_3", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
