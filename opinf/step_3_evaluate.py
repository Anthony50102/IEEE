"""
Step 3: Evaluation and Prediction.

This script orchestrates:
1. Loading trained ensemble models
2. Computing ensemble predictions on training and test trajectories
3. Computing evaluation metrics
4. Generating diagnostic plots
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
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config, save_config, setup_logging, save_step_status,
    check_step_completed, get_output_paths, print_header,
)
from data import load_ensemble, load_preprocessing_info
from evaluation import compute_ensemble_predictions, compute_metrics
from shared.plotting import plot_gamma_timeseries, generate_state_diagnostic_plots
from utils import load_dataset


def generate_gamma_plots(predictions: dict, ref_files: list, boundaries: np.ndarray,
                         dt: float, engine: str, output_dir: str, logger,
                         start_offset: int = 0):
    """
    Generate Gamma plots for OpInf ensemble predictions.
    
    Loads reference data and calls the shared plotting function.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_traj = len(predictions['Gamma_n'])
    
    for i in range(n_traj):
        fh = load_dataset(ref_files[i], engine)
        n_steps = boundaries[i + 1] - boundaries[i]
        
        ref_n = fh["gamma_n"].values[start_offset:start_offset + n_steps]
        ref_c = fh["gamma_c"].values[start_offset:start_offset + n_steps]
        
        pred_n = predictions['Gamma_n'][i]  # Shape: (n_ensemble, n_steps)
        pred_c = predictions['Gamma_c'][i]
        
        output_path = os.path.join(output_dir, f'traj_{i+1}_gamma.png')
        plot_gamma_timeseries(
            pred_n=pred_n, pred_c=pred_c,
            ref_n=ref_n, ref_c=ref_c,
            dt=dt, output_path=output_path, logger=logger,
            title_prefix=f'Trajectory {i+1}: ',
            method_name="OpInf"
        )
    
    logger.info(f"Plots saved to {output_dir}")


def main():
    """Main entry point for Step 3."""
    parser = argparse.ArgumentParser(description="Step 3: Evaluation and Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg.run_dir = args.run_dir
    
    logger = setup_logging("step_3", args.run_dir, cfg.log_level)
    
    print_header("STEP 3: EVALUATION AND PREDICTION")
    print(f"  Run directory: {args.run_dir}")
    
    if not check_step_completed(args.run_dir, "step_2"):
        logger.error("Step 2 has not completed!")
        return
    
    save_step_status(args.run_dir, "step_3", "running")
    save_config(cfg, args.run_dir, step_name="step_3")
    
    paths = get_output_paths(args.run_dir)
    
    try:
        # Load preprocessing info
        preproc_info = load_preprocessing_info(paths["preprocessing_info"], logger)
        
        # Load models
        models = load_ensemble(paths["ensemble_models"], paths["operators_dir"], logger)
        
        if not models:
            logger.error("No models loaded!")
            save_step_status(args.run_dir, "step_3", "failed", {"error": "No models"})
            return
        
        # Load supporting data
        logger.info("Loading supporting data...")
        learning = np.load(paths["learning_matrices"])
        mean_Xhat = learning['mean_Xhat']
        scaling_Xhat = float(learning['scaling_Xhat'])
        
        ICs = np.load(paths["initial_conditions"])
        train_ICs = ICs['train_ICs_reduced']
        test_ICs = ICs['test_ICs_reduced']
        
        bounds = np.load(paths["boundaries"])
        train_bounds = bounds['train_boundaries']
        test_bounds = bounds['test_boundaries']
        
        # Compute predictions
        t_start = time.time()
        
        train_pred = compute_ensemble_predictions(
            models, train_ICs, train_bounds, mean_Xhat, scaling_Xhat, logger, "training"
        )
        test_pred = compute_ensemble_predictions(
            models, test_ICs, test_bounds, mean_Xhat, scaling_Xhat, logger, "test"
        )
        
        logger.info(f"Predictions completed in {time.time() - t_start:.1f}s")
        
        # Save predictions
        if cfg.save_predictions:
            save_dict = {
                'n_train_traj': len(train_bounds) - 1,
                'n_test_traj': len(test_bounds) - 1,
                'num_models': len(models),
                'train_boundaries': train_bounds,
                'test_boundaries': test_bounds,
            }
            
            for i in range(len(train_bounds) - 1):
                save_dict[f'train_traj_{i}_Gamma_n'] = train_pred['Gamma_n'][i]
                save_dict[f'train_traj_{i}_Gamma_c'] = train_pred['Gamma_c'][i]
                save_dict[f'train_traj_{i}_X_OpInf'] = train_pred['X_OpInf'][i]
            
            for i in range(len(test_bounds) - 1):
                save_dict[f'test_traj_{i}_Gamma_n'] = test_pred['Gamma_n'][i]
                save_dict[f'test_traj_{i}_Gamma_c'] = test_pred['Gamma_c'][i]
                save_dict[f'test_traj_{i}_X_OpInf'] = test_pred['X_OpInf'][i]
            
            np.savez(paths["predictions"], **save_dict)
            logger.info(f"Saved predictions to {paths['predictions']}")
        
        # Determine offsets for temporal_split mode
        train_offset = cfg.train_start if cfg.training_mode == "temporal_split" else 0
        test_offset = cfg.test_start if cfg.training_mode == "temporal_split" else 0
        
        # Compute metrics
        train_metrics = compute_metrics(
            train_pred, cfg.training_files, train_bounds, cfg.engine, logger,
            start_offset=train_offset
        )
        test_metrics = compute_metrics(
            test_pred, cfg.test_files if cfg.test_files else cfg.training_files,
            test_bounds, cfg.engine, logger, start_offset=test_offset
        )
        
        all_metrics = {'train': train_metrics, 'test': test_metrics}
        with open(paths["metrics"], 'w') as f:
            yaml.dump(all_metrics, f, default_flow_style=False)
        logger.info(f"Saved metrics to {paths['metrics']}")
        
        # Generate plots
        if cfg.generate_plots:
            generate_gamma_plots(train_pred, cfg.training_files, train_bounds,
                                 cfg.dt, cfg.engine, 
                                 os.path.join(paths["figures_dir"], "train"), logger,
                                 start_offset=train_offset)
            generate_gamma_plots(test_pred, 
                                 cfg.test_files if cfg.test_files else cfg.training_files,
                                 test_bounds, cfg.dt, cfg.engine,
                                 os.path.join(paths["figures_dir"], "test"), logger,
                                 start_offset=test_offset)
        
        # Generate state diagnostic plots (optional)
        if cfg.plot_state_error or cfg.plot_state_snapshots:
            # Load POD basis and temporal mean for state reconstruction
            pod_basis = None
            temporal_mean = None
            
            if os.path.exists(paths["pod_basis"]):
                pod_basis = np.load(paths["pod_basis"])
                logger.info(f"Loaded POD basis: {pod_basis.shape}")
            
            # Try to get temporal mean from initial_conditions
            if 'train_temporal_mean' in ICs:
                temporal_mean = ICs['train_temporal_mean']
                logger.info(f"Loaded temporal mean: {temporal_mean.shape}")
            
            # Get grid dimensions from preprocessing info
            preproc = np.load(paths["preprocessing_info"])
            n_y = int(preproc.get('n_y', cfg.n_y))
            n_x = int(preproc.get('n_x', cfg.n_x))
            
            train_ref_files = cfg.training_files
            test_ref_files = cfg.test_files if cfg.test_files else cfg.training_files
            
            # Prepare reduced states for plotting (ensemble mean)
            train_reduced = [np.mean(X, axis=0).T for X in train_pred['X_OpInf']]  # each becomes (r, n_time)
            test_reduced = [np.mean(X, axis=0).T for X in test_pred['X_OpInf']]
            
            generate_state_diagnostic_plots(
                train_reduced, train_ref_files, train_bounds,
                pod_basis, temporal_mean, n_y, n_x,
                cfg.engine, cfg.dt,
                os.path.join(paths["figures_dir"], "train"), logger,
                method_name="OpInf",
                prefix="train_", ref_offset=train_offset,
                plot_error=cfg.plot_state_error,
                plot_snapshots=cfg.plot_state_snapshots,
                n_snapshots=cfg.n_snapshot_samples
            )
            generate_state_diagnostic_plots(
                test_reduced, test_ref_files, test_bounds,
                pod_basis, temporal_mean, n_y, n_x,
                cfg.engine, cfg.dt,
                os.path.join(paths["figures_dir"], "test"), logger,
                method_name="OpInf",
                prefix="test_", ref_offset=test_offset,
                plot_error=cfg.plot_state_error,
                plot_snapshots=cfg.plot_state_snapshots,
                n_snapshots=cfg.n_snapshot_samples
            )
        
        # Print summary
        print_header("EVALUATION SUMMARY")
        print("\n  Training Data:")
        for traj in train_metrics['trajectories']:
            print(f"    Traj {traj['trajectory'] + 1}: "
                  f"Γn err=[{traj['err_mean_Gamma_n']:.4f}, {traj['err_std_Gamma_n']:.4f}], "
                  f"Γc err=[{traj['err_mean_Gamma_c']:.4f}, {traj['err_std_Gamma_c']:.4f}]")
        
        print("\n  Test Data:")
        for traj in test_metrics['trajectories']:
            print(f"    Traj {traj['trajectory'] + 1}: "
                  f"Γn err=[{traj['err_mean_Gamma_n']:.4f}, {traj['err_std_Gamma_n']:.4f}], "
                  f"Γc err=[{traj['err_mean_Gamma_c']:.4f}, {traj['err_std_Gamma_c']:.4f}]")
        
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
