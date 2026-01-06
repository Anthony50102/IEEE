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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config, save_config, setup_logging, save_step_status,
    check_step_completed, get_output_paths, print_header,
)
from data import load_ensemble, load_preprocessing_info
from evaluation import compute_ensemble_predictions, compute_metrics
from shared.plotting import plot_gamma_predictions


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
            plot_gamma_predictions(train_pred, cfg.training_files, train_bounds,
                                   cfg.dt, cfg.engine, 
                                   os.path.join(paths["figures_dir"], "train"), logger,
                                   start_offset=train_offset)
            plot_gamma_predictions(test_pred, 
                                   cfg.test_files if cfg.test_files else cfg.training_files,
                                   test_bounds, cfg.dt, cfg.engine,
                                   os.path.join(paths["figures_dir"], "test"), logger,
                                   start_offset=test_offset)
        
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
