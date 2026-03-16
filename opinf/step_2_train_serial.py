"""
Step 2: ROM Training via Hyperparameter Sweep (Serial).

Non-MPI serial version of step_2_train.py. Use when MPI is not available.

Usage:
    python step_2_train_serial.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import gc
import os
import time
import numpy as np
from itertools import product

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config, save_config, setup_logging,
    save_step_status, check_step_completed, get_output_paths,
    print_header, print_config_summary,
)
from data import save_ensemble
from training import (
    evaluate_hyperparameters, log_error_statistics,
    select_models, compute_operators,
)


def load_data_serial(paths: dict, logger) -> dict:
    """Load pre-computed data without MPI shared memory."""
    logger.info("Loading pre-computed data...")
    learning = np.load(paths["learning_matrices"])
    gamma_ref = np.load(paths["gamma_ref"])

    data = {
        'X_state': learning['X_state'].copy(),
        'Y_state': learning['Y_state'].copy(),
        'D_state': learning['D_state'].copy(),
        'D_state_2': learning['D_state_2'].copy(),
        'D_out': learning['D_out'].copy(),
        'D_out_2': learning['D_out_2'].copy(),
        'mean_Xhat': learning['mean_Xhat'].copy(),
        'scaling_Xhat': float(learning['scaling_Xhat']),
        'Y_Gamma': gamma_ref['Y_Gamma'].copy(),
        'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
        'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
        'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
        'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
    }
    learning.close()
    gamma_ref.close()
    logger.info("Data loaded successfully")
    return data


def serial_hyperparameter_sweep(cfg, data: dict, logger) -> list:
    """Run hyperparameter sweep serially."""
    param_grid = list(product(cfg.state_lin, cfg.state_quad, cfg.output_lin, cfg.output_quad))
    n_total = len(param_grid)
    logger.info(f"Serial sweep: {n_total:,} combinations")

    results = []
    n_nan = 0

    for i, (asl, asq, aol, aoq) in enumerate(param_grid):
        result = evaluate_hyperparameters(
            asl, asq, aol, aoq, data, cfg.r, cfg.n_steps, cfg.training_end
        )

        if result['is_nan']:
            n_nan += 1
        else:
            results.append(result)

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{n_total} ({len(results)} valid, {n_nan} NaN)")
            gc.collect()

    logger.info(f"Sweep complete: {len(results)} valid, {n_nan} NaN")
    return results


def recompute_operators_serial(selected: list, data: dict, r: int,
                                operators_dir: str, logger) -> list:
    """Recompute operator matrices for selected models serially."""
    logger.info(f"Recomputing operators for {len(selected)} models...")
    os.makedirs(operators_dir, exist_ok=True)

    models = []
    for idx, params in enumerate(selected):
        model = compute_operators(params, data, r)
        filepath = os.path.join(operators_dir, f"model_{idx:04d}.npz")
        np.savez(filepath, **model)
        models.append((model['total_error'], model))

    models.sort(key=lambda x: x[0])
    logger.info(f"  Saved {len(models)} models to {operators_dir}")
    return models


def main():
    """Main entry point for serial Step 2."""
    parser = argparse.ArgumentParser(description="Step 2: ROM Training (Serial)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory from Step 1")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only show error statistics (for threshold tuning)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.run_dir = args.run_dir

    logger = setup_logging("step_2", args.run_dir, cfg.log_level, rank=0)

    print_header("STEP 2: ROM TRAINING (SERIAL)")
    print(f"  Run directory: {args.run_dir}")
    print_config_summary(cfg)

    if not check_step_completed(args.run_dir, "step_1"):
        logger.error("Step 1 has not completed!")
        return

    save_step_status(args.run_dir, "step_2", "running")
    save_config(cfg, args.run_dir, step_name="step_2")

    paths = get_output_paths(args.run_dir)

    try:
        data = load_data_serial(paths, logger)

        actual_r = data['X_state'].shape[1]
        if actual_r != cfg.r:
            logger.warning(f"Overriding cfg.r={cfg.r} with actual r={actual_r} from step 1 data")
        cfg.r = actual_r

        t_start = time.time()
        results = serial_hyperparameter_sweep(cfg, data, logger)
        t_elapsed = time.time() - t_start

        logger.info(f"Sweep completed in {t_elapsed:.1f}s")

        if not results:
            logger.error("No valid models found!")
            save_step_status(args.run_dir, "step_2", "failed", {"error": "No valid models"})
            return

        log_error_statistics(results, logger)

        if args.stats_only:
            logger.info("STATS-ONLY MODE: Exiting without saving.")
            print_header("STATS-ONLY COMPLETE")
            return

        selected = select_models(results, cfg.threshold_mean, cfg.threshold_std, logger)

        if not selected:
            logger.error("No models met selection criteria!")
            save_step_status(args.run_dir, "step_2", "failed", {"error": "No models selected"})
            return

        models = recompute_operators_serial(selected, data, cfg.r, paths["operators_dir"], logger)

        save_ensemble(models, paths["ensemble_models"], cfg, logger)

        np.savez(paths["sweep_results"],
                 n_total=len(results), n_selected=len(selected),
                 best_error=selected[0]['total_error'])

        print_header("MODEL SELECTION SUMMARY")
        print(f"  Total valid: {len(results)}")
        print(f"  Selected: {len(selected)}")
        print(f"  Best error: {selected[0]['total_error']:.6e}")

        save_step_status(args.run_dir, "step_2", "completed", {
            "n_models": len(selected),
            "best_error": float(selected[0]['total_error']),
            "sweep_time_seconds": t_elapsed,
        })

        print_header("STEP 2 COMPLETE")
        logger.info("Step 2 completed successfully")

    except Exception as e:
        logger.error(f"Step 2 failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_2", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
