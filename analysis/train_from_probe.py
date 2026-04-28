"""
Direct model training from probe results.

Skips the full sweep — uses the best parameters identified by
probe_regularization.py to train operators and save them for step 3.

Usage:
    python analysis/train_from_probe.py \
        --run-dir /path/to/step1_output \
        --config opinf/config/opinf_hw_closure_targeted.yaml
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))

from core import get_quadratic_terms, get_cubic_diagonal_terms, solve_difference_model
from training import evaluate_hyperparameters, compute_operators, select_models
from utils import load_config, save_config, get_output_paths, save_step_status


def load_data(run_dir):
    """Load learning matrices and gamma reference."""
    lm_path = os.path.join(run_dir, 'learning_matrices.npz')
    gamma_path = os.path.join(run_dir, 'gamma_reference.npz')

    learning = np.load(lm_path, allow_pickle=True)
    gamma_ref = np.load(gamma_path, allow_pickle=True)

    data = {
        'D_state': learning['D_state'],
        'Y_state': learning['Y_state'],
        'D_state_2': learning['D_state_2'],
        'D_out': learning['D_out'],
        'D_out_2': learning['D_out_2'],
        'X_state': learning['X_state'],
        'mean_Xhat': learning['mean_Xhat'],
        'scaling_Xhat': learning['scaling_Xhat'],
        'Y_Gamma': gamma_ref['Y_Gamma'],
        'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
        'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
        'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
        'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
        'include_cubic': bool(learning['include_cubic']) if 'include_cubic' in learning else False,
        'include_constant': bool(learning['include_constant']) if 'include_constant' in learning else False,
        'closure_enabled': bool(learning['closure_enabled']) if 'closure_enabled' in learning else False,
    }
    learning.close()
    gamma_ref.close()
    return data


def main():
    parser = argparse.ArgumentParser(description='Train from probe results')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--r', type=int, default=75)
    parser.add_argument('--n-steps', type=int, default=20000)
    parser.add_argument('--training-end', type=int, default=8000)
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING FROM PROBE RESULTS")
    print("=" * 60)

    # Load probe results
    probe_path = os.path.join(args.run_dir, 'probe_results.npz')
    probe = np.load(probe_path, allow_pickle=True)
    best_params = probe['best_params']
    best_result = probe['best_result'].item()
    print(f"  Probe best error: {best_result['total_error']:.4f}")
    print(f"  Best params: {best_params}")

    # Load data
    data = load_data(args.run_dir)
    r = args.r
    print(f"  D_state shape: {data['D_state'].shape}")
    print(f"  Closure: cubic={data['include_cubic']}, constant={data['include_constant']}")

    # Build a focused grid around the probe's best params
    # ±0.5 decades in 5 steps for state params, wider for output params
    asl_best, asq_best, asc_best, aol_best, aoq_best = best_params

    alpha_sl_grid = np.logspace(np.log10(asl_best) - 0.3, np.log10(asl_best) + 0.3, 3)
    alpha_sq_grid = np.logspace(np.log10(asq_best) - 0.3, np.log10(asq_best) + 0.3, 3)
    alpha_ol_grid = np.logspace(np.log10(max(aol_best, 1e-6)) - 1, np.log10(max(aol_best, 1e-6)) + 1, 3)
    alpha_oq_grid = np.logspace(np.log10(max(aoq_best, 1e-2)) - 1, np.log10(max(aoq_best, 1e-2)) + 1, 3)

    if data['include_cubic'] and asc_best > 0:
        alpha_sc_grid = np.logspace(np.log10(asc_best) - 0.3, np.log10(asc_best) + 0.3, 3)
    else:
        alpha_sc_grid = [0.0]

    from itertools import product
    param_grid = list(product(alpha_sl_grid, alpha_sq_grid, alpha_ol_grid, alpha_oq_grid, alpha_sc_grid))
    n_total = len(param_grid)
    print(f"\n  Fine grid: {len(alpha_sl_grid)}x{len(alpha_sq_grid)}x{len(alpha_ol_grid)}x{len(alpha_oq_grid)}x{len(alpha_sc_grid)} = {n_total} combos")

    # Evaluate all combos
    results = []
    n_nan = 0
    t0 = time.time()

    for i, (asl, asq, aol, aoq, asc) in enumerate(param_grid):
        result = evaluate_hyperparameters(
            asl, asq, aol, aoq, data, r, args.n_steps, args.training_end,
            alpha_state_cubic=asc,
        )
        if result['is_nan']:
            n_nan += 1
        else:
            results.append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_total}] {len(results)} valid, {n_nan} NaN ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Sweep complete: {len(results)} valid, {n_nan} NaN ({elapsed:.0f}s)")

    if not results:
        print("  ERROR: No valid models!")
        return

    # Sort by total error
    results.sort(key=lambda x: x['total_error'])

    print(f"\n  TOP 10 RESULTS:")
    print(f"  {'rank':>4s} {'total':>8s} {'mn_Gn':>8s} {'sd_Gn':>8s} {'mn_Gc':>8s} {'sd_Gc':>8s} "
          f"{'a_sl':>10s} {'a_sq':>10s} {'a_sc':>10s} {'a_ol':>10s} {'a_oq':>10s}")
    for idx, res in enumerate(results[:10]):
        print(f"  {idx+1:4d} {res['total_error']:8.4f} {res['mean_err_Gamma_n']:8.4f} "
              f"{res['std_err_Gamma_n']:8.4f} {res['mean_err_Gamma_c']:8.4f} "
              f"{res['std_err_Gamma_c']:8.4f} {res['alpha_state_lin']:10.2e} "
              f"{res['alpha_state_quad']:10.2e} {res['alpha_state_cubic']:10.2e} "
              f"{res['alpha_out_lin']:10.2e} {res['alpha_out_quad']:10.2e}")

    # Select models using thresholds from config
    cfg = load_config(args.config)
    import logging
    _logger = logging.getLogger("train_from_probe")
    _logger.addHandler(logging.StreamHandler())
    _logger.setLevel(logging.INFO)
    selected = select_models(results, cfg.threshold_mean, cfg.threshold_std, _logger)

    if not selected:
        print("  No models met threshold. Using top 20 by total error.")
        selected = results[:20]

    print(f"\n  Selected {len(selected)} models for operator computation")

    # Compute and save operators
    operators_dir = os.path.join(args.run_dir, 'operators')
    os.makedirs(operators_dir, exist_ok=True)

    models = []
    for idx, params in enumerate(selected):
        model = compute_operators(params, data, r)
        filepath = os.path.join(operators_dir, f"model_{idx:04d}.npz")
        np.savez(filepath, **model)
        models.append(model)
    print(f"  Saved {len(models)} operator sets to {operators_dir}")

    # Save ensemble info
    paths = get_output_paths(args.run_dir)
    ensemble = {
        'n_models': len(models),
        'r': r,
        'config': str(args.config),
        'closure_enabled': data['closure_enabled'],
        'include_cubic': data['include_cubic'],
        'include_constant': data['include_constant'],
    }
    np.savez(paths["ensemble_models"], **ensemble)

    # Save sweep results
    np.savez(paths["sweep_results"],
             n_total=len(results), n_selected=len(selected),
             best_error=selected[0]['total_error'])

    # Mark step 2 as completed
    save_step_status(args.run_dir, "step_2", "completed", {
        "n_models": len(selected),
        "best_error": float(selected[0]['total_error']),
        "sweep_time_seconds": elapsed,
    })

    save_config(cfg, args.run_dir, step_name="step_2")

    print(f"\n  Best total error: {results[0]['total_error']:.4f}")
    print("  Step 2 marked complete. Ready for step 3.")
    print("=" * 60)


if __name__ == '__main__':
    main()
