#!/usr/bin/env python3
"""
Ensemble Re-evaluation: test different model selection strategies on existing runs.

Loads all models from an OpInf ensemble, re-selects with different strategies
(thresholds, top-K), and computes test errors for each. Does NOT re-train —
just re-evaluates existing models.

Usage:
    python analysis/ensemble_resweep.py --run-dir local_output/20260330_153626_opinf_ks_closure_test

Author: Anthony Poole
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opinf"))
from core import get_quadratic_terms, get_cubic_diagonal_terms, solve_difference_model


def load_run_data(run_dir: str) -> dict:
    """Load all data needed for re-evaluation from a completed run directory."""
    ens = np.load(os.path.join(run_dir, 'ensemble_models.npz'), allow_pickle=True)
    lm = np.load(os.path.join(run_dir, 'learning_matrices.npz'), allow_pickle=True)
    ic = np.load(os.path.join(run_dir, 'initial_conditions.npz'), allow_pickle=True)
    bd = np.load(os.path.join(run_dir, 'data_boundaries.npz'), allow_pickle=True)
    pod_basis = np.load(os.path.join(run_dir, 'POD_basis_Ur.npy'))
    X_hat_test = np.load(os.path.join(run_dir, 'X_hat_test.npy'))

    n_models = int(ens['num_models'])
    r = int(ens['r'])

    # Load all model operators and errors
    models = []
    for i in range(n_models):
        m = {
            'A': ens[f'model_{i}_A'], 'F': ens[f'model_{i}_F'],
            'C': ens[f'model_{i}_C'], 'G': ens[f'model_{i}_G'],
            'c': ens[f'model_{i}_c'],
            'total_error': float(ens[f'model_{i}_total_error']),
            'mean_err_n': float(ens[f'model_{i}_mean_err_Gamma_n']),
            'std_err_n': float(ens[f'model_{i}_std_err_Gamma_n']),
            'mean_err_c': float(ens[f'model_{i}_mean_err_Gamma_c']),
            'std_err_c': float(ens[f'model_{i}_std_err_Gamma_c']),
        }
        H_key = f'model_{i}_H'
        if H_key in ens.keys():
            m['H'] = ens[H_key]
        cs_key = f'model_{i}_c_state'
        if cs_key in ens.keys():
            m['c_state'] = ens[cs_key]
        models.append(m)

    # Sort by total training error (best first)
    models.sort(key=lambda x: x['total_error'])

    return {
        'models': models,
        'r': r,
        'n_models': n_models,
        'mean_Xhat': lm['mean_Xhat'],
        'scaling_Xhat': float(lm['scaling_Xhat']),
        'test_ic': ic['test_ICs_reduced'][0],
        'train_ic': ic['train_ICs_reduced'][0],
        'train_mean': ic['train_temporal_mean'],
        'test_mean': ic['test_temporal_mean'],
        'pod_basis': pod_basis,
        'X_hat_test': X_hat_test,
        'X_hat_train': np.load(os.path.join(run_dir, 'X_hat_train.npy')),
        'n_test_steps': int(bd['test_boundaries'][1] - bd['test_boundaries'][0]),
        'n_train_steps': int(bd['train_boundaries'][1] - bd['train_boundaries'][0]),
    }


def predict_single_model(model, ic_vec, n_steps):
    """Run one model forward from initial condition."""
    A, F = model['A'], model['F']
    H = model.get('H', None)
    c_s = model.get('c_state', None)

    def f(x):
        result = A @ x + F @ get_quadratic_terms(x)
        if H is not None:
            result += H @ get_cubic_diagonal_terms(x)
        if c_s is not None:
            result += c_s
        return result

    is_nan, Xhat_pred = solve_difference_model(ic_vec, n_steps, f)
    if is_nan:
        return None
    return Xhat_pred.T  # (n_steps, r)


def compute_qoi_from_reduced(Xhat_pred, pod_basis, temporal_mean, mean_Xhat,
                              scaling_Xhat, model, pde="ks", dx=None):
    """Compute QoI from reduced predictions using the learned output model."""
    # Output model approach (C, G, c operators)
    X = Xhat_pred  # (n_steps, r)
    Xhat_scaled = (X - mean_Xhat) / scaling_Xhat
    Xhat_2 = get_quadratic_terms(Xhat_scaled)
    C, G, c = model['C'], model['G'], model['c']
    Y = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
    return Y[0, :], Y[1, :]  # Gamma_n, Gamma_c


def compute_ref_qoi_learned(X_hat, mean_Xhat, scaling_Xhat, models, pde="ks"):
    """
    Compute reference QoI from actual reduced data using the output model.
    
    Uses the best model's C, G, c operators on the TRUE reduced trajectory.
    """
    m = models[0]
    X = X_hat  # (n_steps, r)
    Xhat_scaled = (X - mean_Xhat) / scaling_Xhat
    Xhat_2 = get_quadratic_terms(Xhat_scaled)
    C, G, c = m['C'], m['G'], m['c']
    Y = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]
    return Y[0, :], Y[1, :]


def run_ensemble_sweep(data: dict, pde: str = "ks"):
    """Run the full ensemble size sweep and report results."""
    models = data['models']
    n_models = data['n_models']

    # Compute reference QoI for test data using the output model on TRUE test Xhat
    ref_test_n, ref_test_c = compute_ref_qoi_learned(
        data['X_hat_test'], data['mean_Xhat'], data['scaling_Xhat'], models)

    ref_test_mean_n = np.mean(ref_test_n)
    ref_test_std_n = np.std(ref_test_n, ddof=1)
    ref_test_mean_c = np.mean(ref_test_c)
    ref_test_std_c = np.std(ref_test_c, ddof=1)

    print(f"Test reference (learned output model on true Xhat):")
    print(f"  Gamma_n: mean={ref_test_mean_n:.4f}, std={ref_test_std_n:.4f}")
    print(f"  Gamma_c: mean={ref_test_mean_c:.4f}, std={ref_test_std_c:.4f}")

    # Same for training
    ref_train_n, ref_train_c = compute_ref_qoi_learned(
        data['X_hat_train'], data['mean_Xhat'], data['scaling_Xhat'], models)

    ref_train_mean_n = np.mean(ref_train_n)
    ref_train_std_n = np.std(ref_train_n, ddof=1)
    ref_train_mean_c = np.mean(ref_train_c)
    ref_train_std_c = np.std(ref_train_c, ddof=1)

    print(f"Train reference (learned output model on true Xhat):")
    print(f"  Gamma_n: mean={ref_train_mean_n:.4f}, std={ref_train_std_n:.4f}")
    print(f"  Gamma_c: mean={ref_train_mean_c:.4f}, std={ref_train_std_c:.4f}")

    # Pre-compute all test predictions
    print(f"\nPre-computing test predictions for {n_models} models...")
    all_pred_n = []
    all_pred_c = []
    n_failed = 0

    for i, m in enumerate(models):
        Xp = predict_single_model(m, data['test_ic'], data['n_test_steps'])
        if Xp is None:
            n_failed += 1
            continue
        gn, gc = compute_qoi_from_reduced(
            Xp, data['pod_basis'], data['test_mean'],
            data['mean_Xhat'], data['scaling_Xhat'], m, pde=pde)
        all_pred_n.append(gn)
        all_pred_c.append(gc)

        if (i + 1) % 500 == 0:
            print(f"  Computed {i+1}/{n_models} ({n_failed} failed)")

    print(f"  Done: {len(all_pred_n)} valid, {n_failed} failed")

    all_pred_n = np.array(all_pred_n)
    all_pred_c = np.array(all_pred_c)

    # Test different ensemble sizes (top-K)
    print(f"\n{'='*70}")
    print(f"ENSEMBLE SIZE SWEEP (top-K by training error)")
    print(f"{'='*70}")
    print(f"{'K':>6} {'E mean%':>8} {'E std%':>8} {'Ens mean%':>9} {'Ens std%':>9}")
    print("-" * 50)

    sizes = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    sizes = [s for s in sizes if s <= len(all_pred_n)]
    sizes.append(len(all_pred_n))

    best_total = float('inf')
    best_k = 0

    for k in sizes:
        ens_n = np.mean(all_pred_n[:k], axis=0)
        ens_c = np.mean(all_pred_c[:k], axis=0)

        me_n = abs(ref_test_mean_n - np.mean(ens_n)) / abs(ref_test_mean_n)
        se_n = abs(ref_test_std_n - np.std(ens_n, ddof=1)) / ref_test_std_n
        me_c = abs(ref_test_mean_c - np.mean(ens_c)) / abs(ref_test_mean_c)
        se_c = abs(ref_test_std_c - np.std(ens_c, ddof=1)) / ref_test_std_c
        total = me_n + se_n + me_c + se_c

        if total < best_total:
            best_total = total
            best_k = k

        print(f"{k:>6} {me_n:>7.1%} {se_n:>7.1%} {me_c:>8.1%} {se_c:>8.1%}")

    print(f"\n★ Best top-K: K={best_k} (total error: {best_total:.4f})")

    # Test threshold-based selection
    print(f"\n{'='*70}")
    print(f"THRESHOLD-BASED SELECTION")
    print(f"{'='*70}")
    print(f"{'thresh_m':>9} {'thresh_s':>9} {'N':>6} {'E mean%':>8} {'E std%':>8} {'Ens mean%':>9} {'Ens std%':>9}")
    print("-" * 70)

    for tm, ts in [(0.50, 0.50), (0.10, 0.30), (0.05, 0.30), (0.05, 0.20),
                   (0.05, 0.15), (0.03, 0.15), (0.03, 0.10), (0.02, 0.10)]:
        # Filter by thresholds on training errors
        mask = np.array([
            m['mean_err_n'] < tm and m['std_err_n'] < ts
            and m['mean_err_c'] < tm and m['std_err_c'] < ts
            for m in models[:len(all_pred_n)]
        ])
        if mask.sum() == 0:
            print(f"{tm:>9.2f} {ts:>9.2f} {'0':>6} {'N/A':>8} {'N/A':>8} {'N/A':>9} {'N/A':>9}")
            continue

        ens_n = np.mean(all_pred_n[mask], axis=0)
        ens_c = np.mean(all_pred_c[mask], axis=0)

        me_n = abs(ref_test_mean_n - np.mean(ens_n)) / abs(ref_test_mean_n)
        se_n = abs(ref_test_std_n - np.std(ens_n, ddof=1)) / ref_test_std_n
        me_c = abs(ref_test_mean_c - np.mean(ens_c)) / abs(ref_test_mean_c)
        se_c = abs(ref_test_std_c - np.std(ens_c, ddof=1)) / ref_test_std_c

        print(f"{tm:>9.2f} {ts:>9.2f} {mask.sum():>6} {me_n:>7.1%} {se_n:>7.1%} {me_c:>8.1%} {se_c:>8.1%}")

    return all_pred_n, all_pred_c, ref_test_n, ref_test_c


def main():
    parser = argparse.ArgumentParser(description="Ensemble re-evaluation sweep")
    parser.add_argument('--run-dir', required=True, help="Path to completed OpInf run directory")
    parser.add_argument('--pde', default='ks', choices=['ks', 'hw2d'], help="PDE type")
    args = parser.parse_args()

    print(f"Loading data from: {args.run_dir}")
    data = load_run_data(args.run_dir)
    print(f"Loaded {data['n_models']} models, r={data['r']}")
    print(f"Test: {data['n_test_steps']} steps, Train: {data['n_train_steps']} steps")

    run_ensemble_sweep(data, pde=args.pde)


if __name__ == "__main__":
    main()
