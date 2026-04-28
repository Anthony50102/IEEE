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
    _prepare_learning_matrices_impl,
)
from core import get_quadratic_terms, solve_difference_model


def load_data_serial(paths: dict, logger) -> dict:
    """Load pre-computed data without MPI shared memory."""
    logger.info("Loading pre-computed data...")
    learning = np.load(paths["learning_matrices"], allow_pickle=True)
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
        # Closure flags (may not exist in older runs)
        'include_cubic': bool(learning['include_cubic']) if 'include_cubic' in learning else False,
        'include_constant': bool(learning['include_constant']) if 'include_constant' in learning else False,
        'closure_enabled': bool(learning['closure_enabled']) if 'closure_enabled' in learning else False,
    }
    learning.close()
    gamma_ref.close()
    logger.info(f"Data loaded successfully (closure={data['closure_enabled']})")
    return data


def serial_hyperparameter_sweep(cfg, data: dict, logger,
                                test_IC=None, n_test_steps=0,
                                test_ref_energy=None,
                                test_ref_reduced=None) -> list:
    """Run hyperparameter sweep serially."""
    include_cubic = data.get('include_cubic', False)
    if include_cubic and len(cfg.state_cubic) > 0:
        param_grid = list(product(cfg.state_lin, cfg.state_quad, cfg.output_lin, cfg.output_quad, cfg.state_cubic))
    else:
        param_grid = list(product(cfg.state_lin, cfg.state_quad, cfg.output_lin, cfg.output_quad))
    n_total = len(param_grid)
    logger.info(f"Serial sweep: {n_total:,} combinations (closure={include_cubic})")
    if test_IC is not None:
        logger.info(f"  Test rollout enabled: {n_test_steps} steps from test IC")

    results = []
    n_nan = 0

    for i, params in enumerate(param_grid):
        if len(params) == 5:
            asl, asq, aol, aoq, asc = params
        else:
            asl, asq, aol, aoq = params
            asc = 0.0

        result = evaluate_hyperparameters(
            asl, asq, aol, aoq, data, cfg.r, cfg.n_steps, cfg.training_end,
            alpha_state_cubic=asc,
            stability_projection=getattr(cfg, 'stability_projection', False),
            stability_max_rho=getattr(cfg, 'stability_max_rho', 0.999),
            test_IC=test_IC,
            n_test_steps=n_test_steps,
            test_ref_energy=test_ref_energy,
            test_ref_reduced=test_ref_reduced,
        )

        if result['is_nan']:
            n_nan += 1
        else:
            results.append(result)

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{n_total} ({len(results)} valid, {n_nan} NaN)")
            gc.collect()

    logger.info(f"Sweep complete: {len(results)} valid, {n_nan} NaN")

    if test_IC is not None:
        n_stable = sum(1 for r in results if r.get('test_stable', True))
        logger.info(f"Test stability: {n_stable}/{len(results)} models stable on test rollout")

    return results


def recompute_operators_serial(selected: list, data: dict, r: int,
                                operators_dir: str, logger,
                                stability_projection: bool = False,
                                stability_max_rho: float = 0.999) -> list:
    """Recompute operator matrices for selected models serially."""
    logger.info(f"Recomputing operators for {len(selected)} models...")
    os.makedirs(operators_dir, exist_ok=True)

    models = []
    for idx, params in enumerate(selected):
        model = compute_operators(params, data, r,
                                  stability_projection=stability_projection,
                                  stability_max_rho=stability_max_rho)
        filepath = os.path.join(operators_dir, f"model_{idx:04d}.npz")
        np.savez(filepath, **model)
        models.append((model['total_error'], model))

    models.sort(key=lambda x: x[0])
    logger.info(f"  Saved {len(models)} models to {operators_dir}")
    return models


def build_data_for_rank(Xhat_full, train_boundaries, r_c, gamma_ref, logger):
    """Build learning matrices for a specific rank by truncating max-rank data."""
    Xhat_trunc = Xhat_full[:, :r_c]
    learning = _prepare_learning_matrices_impl(
        Xhat_trunc, train_boundaries, r_c, logger,
        closure_enabled=False, closure_cubic=False, closure_constant=False,
    )
    learning['Y_Gamma'] = gamma_ref['Y_Gamma']
    learning['mean_Gamma_n'] = gamma_ref['mean_Gamma_n']
    learning['std_Gamma_n'] = gamma_ref['std_Gamma_n']
    learning['mean_Gamma_c'] = gamma_ref['mean_Gamma_c']
    learning['std_Gamma_c'] = gamma_ref['std_Gamma_c']
    return learning


def compute_val_energy_error(X_pred_val, Ur, u_mean, val_ref_qoi, n_x, n_y):
    """Compute relative L2 QoI error on validation window.
    
    Computes 0.5/N * ||u_mean + Ur @ x||^2 which equals enstrophy
    for NS (vorticity state) or energy for KS.
    """
    N = n_x * n_y
    energy_a = float(np.sum(u_mean ** 2))
    energy_b = Ur.T @ u_mean

    norms_sq = np.sum(X_pred_val ** 2, axis=1)
    cross = X_pred_val @ energy_b
    pred_qoi = 0.5 / N * (norms_sq + 2.0 * cross + energy_a)

    err = np.linalg.norm(pred_qoi - val_ref_qoi) / np.linalg.norm(val_ref_qoi)
    return float(err), pred_qoi


def validation_rank_sweep(cfg, paths, logger):
    """
    Sweep over (rank, regularization) using a validation window.

    Strategy:
    1. Step 1 computed POD at r_max. Load Xhat_train at r_max.
    2. Split: [0, train_end) for fitting, [val_start, val_end) for scoring.
    3. For each r_c in r_candidates, truncate and rebuild learning matrices.
    4. For each (r_c, alpha), fit operators, rollout through val, score.
    5. Select best (r*, alpha*) by validation QoI error with simplicity tiebreaker.
    6. Retrain winner on full [0, val_end) for the final model.
    """
    import h5py

    val_start = cfg.val_start
    val_end = cfg.val_end
    train_end = cfg.train_end  # Actual training fit end (before val)
    r_candidates = cfg.r_candidates

    # Load max-rank reduced data from step 1
    Xhat_train_full = np.load(paths["xhat_train"])     # (n_train_orig, r_max)
    Ur_full = np.load(paths["pod_basis"])               # (n_spatial, r_max)
    ic_data = np.load(paths["initial_conditions"])
    u_mean = ic_data['train_temporal_mean']
    ic_data.close()

    r_max = Xhat_train_full.shape[1]
    n_x = getattr(cfg, 'n_x', 64)
    n_y = getattr(cfg, 'n_y', 64)
    n_train_fit = train_end - cfg.train_start       # snapshots for fitting
    n_val = val_end - val_start                     # snapshots for validation

    logger.info(f"Validation sweep: train [0,{n_train_fit}), val [{val_start},{val_end}) "
                f"= {n_val} steps, r_candidates={r_candidates}")

    # Load reference validation QoI from HDF5
    # Our ROM state is vorticity (ω), so 0.5/N * ||ω||² = ENSTROPHY.
    # We must compare against enstrophy, not kinetic energy.
    hdf5_path = cfg.training_files[0]
    pde = getattr(cfg, 'pde', 'hw2d')
    with h5py.File(hdf5_path, 'r') as hf:
        if pde == 'ns':
            val_ref_qoi = np.array(hf['enstrophy'][val_start:val_end])
        else:
            val_ref_qoi = np.array(hf['energy'][val_start:val_end])

    # Load reference gamma (only for training portion)
    gamma_ref_data = np.load(paths["gamma_ref"])
    gamma_ref = {
        'Y_Gamma': gamma_ref_data['Y_Gamma'][:, :n_train_fit],
        'mean_Gamma_n': float(np.mean(gamma_ref_data['Y_Gamma'][0, :n_train_fit])),
        'std_Gamma_n': float(np.std(gamma_ref_data['Y_Gamma'][0, :n_train_fit], ddof=1)),
        'mean_Gamma_c': float(np.mean(gamma_ref_data['Y_Gamma'][1, :n_train_fit])),
        'std_Gamma_c': float(np.std(gamma_ref_data['Y_Gamma'][1, :n_train_fit], ddof=1)),
    }
    gamma_ref_data.close()

    # Fit portion of training data
    Xhat_fit = Xhat_train_full[:n_train_fit, :]   # (n_train_fit, r_max)
    # Validation IC = last snapshot of fit window
    val_ic_full = Xhat_train_full[n_train_fit - 1, :]  # (r_max,)
    # Validation reference reduced states
    Xhat_val_ref = Xhat_train_full[n_train_fit:n_train_fit + n_val, :]  # (n_val, r_max)

    train_boundaries = [0, n_train_fit]

    # Build alpha grid
    param_grid = list(product(cfg.state_lin, cfg.state_quad))
    logger.info(f"  Alpha grid: {len(param_grid)} combos × {len(r_candidates)} ranks "
                f"= {len(param_grid) * len(r_candidates)} total")

    all_results = []

    for r_c in r_candidates:
        if r_c > r_max:
            logger.warning(f"  r_c={r_c} > r_max={r_max}, skipping")
            continue

        logger.info(f"  --- Rank r={r_c} ---")
        data = build_data_for_rank(Xhat_fit, train_boundaries, r_c, gamma_ref, logger)

        val_ic = val_ic_full[:r_c]
        Ur_trunc = Ur_full[:, :r_c]

        n_valid = 0
        n_nan = 0

        for asl, asq in param_grid:
            s = r_c * (r_c + 1) // 2
            d_state = r_c + s

            reg_state = np.zeros(d_state)
            reg_state[:r_c] = asl
            reg_state[r_c:r_c + s] = asq

            DtD = data['D_state_2'] + np.diag(reg_state)
            try:
                O = np.linalg.solve(DtD, data['D_state'].T @ data['Y_state']).T
            except np.linalg.LinAlgError:
                n_nan += 1
                continue

            A = O[:, :r_c]
            F = O[:, r_c:r_c + s]

            def f(x, _A=A, _F=F):
                return _A @ x + _F @ get_quadratic_terms(x)

            # Rollout on validation window
            is_nan, Xhat_pred = solve_difference_model(val_ic, n_val + 1, f)
            if is_nan:
                n_nan += 1
                continue

            X_pred_val = Xhat_pred.T[1:, :]  # skip IC, shape (n_val, r_c)

            # Compute validation energy error
            val_err, _ = compute_val_energy_error(
                X_pred_val, Ur_trunc, u_mean, val_ref_qoi, n_x, n_y)

            # Also compute validation state MSE for diagnostics
            Xhat_val_ref_trunc = Xhat_val_ref[:n_val, :r_c]
            n_cmp = min(len(X_pred_val), len(Xhat_val_ref_trunc))
            val_state_mse = float(np.mean(
                (X_pred_val[:n_cmp] - Xhat_val_ref_trunc[:n_cmp]) ** 2))

            all_results.append({
                'r': r_c,
                'alpha_state_lin': asl,
                'alpha_state_quad': asq,
                'val_energy_err': val_err,
                'val_state_mse': val_state_mse,
                'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(A)))),
            })
            n_valid += 1

        logger.info(f"    r={r_c}: {n_valid} valid, {n_nan} NaN/singular")

    if not all_results:
        logger.error("No valid models found in validation sweep!")
        return None

    # Sort by validation energy error
    all_results.sort(key=lambda x: x['val_energy_err'])
    best = all_results[0]

    # Simplicity tiebreaker: among models within 5% of best, prefer smallest r
    tol = 0.05
    best_val = best['val_energy_err']
    candidates_within_tol = [r for r in all_results
                             if r['val_energy_err'] <= best_val * (1 + tol)]
    candidates_within_tol.sort(key=lambda x: (x['r'], x['alpha_state_lin']))
    best = candidates_within_tol[0]

    logger.info(f"\n  === VALIDATION SWEEP RESULTS ===")
    logger.info(f"  Best: r={best['r']}, α_lin={best['alpha_state_lin']:.2e}, "
                f"α_quad={best['alpha_state_quad']:.2e}")
    logger.info(f"  Val energy error: {best['val_energy_err']:.4%}")
    logger.info(f"  Val state MSE: {best['val_state_mse']:.6e}")
    logger.info(f"  ρ(A): {best['spectral_radius']:.6f}")
    logger.info(f"  Models within {tol:.0%} of best: {len(candidates_within_tol)}")

    # Log top 10
    logger.info(f"  Top 10:")
    for i, res in enumerate(all_results[:10]):
        logger.info(f"    {i+1}. r={res['r']}, α_lin={res['alpha_state_lin']:.2e}, "
                     f"α_quad={res['alpha_state_quad']:.2e}, "
                     f"val_E_err={res['val_energy_err']:.4%}, ρ={res['spectral_radius']:.4f}")

    # === RETRAIN on full [0, val_end) with winning hyperparameters ===
    logger.info(f"\n  Retraining best model on [0,{val_end}) (train+val)...")
    r_best = best['r']
    Xhat_retrain = Xhat_train_full[:val_end - cfg.train_start, :r_best]

    # Load full gamma for retrain window
    with h5py.File(hdf5_path, 'r') as hf:
        energy_retrain = np.array(hf['energy'][cfg.train_start:val_end])
        enstrophy_retrain = np.array(hf['enstrophy'][cfg.train_start:val_end])
    Y_Gamma_retrain = np.vstack([energy_retrain, enstrophy_retrain])

    gamma_ref_retrain = {
        'Y_Gamma': Y_Gamma_retrain,
        'mean_Gamma_n': float(np.mean(energy_retrain)),
        'std_Gamma_n': float(np.std(energy_retrain, ddof=1)),
        'mean_Gamma_c': float(np.mean(enstrophy_retrain)),
        'std_Gamma_c': float(np.std(enstrophy_retrain, ddof=1)),
    }
    retrain_boundaries = [0, val_end - cfg.train_start]
    data_retrain = build_data_for_rank(
        Xhat_retrain, retrain_boundaries, r_best, gamma_ref_retrain, logger)

    s = r_best * (r_best + 1) // 2
    d_state = r_best + s
    reg_state = np.zeros(d_state)
    reg_state[:r_best] = best['alpha_state_lin']
    reg_state[r_best:r_best + s] = best['alpha_state_quad']

    DtD = data_retrain['D_state_2'] + np.diag(reg_state)
    O = np.linalg.solve(DtD, data_retrain['D_state'].T @ data_retrain['Y_state']).T
    A_final = O[:, :r_best]
    F_final = O[:, r_best:r_best + s]

    rho_final = float(np.max(np.abs(np.linalg.eigvals(A_final))))
    logger.info(f"  Retrained model: r={r_best}, ρ(A)={rho_final:.6f}")

    # Save truncated step-1 artifacts so step 3 sees the correct rank
    logger.info(f"  Saving truncated artifacts for r={r_best}...")
    Ur_trunc_final = Ur_full[:, :r_best]
    np.save(paths["pod_basis"], Ur_trunc_final)

    # Truncate and re-save Xhat_train/test and ICs
    Xhat_test_full = np.load(paths["xhat_test"])  # (n_test, r_max)
    np.save(paths["xhat_train"], Xhat_train_full[:, :r_best])
    np.save(paths["xhat_test"], Xhat_test_full[:, :r_best])

    ic_data2 = np.load(paths["initial_conditions"])
    ic_dict = {k: ic_data2[k] for k in ic_data2.files}
    ic_data2.close()
    if 'train_ICs_reduced' in ic_dict:
        ic_dict['train_ICs_reduced'] = ic_dict['train_ICs_reduced'][:, :r_best]
    if 'test_ICs_reduced' in ic_dict:
        ic_dict['test_ICs_reduced'] = ic_dict['test_ICs_reduced'][:, :r_best]
    np.savez(paths["initial_conditions"], **ic_dict)

    # Re-save learning matrices at the retrain rank
    np.savez(paths["learning_matrices"], **data_retrain)

    # Update preprocessing_info with the new rank
    preproc = np.load(paths["preprocessing_info"], allow_pickle=True)
    preproc_dict = {k: preproc[k] for k in preproc.files}
    preproc.close()
    preproc_dict['r_actual'] = r_best
    np.savez(paths["preprocessing_info"], **preproc_dict)

    return {
        'best_result': best,
        'all_results': all_results,
        'A': A_final,
        'F': F_final,
        'r': r_best,
        'rho': rho_final,
        'mean_Xhat': data_retrain['mean_Xhat'],
        'scaling_Xhat': data_retrain['scaling_Xhat'],
        'retrain_n_steps': val_end - cfg.train_start,
    }


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
        # Check if validation-based selection is enabled
        use_val = getattr(cfg, 'val_end', 0) > 0 and len(getattr(cfg, 'r_candidates', [])) > 0

        if use_val:
            logger.info("=== VALIDATION-BASED MODEL SELECTION ===")
            t_start = time.time()
            val_result = validation_rank_sweep(cfg, paths, logger)
            t_elapsed = time.time() - t_start

            if val_result is None:
                save_step_status(args.run_dir, "step_2", "failed",
                                 {"error": "No valid models in validation sweep"})
                return

            # Save the winning model in pipeline-compatible format
            best = val_result['best_result']
            r_best = val_result['r']
            operators_dir = paths["operators_dir"]
            os.makedirs(operators_dir, exist_ok=True)

            model = {
                'A': val_result['A'],
                'F': val_result['F'],
                'r': r_best,
                'alpha_state_lin': best['alpha_state_lin'],
                'alpha_state_quad': best['alpha_state_quad'],
                'alpha_out_lin': 0.0,
                'alpha_out_quad': 0.0,
                'alpha_state_cubic': 0.0,
                'total_error': best['val_energy_err'],
                'mean_Xhat': val_result['mean_Xhat'],
                'scaling_Xhat': val_result['scaling_Xhat'],
            }
            filepath = os.path.join(operators_dir, "model_0000.npz")
            np.savez(filepath, **model)

            # Save ensemble_models for pipeline compatibility
            np.savez(paths["ensemble_models"],
                     n_models=1, best_error=best['val_energy_err'],
                     r=r_best)

            np.savez(paths["sweep_results"],
                     n_total=len(val_result['all_results']),
                     n_selected=1,
                     best_error=best['val_energy_err'],
                     best_r=r_best,
                     best_alpha_lin=best['alpha_state_lin'],
                     best_alpha_quad=best['alpha_state_quad'],
                     val_energy_err=best['val_energy_err'])

            # Update cfg.r so step 3 uses the right rank
            cfg.r = r_best
            # Update n_steps and training_end for the retrained model
            cfg.n_steps = val_result['retrain_n_steps']
            cfg.training_end = val_result['retrain_n_steps']

            print_header("MODEL SELECTION SUMMARY (VALIDATION)")
            print(f"  Best rank: {r_best}")
            print(f"  Best α_lin: {best['alpha_state_lin']:.2e}")
            print(f"  Best α_quad: {best['alpha_state_quad']:.2e}")
            print(f"  Val energy error: {best['val_energy_err']:.4%}")
            print(f"  ρ(A): {val_result['rho']:.6f}")
            print(f"  Sweep time: {t_elapsed:.1f}s")

            save_step_status(args.run_dir, "step_2", "completed", {
                "n_models": 1,
                "best_error": float(best['val_energy_err']),
                "best_r": r_best,
                "sweep_time_seconds": t_elapsed,
                "selection_method": "validation",
            })

            print_header("STEP 2 COMPLETE")
            logger.info("Step 2 completed successfully (validation-based)")
            return

        # === Original path: training-MSE-based selection ===
        data = load_data_serial(paths, logger)

        actual_r = data['X_state'].shape[1]
        if actual_r != cfg.r:
            logger.warning(f"Overriding cfg.r={cfg.r} with actual r={actual_r} from step 1 data")
        cfg.r = actual_r

        t_start = time.time()

        # Physics-based energy precomputes for sweep selection
        if getattr(cfg, 'sweep_qoi_method', '') == "physics_energy":
            logger.info("Computing physics-energy precomputes for sweep selection...")
            Ur = np.load(paths["pod_basis"])
            ic_data = np.load(paths["initial_conditions"])
            u_mean = ic_data['train_temporal_mean']
            energy_N = getattr(cfg, 'ks_N', None) or getattr(cfg, 'n_x', 200)
            if cfg.pde != "ks":
                energy_N = cfg.n_x * cfg.n_y

            energy_a = float(np.sum(u_mean ** 2))
            energy_b = Ur.T @ u_mean  # (r,)

            X_full = np.vstack([data['X_state'], data['Y_state'][-1:]])
            norms_sq = np.sum(X_full ** 2, axis=1)
            cross = X_full @ energy_b
            ref_E = 0.5 / energy_N * (norms_sq + 2.0 * cross + energy_a)
            ref_E_train = ref_E[:cfg.training_end]

            data['energy_a'] = energy_a
            data['energy_b'] = energy_b
            data['energy_N'] = energy_N
            data['ref_energy_mean'] = float(np.mean(ref_E_train))
            data['ref_energy_std'] = float(np.std(ref_E_train, ddof=1))

            logger.info(f"  POD-based ref energy: mean={data['ref_energy_mean']:.6f}, "
                        f"std={data['ref_energy_std']:.6f}")
            del Ur, u_mean, X_full
            ic_data.close()

        # Load test data for stability evaluation during sweep
        test_IC = None
        n_test_steps = 0
        test_ref_energy = None
        test_ref_reduced = None

        if getattr(cfg, 'training_mode', '') == "temporal_split" and getattr(cfg, 'pde', '') in ("ks", "ns"):
            try:
                ic_data = np.load(paths["initial_conditions"])
                test_ICs = ic_data['test_ICs_reduced']
                test_IC = test_ICs[0]
                ic_data.close()

                n_test_steps = cfg.test_end - cfg.test_start

                # Load test reference reduced coefficients for MSE reporting
                test_ref_path = os.path.join(args.run_dir, "X_hat_test.npy")
                test_ref_reduced = np.load(test_ref_path)

                # Load HDF5 test reference energy (full-state, matches step 3)
                if 'energy_a' in data:
                    import h5py
                    hdf5_path = cfg.training_files[0]
                    with h5py.File(hdf5_path, 'r') as hf:
                        test_ref_energy = np.array(
                            hf['energy'][cfg.test_start:cfg.test_end])
                    logger.info(f"  Test ref energy (HDF5): mean={np.mean(test_ref_energy):.6f}, "
                                f"std={np.std(test_ref_energy, ddof=1):.6f}")

                logger.info(f"Test rollout: IC shape={test_IC.shape}, "
                            f"n_test_steps={n_test_steps}, ref_reduced shape={test_ref_reduced.shape}")
            except Exception as e:
                logger.warning(f"Could not load test data for stability eval: {e}. "
                               f"Proceeding without test rollout.")
                test_IC = None
                n_test_steps = 0
                test_ref_reduced = None

        results = serial_hyperparameter_sweep(cfg, data, logger,
                                              test_IC=test_IC,
                                              n_test_steps=n_test_steps,
                                              test_ref_energy=test_ref_energy,
                                              test_ref_reduced=test_ref_reduced)
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

        # Apply max_models limit
        max_models = getattr(cfg, 'max_models', 0)
        if max_models > 0 and len(selected) > max_models:
            logger.info(f"Limiting from {len(selected)} to {max_models} model(s) (max_models)")
            selected = selected[:max_models]

        if not selected:
            logger.error("No models met selection criteria!")
            save_step_status(args.run_dir, "step_2", "failed", {"error": "No models selected"})
            return

        models = recompute_operators_serial(selected, data, cfg.r, paths["operators_dir"], logger,
                                              stability_projection=getattr(cfg, 'stability_projection', False),
                                              stability_max_rho=getattr(cfg, 'stability_max_rho', 0.999))

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
