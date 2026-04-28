"""
Step 2: Fit DMD Model.

This script orchestrates:
1. Loading POD basis and projected data from Step 1
2. Fitting BOPDMD (Bagging/Optimized DMD) model
3. Saving DMD model components

Usage:
    python step_2_train.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import gc
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use single precision for memory efficiency with large ranks
DTYPE = np.float32

from dmd.utils import (
    load_dmd_config, get_dmd_output_paths, print_dmd_config_summary, DMDConfig, save_config,
    fit_output_operators, dmd_forecast_reduced,
)
from opinf.utils import (
    setup_logging, save_step_status, check_step_completed, print_header,
    load_dataset,
)


# =============================================================================
# DMD FITTING
# =============================================================================

def fit_bopdmd(X_train: np.ndarray, t_train: np.ndarray, r: int, cfg: DMDConfig, logger) -> dict:
    """
    Fit BOPDMD model to training data.
    
    Parameters
    ----------
    X_train : np.ndarray, shape (n_time, r)
        Training data in reduced space.
    t_train : np.ndarray, shape (n_time,)
        Time vector.
    r : int
        DMD rank.
    cfg : DMDConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        DMD model components.
    """
    try:
        from pydmd import BOPDMD, DMD
    except ImportError:
        raise ImportError("pydmd required. Install with: pip install pydmd")
    
    logger.info("Fitting BOPDMD model...")
    logger.info(f"  Data: {X_train.shape[0]} snapshots, {X_train.shape[1]} modes")
    
    dt = t_train[1] - t_train[0]
    
    # DMD expects (n_features, n_time)
    X_dmd = X_train.T  # (r, n_time)
    
    # Initial eigenvalue guess via standard DMD
    logger.info("  Computing initial eigenvalue guess...")
    dmd0 = DMD(svd_rank=r)
    dmd0.fit(X_dmd)
    init_alpha = np.log(dmd0.eigs) / dt  # Convert to continuous-time
    
    logger.info(f"  Initial eigenvalues: real in [{init_alpha.real.min():.4f}, {init_alpha.real.max():.4f}]")
    logger.info(f"  Initial eigenvalues: imag in [{init_alpha.imag.min():.4f}, {init_alpha.imag.max():.4f}]")
    
    # Fit BOPDMD
    logger.info(f"  Fitting BOPDMD...")
    t0 = time.time()
    
    # Note: We avoid creating the V_global identity matrix when possible
    # as it's r×r float64 = 8*r^2 bytes (e.g., 8GB for r=1000)
    # Setting use_proj=False and proj_basis=None saves this memory
    
    # Build BOPDMD kwargs
    bopdmd_kwargs = dict(
        svd_rank=r,
        num_trials=cfg.num_trials,
        proj_basis=None,  # Don't use projection to save memory
        use_proj=False,   # Disable projection
        eig_sort=cfg.eig_sort,
        init_alpha=init_alpha,
    )
    if cfg.eig_stable:
        bopdmd_kwargs['eig_constraints'] = {"stable"}
    
    dmd_model = BOPDMD(**bopdmd_kwargs)
    
    # Convert to float64 for pydmd fitting (required by optimizer)
    logger.info(f"  Input data memory: {X_dmd.nbytes / 1e6:.1f} MB")
    dmd_model.fit(X_dmd.astype(np.float64), t=t_train.astype(np.float64))
    
    logger.info(f"  Fit completed in {time.time() - t0:.2f}s")
    
    # Extract components - modes are already in reduced space since no projection
    eigs = dmd_model.eigs  # Continuous-time eigenvalues
    modes_reduced = dmd_model.modes  # (r, r) complex modes
    amplitudes = dmd_model._b
    
    # Free pydmd internal storage
    del dmd_model, dmd0
    gc.collect()
    logger.info("  Freed pydmd model from memory")
    
    # Log info
    n_stable = np.sum(eigs.real < 0)
    n_unstable = np.sum(eigs.real > 0)
    logger.info(f"  Eigenvalues: {len(eigs)} ({n_stable} stable, {n_unstable} unstable)")
    logger.info(f"  Modes dtype: {modes_reduced.dtype}, size: {modes_reduced.nbytes / 1e6:.1f} MB")
    
    return {
        'eigs': eigs,
        'modes_reduced': modes_reduced,
        'amplitudes': amplitudes,
        'dt': dt,
    }


# =============================================================================
# OUTPUT MODEL TRAINING
# =============================================================================

def load_gamma_reference(cfg: DMDConfig, n_train: int, logger) -> np.ndarray:
    """Load reference QoI values for output model training.
    
    For HW2D: loads gamma_n, gamma_c.
    For KS: loads energy, enstrophy.
    Returns array of shape (K, 2).
    """
    logger.info("Loading reference QoI for output model...")
    
    pde = getattr(cfg, 'pde', 'hw2d')
    
    if pde in ("ks", "ns"):
        return _load_qoi_reference_ks(cfg, n_train, logger)
    
    if cfg.training_mode == "temporal_split":
        # Single file, use train portion
        fh = load_dataset(cfg.training_files[0], cfg.engine)
        gamma_n = fh["gamma_n"].values[cfg.train_start:cfg.train_start + n_train]
        gamma_c = fh["gamma_c"].values[cfg.train_start:cfg.train_start + n_train]
    else:
        # Multi-trajectory: concatenate all training files
        gamma_n_list, gamma_c_list = [], []
        for f in cfg.training_files:
            fh = load_dataset(f, cfg.engine)
            gamma_n_list.append(fh["gamma_n"].values[:n_train])
            gamma_c_list.append(fh["gamma_c"].values[:n_train])
        gamma_n = np.concatenate(gamma_n_list)
        gamma_c = np.concatenate(gamma_c_list)
    
    Y_Gamma = np.column_stack([gamma_n, gamma_c])  # (K, 2)
    logger.info(f"  Loaded {Y_Gamma.shape[0]} Gamma samples")
    return Y_Gamma


def _load_qoi_reference_ks(cfg: DMDConfig, n_train: int, logger) -> np.ndarray:
    """Load reference KS QoIs (energy, enstrophy) for output model training."""
    import h5py
    
    if cfg.training_mode == "temporal_split":
        with h5py.File(cfg.training_files[0], 'r') as f:
            energy = np.array(f['energy'][cfg.train_start:cfg.train_start + n_train])
            enstrophy = np.array(f['enstrophy'][cfg.train_start:cfg.train_start + n_train])
    else:
        energy_list, enstrophy_list = [], []
        for fp in cfg.training_files:
            with h5py.File(fp, 'r') as f:
                energy_list.append(np.array(f['energy'][:n_train]))
                enstrophy_list.append(np.array(f['enstrophy'][:n_train]))
        energy = np.concatenate(energy_list)
        enstrophy = np.concatenate(enstrophy_list)
    
    Y_Gamma = np.column_stack([energy, enstrophy])  # (K, 2)
    logger.info(f"  Loaded {Y_Gamma.shape[0]} KS QoI samples")
    return Y_Gamma


def train_output_model(
    dmd_result: dict,
    Xhat_train: np.ndarray,
    Y_Gamma: np.ndarray,
    cfg: DMDConfig,
    logger,
) -> dict:
    """
    Train quadratic output model using DMD predictions on training data.
    
    Parameters
    ----------
    dmd_result : dict
        Fitted DMD model with eigs, modes_reduced, amplitudes.
    Xhat_train : np.ndarray, shape (n_time, r)
        Training data in reduced space.
    Y_Gamma : np.ndarray, shape (n_time, 2)
        Reference Gamma values [Gamma_n, Gamma_c].
    cfg : DMDConfig
        Configuration with regularization parameters.
    logger
        Logger instance.
    
    Returns
    -------
    dict
        Output model operators {C, G, c, mean_X, scaling_X}.
    """
    logger.info("Training quadratic output model...")
    logger.info(f"  Regularization: alpha_lin={cfg.output_alpha_lin}, alpha_quad={cfg.output_alpha_quad}")
    
    # Use the training reduced states directly (not DMD predictions)
    # This ensures the output model learns from the actual training data
    output_model = fit_output_operators(
        X_train=Xhat_train,
        Y_Gamma=Y_Gamma,
        alpha_lin=float(cfg.output_alpha_lin),
        alpha_quad=float(cfg.output_alpha_quad),
    )
    
    logger.info(f"  Output operators: C {output_model['C'].shape}, G {output_model['G'].shape}")
    return output_model


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(paths: dict, cfg: DMDConfig, logger) -> tuple:
    """Load training data from Step 1 (POD-projected)."""
    logger.info("Loading training data...")
    
    Xhat_train = np.load(paths['xhat_train'])
    bounds = np.load(paths['boundaries'])
    train_boundaries = bounds['train_boundaries']
    
    # Create time vector
    n_train = train_boundaries[-1]
    t_train = np.arange(n_train) * cfg.dt
    
    logger.info(f"  Data shape: {Xhat_train.shape}")
    logger.info(f"  Time: {len(t_train)} steps, dt={cfg.dt}")
    
    return Xhat_train, train_boundaries, t_train


# =============================================================================
# VALIDATION-BASED RANK SWEEP
# =============================================================================

def validation_rank_sweep_dmd(cfg, paths, logger):
    """
    Sweep over DMD rank using a validation window.
    
    For each r_c in r_candidates:
      1. Truncate max-rank POD coefficients to r_c
      2. Fit BOPDMD on train portion [0, train_end)
      3. Forecast through validation window [val_start, val_end)
      4. Score by validation kinetic energy error
    
    Select best rank, retrain on [0, val_end) for final model.
    """
    import h5py

    val_start = cfg.val_start
    val_end = cfg.val_end
    train_end = cfg.train_end
    r_candidates = cfg.r_candidates

    # Load max-rank data from step 1
    Xhat_full = np.load(paths['xhat_train'])   # (n_total, r_max)
    pod_data = np.load(paths['pod_basis'])      # npz with U_r and mean
    Ur_full = pod_data['U_r']                   # (n_spatial, r_max)
    u_mean = pod_data['mean']                   # (n_spatial,)
    pod_data.close()

    r_max = Xhat_full.shape[1]
    n_x = getattr(cfg, 'n_x', 64)
    n_y = getattr(cfg, 'n_y', 64)
    N = n_x * n_y

    n_train_fit = train_end - cfg.train_start
    n_val = val_end - val_start

    logger.info(f"DMD Validation sweep: train [0,{n_train_fit}), val [{val_start},{val_end}) "
                f"= {n_val} steps, r_candidates={r_candidates}")

    # Load reference validation QoI from HDF5
    # ROM state is vorticity (ω) for NS, so 0.5/N * ||ω||² = enstrophy
    hdf5_path = cfg.training_files[0]
    pde = getattr(cfg, 'pde', 'hw2d')
    with h5py.File(hdf5_path, 'r') as hf:
        if pde == 'ns':
            val_ref_qoi = np.array(hf['enstrophy'][val_start:val_end])
        else:
            val_ref_qoi = np.array(hf['energy'][val_start:val_end])

    Xhat_fit = Xhat_full[:n_train_fit, :]
    t_fit = np.arange(n_train_fit) * cfg.dt

    all_results = []

    for r_c in r_candidates:
        if r_c > r_max:
            logger.warning(f"  r_c={r_c} > r_max={r_max}, skipping")
            continue

        logger.info(f"  --- Rank r={r_c} ---")
        X_dmd = Xhat_fit[:, :r_c]

        try:
            dmd_result = fit_bopdmd(X_dmd, t_fit, r_c, cfg, logger)
        except Exception as e:
            logger.warning(f"    BOPDMD fit failed: {e}")
            continue

        # Forecast through validation window using trained amplitudes
        eigs_ct = dmd_result['eigs']       # continuous-time eigenvalues
        modes = dmd_result['modes_reduced'] # (r_c, r_c) complex
        amps = dmd_result['amplitudes']     # (r_c,) complex

        # Forecast at validation times (continue from training time axis)
        t_val = np.arange(val_start, val_end) * cfg.dt
        dynamics = np.exp(np.outer(eigs_ct, t_val))   # (r_c, n_val)
        X_pred_val = (modes @ np.diag(amps) @ dynamics).real.T  # (n_val, r_c)

        # Compute validation QoI error (enstrophy for NS, energy for KS)
        Ur_trunc = Ur_full[:, :r_c]
        energy_a = float(np.sum(u_mean ** 2))
        energy_b = Ur_trunc.T @ u_mean
        norms_sq = np.sum(X_pred_val ** 2, axis=1)
        cross = X_pred_val @ energy_b
        pred_qoi = 0.5 / N * (norms_sq + 2.0 * cross + energy_a)

        val_err = float(np.linalg.norm(pred_qoi - val_ref_qoi) /
                       np.linalg.norm(val_ref_qoi))

        # Validation state MSE
        Xhat_val_ref = Xhat_full[n_train_fit:n_train_fit + n_val, :r_c]
        n_cmp = min(len(X_pred_val), len(Xhat_val_ref))
        val_state_mse = float(np.mean(
            (X_pred_val[:n_cmp] - Xhat_val_ref[:n_cmp]) ** 2))

        n_stable = int(np.sum(eigs_ct.real < 0))
        logger.info(f"    val_E_err={val_err:.4%}, val_MSE={val_state_mse:.2e}, "
                     f"eigs: {n_stable}/{r_c} stable")

        all_results.append({
            'r': r_c,
            'val_energy_err': val_err,
            'val_state_mse': val_state_mse,
            'n_stable_eigs': n_stable,
            'dmd_result': dmd_result,
        })

    if not all_results:
        logger.error("No valid DMD models in validation sweep!")
        return None

    # Sort by validation energy error
    all_results.sort(key=lambda x: x['val_energy_err'])
    best = all_results[0]

    # Simplicity tiebreaker: among models within 5% of best, prefer smallest r
    tol = 0.05
    best_val = best['val_energy_err']
    within_tol = [r for r in all_results if r['val_energy_err'] <= best_val * (1 + tol)]
    within_tol.sort(key=lambda x: x['r'])
    best = within_tol[0]

    logger.info(f"\n  === DMD VALIDATION SWEEP RESULTS ===")
    logger.info(f"  Best rank: r={best['r']}, val_E_err={best['val_energy_err']:.4%}")
    logger.info(f"  Models within {tol:.0%} of best: {len(within_tol)}")
    for i, res in enumerate(all_results[:7]):
        logger.info(f"    {i+1}. r={res['r']}, val_E_err={res['val_energy_err']:.4%}")

    # === RETRAIN on full [0, val_end) with winning rank ===
    r_best = best['r']
    n_retrain = val_end - cfg.train_start
    logger.info(f"\n  Retraining at r={r_best} on [0,{n_retrain}) (train+val)...")

    X_retrain = Xhat_full[:n_retrain, :r_best]
    t_retrain = np.arange(n_retrain) * cfg.dt
    dmd_retrain = fit_bopdmd(X_retrain, t_retrain, r_best, cfg, logger)

    # Save truncated step-1 artifacts
    logger.info(f"  Saving truncated artifacts for r={r_best}...")
    Xhat_test = np.load(paths['xhat_test'])
    # DMD pod_basis is npz with U_r, mean, n_y, n_x
    np.savez(paths['pod_basis'], U_r=Ur_full[:, :r_best], mean=u_mean,
             n_y=n_x, n_x=n_x)
    np.save(paths['xhat_train'], Xhat_full[:, :r_best])
    np.save(paths['xhat_test'], Xhat_test[:, :r_best])

    ic_data2 = np.load(paths['initial_conditions'])
    ic_dict = {k: ic_data2[k] for k in ic_data2.files}
    ic_data2.close()
    if 'train_ICs_reduced' in ic_dict:
        ic_dict['train_ICs_reduced'] = ic_dict['train_ICs_reduced'][:, :r_best]
    if 'test_ICs_reduced' in ic_dict:
        ic_dict['test_ICs_reduced'] = ic_dict['test_ICs_reduced'][:, :r_best]
    np.savez(paths['initial_conditions'], **ic_dict)

    preproc = np.load(paths['preprocessing_info'], allow_pickle=True)
    preproc_dict = {k: preproc[k] for k in preproc.files}
    preproc.close()
    preproc_dict['r_actual'] = r_best
    np.savez(paths['preprocessing_info'], **preproc_dict)

    return {
        'best': best,
        'all_results': [{k: v for k, v in r.items() if k != 'dmd_result'} for r in all_results],
        'dmd_result': dmd_retrain,
        'r': r_best,
        'n_retrain': n_retrain,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 2."""
    parser = argparse.ArgumentParser(description="Step 2: Fit DMD Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory from Step 1")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_dmd_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("step_2", args.run_dir, cfg.log_level)
    
    print_header("STEP 2: FIT DMD MODEL")
    print(f"  Run directory: {args.run_dir}")
    print_dmd_config_summary(cfg)
    
    # Check Step 1 completed
    if not check_step_completed(args.run_dir, "step_1"):
        logger.error("Step 1 has not completed!")
        return
    
    save_step_status(args.run_dir, "step_2", "running")
    save_config(cfg, args.run_dir, step_name="step_2")
    
    paths = get_dmd_output_paths(args.run_dir)
    t_start = time.time()
    
    try:
        # Check if validation-based selection is enabled
        use_val = getattr(cfg, 'val_end', 0) > 0 and len(getattr(cfg, 'r_candidates', [])) > 0

        if use_val:
            logger.info("=== VALIDATION-BASED RANK SELECTION ===")
            t_start_val = time.time()
            val_result = validation_rank_sweep_dmd(cfg, paths, logger)
            t_elapsed_val = time.time() - t_start_val

            if val_result is None:
                save_step_status(args.run_dir, "step_2", "failed",
                                 {"error": "No valid DMD models in validation sweep"})
                return

            # Save retrained model
            dmd_result = val_result['dmd_result']
            r_best = val_result['r']
            save_dict = {
                'eigs': dmd_result['eigs'],
                'modes_reduced': dmd_result['modes_reduced'],
                'amplitudes': dmd_result['amplitudes'],
                'dt': dmd_result['dt'],
                'dmd_rank': r_best,
                'use_learned_output': False,
            }
            np.savez(paths['dmd_model'], **save_dict)

            # Update cfg so step_3 uses correct rank
            cfg.dmd_rank = r_best

            print_header("MODEL SELECTION SUMMARY (VALIDATION)")
            print(f"  Best rank: {r_best}")
            print(f"  Val energy error: {val_result['best']['val_energy_err']:.4%}")
            print(f"  Retrained on: [0, {val_result['n_retrain']}) snapshots")
            print(f"  Sweep time: {t_elapsed_val:.1f}s")

            save_step_status(args.run_dir, "step_2", "completed", {
                "dmd_rank": r_best,
                "use_learned_output": False,
                "n_train_snapshots": val_result['n_retrain'],
                "time_seconds": t_elapsed_val,
                "selection_method": "validation",
                "val_energy_err": float(val_result['best']['val_energy_err']),
            })

            print_header("STEP 2 COMPLETE")
            logger.info("Step 2 completed successfully (validation-based)")
            return

        # === Original path: single DMD fit ===
        # Load preprocessing info
        preproc = np.load(paths['preprocessing_info'])
        r_actual = int(preproc['r_actual'])
        
        # Load training data
        Xhat_train, train_boundaries, t_train = load_training_data(paths, cfg, logger)
        n_features = Xhat_train.shape[1]
        
        # Determine DMD rank (default to POD rank)
        dmd_rank = cfg.dmd_rank if cfg.dmd_rank else r_actual
        if dmd_rank > r_actual:
            logger.warning(f"DMD rank {dmd_rank} > POD rank {r_actual}, using {r_actual}")
            dmd_rank = r_actual
        
        logger.info(f"  DMD rank: {dmd_rank}")
        logger.info(f"  Feature dimension: {n_features}")
        
        # For single trajectory (or first trajectory if multiple)
        n_traj = len(train_boundaries) - 1
        if n_traj > 1:
            logger.warning(f"Multiple trajectories ({n_traj}), using first only")
            Xhat_train = Xhat_train[:train_boundaries[1], :]
            t_train = t_train[:train_boundaries[1]]
        
        # Extract only the columns we need for DMD (truncate to dmd_rank)
        # Make a copy so we can free the original
        X_dmd_input = Xhat_train[:, :dmd_rank].copy()
        t_dmd_input = t_train.copy()
        n_train_snapshots = Xhat_train.shape[0]  # Save for later
        
        # Keep Xhat_train only if we need it for output model
        if not cfg.use_learned_output:
            del Xhat_train, t_train, train_boundaries, preproc
            gc.collect()
            logger.info("  Freed preprocessing data from memory before DMD fit")
        
        # Fit DMD
        dmd_result = fit_bopdmd(
            X_train=X_dmd_input,
            t_train=t_dmd_input,
            r=dmd_rank,
            cfg=cfg,
            logger=logger,
        )
        
        # Free DMD input data
        del X_dmd_input, t_dmd_input
        gc.collect()
        
        # Train output model if enabled
        output_model = None
        if cfg.use_learned_output:
            n_train = Xhat_train.shape[0]
            Y_Gamma = load_gamma_reference(cfg, n_train, logger)
            output_model = train_output_model(
                dmd_result=dmd_result,
                Xhat_train=Xhat_train[:, :dmd_rank],
                Y_Gamma=Y_Gamma,
                cfg=cfg,
                logger=logger,
            )
            # Now free the training data
            del Xhat_train, Y_Gamma
            gc.collect()
        
        # Save model
        logger.info("Saving DMD model...")
        save_dict = {
            'eigs': dmd_result['eigs'],
            'modes_reduced': dmd_result['modes_reduced'],
            'amplitudes': dmd_result['amplitudes'],
            'dt': dmd_result['dt'],
            'dmd_rank': dmd_rank,
            'use_learned_output': cfg.use_learned_output,
        }
        
        # Add output model if trained
        if output_model is not None:
            save_dict.update({
                'output_C': output_model['C'],
                'output_G': output_model['G'],
                'output_c': output_model['c'],
                'output_mean_X': output_model['mean_X'],
                'output_scaling_X': output_model['scaling_X'],
            })
            logger.info("  Saved learned output model operators")
        
        np.savez(paths['dmd_model'], **save_dict)
        
        # Final timing
        t_elapsed = time.time() - t_start
        
        save_step_status(args.run_dir, "step_2", "completed", {
            "dmd_rank": dmd_rank,
            "use_learned_output": cfg.use_learned_output,
            "n_train_snapshots": n_train_snapshots,
            "time_seconds": t_elapsed,
        })
        
        print_header("STEP 2 COMPLETE")
        print(f"  DMD rank: {dmd_rank}")
        print(f"  Learned output model: {cfg.use_learned_output}")
        print(f"  Runtime: {t_elapsed:.1f}s")
        
    except Exception as e:
        logger.error(f"Step 2 failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_2", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
