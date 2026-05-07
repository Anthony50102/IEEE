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
import h5py
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
from evaluation import compute_ensemble_predictions, compute_metrics, recompute_qoi_from_physics, predict_trajectory
from shared.plotting import (
    plot_gamma_timeseries, plot_qoi_timeseries,
    generate_state_diagnostic_plots,
    plot_ks_full_trajectory_reconstruction, plot_ks_full_trajectory_qoi,
)
from utils import load_dataset


def generate_gamma_plots(predictions: dict, ref_files: list, boundaries: np.ndarray,
                         dt: float, engine: str, output_dir: str, logger,
                         start_offset: int = 0, pde: str = "hw2d"):
    """
    Generate QoI plots for OpInf ensemble predictions.
    
    For hw2d: plots Gamma_n and Gamma_c.
    For ks: plots energy and enstrophy.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_traj = len(predictions['Gamma_n'])
    t_start = start_offset * dt
    
    for i in range(n_traj):
        n_steps = boundaries[i + 1] - boundaries[i]
        
        pred_n = predictions['Gamma_n'][i]  # Shape: (n_ensemble, n_steps)
        pred_c = predictions['Gamma_c'][i]
        
        # Skip trajectory if all models produced NaN (empty predictions)
        if pred_n.size == 0 or pred_c.size == 0:
            logger.warning(f"  Trajectory {i+1}: no valid predictions, skipping plot")
            continue
        
        # Use actual prediction length (may be shorter than boundary span)
        actual_steps = pred_n.shape[-1]
        if actual_steps < n_steps:
            n_steps = actual_steps
        
        if pde == "ks":
            import h5py
            with h5py.File(ref_files[i], 'r') as f:
                ref_n = np.array(f['energy'][start_offset:start_offset + n_steps])
                ref_c = np.array(f['enstrophy'][start_offset:start_offset + n_steps])
            
            output_path = os.path.join(output_dir, f'traj_{i+1}_qoi.png')
            plot_qoi_timeseries(
                pred_1=pred_n, pred_2=pred_c,
                ref_1=ref_n, ref_2=ref_c,
                dt=dt, output_path=output_path, logger=logger,
                label_1="Energy", label_2="Enstrophy",
                symbol_1=r"$E$", symbol_2=r"$P$",
                title_prefix=f'Trajectory {i+1}: ',
                method_name="OpInf",
                t_start=t_start,
            )
        elif pde == "ns":
            import h5py
            with h5py.File(ref_files[i], 'r') as f:
                ref_n = np.array(f['energy'][start_offset:start_offset + n_steps])
                ref_c = np.array(f['enstrophy'][start_offset:start_offset + n_steps])
            
            output_path = os.path.join(output_dir, f'traj_{i+1}_qoi.png')
            plot_qoi_timeseries(
                pred_1=pred_n, pred_2=pred_c,
                ref_1=ref_n, ref_2=ref_c,
                dt=dt, output_path=output_path, logger=logger,
                label_1="Energy", label_2="Enstrophy",
                symbol_1=r"$E$", symbol_2=r"$Z$",
                title_prefix=f'Trajectory {i+1}: ',
                method_name="OpInf",
                t_start=t_start,
            )
        else:
            import h5py
            with h5py.File(ref_files[i], 'r') as f:
                ref_n = np.asarray(f['gamma_n'][start_offset:start_offset + n_steps])
                ref_c = np.asarray(f['gamma_c'][start_offset:start_offset + n_steps])
            
            output_path = os.path.join(output_dir, f'traj_{i+1}_gamma.png')
            plot_gamma_timeseries(
                pred_n=pred_n, pred_c=pred_c,
                ref_n=ref_n, ref_c=ref_c,
                dt=dt, output_path=output_path, logger=logger,
                title_prefix=f'Trajectory {i+1}: ',
                method_name="OpInf",
                t_start=t_start,
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
            models, train_ICs, train_bounds, mean_Xhat, scaling_Xhat, logger, "training",
            max_ensemble=cfg.max_models, max_steps=cfg.n_steps
        )
        test_pred = compute_ensemble_predictions(
            models, test_ICs, test_bounds, mean_Xhat, scaling_Xhat, logger, "test",
            max_ensemble=cfg.max_models
        )
        
        logger.info(f"Predictions completed in {time.time() - t_start:.1f}s")
        
        # Optionally recompute QoI from physics instead of learned operators
        if getattr(cfg, 'qoi_method', 'learned') == 'physics':
            logger.info("Recomputing QoI from physics (reconstructing full state)...")
            
            # Load POD basis for reconstruction
            pod_basis_path = paths.get("pod_basis")
            if pod_basis_path and os.path.exists(pod_basis_path):
                _pod_basis = np.load(pod_basis_path)
            else:
                raise FileNotFoundError(f"POD basis required for physics QoI but not found at {pod_basis_path}")
            
            # Load temporal mean
            _temporal_mean = None
            ics_data = np.load(paths["initial_conditions"])
            if 'train_temporal_mean' in ics_data:
                _temporal_mean = ics_data['train_temporal_mean']
            
            train_pred = recompute_qoi_from_physics(
                train_pred, _pod_basis, _temporal_mean,
                pde=cfg.pde, ks_L=cfg.ks_L, ks_N=cfg.ks_N,
                ns_Lx=getattr(cfg, 'ns_Lx', 2*np.pi),
                ns_Ly=getattr(cfg, 'ns_Ly', None),
                ns_ny=cfg.n_y, ns_nx=cfg.n_x,
                logger=logger
            )
            test_pred = recompute_qoi_from_physics(
                test_pred, _pod_basis, _temporal_mean,
                pde=cfg.pde, ks_L=cfg.ks_L, ks_N=cfg.ks_N,
                ns_Lx=getattr(cfg, 'ns_Lx', 2*np.pi),
                ns_Ly=getattr(cfg, 'ns_Ly', None),
                ns_ny=cfg.n_y, ns_nx=cfg.n_x,
                logger=logger
            )
            logger.info("QoI recomputation complete.")
        
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
            start_offset=train_offset, pde=cfg.pde
        )
        test_metrics = compute_metrics(
            test_pred, cfg.test_files if cfg.test_files else cfg.training_files,
            test_bounds, cfg.engine, logger, start_offset=test_offset,
            pde=cfg.pde
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
                                 start_offset=train_offset, pde=cfg.pde)
            generate_gamma_plots(test_pred, 
                                 cfg.test_files if cfg.test_files else cfg.training_files,
                                 test_bounds, cfg.dt, cfg.engine,
                                 os.path.join(paths["figures_dir"], "test"), logger,
                                 start_offset=test_offset, pde=cfg.pde)
        
        # Initialize basis variables (used by both state diagnostics and full trajectory plots)
        pod_basis = None
        temporal_mean = None
        manifold_W = None
        manifold_shift = None
        
        # Generate state diagnostic plots (optional)
        if cfg.plot_state_error or cfg.plot_state_snapshots:
            
            # Get reduction method from preprocessing info
            preproc = np.load(paths["preprocessing_info"])
            reduction_method = str(preproc.get('reduction_method', cfg.reduction_method))
            n_y = int(preproc.get('n_y', cfg.n_y))
            n_x = int(preproc.get('n_x', cfg.n_x))
            
            logger.info(f"Loading basis for {reduction_method} reconstruction...")
            
            if reduction_method == "manifold":
                # Load full manifold basis (V, W, shift)
                if os.path.exists(paths["manifold_basis"]):
                    from pod import load_basis
                    basis = load_basis(paths["manifold_basis"])
                    pod_basis = basis.V
                    manifold_W = basis.W
                    manifold_shift = basis.shift
                    logger.info(f"Loaded manifold basis: V={basis.V.shape}, W={basis.W.shape}")
                else:
                    logger.warning(f"Manifold basis not found at {paths['manifold_basis']}")
                    # Fall back to linear POD basis
                    if os.path.exists(paths["pod_basis"]):
                        pod_basis = np.load(paths["pod_basis"])
                        logger.warning(f"Falling back to linear basis: {pod_basis.shape}")
                        reduction_method = "linear"
            else:
                # Load linear POD basis
                if os.path.exists(paths["pod_basis"]):
                    pod_basis = np.load(paths["pod_basis"])
                    logger.info(f"Loaded POD basis: {pod_basis.shape}")
            
            # Try to get temporal mean from initial_conditions (for linear POD)
            if reduction_method == "linear" and 'train_temporal_mean' in ICs:
                temporal_mean = ICs['train_temporal_mean']
                logger.info(f"Loaded temporal mean: {temporal_mean.shape}")
            
            train_ref_files = cfg.training_files
            test_ref_files = cfg.test_files if cfg.test_files else cfg.training_files
            
            # Prepare reduced states for plotting (ensemble mean)
            # Skip trajectories where all models NaN'd (empty X_OpInf)
            train_reduced = [
                np.mean(X, axis=0).T if X.size > 0 else None
                for X in train_pred['X_OpInf']
            ]
            test_reduced = [
                np.mean(X, axis=0).T if X.size > 0 else None
                for X in test_pred['X_OpInf']
            ]
            
            # Filter out None entries (trajectories with no valid predictions)
            train_reduced = [x for x in train_reduced if x is not None]
            test_reduced = [x for x in test_reduced if x is not None]
            
            ks_dx = cfg.ks_L / cfg.ks_N if cfg.pde == "ks" else None
            if train_reduced:
                generate_state_diagnostic_plots(
                    train_reduced, train_ref_files, train_bounds,
                    pod_basis, temporal_mean, n_y, n_x,
                    cfg.engine, cfg.dt,
                    os.path.join(paths["figures_dir"], "train"), logger,
                    method_name="OpInf",
                    prefix="train_", ref_offset=train_offset,
                    plot_error=cfg.plot_state_error,
                    plot_snapshots=cfg.plot_state_snapshots,
                    n_snapshots=cfg.n_snapshot_samples,
                    reduction_method=reduction_method,
                    manifold_W=manifold_W,
                    manifold_shift=manifold_shift,
                    pde=cfg.pde,
                    dx=ks_dx,
                )
            else:
                logger.warning("No valid training predictions for state diagnostics")
            if test_reduced:
                generate_state_diagnostic_plots(
                    test_reduced, test_ref_files, test_bounds,
                    pod_basis, temporal_mean, n_y, n_x,
                    cfg.engine, cfg.dt,
                    os.path.join(paths["figures_dir"], "test"), logger,
                    method_name="OpInf",
                    prefix="test_", ref_offset=test_offset,
                    plot_error=cfg.plot_state_error,
                    plot_snapshots=cfg.plot_state_snapshots,
                    n_snapshots=cfg.n_snapshot_samples,
                    reduction_method=reduction_method,
                    manifold_W=manifold_W,
                    manifold_shift=manifold_shift,
                    pde=cfg.pde,
                    dx=ks_dx,
                )
            else:
                logger.warning("No valid test predictions for state diagnostics")
        
        # FULL TRAJECTORY PLOTS (KS only — continuous rollout)
        # =================================================================
        if cfg.pde == "ks" and cfg.training_mode == "temporal_split":
            logger.info("Generating full-trajectory KS plots (continuous rollout)...")
            
            try:
                # Load POD basis if not already loaded
                if pod_basis is None and os.path.exists(paths["pod_basis"]):
                    pod_basis = np.load(paths["pod_basis"])
                    logger.info(f"Loaded POD basis for full trajectory: {pod_basis.shape}")
                if temporal_mean is None:
                    ics_data = np.load(paths["initial_conditions"])
                    if 'train_temporal_mean' in ics_data:
                        temporal_mean = ics_data['train_temporal_mean']
                
                if pod_basis is not None:
                    from shared.data_io import reconstruct_full_state
                    from shared.plotting import (
                        plot_ks_full_trajectory_reconstruction,
                        plot_ks_full_trajectory_qoi,
                    )
                    import h5py
                    
                    # Single continuous rollout from train IC through entire domain
                    n_train_full = train_bounds[1] - train_bounds[0]
                    n_test = test_bounds[1] - test_bounds[0]
                    n_total = n_train_full + n_test
                    train_ic = train_ICs[0]
                    
                    # Run continuous rollout for each model in ensemble
                    continuous_X = []
                    continuous_Gn = []
                    continuous_Gc = []
                    for score, mdl in models[:cfg.max_models or len(models)]:
                        result = predict_trajectory(train_ic, n_total, mdl, mean_Xhat, scaling_Xhat)
                        if not result['is_nan']:
                            continuous_X.append(result['X_OpInf'])
                            continuous_Gn.append(result['Gamma_n'])
                            continuous_Gc.append(result['Gamma_c'])
                    
                    if continuous_X:
                        # Ensemble mean
                        X_hat_full = np.mean(continuous_X, axis=0)
                        if X_hat_full.shape[0] != pod_basis.shape[1]:
                            X_hat_full = X_hat_full.T
                        
                        n_train = n_train_full
                        n_test_actual = n_test
                        
                        Q_full = reconstruct_full_state(X_hat_full, pod_basis, temporal_mean)
                        pred_u = Q_full.T  # (n_total, N)
                        
                        # Load reference states from HDF5
                        ref_file = cfg.training_files[0]
                        with h5py.File(ref_file, 'r') as f:
                            ref_u_train = np.array(f['u'][cfg.train_start:cfg.train_start + n_train])
                            ref_u_test = np.array(f['u'][cfg.test_start:cfg.test_start + n_test_actual])
                        ref_u = np.concatenate([ref_u_train, ref_u_test], axis=0)
                        
                        # QoI from continuous rollout ensemble mean
                        full_pred_energy = np.mean(continuous_Gn, axis=0)
                        full_pred_enstrophy = np.mean(continuous_Gc, axis=0)
                        
                        # Load reference QoIs
                        with h5py.File(ref_file, 'r') as f:
                            ref_energy_train = np.array(f['energy'][cfg.train_start:cfg.train_start + n_train])
                            ref_enstrophy_train = np.array(f['enstrophy'][cfg.train_start:cfg.train_start + n_train])
                            ref_energy_test = np.array(f['energy'][cfg.test_start:cfg.test_start + n_test_actual])
                            ref_enstrophy_test = np.array(f['enstrophy'][cfg.test_start:cfg.test_start + n_test_actual])
                        full_ref_energy = np.concatenate([ref_energy_train, ref_energy_test])
                        full_ref_enstrophy = np.concatenate([ref_enstrophy_train, ref_enstrophy_test])
                        
                        # Consistent limits from reference
                        ref_vmin = float(ref_u.min())
                        ref_vmax = float(ref_u.max())
                        train_n_steps = n_train
                        ks_dx = cfg.ks_L / cfg.ks_N
                        t_start_val = cfg.train_start * cfg.dt
                        
                        energy_range = full_ref_energy.max() - full_ref_energy.min()
                        energy_pad = 0.1 * energy_range if energy_range > 0 else 1.0
                        enstrophy_range = full_ref_enstrophy.max() - full_ref_enstrophy.min()
                        enstrophy_pad = 0.1 * enstrophy_range if enstrophy_range > 0 else 1.0
                        energy_ylim = (float(full_ref_energy.min() - energy_pad),
                                       float(full_ref_energy.max() + energy_pad))
                        enstrophy_ylim = (float(full_ref_enstrophy.min() - enstrophy_pad),
                                           float(full_ref_enstrophy.max() + enstrophy_pad))
                        
                        ft_dir = os.path.join(paths["figures_dir"], "full_trajectory")
                        os.makedirs(ft_dir, exist_ok=True)
                        
                        plot_ks_full_trajectory_reconstruction(
                            pred_states=pred_u,
                            ref_states=ref_u,
                            dt=cfg.dt, dx=ks_dx,
                            train_n_steps=train_n_steps,
                            output_path=os.path.join(ft_dir, "ks_full_trajectory_reconstruction.png"),
                            logger=logger,
                            method_name="OpInf",
                            vmin=ref_vmin, vmax=ref_vmax,
                            t_start=t_start_val,
                        )
                        
                        plot_ks_full_trajectory_qoi(
                            pred_qoi_1=full_pred_energy,
                            pred_qoi_2=full_pred_enstrophy,
                            ref_qoi_1=full_ref_energy,
                            ref_qoi_2=full_ref_enstrophy,
                            dt=cfg.dt,
                            train_n_steps=train_n_steps,
                            output_path=os.path.join(ft_dir, "ks_full_trajectory_qoi.png"),
                            logger=logger,
                            method_name="OpInf",
                            ylim_1=energy_ylim,
                            ylim_2=enstrophy_ylim,
                            t_start=t_start_val,
                        )
                    else:
                        logger.warning("No valid predictions for full trajectory plot")
                else:
                    logger.warning("POD basis not available for full trajectory reconstruction")
            except Exception as e:
                logger.warning(f"Full trajectory plot failed (non-fatal): {e}")
        
        # FULL TRAJECTORY PLOTS (NS — continuous rollout)
        # =================================================================
        if cfg.pde == "ns" and cfg.training_mode == "temporal_split":
            logger.info("Generating full-trajectory NS plots (continuous rollout)...")
            
            try:
                # Load POD basis if not already loaded
                if pod_basis is None and os.path.exists(paths["pod_basis"]):
                    pod_basis = np.load(paths["pod_basis"])
                    logger.info(f"Loaded POD basis for full trajectory: {pod_basis.shape}")
                if temporal_mean is None:
                    ics_data = np.load(paths["initial_conditions"])
                    if 'train_temporal_mean' in ics_data:
                        temporal_mean = ics_data['train_temporal_mean']
                
                if pod_basis is not None:
                    from shared.data_io import reconstruct_full_state
                    from shared.plotting import (
                        plot_ns_full_trajectory_reconstruction,
                        plot_ks_full_trajectory_qoi,
                        plot_ns_full_trajectory_state_error,
                    )
                    from shared.physics import compute_ns_qoi_timeseries
                    from shared.evaluation_io import (
                        load_full_trajectory_reference, compute_qoi_ylims,
                    )
                    import h5py
                    
                    # Use config-based spans (not capped forecast length)
                    n_test = test_bounds[1] - test_bounds[0]
                    ref_file = cfg.training_files[0]
                    ref_data = load_full_trajectory_reference(
                        ref_file, cfg.train_start, cfg.test_start, n_test,
                        pde="ns",
                    )
                    n_total = ref_data['n_total']
                    train_n_steps = ref_data['train_n_steps']
                    train_ic = train_ICs[0]
                    
                    # Run continuous rollout for each model in ensemble
                    continuous_X = []
                    for score, mdl in models[:cfg.max_models or len(models)]:
                        result = predict_trajectory(train_ic, n_total, mdl, mean_Xhat, scaling_Xhat)
                        if not result['is_nan']:
                            continuous_X.append(result['X_OpInf'])
                    
                    if continuous_X:
                        X_hat_full = np.mean(continuous_X, axis=0)
                        if X_hat_full.shape[0] != pod_basis.shape[1]:
                            X_hat_full = X_hat_full.T
                        
                        Q_full = reconstruct_full_state(X_hat_full, pod_basis, temporal_mean)
                        pred_u = Q_full.T  # (n_total, ny*nx)
                        
                        preproc = np.load(paths['preprocessing_info'])
                        n_y = int(preproc.get('n_y', cfg.n_y))
                        n_x = int(preproc.get('n_x', cfg.n_x))
                        
                        pred_omega_3d = pred_u.reshape(-1, n_y, n_x)
                        ref_omega_3d = ref_data['ref_state']
                        if ref_omega_3d.ndim == 2:
                            ref_omega_3d = ref_omega_3d.reshape(-1, n_y, n_x)
                        
                        full_pred_energy, full_pred_enstrophy = compute_ns_qoi_timeseries(
                            pred_omega_3d, Lx=cfg.ns_Lx, Ly=cfg.ns_Ly
                        )
                        
                        full_ref_energy = ref_data['ref_energy']
                        full_ref_enstrophy = ref_data['ref_enstrophy']
                        t_start_val = cfg.train_start * cfg.dt
                        energy_ylim, enstrophy_ylim = compute_qoi_ylims(
                            full_ref_energy, full_ref_enstrophy
                        )
                        
                        ft_dir = os.path.join(paths['figures_dir'], 'full_trajectory')
                        os.makedirs(ft_dir, exist_ok=True)
                        
                        plot_ns_full_trajectory_reconstruction(
                            pred_omega=pred_omega_3d,
                            ref_omega=ref_omega_3d,
                            dt=cfg.dt,
                            output_path=os.path.join(ft_dir, "ns_full_trajectory_reconstruction.png"),
                            logger=logger,
                            method_name="OpInf",
                            Lx=cfg.ns_Lx,
                        )

                        plot_ns_full_trajectory_state_error(
                            pred_omega=pred_omega_3d,
                            ref_omega=ref_omega_3d,
                            dt=cfg.dt,
                            train_n_steps=train_n_steps,
                            output_path=os.path.join(ft_dir, "ns_full_trajectory_state_error.png"),
                            logger=logger,
                            method_name="OpInf",
                            t_start=t_start_val,
                        )
                        
                        plot_ks_full_trajectory_qoi(
                            pred_qoi_1=full_pred_energy,
                            pred_qoi_2=full_pred_enstrophy,
                            ref_qoi_1=full_ref_energy,
                            ref_qoi_2=full_ref_enstrophy,
                            dt=cfg.dt,
                            train_n_steps=train_n_steps,
                            output_path=os.path.join(ft_dir, "ns_full_trajectory_qoi.png"),
                            logger=logger,
                            method_name="OpInf",
                            ylim_1=energy_ylim,
                            ylim_2=enstrophy_ylim,
                            t_start=t_start_val,
                        )
                    else:
                        logger.warning("No valid continuous predictions for NS full trajectory plot")
                else:
                    logger.warning("POD basis not available for NS full trajectory reconstruction")
            except Exception as e:
                logger.warning(f"NS full trajectory plot failed (non-fatal): {e}")
        
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
        
        # Write machine-readable run summary
        try:
            from shared.metrics import RunSummary
            summary = RunSummary(method="opinf", pde=cfg.pde, run_dir=args.run_dir,
                                config_path=args.config, dt=cfg.dt)
            
            for split, pred, bounds, ref_files_list, offset in [
                ("train", train_pred, train_bounds, cfg.training_files, train_offset),
                ("test", test_pred, test_bounds,
                 cfg.test_files if cfg.test_files else cfg.training_files, test_offset),
            ]:
                n_traj = len(bounds) - 1
                for i in range(n_traj):
                    pred_n = pred['Gamma_n'][i]
                    pred_c = pred['Gamma_c'][i]
                    if pred_n.size == 0:
                        continue
                    # Ensemble mean
                    mean_pred_n = np.mean(pred_n, axis=0) if pred_n.ndim > 1 else pred_n
                    mean_pred_c = np.mean(pred_c, axis=0) if pred_c.ndim > 1 else pred_c
                    # Use actual prediction length
                    traj_len = bounds[i + 1] - bounds[i]
                    actual_len = len(mean_pred_n)
                    if actual_len < traj_len:
                        traj_len = actual_len
                    if cfg.pde == "ks" or cfg.pde == "ns":
                        with h5py.File(ref_files_list[i], 'r') as fh:
                            ref_n = np.array(fh['energy'][offset:offset + traj_len])
                            ref_c = np.array(fh['enstrophy'][offset:offset + traj_len])
                    else:
                        with h5py.File(ref_files_list[i], 'r') as fh:
                            ref_n = np.asarray(fh['gamma_n'][offset:offset + traj_len])
                            ref_c = np.asarray(fh['gamma_c'][offset:offset + traj_len])
                    summary.add_qoi_metrics(mean_pred_n, ref_n, mean_pred_c, ref_c,
                                            split=split, trajectory=i)
            
            summary_path = summary.save()
            logger.info(f"Saved run summary to {summary_path}")
        except Exception as e:
            logger.warning(f"Run summary generation failed (non-fatal): {e}")
        
        print_header("STEP 3 COMPLETE")
        logger.info("Step 3 completed successfully")
    
    except Exception as e:
        logger.error(f"Step 3 failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_3", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
