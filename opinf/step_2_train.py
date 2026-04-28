"""
Step 2: ROM Training via Hyperparameter Sweep.

This script orchestrates:
1. Loading pre-computed POD basis and learning matrices
2. Parallel hyperparameter sweep over regularization parameters
3. Model selection (top-k or threshold based)
4. Saving ensemble of best models

Supports MPI-parallel execution for HPC.

Usage:
    python step_2_train.py --config config.yaml --run-dir /path/to/run
    mpirun -np 56 python step_2_train.py --config config.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import os
import numpy as np
import time as _time

try:
    from mpi4py import MPI
    _HAS_MPI = True
except (ImportError, RuntimeError):
    _HAS_MPI = False

    class _SerialComm:
        """Minimal MPI communicator mock for single-rank serial execution."""
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def Barrier(self): pass
        def bcast(self, data, root=0): return data
        def gather(self, data, root=0): return [data]
        def Abort(self, code):
            import sys
            sys.exit(code)

    class MPI:
        COMM_WORLD = _SerialComm()
        @staticmethod
        def Wtime():
            return _time.time()

from utils import (
    load_config, save_config, setup_logging, DummyLogger,
    save_step_status, check_step_completed, get_output_paths,
    print_header, print_config_summary,
)
from data import load_data_shared_memory, save_ensemble
from training import (
    parallel_hyperparameter_sweep, log_error_statistics, 
    select_models, recompute_operators_parallel,
)


def main():
    """Main entry point for Step 2."""
    parser = argparse.ArgumentParser(description="Step 2: ROM Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory from Step 1")
    parser.add_argument("--stats-only", action="store_true", 
                        help="Only show error statistics (for threshold tuning)")
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Load configuration
    cfg = load_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("step_2", args.run_dir, cfg.log_level, rank) if rank == 0 else DummyLogger()
    
    if rank == 0:
        print_header("STEP 2: ROM TRAINING (MPI PARALLEL)")
        print(f"  Run directory: {args.run_dir}")
        print(f"  MPI ranks: {size}")
        print_config_summary(cfg)
        
        if not check_step_completed(args.run_dir, "step_1"):
            logger.error("Step 1 has not completed!")
            comm.Abort(1)
            return
        
        save_step_status(args.run_dir, "step_2", "running")
        save_config(cfg, args.run_dir, step_name="step_2")
    
    comm.Barrier()
    paths = get_output_paths(args.run_dir) if rank == 0 else None
    paths = comm.bcast(paths, root=0)
    
    windows = []
    
    try:
        # Load data — use serial loader if MPI unavailable
        if _HAS_MPI:
            data, windows = load_data_shared_memory(paths, comm, logger)
            comm.Barrier()
        else:
            logger.info("Loading pre-computed data (serial)...")
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
                'include_cubic': bool(learning.get('include_cubic', False)),
                'include_constant': bool(learning.get('include_constant', False)),
                'closure_enabled': bool(learning.get('closure_enabled', False)),
            }
            learning.close()
            gamma_ref.close()
            logger.info("Data loaded successfully")
        
        # Infer actual r from loaded data (step 1 may truncate below cfg.r)
        actual_r = data['X_state'].shape[1]
        if rank == 0 and actual_r != cfg.r:
            logger.warning(f"Overriding cfg.r={cfg.r} with actual r={actual_r} from step 1 data")
        cfg.r = actual_r
        
        # Physics-based energy precomputes for sweep selection
        if cfg.sweep_qoi_method == "physics_energy" and rank == 0:
            logger.info("Computing physics-energy precomputes for sweep selection...")
            Ur = np.load(paths["pod_basis"])
            ic_data = np.load(paths["initial_conditions"])
            u_mean = ic_data['train_temporal_mean']
            energy_N = cfg.ks_N if cfg.pde == "ks" else cfg.n_x * cfg.n_y
            
            energy_a = float(np.sum(u_mean ** 2))
            energy_b = Ur.T @ u_mean  # (r,)
            
            # Reference energy from POD coefficients (same truncation level)
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
            logger.info(f"  (HDF5 ref energy:     mean={data.get('mean_Gamma_n', 'N/A')}, "
                        f"std={data.get('std_Gamma_n', 'N/A')})")
            del Ur, ic_data, u_mean, X_full
        
        # Broadcast physics energy data to all ranks
        if cfg.sweep_qoi_method == "physics_energy":
            physics_data = None
            if rank == 0:
                physics_data = {
                    'energy_a': data['energy_a'],
                    'energy_b': data['energy_b'],
                    'energy_N': data['energy_N'],
                    'ref_energy_mean': data['ref_energy_mean'],
                    'ref_energy_std': data['ref_energy_std'],
                }
            physics_data = comm.bcast(physics_data, root=0)
            data.update(physics_data)
        
        # Run sweep
        t_start = MPI.Wtime()
        results = parallel_hyperparameter_sweep(cfg, data, logger, comm)
        t_elapsed = MPI.Wtime() - t_start
        
        if rank == 0:
            logger.info(f"Sweep completed in {t_elapsed:.1f}s")
            
            if not results:
                logger.error("No valid models found!")
                save_step_status(args.run_dir, "step_2", "failed", {"error": "No valid models"})
                return
            
            # Log statistics
            log_error_statistics(results, logger)
            
            if args.stats_only:
                logger.info("STATS-ONLY MODE: Exiting without saving.")
                print_header("STATS-ONLY COMPLETE")
                return
            
            # Select models
            selected = select_models(
                results, cfg.threshold_mean, cfg.threshold_std, logger,
                thresh_mean_c=cfg.threshold_mean_c, thresh_std_c=cfg.threshold_std_c
            )
            
            if not selected:
                logger.error("No models met selection criteria!")
                # Save full sweep diagnostics for offline analysis
                all_errors = np.array([[r['mean_err_Gamma_n'], r['std_err_Gamma_n'],
                                        r['mean_err_Gamma_c'], r['std_err_Gamma_c'],
                                        r['total_error']] for r in results])
                all_params = np.array([[r['alpha_state_lin'], r['alpha_state_quad'],
                                        r.get('alpha_output_lin', 0), r.get('alpha_output_quad', 0)]
                                       for r in results])
                np.savez(os.path.join(args.run_dir, "sweep_diagnostics.npz"),
                         errors=all_errors, params=all_params,
                         columns=['mean_n', 'std_n', 'mean_c', 'std_c', 'total'],
                         param_columns=['alpha_state_lin', 'alpha_state_quad', 'alpha_output_lin', 'alpha_output_quad'])
                logger.info(f"Saved sweep diagnostics ({len(results)} models) for offline analysis")
                save_step_status(args.run_dir, "step_2", "failed", {"error": "No models selected"})
                return
        else:
            selected = None
        
        selected = comm.bcast(selected, root=0)
        
        if selected is None:
            return
        
        # Recompute operators in parallel
        models = recompute_operators_parallel(selected, data, cfg.r, paths["operators_dir"], comm, logger)
        
        if rank == 0:
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
        if rank == 0:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            save_step_status(args.run_dir, "step_2", "failed", {"error": str(e)})
        raise
    
    finally:
        for win in windows:
            try:
                win.Free()
            except:
                pass


if __name__ == "__main__":
    main()
