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
import numpy as np
from mpi4py import MPI

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
        # Load data with shared memory
        data, windows = load_data_shared_memory(paths, comm, logger)
        comm.Barrier()
        
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
                results, cfg.threshold_mean, cfg.threshold_std, logger
            )
            
            if not selected:
                logger.error("No models met selection criteria!")
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
