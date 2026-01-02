"""
Step 1: Parallel Data Preprocessing and POD Computation.

This script orchestrates:
1. Distributed loading of raw simulation data
2. Computing POD basis via distributed Gram matrix eigendecomposition
3. Projecting training and test data onto POD basis
4. Preparing learning matrices for ROM training

Usage:
    mpirun -np 4 python step_1_preprocess.py --config config.yaml
    mpirun -np 4 python step_1_preprocess.py --config config.yaml --save-pod-energy

Author: Anthony Poole
"""

import argparse
import gc
import os
import numpy as np
from mpi4py import MPI

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config, save_config, get_run_directory, setup_logging, DummyLogger,
    save_step_status, get_output_paths, print_header, print_config_summary,
)
from data import (
    load_all_data_distributed, center_data_distributed, scale_data_distributed,
    load_reference_gamma, gather_initial_conditions,
)
from pod import compute_pod_distributed, project_data_distributed
from training import prepare_learning_matrices
from shared.plotting import plot_pod_energy


def main():
    """Main entry point for parallel Step 1."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser(description="Step 1: Data Preprocessing and POD")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory")
    parser.add_argument("--save-pod-energy", action="store_true", help="Save POD energy plot")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Get/create run directory
    if rank == 0:
        run_dir = get_run_directory(cfg, args.run_dir)
    else:
        run_dir = None
    run_dir = comm.bcast(run_dir, root=0)
    
    # Set up logging
    logger = setup_logging("step_1", run_dir, cfg.log_level, rank) if rank == 0 else DummyLogger()
    
    if rank == 0:
        print_header("STEP 1: DATA PREPROCESSING AND POD COMPUTATION")
        print(f"  Run directory: {run_dir}")
        print(f"  MPI ranks: {size}")
        print_config_summary(cfg)
        save_step_status(run_dir, "step_1", "running")
        save_config(cfg, run_dir, step_name="step_1")
    
    paths = get_output_paths(run_dir) if rank == 0 else None
    paths = comm.bcast(paths, root=0)
    
    t_start = MPI.Wtime()
    
    try:
        # 1. Load data
        (Q_train_local, Q_test_local, train_boundaries, test_boundaries,
         n_spatial, n_local, start_idx, end_idx) = load_all_data_distributed(
            cfg, run_dir, comm, rank, size, logger
        )
        
        if rank == 0:
            np.savez(paths["boundaries"], train_boundaries=train_boundaries,
                     test_boundaries=test_boundaries, n_spatial=n_spatial)
        
        # 2. Center data
        if cfg.centering_enabled:
            Q_train_centered, train_mean = center_data_distributed(Q_train_local, comm, rank, logger)
            Q_test_centered, test_mean = center_data_distributed(Q_test_local, comm, rank, logger)
        else:
            Q_train_centered, Q_test_centered = Q_train_local, Q_test_local
            train_mean = test_mean = np.zeros(n_local)
        
        # 3. Scale data (optional)
        scaling_factors = None
        if cfg.scaling_enabled:
            n_local_per_field = n_local // cfg.n_fields
            Q_train_centered, scaling_factors = scale_data_distributed(
                Q_train_centered, cfg.n_fields, n_local_per_field, comm, rank, logger
            )
            Q_test_centered, _ = scale_data_distributed(
                Q_test_centered, cfg.n_fields, n_local_per_field, comm, rank, logger
            )
        
        # 4. Compute POD
        eigs, eigv, D_global, r_energy = compute_pod_distributed(
            Q_train_centered, comm, rank, size, logger, cfg.target_energy
        )
        r_actual = min(cfg.r, r_energy)
        
        if rank == 0:
            logger.info(f"  Using r={r_actual} (config: {cfg.r}, energy-based: {r_energy})")
            np.savez(paths["pod_file"], S=np.sqrt(np.maximum(eigs, 0)), eigs=eigs, eigv=eigv)
            if args.save_pod_energy:
                plot_pod_energy(eigs, r_actual, run_dir, logger)
        
        # Update config with actual r
        cfg.r = r_actual
        
        # 5. Project data
        Xhat_train, Xhat_test, Ur_local, Ur_full = project_data_distributed(
            Q_train_centered, Q_test_centered, eigv, eigs, r_actual, D_global, comm, rank, logger
        )
        
        if rank == 0:
            np.save(paths["xhat_train"], Xhat_train)
            np.save(paths["xhat_test"], Xhat_test)
            np.save(paths["pod_basis"], Ur_full)
        
        # 6. Gather initial conditions
        ics = gather_initial_conditions(
            Q_train_local, Q_test_local, Xhat_train, Xhat_test,
            train_boundaries, test_boundaries,
            len(cfg.training_files), len(cfg.test_files), n_spatial, comm, rank
        )
        
        train_means = comm.gather(train_mean, root=0)
        test_means = comm.gather(test_mean, root=0)
        
        if rank == 0:
            np.savez(
                paths["initial_conditions"],
                **ics,
                train_temporal_mean=np.concatenate(train_means),
                test_temporal_mean=np.concatenate(test_means),
            )
        
        # 7. Prepare learning matrices
        learning = prepare_learning_matrices(Xhat_train, train_boundaries, cfg, rank, logger)
        gamma_ref = load_reference_gamma(cfg, rank, logger)
        
        if rank == 0:
            np.savez(paths["learning_matrices"], **learning)
            np.savez(paths["gamma_ref"], **gamma_ref)
            
            preproc = {
                'centering_applied': cfg.centering_enabled,
                'scaling_applied': cfg.scaling_enabled,
                'r_actual': r_actual, 'r_config': cfg.r, 'r_from_energy': r_energy,
                'n_spatial': n_spatial, 'n_fields': cfg.n_fields,
                'n_x': cfg.n_x, 'n_y': cfg.n_y, 'dt': cfg.dt,
            }
            if scaling_factors is not None:
                preproc['scaling_factors'] = scaling_factors
            np.savez(paths["preprocessing_info"], **preproc)
        
        # Cleanup
        del Q_train_local, Q_test_local, Q_train_centered, Q_test_centered
        gc.collect()
        
        total_time = MPI.Wtime() - t_start
        
        if rank == 0:
            save_step_status(run_dir, "step_1", "completed", {
                "n_spatial": int(n_spatial),
                "r": r_actual,
                "mpi_ranks": size,
                "total_time_seconds": total_time,
            })
            print_header("STEP 1 COMPLETE")
            print(f"  Output: {run_dir}")
            print(f"  Runtime: {total_time:.1f}s")
            logger.info(f"Step 1 completed in {total_time:.1f}s")
    
    except Exception as e:
        if rank == 0:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            save_step_status(run_dir, "step_1", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
