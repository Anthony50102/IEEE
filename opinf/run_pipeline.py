"""
Run Full OpInf Pipeline.

This script orchestrates the complete Operator Inference ROM pipeline:
1. Data preprocessing and POD computation
2. ROM training via hyperparameter sweep
3. Evaluation and prediction

Can also run individual steps if previous steps have completed.

Usage:
    # Full pipeline (serial step 2)
    python run_pipeline.py --config config.yaml
    
    # Full pipeline (parallel step 2)
    mpirun -np 56 python run_pipeline.py --config config.yaml
    
    # Run specific steps
    python run_pipeline.py --config config.yaml --steps 1,3 --run-dir /path/to/run

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import subprocess
from datetime import datetime

from utils import (
    load_config,
    save_config,
    create_run_directory,
    setup_logging,
    save_step_status,
    load_step_status,
    check_step_completed,
    print_header,
    print_config_summary,
)

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


def run_step_1(config_path: str, run_dir: str, logger) -> bool:
    """
    Run Step 1: Data Preprocessing and POD.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    run_dir : str
        Run directory.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    bool
        True if successful.
    """
    logger.info("=" * 60)
    logger.info("RUNNING STEP 1: Data Preprocessing and POD")
    logger.info("=" * 60)
    
    # Import and run step 1
    from step_1_preprocess import main as step_1_main
    
    # Modify sys.argv for step 1
    original_argv = sys.argv.copy()
    sys.argv = [
        'step_1_preprocess.py',
        '--config', config_path,
        '--run-dir', run_dir,
    ]
    
    try:
        step_1_main()
        success = check_step_completed(run_dir, "step_1")
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        success = False
    finally:
        sys.argv = original_argv
    
    return success


def run_step_2(config_path: str, run_dir: str, logger, rank: int = 0) -> bool:
    """
    Run Step 2: ROM Training.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    run_dir : str
        Run directory.
    logger : logging.Logger
        Logger instance.
    rank : int
        MPI rank.
    
    Returns
    -------
    bool
        True if successful.
    """
    if rank == 0:
        logger.info("=" * 60)
        logger.info("RUNNING STEP 2: ROM Training")
        logger.info("=" * 60)
    
    # Import and run step 2
    from step_2_train_rom import main as step_2_main
    
    # Modify sys.argv for step 2
    original_argv = sys.argv.copy()
    sys.argv = [
        'step_2_train_rom.py',
        '--config', config_path,
        '--run-dir', run_dir,
    ]
    
    try:
        step_2_main()
        success = check_step_completed(run_dir, "step_2")
    except Exception as e:
        if rank == 0:
            logger.error(f"Step 2 failed: {e}")
        success = False
    finally:
        sys.argv = original_argv
    
    return success


def run_step_3(config_path: str, run_dir: str, logger) -> bool:
    """
    Run Step 3: Evaluation.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    run_dir : str
        Run directory.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    bool
        True if successful.
    """
    logger.info("=" * 60)
    logger.info("RUNNING STEP 3: Evaluation")
    logger.info("=" * 60)
    
    # Import and run step 3
    from step_3_evaluate import main as step_3_main
    
    # Modify sys.argv for step 3
    original_argv = sys.argv.copy()
    sys.argv = [
        'step_3_evaluate.py',
        '--config', config_path,
        '--run-dir', run_dir,
    ]
    
    try:
        step_3_main()
        success = check_step_completed(run_dir, "step_3")
    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        success = False
    finally:
        sys.argv = original_argv
    
    return success


def main():
    """Main entry point for pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run OpInf Pipeline (Full or Partial)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run directory (creates new if not specified)"
    )
    parser.add_argument(
        "--steps", type=str, default="1,2,3",
        help="Comma-separated list of steps to run (default: 1,2,3)"
    )
    args = parser.parse_args()
    
    # Parse steps to run
    steps = [int(s.strip()) for s in args.steps.split(",")]
    
    # Initialize MPI if available
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Create or use run directory (only rank 0 creates)
    if rank == 0:
        if args.run_dir and os.path.isdir(args.run_dir):
            run_dir = args.run_dir
            cfg.run_dir = run_dir
        else:
            run_dir = create_run_directory(cfg)
        
        # Save config to run directory
        save_config(cfg, run_dir)
    else:
        run_dir = None
    
    # Broadcast run_dir to all ranks
    if HAS_MPI:
        run_dir = comm.bcast(run_dir, root=0)
        cfg.run_dir = run_dir
    
    # Set up logging
    logger = setup_logging("pipeline", run_dir, cfg.log_level, rank)
    
    if rank == 0:
        print_header("OPINF ROM PIPELINE")
        print(f"  Configuration: {args.config}")
        print(f"  Run directory: {run_dir}")
        print(f"  Steps to run: {steps}")
        print(f"  MPI processes: {size}")
        print_config_summary(cfg)
    
    start_time = time.time()
    all_success = True
    
    # Run requested steps
    if 1 in steps:
        if rank == 0:
            success = run_step_1(args.config, run_dir, logger)
            if not success:
                all_success = False
                logger.error("Step 1 failed, aborting pipeline")
        
        # Synchronize
        if HAS_MPI:
            all_success = comm.bcast(all_success, root=0)
        
        if not all_success:
            return
    
    if 2 in steps:
        # Check Step 1 completed
        if rank == 0:
            if not check_step_completed(run_dir, "step_1"):
                logger.error("Step 1 has not completed. Cannot run Step 2.")
                all_success = False
        
        if HAS_MPI:
            all_success = comm.bcast(all_success, root=0)
        
        if not all_success:
            return
        
        # All ranks participate in Step 2
        success = run_step_2(args.config, run_dir, logger, rank)
        
        if rank == 0 and not success:
            all_success = False
            logger.error("Step 2 failed, aborting pipeline")
        
        if HAS_MPI:
            all_success = comm.bcast(all_success, root=0)
        
        if not all_success:
            return
    
    if 3 in steps:
        if rank == 0:
            # Check Step 2 completed
            if not check_step_completed(run_dir, "step_2"):
                logger.error("Step 2 has not completed. Cannot run Step 3.")
                all_success = False
            else:
                success = run_step_3(args.config, run_dir, logger)
                if not success:
                    all_success = False
    
    elapsed = time.time() - start_time
    
    if rank == 0:
        print_header("PIPELINE COMPLETE")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Run directory: {run_dir}")
        
        # Print final status
        status = load_step_status(run_dir)
        print("\n  Step Status:")
        for step_name, step_info in status.items():
            status_str = step_info.get('status', 'unknown')
            print(f"    {step_name}: {status_str}")
        
        if all_success:
            logger.info("Pipeline completed successfully")
        else:
            logger.warning("Pipeline completed with errors")


if __name__ == "__main__":
    main()
