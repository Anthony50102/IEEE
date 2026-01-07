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
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmd.utils import (
    load_dmd_config, get_dmd_output_paths, print_dmd_config_summary, DMDConfig, save_config,
)
from opinf.utils import (
    setup_logging, save_step_status, check_step_completed, print_header,
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
    print(r)
    print(type(r))
    dmd0 = DMD(svd_rank=r)
    dmd0.fit(X_dmd)
    init_alpha = np.log(dmd0.eigs) / dt  # Convert to continuous-time
    
    logger.info(f"  Initial eigenvalues: real in [{init_alpha.real.min():.4f}, {init_alpha.real.max():.4f}]")
    logger.info(f"  Initial eigenvalues: imag in [{init_alpha.imag.min():.4f}, {init_alpha.imag.max():.4f}]")
    
    # Fit BOPDMD
    logger.info("  Fitting BOPDMD...")
    t0 = time.time()
    
    V_global = np.eye(r, dtype=np.float64)  # Identity in reduced space
    
    dmd_model = BOPDMD(
        svd_rank=r,
        num_trials=cfg.num_trials,
        proj_basis=V_global if cfg.use_proj else None,
        use_proj=cfg.use_proj,
        eig_sort=cfg.eig_sort,
        eig_constraints={"stable"},  # forces all eigenvalues to have non-positive real parts.
        # init_alpha=init_alpha,
    )
    dmd_model.fit(X_dmd, t=t_train)
    
    logger.info(f"  Fit completed in {time.time() - t0:.2f}s")
    
    # Extract components
    eigs = dmd_model.eigs  # Continuous-time eigenvalues
    modes_reduced = V_global.conj().T @ dmd_model.modes
    amplitudes = dmd_model._b
    
    # Log info
    n_stable = np.sum(eigs.real < 0)
    n_unstable = np.sum(eigs.real > 0)
    logger.info(f"  Eigenvalues: {len(eigs)} ({n_stable} stable, {n_unstable} unstable)")
    
    return {
        'eigs': eigs,
        'modes_reduced': modes_reduced,
        'amplitudes': amplitudes,
        'dt': dt,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(paths: dict, cfg: DMDConfig, logger) -> tuple:
    """Load training data from Step 1 (POD-projected or raw)."""
    logger.info("Loading training data...")
    
    Xhat_train = np.load(paths['xhat_train'])
    bounds = np.load(paths['boundaries'])
    train_boundaries = bounds['train_boundaries']
    use_pod = bool(bounds.get('use_pod', True))  # Default to True for backward compatibility
    
    # Create time vector
    n_train = train_boundaries[-1]
    t_train = np.arange(n_train) * cfg.dt
    
    logger.info(f"  Data shape: {Xhat_train.shape}")
    logger.info(f"  Time: {len(t_train)} steps, dt={cfg.dt}")
    logger.info(f"  Use POD: {use_pod}")
    
    return Xhat_train, train_boundaries, t_train, use_pod


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
        # Load preprocessing info
        preproc = np.load(paths['preprocessing_info'])
        r_actual = int(preproc['r_actual'])
        use_pod = bool(preproc.get('use_pod', True))
        
        # Load training data
        Xhat_train, train_boundaries, t_train, _ = load_training_data(paths, cfg, logger)
        n_features = Xhat_train.shape[1]
        
        # Determine DMD rank
        if use_pod:
            # POD mode: default to POD rank
            dmd_rank = cfg.dmd_rank if cfg.dmd_rank else r_actual
            if dmd_rank > r_actual:
                logger.warning(f"DMD rank {dmd_rank} > POD rank {r_actual}, using {r_actual}")
                dmd_rank = r_actual
        else:
            # Raw mode: must specify rank, or DMD will do its own SVD
            if cfg.dmd_rank is None:
                # Use cfg.r as default if available, otherwise warn
                if cfg.r and cfg.r < n_features:
                    dmd_rank = cfg.r
                    logger.info(f"No POD, using config r={dmd_rank} for DMD rank")
                else:
                    # Let DMD determine rank via SVD
                    dmd_rank = min(n_features, Xhat_train.shape[0] - 1)
                    logger.warning(f"No POD and no dmd_rank set, DMD will use rank={dmd_rank}")
            else:
                dmd_rank = cfg.dmd_rank
        
        logger.info(f"  Use POD: {use_pod}")
        logger.info(f"  DMD rank: {dmd_rank}")
        logger.info(f"  Feature dimension: {n_features}")
        
        # For single trajectory (or first trajectory if multiple)
        n_traj = len(train_boundaries) - 1
        if n_traj > 1:
            logger.warning(f"Multiple trajectories ({n_traj}), using first only")
            Xhat_train = Xhat_train[:train_boundaries[1], :]
            t_train = t_train[:train_boundaries[1]]
        
        # Fit DMD
        dmd_result = fit_bopdmd(
            X_train=Xhat_train[:, :dmd_rank] if use_pod else Xhat_train,
            t_train=t_train,
            r=dmd_rank,
            cfg=cfg,
            logger=logger,
        )
        
        # Save model
        logger.info("Saving DMD model...")
        np.savez(
            paths['dmd_model'],
            eigs=dmd_result['eigs'],
            modes_reduced=dmd_result['modes_reduced'],
            amplitudes=dmd_result['amplitudes'],
            dt=dmd_result['dt'],
            dmd_rank=dmd_rank,
            use_pod=use_pod,
        )
        
        # Final timing
        t_elapsed = time.time() - t_start
        
        save_step_status(args.run_dir, "step_2", "completed", {
            "dmd_rank": dmd_rank,
            "use_pod": use_pod,
            "n_train_snapshots": Xhat_train.shape[0],
            "time_seconds": t_elapsed,
        })
        
        print_header("STEP 2 COMPLETE")
        print(f"  Use POD: {use_pod}")
        print(f"  DMD rank: {dmd_rank}")
        print(f"  Runtime: {t_elapsed:.1f}s")
        
    except Exception as e:
        logger.error(f"Step 2 failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_2", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
