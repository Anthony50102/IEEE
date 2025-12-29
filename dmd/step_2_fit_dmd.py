"""
Step 2: Fit DMD Model.

This script handles:
1. Loading pre-computed POD basis from Step 1
2. Loading projected training data
3. Fitting BOPDMD (Bagging/Optimized DMD) model
4. Learning output operator for Gamma prediction (optional)
5. Saving DMD model components

Reuses Step 1 preprocessing (POD computation) from the OpInf pipeline.

Usage:
    python step_2_fit_dmd.py --config config/dmd_1train_5test.yaml --run-dir /path/to/run

Author: Anthony Poole
"""

import os
import sys
import time
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dmd.utils import (
    load_dmd_config,
    get_dmd_output_paths,
    print_dmd_config_summary,
    DMDConfig,
)

# Import shared utilities from opinf
from opinf.utils import (
    setup_logging,
    save_config,
    save_step_status,
    check_step_completed,
    print_header,
    loader,
)


# =============================================================================
# DMD FITTING
# =============================================================================

def fit_bopdmd(
    X_train: np.ndarray,
    t_train: np.ndarray,
    V_global: np.ndarray,
    r: int,
    cfg: DMDConfig,
    logger,
) -> dict:
    """
    Fit BOPDMD (Bagging/Optimized DMD) model to training data.
    
    Parameters
    ----------
    X_train : np.ndarray, shape (n_features, n_time) or (n_time, r) if projected
        Training snapshots (can be full or projected).
    t_train : np.ndarray, shape (n_time,)
        Time vector for training data.
    V_global : np.ndarray, shape (n_features, r)
        POD basis for projection.
    r : int
        DMD rank.
    cfg : DMDConfig
        Configuration object.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'eigs': Continuous-time eigenvalues
        - 'modes_reduced': Reduced-space modes
        - 'amplitudes': Mode amplitudes
        - 'dt': Time step
    """
    try:
        from pydmd import BOPDMD, DMD
    except ImportError:
        logger.error("pydmd not installed. Install with: pip install pydmd")
        raise ImportError("pydmd required for DMD fitting. Install with: pip install pydmd")
    
    logger.info("Fitting BOPDMD model...")
    logger.info(f"  Training data shape: {X_train.shape}")
    logger.info(f"  Time vector: {len(t_train)} points, dt={t_train[1]-t_train[0]:.4f}")
    logger.info(f"  DMD rank: {r}")
    
    dt = t_train[1] - t_train[0]
    
    # Use standard DMD to get initial guess for eigenvalues
    logger.info("  Computing initial eigenvalue guess via standard DMD...")
    t0 = time.time()
    
    dmd0 = DMD(svd_rank=r)
    
    # DMD expects data as (n_features, n_time)
    if X_train.shape[0] > X_train.shape[1]:
        X_dmd = X_train.T
    else:
        X_dmd = X_train
    
    dmd0.fit(X_dmd)
    
    # Convert discrete-time eigenvalues to continuous-time
    # λ_discrete = exp(α_continuous * dt)
    # α_continuous = log(λ_discrete) / dt
    init_alpha = np.log(dmd0.eigs) / dt
    
    logger.info(f"    Standard DMD fit in {time.time() - t0:.2f}s")
    logger.info(f"    Initial eigenvalue range: real [{init_alpha.real.min():.4f}, {init_alpha.real.max():.4f}]")
    
    # Fit BOPDMD with projection onto POD basis
    logger.info("  Fitting BOPDMD with POD projection...")
    t0 = time.time()
    
    # BOPDMD configuration
    dmd_model = BOPDMD(
        svd_rank=r,
        num_trials=cfg.num_trials,
        proj_basis=V_global if cfg.use_proj else None,
        use_proj=cfg.use_proj,
        eig_sort=cfg.eig_sort,
        init_alpha=init_alpha,
    )
    
    dmd_model.fit(X_dmd, t=t_train)
    
    logger.info(f"    BOPDMD fit in {time.time() - t0:.2f}s")
    
    # Extract model components
    eigs = dmd_model.eigs  # Continuous-time eigenvalues
    
    # Compute reduced modes: W_reduced = V^H @ Phi
    # where Phi are the full-space DMD modes
    modes_reduced = V_global.conj().T @ dmd_model.modes  # (r, r)
    
    # Get amplitudes (initial conditions)
    amplitudes = dmd_model._b
    
    logger.info(f"  Eigenvalues shape: {eigs.shape}")
    logger.info(f"  Reduced modes shape: {modes_reduced.shape}")
    logger.info(f"  Amplitudes shape: {amplitudes.shape}")
    
    # Log dominant eigenvalue info
    sorted_idx = np.argsort(-eigs.real)
    dominant_eig = eigs[sorted_idx[0]]
    logger.info(f"  Dominant eigenvalue: {dominant_eig.real:.4f} + {dominant_eig.imag:.4f}j")
    
    # Log stability info
    n_unstable = np.sum(eigs.real > 0)
    n_stable = np.sum(eigs.real < 0)
    logger.info(f"  Stable modes: {n_stable}, Unstable modes: {n_unstable}")
    
    return {
        'eigs': eigs,
        'modes_reduced': modes_reduced,
        'amplitudes': amplitudes,
        'dt': dt,
    }


def load_pod_basis(paths: dict, logger) -> tuple:
    """
    Load POD basis from Step 1 output.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (V_global, singular_values, r_actual)
    """
    logger.info(f"Loading POD data from {paths['pod_file']}")
    
    pod_data = np.load(paths['pod_file'])
    print(pod_data.files)
    singular_values = pod_data['S']
    # eigv = pod_data['eigv']
    # eigs = pod_data['eigs']
    
    # Load preprocessing info to get actual r used
    preproc_info = np.load(paths['preprocessing_info'])
    r_actual = int(preproc_info['r_actual'])
    
    logger.info(f"  Singular values shape: {singular_values.shape}")
    logger.info(f"  POD modes (r): {r_actual}")
    
    return  singular_values, r_actual
    # return eigv, singular_values, eigs, r_actual


def load_training_data(paths: dict, cfg: DMDConfig, logger) -> tuple:
    """
    Load training data from Step 1 output.
    
    Parameters
    ----------
    paths : dict
        Output paths dictionary.
    cfg : DMDConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Xhat_train, train_boundaries, t_train)
    """
    logger.info("Loading training data...")
    
    # Load projected training data
    Xhat_train = np.load(paths['xhat_train'])
    logger.info(f"  Xhat_train shape: {Xhat_train.shape}")
    
    # Load boundaries
    boundaries_data = np.load(paths['boundaries'])
    train_boundaries = boundaries_data['train_boundaries']
    logger.info(f"  Train boundaries: {train_boundaries}")
    
    # Create time vector
    n_train_snapshots = train_boundaries[-1]
    t_train = np.arange(n_train_snapshots) * cfg.dt
    logger.info(f"  Time vector: {len(t_train)} points, t=[{t_train[0]:.3f}, {t_train[-1]:.3f}]")
    
    return Xhat_train, train_boundaries, t_train


def save_dmd_model(
    dmd_result: dict,
    paths: dict,
    cfg: DMDConfig,
    logger,
):
    """
    Save DMD model components.
    
    Parameters
    ----------
    dmd_result : dict
        DMD fitting results.
    paths : dict
        Output paths.
    cfg : DMDConfig
        Configuration.
    logger : logging.Logger
        Logger instance.
    """
    logger.info("Saving DMD model...")
    
    # Save main model file
    np.savez(
        paths['dmd_model'],
        eigs=dmd_result['eigs'],
        modes_reduced=dmd_result['modes_reduced'],
        amplitudes=dmd_result['amplitudes'],
        dt=dmd_result['dt'],
        dmd_rank=len(dmd_result['eigs']),
        num_trials=cfg.num_trials,
        use_proj=cfg.use_proj,
    )
    logger.info(f"  Saved DMD model to {paths['dmd_model']}")
    
    # Save individual components for easy loading
    np.save(paths['dmd_eigenvalues'], dmd_result['eigs'])
    np.save(paths['dmd_modes'], dmd_result['modes_reduced'])
    np.save(paths['dmd_amplitudes'], dmd_result['amplitudes'])
    
    logger.info("DMD model saved successfully")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for Step 2: Fit DMD."""
    parser = argparse.ArgumentParser(
        description="Step 2: Fit DMD Model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory from Step 1"
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_dmd_config(args.config)
    cfg.run_dir = args.run_dir
    
    # Set up logging
    logger = setup_logging("step_2_dmd", args.run_dir, cfg.log_level)
    
    print_header("STEP 2: FIT DMD MODEL")
    print(f"  Run directory: {args.run_dir}")
    print_dmd_config_summary(cfg)
    
    # Check Step 1 completed
    if not check_step_completed(args.run_dir, "step_1"):
        logger.error("Step 1 has not completed. Run step_1_parallel_preprocess.py first.")
        return
    
    save_step_status(args.run_dir, "step_2_dmd", "running")
    
    # Save configuration with step-specific name
    save_config(cfg, args.run_dir, step_name="step_2_dmd")
    logger.info("Configuration saved to run directory")
    
    paths = get_dmd_output_paths(args.run_dir)
    
    start_time = time.time()
    
    try:
        # 1. Load POD basis from Step 1
        singular_values, r_actual = load_pod_basis(paths, logger)
        
        # Determine DMD rank
        dmd_rank = cfg.dmd_rank if cfg.dmd_rank is not None else r_actual
        if dmd_rank > r_actual:
            logger.warning(f"Requested DMD rank {dmd_rank} > POD rank {r_actual}, using {r_actual}")
            dmd_rank = r_actual
        
        logger.info(f"Using DMD rank: {dmd_rank}")
        
        # 2. Load training data
        Xhat_train, train_boundaries, t_train = load_training_data(paths, cfg, logger)
        
        # For single training trajectory, use full data
        # For multiple trajectories, we would concatenate or pick one
        n_train_files = len(train_boundaries) - 1
        if n_train_files > 1:
            logger.warning(f"Multiple training files ({n_train_files}), using first trajectory only")
            start_idx = train_boundaries[0]
            end_idx = train_boundaries[1]
            Xhat_train_single = Xhat_train[start_idx:end_idx, :]
            t_train_single = t_train[start_idx:end_idx] - t_train[start_idx]
        else:
            Xhat_train_single = Xhat_train
            t_train_single = t_train
        
        logger.info(f"Training on single trajectory: {Xhat_train_single.shape[0]} snapshots")
        
        # 3. Construct POD basis for BOPDMD
        # In reduced space, the basis is identity
        # V_global = reconstruct_pod_basis(eigv, eigs_pod, dmd_rank, Xhat_train_single, logger)
        V_global = np.eye(dmd_rank, dtype=np.float64)
        
        # 4. Fit BOPDMD
        # DMD expects data as (n_features, n_time)
        # Here "features" are the POD coordinates, so we transpose
        dmd_result = fit_bopdmd(
            X_train=Xhat_train_single[:, :dmd_rank].T,  # (r, n_time)
            t_train=t_train_single,
            V_global=V_global,
            r=dmd_rank,
            cfg=cfg,
            logger=logger,
        )
        
        # 5. Save model
        save_dmd_model(dmd_result, paths, cfg, logger)
        
        # Final timing
        total_time = time.time() - start_time
        
        save_step_status(args.run_dir, "step_2_dmd", "completed", {
            "dmd_rank": dmd_rank,
            "n_training_snapshots": Xhat_train_single.shape[0],
            "total_time_seconds": total_time,
        })
        
        print_header("STEP 2 (DMD) COMPLETE")
        print(f"  Output directory: {args.run_dir}")
        print(f"  DMD rank: {dmd_rank}")
        print(f"  Total runtime: {total_time:.1f} seconds")
        logger.info(f"Step 2 (DMD) completed successfully in {total_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Step 2 (DMD) failed: {e}", exc_info=True)
        save_step_status(args.run_dir, "step_2_dmd", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
