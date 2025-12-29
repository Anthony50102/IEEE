"""
Step 1 Alternative: Quadratic Manifold Computation.

This script replaces standard POD with quadratic manifold for better
reconstruction of nonlinear dynamics. It follows the same structure as
step_1_parallel_preprocess.py but uses the greedy quadratic manifold algorithm.

The quadratic manifold provides:
- Better reconstruction error for the same r
- Nonlinear approximation: x ≈ V @ z + W @ h(z) + shift
- Greedy selection of modes (not necessarily first r by energy)

Usage:
    python step_1_quadratic_manifold.py --config config.yaml

Author: Adapted for HPC pipeline
"""

import argparse
import gc
import time
import numpy as np
import xarray as xr
import yaml
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

# Import quadratic manifold module
from quadratic_manifold import (
    quadmani_greedy,
    quadmani_greedy_from_svd,
    ShiftedSVD,
    QuadraticManifold,
    linear_reduce,
    lift_quadratic,
    default_feature_map,
    get_num_quadratic_features,
    save_quadratic_manifold,
    compare_with_linear_pod,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class QuadManifoldConfig:
    """Configuration loaded from YAML file."""
    # Run info
    run_name: str
    
    # Paths
    output_base: str
    data_dir: str
    training_files: List[str]
    test_files: List[str]
    
    # Physics
    dt: float
    n_fields: int
    n_x: int
    n_y: int
    
    # Quadratic manifold parameters
    r: int
    n_vectors_to_check: int
    reg_magnitude: float
    compare_with_pod: bool
    initial_indices: List[int]
    
    # Truncation
    truncation_enabled: bool
    truncation_method: str
    truncation_snapshots: Optional[int]
    truncation_time: Optional[float]
    
    # Execution
    verbose: bool
    log_level: str
    engine: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "QuadManifoldConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Build full paths for data files
        data_dir = cfg['paths']['data_dir']
        training_files = [
            os.path.join(data_dir, f) for f in cfg['paths']['training_files']
        ]
        test_files = [
            os.path.join(data_dir, f) for f in cfg['paths']['test_files']
        ]
        
        # Get quadratic manifold config (with defaults)
        qm_cfg = cfg.get('quadratic_manifold', {})
        
        return cls(
            run_name=cfg['run_name'],
            output_base=cfg['paths']['output_base'],
            data_dir=data_dir,
            training_files=training_files,
            test_files=test_files,
            dt=cfg['physics']['dt'],
            n_fields=cfg['physics']['n_fields'],
            n_x=cfg['physics']['n_x'],
            n_y=cfg['physics']['n_y'],
            r=qm_cfg.get('r', cfg.get('pod', {}).get('r', 100)),
            n_vectors_to_check=qm_cfg.get('n_vectors_to_check', 200),
            reg_magnitude=float(qm_cfg.get('reg_magnitude', 1e-6)),
            compare_with_pod=qm_cfg.get('compare_with_pod', True),
            initial_indices=qm_cfg.get('initial_indices', []),
            truncation_enabled=cfg['truncation']['enabled'],
            truncation_method=cfg['truncation']['method'],
            truncation_snapshots=cfg['truncation'].get('snapshots'),
            truncation_time=cfg['truncation'].get('time'),
            verbose=cfg['execution']['verbose'],
            log_level=cfg['execution']['log_level'],
            engine=cfg['execution']['engine'],
        )


def get_run_directory(cfg: QuadManifoldConfig) -> str:
    """Create and return the run directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_base, f"{cfg.run_name}_qm_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_output_paths(run_dir: str) -> dict:
    """Get standard output file paths."""
    return {
        "boundaries": os.path.join(run_dir, "boundaries.npz"),
        "qm_file": os.path.join(run_dir, "quadratic_manifold.npz"),
        "pod_file": os.path.join(run_dir, "pod_basis.npz"),
        "xhat_train": os.path.join(run_dir, "Xhat_train.npy"),
        "xhat_test": os.path.join(run_dir, "Xhat_test.npy"),
        "initial_conditions": os.path.join(run_dir, "initial_conditions.npz"),
        "learning_matrices": os.path.join(run_dir, "learning_matrices.npz"),
        "gamma_ref": os.path.join(run_dir, "gamma_reference.npz"),
        "preprocessing_info": os.path.join(run_dir, "preprocessing_info.npz"),
        "comparison": os.path.join(run_dir, "qm_vs_pod_comparison.npz"),
    }


# =============================================================================
# LOGGING
# =============================================================================

import logging

def setup_logging(name: str, run_dir: str, level: str) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # File handler
    fh = logging.FileHandler(os.path.join(run_dir, f"{name}.log"))
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_config_summary(cfg: QuadManifoldConfig):
    """Print configuration summary."""
    print(f"\n  Configuration Summary:")
    print(f"    Run name: {cfg.run_name}")
    print(f"    Training files: {len(cfg.training_files)}")
    print(f"    Test files: {len(cfg.test_files)}")
    print(f"    Reduced dimension r: {cfg.r}")
    print(f"    Vectors to check: {cfg.n_vectors_to_check}")
    print(f"    Regularization: {cfg.reg_magnitude}")
    print(f"    Grid: {cfg.n_x} x {cfg.n_y}")
    print(f"    dt: {cfg.dt}")
    if cfg.truncation_enabled:
        if cfg.truncation_method == "time":
            print(f"    Truncation: {cfg.truncation_time}s")
        else:
            print(f"    Truncation: {cfg.truncation_snapshots} snapshots")


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_truncation_snapshots(
    file_path: str,
    truncation_snapshots: Optional[int],
    truncation_time: Optional[float],
    dt: float,
) -> Optional[int]:
    """Compute max snapshots based on truncation settings."""
    if truncation_time is not None:
        return int(truncation_time / dt)
    return truncation_snapshots


def get_file_metadata(
    file_path: str,
    cfg: QuadManifoldConfig,
) -> Tuple[int, int, Optional[int]]:
    # """Get metadata from a single file without loading full data."""
    with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
        n_time_original = fh["density"].shape[0]
        if fh["density"].ndim == 3:
            n_y, n_x = fh["density"].shape[1], fh["density"].shape[2]
            n_spatial = 2 * n_y * n_x
        else:
            n_spatial = 2 * fh["density"].shape[1]
    
    if cfg.truncation_enabled:
        max_snaps = compute_truncation_snapshots(
            file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
        )
        n_time = min(n_time_original, max_snaps) if max_snaps else n_time_original
    else:
        n_time = n_time_original
        max_snaps = None
    
    return n_spatial, n_time, max_snaps


def load_snapshots(
    file_path: str,
    cfg: QuadManifoldConfig,
    max_snapshots: Optional[int],
) -> np.ndarray:
    """
    Load snapshots from HDF5 file.
    
    Returns
    -------
    np.ndarray
        Data array of shape (n_spatial, n_time).
    """
    t0 = time.time()
    
    with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
        density = fh["density"].values
        phi = fh["phi"].values
    
    # Truncate time
    if max_snapshots is not None and max_snapshots < density.shape[0]:
        density = density[:max_snapshots]
        phi = phi[:max_snapshots]
    
    n_time = density.shape[0]
    
    # Reshape if needed
    if density.ndim == 2:
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    # Stack fields: shape (2, n_time, n_y, n_x)
    Q_full = np.stack([density, phi], axis=0)
    del density, phi
    
    # Transpose to (2, n_y, n_x, n_time)
    Q_full = Q_full.transpose(0, 2, 3, 1)
    n_field, n_y, n_x, n_time = Q_full.shape
    
    # Reshape to (n_spatial, n_time)
    Q_full = Q_full.reshape(n_field * n_y * n_x, n_time)
    
    if cfg.verbose:
        print(f"    Loaded {os.path.basename(file_path)}: {Q_full.shape}, {time.time() - t0:.1f}s")
    
    return Q_full


def load_all_data(cfg: QuadManifoldConfig, logger) -> Tuple[np.ndarray, np.ndarray, list, list, int]:
    """
    Load all training and test data.
    
    Returns
    -------
    Tuple
        (Q_train, Q_test, train_boundaries, test_boundaries, n_spatial)
    """
    logger.info("Loading simulation data...")
    
    # First pass: determine shapes
    train_timesteps = []
    test_timesteps = []
    train_truncations = []
    test_truncations = []
    n_spatial = None
    
    for file_path in cfg.training_files:
        ns, nt, max_snaps = get_file_metadata(file_path, cfg)
        if n_spatial is None:
            n_spatial = ns
        train_timesteps.append(nt)
        train_truncations.append(max_snaps)
    
    for file_path in cfg.test_files:
        ns, nt, max_snaps = get_file_metadata(file_path, cfg)
        test_timesteps.append(nt)
        test_truncations.append(max_snaps)
    
    total_train = sum(train_timesteps)
    total_test = sum(test_timesteps)
    
    logger.info(f"  Spatial DOF: {n_spatial:,}")
    logger.info(f"  Total train snapshots: {total_train:,}")
    logger.info(f"  Total test snapshots: {total_test:,}")
    
    # Allocate arrays
    Q_train = np.zeros((n_spatial, total_train), dtype=np.float64)
    Q_test = np.zeros((n_spatial, total_test), dtype=np.float64)
    
    # Compute boundaries
    train_boundaries = [0] + list(np.cumsum(train_timesteps))
    test_boundaries = [0] + list(np.cumsum(test_timesteps))
    
    # Load training data
    logger.info("Loading training trajectories...")
    for i, file_path in enumerate(cfg.training_files):
        Q_ic = load_snapshots(file_path, cfg, train_truncations[i])
        Q_train[:, train_boundaries[i]:train_boundaries[i + 1]] = Q_ic
        del Q_ic
        gc.collect()
    
    # Load test data
    logger.info("Loading test trajectories...")
    for i, file_path in enumerate(cfg.test_files):
        Q_ic = load_snapshots(file_path, cfg, test_truncations[i])
        Q_test[:, test_boundaries[i]:test_boundaries[i + 1]] = Q_ic
        del Q_ic
        gc.collect()
    
    return Q_train, Q_test, train_boundaries, test_boundaries, n_spatial


# =============================================================================
# QUADRATIC MANIFOLD PROJECTION
# =============================================================================

def project_with_quadratic_manifold(
    Q_train: np.ndarray,
    Q_test: np.ndarray,
    qm: QuadraticManifold,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project data using quadratic manifold.
    """
    logger.info("Projecting data onto quadratic manifold...")
    
    t_proj = time.time()
    
    # Linear projection: z = V^T @ (x - shift)
    Xhat_train = linear_reduce(qm, Q_train).T  # (n_time, r)
    Xhat_test = linear_reduce(qm, Q_test).T
    
    logger.info(f"  Projection time: {time.time() - t_proj:.1f}s")
    logger.info(f"  Xhat_train shape: {Xhat_train.shape}")
    logger.info(f"  Xhat_test shape: {Xhat_test.shape}")
    
    # Compute reconstruction errors
    logger.info("Computing reconstruction errors...")
    
    # Training data
    train_reconstructed = lift_quadratic(qm, Xhat_train.T)
    train_error = np.linalg.norm(train_reconstructed - Q_train, 'fro') / np.linalg.norm(Q_train, 'fro')
    
    # Test data
    test_reconstructed = lift_quadratic(qm, Xhat_test.T)
    test_error = np.linalg.norm(test_reconstructed - Q_test, 'fro') / np.linalg.norm(Q_test, 'fro')
    
    logger.info(f"  Train relative error: {train_error:.6e}")
    logger.info(f"  Test relative error: {test_error:.6e}")
    
    return Xhat_train, Xhat_test


# =============================================================================
# LEARNING MATRIX PREPARATION
# =============================================================================

def get_x_sq(X: np.ndarray) -> np.ndarray:
    """Compute quadratic features (upper triangular)."""
    K, r = X.shape
    s = r * (r + 1) // 2
    result = np.zeros((K, s))
    idx = 0
    for i in range(r):
        for j in range(i + 1):
            result[:, idx] = X[:, i] * X[:, j]
            idx += 1
    return result


def prepare_learning_matrices(
    Xhat_train: np.ndarray,
    train_boundaries: list,
    r: int,
    logger,
) -> dict:
    """Prepare matrices for ROM training."""
    logger.info("Preparing learning matrices...")
    
    # STATE LEARNING: Create valid pairs within each trajectory
    n_train_traj = len(train_boundaries) - 1
    X_state_list = []
    Y_state_list = []
    
    for traj_idx in range(n_train_traj):
        start_idx = train_boundaries[traj_idx]
        end_idx = train_boundaries[traj_idx + 1]
        
        Xhat_traj = Xhat_train[start_idx:end_idx, :]
        X_state_list.append(Xhat_traj[:-1, :])
        Y_state_list.append(Xhat_traj[1:, :])
    
    X_state = np.vstack(X_state_list)
    Y_state = np.vstack(Y_state_list)
    
    X_state2 = get_x_sq(X_state)
    D_state = np.concatenate((X_state, X_state2), axis=1)
    D_state_2 = D_state.T @ D_state
    
    logger.info(f"  State pairs: {X_state.shape[0]}")
    
    # OUTPUT LEARNING
    X_out = Xhat_train
    K = X_out.shape[0]
    E = np.ones((K, 1))
    
    mean_Xhat = np.mean(X_out, axis=0)
    Xhat_out = X_out - mean_Xhat[np.newaxis, :]
    
    scaling_Xhat = np.maximum(np.abs(np.min(X_out)), np.abs(np.max(X_out)))
    if scaling_Xhat < 1e-14:
        logger.warning(f"  WARNING: scaling_Xhat is {scaling_Xhat}, setting to 1.0")
        scaling_Xhat = 1.0
    
    Xhat_out /= scaling_Xhat
    Xhat_out2 = get_x_sq(Xhat_out)
    
    D_out = np.concatenate((Xhat_out, Xhat_out2, E), axis=1)
    D_out_2 = D_out.T @ D_out
    
    logger.info(f"  D_out shape: {D_out.shape}")
    logger.info(f"  D_out_2 condition: {np.linalg.cond(D_out_2):.2e}")
    
    return {
        'X_state': X_state,
        'Y_state': Y_state,
        'D_state': D_state,
        'D_state_2': D_state_2,
        'D_out': D_out,
        'D_out_2': D_out_2,
        'mean_Xhat': mean_Xhat,
        'scaling_Xhat': scaling_Xhat,
    }


def load_reference_gamma(cfg: QuadManifoldConfig, logger) -> dict:
    """Load reference Gamma values from training files."""
    logger.info("Loading reference Gamma values...")
    
    Gamma_n_list = []
    Gamma_c_list = []
    
    for file_path in cfg.training_files:
        with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
            gamma_n = fh["gamma_n"].values
            gamma_c = fh["gamma_c"].values
        
        if cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            if max_snaps is not None:
                n_time_original = len(gamma_n)
                n_time = min(n_time_original, max_snaps)
                gamma_n = gamma_n[:n_time]
                gamma_c = gamma_c[:n_time]
        
        Gamma_n_list.append(gamma_n)
        Gamma_c_list.append(gamma_c)
    
    Gamma_n = np.concatenate(Gamma_n_list)
    Gamma_c = np.concatenate(Gamma_c_list)
    
    Y_Gamma = np.vstack((Gamma_n, Gamma_c))
    
    logger.info(f"  Y_Gamma shape: {Y_Gamma.shape}")
    
    return {
        'Y_Gamma': Y_Gamma,
        'mean_Gamma_n': np.mean(Gamma_n),
        'std_Gamma_n': np.std(Gamma_n, ddof=1),
        'mean_Gamma_c': np.mean(Gamma_c),
        'std_Gamma_c': np.std(Gamma_c, ddof=1),
    }


# =============================================================================
# STATUS TRACKING
# =============================================================================

def save_step_status(run_dir: str, step: str, status: str, details: dict = None):
    """Save step status to JSON file."""
    import json
    status_file = os.path.join(run_dir, "status.json")
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            all_status = json.load(f)
    else:
        all_status = {}
    
    all_status[step] = {
        'status': status,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'details': details or {},
    }
    
    with open(status_file, 'w') as f:
        json.dump(all_status, f, indent=2)


def save_config_copy(cfg: QuadManifoldConfig, run_dir: str):
    """Save a copy of the configuration."""
    config_copy = {
        'run_name': cfg.run_name,
        'paths': {
            'output_base': cfg.output_base,
            'data_dir': cfg.data_dir,
            'training_files': [os.path.basename(f) for f in cfg.training_files],
            'test_files': [os.path.basename(f) for f in cfg.test_files],
        },
        'physics': {
            'dt': cfg.dt,
            'n_fields': cfg.n_fields,
            'n_x': cfg.n_x,
            'n_y': cfg.n_y,
        },
        'quadratic_manifold': {
            'r': cfg.r,
            'n_vectors_to_check': cfg.n_vectors_to_check,
            'reg_magnitude': cfg.reg_magnitude,
            'compare_with_pod': cfg.compare_with_pod,
            'initial_indices': cfg.initial_indices,
        },
        'truncation': {
            'enabled': cfg.truncation_enabled,
            'method': cfg.truncation_method,
            'snapshots': cfg.truncation_snapshots,
            'time': cfg.truncation_time,
        },
        'execution': {
            'verbose': cfg.verbose,
            'log_level': cfg.log_level,
            'engine': cfg.engine,
        },
    }
    
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for quadratic manifold preprocessing."""
    parser = argparse.ArgumentParser(
        description="Step 1: Quadratic Manifold Computation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run directory (optional, creates new if not specified)"
    )
    args = parser.parse_args()
    
    # Load configuration from YAML
    cfg = QuadManifoldConfig.from_yaml(args.config)
    
    # Get/create run directory
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = get_run_directory(cfg)
    
    # Setup logging
    logger = setup_logging("step_1_qm", run_dir, cfg.log_level)
    
    print_header("STEP 1 (QUADRATIC MANIFOLD): DATA PREPROCESSING")
    print(f"  Run directory: {run_dir}")
    print(f"  Config file: {args.config}")
    print_config_summary(cfg)
    
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config file: {args.config}")
    
    save_step_status(run_dir, "step_1_qm", "running")
    save_config_copy(cfg, run_dir)
    
    paths = get_output_paths(run_dir)
    
    start_time = time.time()
    
    try:
        # 1. Load data
        t_load = time.time()
        Q_train, Q_test, train_boundaries, test_boundaries, n_spatial = load_all_data(cfg, logger)
        logger.info(f"Data loading time: {time.time() - t_load:.1f}s")
        
        # Save boundaries
        np.savez(
            paths["boundaries"],
            train_boundaries=train_boundaries,
            test_boundaries=test_boundaries,
            n_spatial=n_spatial,
        )
        
        # 2. Compute quadratic manifold
        t_qm = time.time()
        logger.info(f"\nComputing quadratic manifold with r={cfg.r}...")
        
        # Handle initial indices
        initial_indices = np.array(cfg.initial_indices) if cfg.initial_indices else None
        
        qm = quadmani_greedy(
            Q_train,
            r=cfg.r,
            n_vectors_to_check=cfg.n_vectors_to_check,
            reg_magnitude=cfg.reg_magnitude,
            idx_in_initial=initial_indices,
            verbose=cfg.verbose,
            logger=logger,
        )
        
        logger.info(f"Quadratic manifold computation time: {time.time() - t_qm:.1f}s")
        logger.info(f"Selected mode indices: {qm.selected_indices}")
        
        # Save quadratic manifold
        save_quadratic_manifold(qm, paths["qm_file"])
        logger.info(f"Saved quadratic manifold to {paths['qm_file']}")
        
        # Also save in POD-compatible format
        np.savez(
            paths["pod_file"],
            S=qm.singular_values,
            V=qm.V,
            W=qm.W,
            shift=qm.shift,
            selected_indices=qm.selected_indices,
            is_quadratic_manifold=True,
        )
        logger.info(f"Saved POD-compatible file to {paths['pod_file']}")
        
        # 3. Compare with linear POD (if enabled)
        if cfg.compare_with_pod:
            logger.info("\nComparing with standard linear POD...")
            comparison = compare_with_linear_pod(qm, Q_test, logger=logger)
            np.savez(paths["comparison"], **comparison)
            logger.info(f"Saved comparison to {paths['comparison']}")
        
        # 4. Project data
        t_proj = time.time()
        Xhat_train, Xhat_test = project_with_quadratic_manifold(Q_train, Q_test, qm, logger)
        logger.info(f"Projection time: {time.time() - t_proj:.1f}s")
        
        np.save(paths["xhat_train"], Xhat_train)
        np.save(paths["xhat_test"], Xhat_test)
        logger.info("Saved projected data")
        
        # 5. Save initial conditions
        logger.info("Saving initial conditions...")
        n_train_files = len(cfg.training_files)
        n_test_files = len(cfg.test_files)
        
        train_ICs = np.array([Q_train[:, train_boundaries[i]] for i in range(n_train_files)])
        test_ICs = np.array([Q_test[:, test_boundaries[i]] for i in range(n_test_files)])
        train_ICs_reduced = np.array([Xhat_train[train_boundaries[i], :] for i in range(n_train_files)])
        test_ICs_reduced = np.array([Xhat_test[test_boundaries[i], :] for i in range(n_test_files)])
        
        np.savez(
            paths["initial_conditions"],
            train_ICs=train_ICs,
            test_ICs=test_ICs,
            train_ICs_reduced=train_ICs_reduced,
            test_ICs_reduced=test_ICs_reduced,
            qm_shift=qm.shift,
        )
        logger.info("Saved initial conditions")
        
        # 6. Prepare learning matrices
        t_learn = time.time()
        learning = prepare_learning_matrices(Xhat_train, train_boundaries, cfg.r, logger)
        np.savez(paths["learning_matrices"], **learning)
        logger.info(f"Saved learning matrices to {paths['learning_matrices']}")
        
        # 7. Load and save reference Gamma
        gamma_ref = load_reference_gamma(cfg, logger)
        np.savez(paths["gamma_ref"], **gamma_ref)
        logger.info(f"Saved gamma reference to {paths['gamma_ref']}")
        
        # 8. Save preprocessing info
        np.savez(
            paths["preprocessing_info"],
            method="quadratic_manifold",
            r=cfg.r,
            n_vectors_checked=cfg.n_vectors_to_check,
            reg_magnitude=cfg.reg_magnitude,
            selected_indices=qm.selected_indices,
            n_spatial=n_spatial,
            n_fields=cfg.n_fields,
            n_x=cfg.n_x,
            n_y=cfg.n_y,
            dt=cfg.dt,
        )
        logger.info(f"Saved preprocessing info")
        
        logger.info(f"Learning matrix prep time: {time.time() - t_learn:.1f}s")
        
        # Cleanup
        del Q_train, Q_test
        gc.collect()
        
        # Final timing
        total_time = time.time() - start_time
        
        save_step_status(run_dir, "step_1_qm", "completed", {
            "n_spatial": int(n_spatial),
            "n_train_snapshots": int(train_boundaries[-1]),
            "n_test_snapshots": int(test_boundaries[-1]),
            "r": cfg.r,
            "selected_indices": qm.selected_indices.tolist(),
            "total_time_seconds": total_time,
        })
        
        print_header("STEP 1 (QUADRATIC MANIFOLD) COMPLETE")
        print(f"  Output directory: {run_dir}")
        print(f"  Total runtime: {total_time:.1f} seconds")
        print(f"  Selected modes: {qm.selected_indices}")
        logger.info(f"Step 1 (QM) completed successfully in {total_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Step 1 (QM) failed: {e}", exc_info=True)
        save_step_status(run_dir, "step_1_qm", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()