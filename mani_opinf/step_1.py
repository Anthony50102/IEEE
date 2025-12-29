"""
Step 1: Quadratic Manifold Preprocessing.

Computes a quadratic manifold basis for nonlinear model reduction:
    x ≈ V·z + W·h(z) + μ

This replaces standard POD with greedy mode selection that minimizes
reconstruction error when using quadratic features.

Usage:
    python step_1_quadratic_manifold.py --config config.yaml
    python step_1_quadratic_manifold.py --config config.yaml --run-dir /path/to/output
"""

import argparse
import gc
import time
import json
import logging
import numpy as np
import xarray as xr
import yaml
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from quadratic_manifold import (
    quadmani_greedy, quadmani_greedy_from_svd,
    ShiftedSVD, QuadraticManifold,
    linear_reduce, lift_quadratic, compute_shifted_svd,
    save_quadratic_manifold, save_shifted_svd, load_shifted_svd,
    compute_energy_metrics, get_num_quadratic_features
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration from YAML."""
    run_name: str
    output_base: str
    data_dir: str
    training_files: List[str]
    test_files: List[str]
    
    dt: float
    n_fields: int
    n_x: int
    n_y: int
    
    r: int
    n_vectors_to_check: int
    reg_magnitude: float
    initial_indices: List[int]
    use_precomputed_svd: bool
    svd_file: Optional[str]
    
    truncation_enabled: bool
    truncation_method: str
    truncation_snapshots: Optional[int]
    truncation_time: Optional[float]
    
    verbose: bool
    log_level: str
    engine: str
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        
        data_dir = cfg['paths']['data_dir']
        qm = cfg.get('quadratic_manifold', {})
        
        return cls(
            run_name=cfg['run_name'],
            output_base=cfg['paths']['output_base'],
            data_dir=data_dir,
            training_files=[os.path.join(data_dir, f) for f in cfg['paths']['training_files']],
            test_files=[os.path.join(data_dir, f) for f in cfg['paths']['test_files']],
            dt=cfg['physics']['dt'],
            n_fields=cfg['physics']['n_fields'],
            n_x=cfg['physics']['n_x'],
            n_y=cfg['physics']['n_y'],
            r=qm.get('r', cfg.get('pod', {}).get('r', 100)),
            n_vectors_to_check=qm.get('n_vectors_to_check', 200),
            reg_magnitude=float(qm.get('reg_magnitude', 1e-6)),
            initial_indices=qm.get('initial_indices', []),
            use_precomputed_svd=qm.get('use_precomputed_svd', False),
            svd_file=qm.get('svd_file', None),
            truncation_enabled=cfg['truncation']['enabled'],
            truncation_method=cfg['truncation']['method'],
            truncation_snapshots=cfg['truncation'].get('snapshots'),
            truncation_time=cfg['truncation'].get('time'),
            verbose=cfg['execution']['verbose'],
            log_level=cfg['execution']['log_level'],
            engine=cfg['execution']['engine'],
        )


def setup_run(cfg: Config, run_dir: Optional[str] = None) -> Tuple[str, dict, logging.Logger]:
    """Create run directory and setup logging."""
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg.output_base, f"{cfg.run_name}_qm_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Paths
    paths = {
        "boundaries": os.path.join(run_dir, "boundaries.npz"),
        "qm_file": os.path.join(run_dir, "quadratic_manifold.npz"),
        "svd_file": os.path.join(run_dir, "shifted_svd.npz"),
        "pod_file": os.path.join(run_dir, "pod_basis.npz"),
        "xhat_train": os.path.join(run_dir, "Xhat_train.npy"),
        "xhat_test": os.path.join(run_dir, "Xhat_test.npy"),
        "initial_conditions": os.path.join(run_dir, "initial_conditions.npz"),
        "learning_matrices": os.path.join(run_dir, "learning_matrices.npz"),
        "gamma_ref": os.path.join(run_dir, "gamma_reference.npz"),
        "preprocessing_info": os.path.join(run_dir, "preprocessing_info.npz"),
        "comparison": os.path.join(run_dir, "qm_vs_pod_comparison.npz"),
    }
    
    # Logger
    logger = logging.getLogger("step_1_qm")
    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    fh = logging.FileHandler(os.path.join(run_dir, "step_1_qm.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return run_dir, paths, logger


def save_status(run_dir: str, step: str, status: str, details: dict = None):
    """Save step status to JSON."""
    status_file = os.path.join(run_dir, "status.json")
    all_status = {}
    if os.path.exists(status_file):
        with open(status_file) as f:
            all_status = json.load(f)
    all_status[step] = {
        'status': status,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'details': details or {}
    }
    with open(status_file, 'w') as f:
        json.dump(all_status, f, indent=2)


# =============================================================================
# Memory Management Helpers
# =============================================================================

def get_memmap_path(run_dir: str, name: str) -> str:
    """Get path for memory-mapped file."""
    return os.path.join(run_dir, f"{name}.dat")


def cleanup_memmap(run_dir: str, name: str):
    """Remove memory-mapped file if it exists."""
    path = get_memmap_path(run_dir, name)
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# Data Loading
# =============================================================================

def get_truncation_snapshots(cfg: Config) -> Optional[int]:
    """Compute max snapshots from truncation settings."""
    if cfg.truncation_time is not None:
        return int(cfg.truncation_time / cfg.dt)
    return cfg.truncation_snapshots


def load_snapshots(file_path: str, cfg: Config, max_snaps: Optional[int]) -> np.ndarray:
    """Load snapshot data from HDF5, shape (n_spatial, n_time)."""
    with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
        density = fh["density"].values
        phi = fh["phi"].values
    
    if max_snaps and max_snaps < density.shape[0]:
        density, phi = density[:max_snaps], phi[:max_snaps]
    
    n_time = density.shape[0]
    
    # Reshape if flattened
    if density.ndim == 2:
        grid = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid, grid)
        phi = phi.reshape(n_time, grid, grid)
    
    # Stack and reshape: (2, n_y, n_x, n_time) -> (n_spatial, n_time)
    Q = np.stack([density, phi], axis=0).transpose(0, 2, 3, 1)
    return Q.reshape(-1, n_time)


def load_all_data(cfg: Config, run_dir: str, logger) -> Tuple[np.ndarray, np.ndarray, list, list, int]:
    """Load training and test data using memory-mapped arrays for efficiency."""
    logger.info("Loading simulation data...")
    
    max_snaps = get_truncation_snapshots(cfg) if cfg.truncation_enabled else None
    
    # First pass: determine shapes and timesteps
    train_timesteps = []
    test_timesteps = []
    n_spatial = None
    
    for file_path in cfg.training_files:
        with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
            shape = fh["density"].shape
            n_time_raw = shape[0]
            if n_spatial is None:
                if len(shape) == 3:
                    n_spatial = 2 * shape[1] * shape[2]
                else:
                    n_spatial = 2 * shape[1]
        n_time = min(n_time_raw, max_snaps) if max_snaps else n_time_raw
        train_timesteps.append(n_time)
    
    for file_path in cfg.test_files:
        with xr.open_dataset(file_path, engine=cfg.engine, phony_dims="sort") as fh:
            n_time_raw = fh["density"].shape[0]
        n_time = min(n_time_raw, max_snaps) if max_snaps else n_time_raw
        test_timesteps.append(n_time)
    
    total_train = sum(train_timesteps)
    total_test = sum(test_timesteps)
    
    logger.info(f"  Spatial DOF: {n_spatial:,}")
    logger.info(f"  Total train snapshots: {total_train:,}")
    logger.info(f"  Total test snapshots: {total_test:,}")
    
    # Cleanup any existing memmap files
    cleanup_memmap(run_dir, "Q_train")
    cleanup_memmap(run_dir, "Q_test")
    
    # Create memory-mapped arrays
    Q_train = np.memmap(
        get_memmap_path(run_dir, "Q_train"),
        dtype='float64',
        mode='w+',
        shape=(n_spatial, total_train)
    )
    Q_test = np.memmap(
        get_memmap_path(run_dir, "Q_test"),
        dtype='float64',
        mode='w+',
        shape=(n_spatial, total_test)
    )
    
    # Compute boundaries
    train_bounds = [0] + list(np.cumsum(train_timesteps))
    test_bounds = [0] + list(np.cumsum(test_timesteps))
    
    # Load training data incrementally
    logger.info("Loading training trajectories...")
    for i, f in enumerate(cfg.training_files):
        t0 = time.time()
        Q_ic = load_snapshots(f, cfg, max_snaps)
        Q_train[:, train_bounds[i]:train_bounds[i+1]] = Q_ic
        del Q_ic
        gc.collect()
        if cfg.verbose:
            logger.info(f"  Loaded {os.path.basename(f)} ({time.time()-t0:.1f}s)")
    
    # Load test data incrementally
    logger.info("Loading test trajectories...")
    for i, f in enumerate(cfg.test_files):
        t0 = time.time()
        Q_ic = load_snapshots(f, cfg, max_snaps)
        Q_test[:, test_bounds[i]:test_bounds[i+1]] = Q_ic
        del Q_ic
        gc.collect()
        if cfg.verbose:
            logger.info(f"  Loaded {os.path.basename(f)} ({time.time()-t0:.1f}s)")
    
    logger.info(f"  Train: {Q_train.shape}, Test: {Q_test.shape}")
    return Q_train, Q_test, train_bounds, test_bounds, n_spatial


# =============================================================================
# Learning Matrix Preparation
# =============================================================================

def compute_quadratic_features(X: np.ndarray) -> np.ndarray:
    """Compute upper-triangular quadratic features for learning."""
    K, r = X.shape
    s = r * (r + 1) // 2
    result = np.zeros((K, s))
    idx = 0
    for i in range(r):
        for j in range(i + 1):
            result[:, idx] = X[:, i] * X[:, j]
            idx += 1
    return result


def prepare_learning_matrices(Xhat_train: np.ndarray, train_bounds: list, r: int, logger) -> dict:
    """Prepare matrices for ROM training."""
    logger.info("Preparing learning matrices...")
    
    # State learning: consecutive pairs within trajectories
    X_list, Y_list = [], []
    for i in range(len(train_bounds) - 1):
        traj = Xhat_train[train_bounds[i]:train_bounds[i+1]]
        X_list.append(traj[:-1])
        Y_list.append(traj[1:])
    
    X_state = np.vstack(X_list)
    Y_state = np.vstack(Y_list)
    X_state2 = compute_quadratic_features(X_state)
    D_state = np.concatenate([X_state, X_state2], axis=1)
    D_state_2 = D_state.T @ D_state
    
    # Output learning
    mean_Xhat = np.mean(Xhat_train, axis=0)
    Xhat_centered = Xhat_train - mean_Xhat
    scaling = max(np.abs(Xhat_train.min()), np.abs(Xhat_train.max()))
    if scaling < 1e-14:
        scaling = 1.0
    Xhat_scaled = Xhat_centered / scaling
    
    E = np.ones((len(Xhat_train), 1))
    D_out = np.concatenate([Xhat_scaled, compute_quadratic_features(Xhat_scaled), E], axis=1)
    D_out_2 = D_out.T @ D_out
    
    logger.info(f"  State pairs: {X_state.shape[0]}, D_out shape: {D_out.shape}")
    
    return {
        'X_state': X_state, 'Y_state': Y_state,
        'D_state': D_state, 'D_state_2': D_state_2,
        'D_out': D_out, 'D_out_2': D_out_2,
        'mean_Xhat': mean_Xhat, 'scaling_Xhat': scaling
    }


def load_gamma_reference(cfg: Config, logger) -> dict:
    """Load reference Gamma values."""
    logger.info("Loading reference Gamma values...")
    
    max_snaps = get_truncation_snapshots(cfg) if cfg.truncation_enabled else None
    Gamma_n, Gamma_c = [], []
    
    for f in cfg.training_files:
        with xr.open_dataset(f, engine=cfg.engine, phony_dims="sort") as fh:
            gn, gc = fh["gamma_n"].values, fh["gamma_c"].values
        if max_snaps:
            gn, gc = gn[:max_snaps], gc[:max_snaps]
        Gamma_n.append(gn)
        Gamma_c.append(gc)
    
    Gamma_n, Gamma_c = np.concatenate(Gamma_n), np.concatenate(Gamma_c)
    
    return {
        'Y_Gamma': np.vstack([Gamma_n, Gamma_c]),
        'mean_Gamma_n': np.mean(Gamma_n), 'std_Gamma_n': np.std(Gamma_n, ddof=1),
        'mean_Gamma_c': np.mean(Gamma_c), 'std_Gamma_c': np.std(Gamma_c, ddof=1)
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 1: Quadratic Manifold Preprocessing")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--run-dir", default=None, help="Output directory (optional)")
    args = parser.parse_args()
    
    cfg = Config.from_yaml(args.config)
    run_dir, paths, logger = setup_run(cfg, args.run_dir)
    
    print("=" * 70)
    print("  STEP 1: QUADRATIC MANIFOLD PREPROCESSING")
    print("=" * 70)
    print(f"  Run directory: {run_dir}")
    print(f"  r={cfg.r}, n_check={cfg.n_vectors_to_check}, reg={cfg.reg_magnitude:.1e}")
    print(f"  use_precomputed_svd: {cfg.use_precomputed_svd}")
    
    save_status(run_dir, "step_1_qm", "running")
    start_time = time.time()
    
    try:
        # 1. Load data (now uses memmap)
        Q_train, Q_test, train_bounds, test_bounds, n_spatial = load_all_data(cfg, run_dir, logger)
        np.savez(paths["boundaries"], train_boundaries=train_bounds,
                 test_boundaries=test_bounds, n_spatial=n_spatial)
        
        # 2. Compute quadratic manifold
        initial_idx = np.array(cfg.initial_indices) if cfg.initial_indices else None
        
        if cfg.use_precomputed_svd and cfg.svd_file and os.path.exists(cfg.svd_file):
            logger.info(f"Loading pre-computed SVD from {cfg.svd_file}")
            svd = load_shifted_svd(cfg.svd_file)
            qm = quadmani_greedy_from_svd(
                svd, r=cfg.r, n_vectors_to_check=cfg.n_vectors_to_check,
                reg_magnitude=cfg.reg_magnitude, idx_in_initial=initial_idx,
                verbose=cfg.verbose, logger=logger
            )
        else:
            # Use copy=False to save memory during SVD computation
            qm = quadmani_greedy(
                Q_train, r=cfg.r, n_vectors_to_check=cfg.n_vectors_to_check,
                reg_magnitude=cfg.reg_magnitude, idx_in_initial=initial_idx,
                verbose=cfg.verbose, logger=logger
            )
            # Save SVD for potential reuse (compute from manifold's singular values)
            svd = compute_shifted_svd(np.asarray(Q_train), copy=False)
            save_shifted_svd(svd, paths["svd_file"])
            logger.info(f"Saved SVD to {paths['svd_file']}")
            del svd
            gc.collect()
        
        logger.info(f"Selected modes: {qm.selected_indices}")
        
        # Save manifold
        save_quadratic_manifold(qm, paths["qm_file"])
        np.savez(paths["pod_file"], S=qm.singular_values, V=qm.V, W=qm.W,
                 shift=qm.shift, selected_indices=qm.selected_indices,
                 is_quadratic_manifold=True)
        
        # 3. Compute energy metrics (replaces expensive POD comparison)
        energy_metrics = compute_energy_metrics(qm, verbose=cfg.verbose, logger=logger)
        np.savez(paths["comparison"], **energy_metrics)
        
        # 4. Project data
        Xhat_train = linear_reduce(qm, Q_train).T  # (n_time, r)
        Xhat_test = linear_reduce(qm, Q_test).T
        
        # Reconstruction errors (compute on smaller batches if needed)
        logger.info("Computing reconstruction errors...")
        recon_train = lift_quadratic(qm, Xhat_train.T)
        train_err = np.linalg.norm(recon_train - Q_train, 'fro') / np.linalg.norm(Q_train, 'fro')
        del recon_train
        gc.collect()
        
        recon_test = lift_quadratic(qm, Xhat_test.T)
        test_err = np.linalg.norm(recon_test - Q_test, 'fro') / np.linalg.norm(Q_test, 'fro')
        del recon_test
        gc.collect()
        
        logger.info(f"Reconstruction: train={train_err:.6e}, test={test_err:.6e}")
        
        np.save(paths["xhat_train"], Xhat_train)
        np.save(paths["xhat_test"], Xhat_test)
        
        # 5. Initial conditions
        n_train, n_test = len(cfg.training_files), len(cfg.test_files)
        np.savez(
            paths["initial_conditions"],
            train_ICs=np.array([Q_train[:, train_bounds[i]] for i in range(n_train)]),
            test_ICs=np.array([Q_test[:, test_bounds[i]] for i in range(n_test)]),
            train_ICs_reduced=np.array([Xhat_train[train_bounds[i]] for i in range(n_train)]),
            test_ICs_reduced=np.array([Xhat_test[test_bounds[i]] for i in range(n_test)]),
            qm_shift=qm.shift
        )
        
        # 6. Learning matrices
        learning = prepare_learning_matrices(Xhat_train, train_bounds, cfg.r, logger)
        np.savez(paths["learning_matrices"], **learning)
        
        # 7. Gamma reference
        gamma_ref = load_gamma_reference(cfg, logger)
        np.savez(paths["gamma_ref"], **gamma_ref)
        
        # 8. Preprocessing info
        np.savez(
            paths["preprocessing_info"],
            method="quadratic_manifold", r=cfg.r,
            n_vectors_checked=cfg.n_vectors_to_check,
            reg_magnitude=cfg.reg_magnitude,
            selected_indices=qm.selected_indices,
            n_spatial=n_spatial, n_fields=cfg.n_fields,
            n_x=cfg.n_x, n_y=cfg.n_y, dt=cfg.dt
        )
        
        del Q_train, Q_test
        gc.collect()
        
        # Cleanup memmap files
        cleanup_memmap(run_dir, "Q_train")
        cleanup_memmap(run_dir, "Q_test")
        logger.info("Cleaned up temporary memmap files")
        
        total_time = time.time() - start_time
        save_status(run_dir, "step_1_qm", "completed", {
            "r": cfg.r, "selected_indices": qm.selected_indices.tolist(),
            "train_error": float(train_err), "test_error": float(test_err),
            "energy_pct": energy_metrics['selected_modes_energy_pct'],
            "total_time_seconds": total_time
        })
        
        print("=" * 70)
        print(f"  COMPLETE: {total_time:.1f}s")
        print(f"  Selected modes: {qm.selected_indices}")
        print(f"  Energy conserved: {energy_metrics['selected_modes_energy_pct']:.4f}%")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        save_status(run_dir, "step_1_qm", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()