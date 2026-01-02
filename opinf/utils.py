"""
Utility functions for the OpInf pipeline.

This module provides shared utilities:
- Configuration loading and validation
- Run directory management
- Logging setup
- Data loading utilities
- Step status tracking

Author: Anthony Poole
"""

import os
import yaml
import logging
import gc
import h5py
import xarray as xr
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from mpi4py import MPI


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OpInfConfig:
    """Configuration container for the OpInf pipeline."""
    
    # Run identification
    run_name: str = ""
    run_dir: str = ""
    
    # Paths
    output_base: str = ""
    data_dir: str = ""
    training_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    
    # Physics
    dt: float = 0.025
    n_fields: int = 2
    n_x: int = 512
    n_y: int = 512
    
    # POD
    r: int = 100
    target_energy: float = 0.9999
    
    # Truncation
    truncation_enabled: bool = False
    truncation_method: str = "time"
    truncation_snapshots: Optional[int] = None
    truncation_time: Optional[float] = None
    
    # Preprocessing
    centering_enabled: bool = True
    scaling_enabled: bool = False
    
    # Training
    training_end: int = 5000
    n_steps: int = 16001
    
    # Regularization grids
    state_lin: np.ndarray = field(default_factory=lambda: np.array([]))
    state_quad: np.ndarray = field(default_factory=lambda: np.array([]))
    output_lin: np.ndarray = field(default_factory=lambda: np.array([]))
    output_quad: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Model selection (threshold-based)
    threshold_mean: float = 0.05
    threshold_std: float = 0.30
    
    # Evaluation
    save_predictions: bool = True
    generate_plots: bool = True
    
    # Execution
    verbose: bool = True
    log_level: str = "INFO"
    engine: str = "h5netcdf"


def _build_reg_array(reg_config: dict) -> np.ndarray:
    """Build regularization parameter array from config dict."""
    scale = reg_config.get("scale", "linear")
    min_val = float(reg_config["min"])
    max_val = float(reg_config["max"])
    num_val = int(reg_config["num"])
    
    if scale == "log":
        return np.logspace(np.log10(min_val), np.log10(max_val), num_val)
    else:
        return np.linspace(min_val, max_val, num_val)


def load_config(config_path: str) -> OpInfConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    cfg = OpInfConfig()
    cfg.run_name = raw.get("run_name", "")
    
    # Paths
    paths = raw.get("paths", {})
    cfg.output_base = paths.get("output_base", "")
    cfg.data_dir = paths.get("data_dir", "")
    cfg.training_files = [
        os.path.join(cfg.data_dir, f) if not os.path.isabs(f) else f
        for f in paths.get("training_files", [])
    ]
    cfg.test_files = [
        os.path.join(cfg.data_dir, f) if not os.path.isabs(f) else f
        for f in paths.get("test_files", [])
    ]
    
    # Physics
    physics = raw.get("physics", {})
    cfg.dt = physics.get("dt", 0.025)
    cfg.n_fields = physics.get("n_fields", 2)
    cfg.n_x = physics.get("n_x", 512)
    cfg.n_y = physics.get("n_y", 512)
    
    # POD
    pod = raw.get("pod", {})
    cfg.r = pod.get("r", 100)
    cfg.target_energy = pod.get("target_energy", 0.9999)
    
    # Truncation
    trunc = raw.get("truncation", {})
    cfg.truncation_enabled = trunc.get("enabled", False)
    cfg.truncation_method = trunc.get("method", "time")
    cfg.truncation_snapshots = trunc.get("snapshots")
    cfg.truncation_time = trunc.get("time")
    
    # Preprocessing
    preproc = raw.get("preprocessing", {})
    cfg.centering_enabled = preproc.get("centering", True)
    cfg.scaling_enabled = preproc.get("scaling", False)
    
    # Training
    training = raw.get("training", {})
    cfg.training_end = training.get("training_end", 5000)
    cfg.n_steps = training.get("n_steps", 16001)
    
    # Regularization
    reg = raw.get("regularization", {})
    cfg.state_lin = _build_reg_array(reg.get("state_lin", {"min": 1, "max": 1000, "num": 10}))
    cfg.state_quad = _build_reg_array(reg.get("state_quad", {"min": 1e7, "max": 1e12, "num": 10}))
    cfg.output_lin = _build_reg_array(reg.get("output_lin", {"min": 1e-8, "max": 1e-2, "num": 10}))
    cfg.output_quad = _build_reg_array(reg.get("output_quad", {"min": 1e-10, "max": 1e-2, "num": 10}))
    
    # Model selection
    selection = raw.get("model_selection", {})
    cfg.threshold_mean = selection.get("threshold_mean", 0.05)
    cfg.threshold_std = selection.get("threshold_std", 0.30)
    
    # Evaluation
    evaluation = raw.get("evaluation", {})
    cfg.save_predictions = evaluation.get("save_predictions", True)
    cfg.generate_plots = evaluation.get("generate_plots", True)
    
    # Execution
    execution = raw.get("execution", {})
    cfg.verbose = execution.get("verbose", True)
    cfg.log_level = execution.get("log_level", "INFO")
    cfg.engine = execution.get("engine", "h5netcdf")
    
    return cfg


def save_config(cfg: OpInfConfig, output_path: str, step_name: str = None) -> str:
    """Save configuration to YAML file."""
    config_dict = {
        "run_name": cfg.run_name,
        "run_dir": cfg.run_dir,
        "paths": {
            "output_base": cfg.output_base,
            "data_dir": cfg.data_dir,
            "training_files": cfg.training_files,
            "test_files": cfg.test_files,
        },
        "physics": {"dt": cfg.dt, "n_fields": cfg.n_fields, "n_x": cfg.n_x, "n_y": cfg.n_y},
        "pod": {"r": cfg.r, "target_energy": cfg.target_energy},
        "truncation": {
            "enabled": cfg.truncation_enabled,
            "method": cfg.truncation_method,
            "snapshots": cfg.truncation_snapshots,
            "time": cfg.truncation_time,
        },
        "preprocessing": {"centering": cfg.centering_enabled, "scaling": cfg.scaling_enabled},
        "training": {"training_end": cfg.training_end, "n_steps": cfg.n_steps},
        "regularization": {
            "state_lin": cfg.state_lin.tolist(),
            "state_quad": cfg.state_quad.tolist(),
            "output_lin": cfg.output_lin.tolist(),
            "output_quad": cfg.output_quad.tolist(),
        },
        "model_selection": {
            "threshold_mean": cfg.threshold_mean,
            "threshold_std": cfg.threshold_std,
        },
        "evaluation": {"save_predictions": cfg.save_predictions, "generate_plots": cfg.generate_plots},
        "execution": {"verbose": cfg.verbose, "log_level": cfg.log_level, "engine": cfg.engine},
    }
    
    filename = f"config_{step_name}.yaml" if step_name else "config.yaml"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    return filepath


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def create_run_directory(cfg: OpInfConfig) -> str:
    """Create a new run directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{cfg.run_name}" if cfg.run_name else timestamp
    run_dir = os.path.join(cfg.output_base, dir_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run_dir = run_dir
    return run_dir


def get_run_directory(cfg: OpInfConfig, run_dir: str = None) -> str:
    """Get or create run directory."""
    if run_dir and os.path.isdir(run_dir):
        cfg.run_dir = run_dir
        return run_dir
    return create_run_directory(cfg)


def get_output_paths(run_dir: str) -> dict:
    """Get standard output file paths for a run."""
    return {
        # Step 1 outputs
        "pod_file": os.path.join(run_dir, "POD.npz"),
        "pod_basis": os.path.join(run_dir, "POD_basis_Ur.npy"),
        "xhat_train": os.path.join(run_dir, "X_hat_train.npy"),
        "xhat_test": os.path.join(run_dir, "X_hat_test.npy"),
        "boundaries": os.path.join(run_dir, "data_boundaries.npz"),
        "initial_conditions": os.path.join(run_dir, "initial_conditions.npz"),
        "gamma_ref": os.path.join(run_dir, "gamma_reference.npz"),
        "learning_matrices": os.path.join(run_dir, "learning_matrices.npz"),
        "preprocessing_info": os.path.join(run_dir, "preprocessing_info.npz"),
        # Step 2 outputs
        "ensemble_models": os.path.join(run_dir, "ensemble_models.npz"),
        "sweep_results": os.path.join(run_dir, "sweep_results.npz"),
        "operators_dir": os.path.join(run_dir, "operators"),
        # Step 3 outputs
        "predictions": os.path.join(run_dir, "ensemble_predictions.npz"),
        "metrics": os.path.join(run_dir, "evaluation_metrics.yaml"),
        "figures_dir": os.path.join(run_dir, "figures"),
    }


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(name: str, run_dir: str, log_level: str = "INFO", rank: int = 0) -> logging.Logger:
    """Set up logging for a pipeline step."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level.upper()))
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (only rank 0)
    if rank == 0 and run_dir:
        log_file = os.path.join(run_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class DummyLogger:
    """Silent logger for non-root MPI ranks."""
    def info(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def debug(self, *a, **kw): pass


# =============================================================================
# STEP STATUS TRACKING
# =============================================================================

def save_step_status(run_dir: str, step: str, status: str, metadata: dict = None):
    """Save step completion status."""
    status_file = os.path.join(run_dir, "pipeline_status.yaml")
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = yaml.safe_load(f) or {}
    else:
        status_data = {}
    
    status_data[step] = {"status": status, "timestamp": datetime.now().isoformat()}
    if metadata:
        status_data[step].update(metadata)
    
    with open(status_file, 'w') as f:
        yaml.dump(status_data, f, default_flow_style=False)


def check_step_completed(run_dir: str, step: str) -> bool:
    """Check if a step has completed successfully."""
    status_file = os.path.join(run_dir, "pipeline_status.yaml")
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = yaml.safe_load(f) or {}
        return status_data.get(step, {}).get("status") == "completed"
    return False


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def get_dt_from_file(file_path: str, default: float = 0.025) -> float:
    """Extract time step (dt) from HDF5 file attributes."""
    try:
        with h5py.File(file_path, 'r') as f:
            for loc in [f.attrs, f]:
                if 'dt' in loc:
                    return float(f['dt'][()] if 'dt' in f else f.attrs['dt'])
            for grp_name in ['params', 'metadata', 'parameters']:
                if grp_name in f:
                    grp = f[grp_name]
                    if 'dt' in grp.attrs:
                        return float(grp.attrs['dt'])
                    if 'dt' in grp:
                        return float(grp['dt'][()])
    except Exception:
        pass
    return default


def compute_truncation_snapshots(
    file_path: str,
    truncate_snapshots: int = None,
    truncate_time: float = None,
    default_dt: float = 0.025,
) -> Optional[int]:
    """Compute number of snapshots to keep based on truncation settings."""
    if truncate_snapshots is not None:
        return truncate_snapshots
    elif truncate_time is not None:
        dt = get_dt_from_file(file_path, default_dt)
        return int(truncate_time / dt)
    return None


def load_dataset(path: str, engine: str = "h5netcdf"):
    """Load xarray dataset from HDF5 file."""
    try:
        return xr.open_dataset(path, engine=engine)
    except Exception:
        return xr.open_dataset(path, engine=engine, phony_dims="sort")


# =============================================================================
# MPI UTILITIES
# =============================================================================

def distribute_indices(rank: int, n_total: int, size: int) -> tuple:
    """Distribute indices across MPI ranks."""
    n_per_rank = n_total // size
    start = rank * n_per_rank
    end = (rank + 1) * n_per_rank
    
    # Last rank handles remainder
    if rank == size - 1 and end != n_total:
        end = n_total
    
    return start, end, end - start


def chunked_bcast(comm, data, root: int = 0, max_bytes: int = 2**30):
    """
    Broadcast a numpy array in chunks to avoid MPI 32-bit integer overflow.
    
    MPI's Bcast uses a 32-bit signed integer for count, limiting messages to ~2GB.
    """
    rank = comm.Get_rank()
    
    # Broadcast shape and dtype first
    if rank == root:
        shape, dtype = data.shape, data.dtype
    else:
        shape, dtype = None, None
    
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    
    if rank != root:
        data = np.empty(shape, dtype=dtype)
    
    # If small enough, single broadcast
    itemsize = np.dtype(dtype).itemsize
    total_bytes = int(np.prod(shape)) * itemsize
    
    if total_bytes <= max_bytes:
        comm.Bcast(data, root=root)
        return data
    
    # Chunked broadcast for large arrays
    n_rows = shape[0]
    bytes_per_row = total_bytes // n_rows
    rows_per_chunk = max(1, max_bytes // bytes_per_row)
    
    data_flat = data.reshape(n_rows, -1) if len(shape) > 1 else data.reshape(n_rows, 1)
    
    for start_row in range(0, n_rows, rows_per_chunk):
        end_row = min(start_row + rows_per_chunk, n_rows)
        if rank == root:
            chunk = np.ascontiguousarray(data_flat[start_row:end_row, :])
        else:
            chunk = np.empty((end_row - start_row, data_flat.shape[1]), dtype=dtype)
        comm.Bcast(chunk, root=root)
        if rank != root:
            data_flat[start_row:end_row, :] = chunk
    
    return data


def create_shared_array(node_comm, shape, dtype=np.float64):
    """Create a numpy array backed by MPI shared memory within a node."""
    node_rank = node_comm.Get_rank()
    itemsize = np.dtype(dtype).itemsize
    nbytes = int(np.prod(shape)) * itemsize
    
    if node_rank == 0:
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    else:
        win = MPI.Win.Allocate_shared(0, itemsize, comm=node_comm)
    
    buf, _ = win.Shared_query(0)
    arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    
    return arr, win


# =============================================================================
# CONSOLE OUTPUT HELPERS
# =============================================================================

def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_config_summary(cfg: OpInfConfig):
    """Print a summary of the configuration."""
    print_header("CONFIGURATION SUMMARY")
    print(f"  Run name: {cfg.run_name or '(auto)'}")
    print(f"  Training files: {len(cfg.training_files)}")
    print(f"  Test files: {len(cfg.test_files)}")
    print(f"  POD modes (r): {cfg.r}")
    print(f"  Truncation: {'enabled' if cfg.truncation_enabled else 'disabled'}")
    print(f"  Centering: {'enabled' if cfg.centering_enabled else 'disabled'}")
    print(f"  Scaling: {'enabled' if cfg.scaling_enabled else 'disabled'}")
    n_reg = len(cfg.state_lin) * len(cfg.state_quad) * len(cfg.output_lin) * len(cfg.output_quad)
    print(f"  Regularization combinations: {n_reg:,}")
    print("=" * 70 + "\n")
