"""
Pipeline Utilities for Operator Inference ROM.

This module provides shared utilities for the OpInf pipeline:
- Configuration loading and validation
- Run directory management
- Logging setup
- Common data structures

Author: Anthony Poole
"""

import os
import yaml
import shutil
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

#TODO: Clean this part
# gets non-redundant quadratic terms of X
def get_x_sq(X):
    if len(np.shape(X)) == 1:  # if X is a vector
        r = np.size(X)
        prods = []
        for i in range(r):
            temp = X[i] * X[i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods))

    elif len(np.shape(X)) == 2:  # if X is a matrix
        K, r = np.shape(X)

        prods = []
        for i in range(r):
            temp = np.transpose(np.broadcast_to(X[:, i], (r - i, K))) * X[:, i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods), axis=1)

    else:
        print("invalid input size for helpers.get_x_sq")
    return X2

def solve_opinf_difference_model(s0: np.ndarray, n_steps: int, f: callable):
    """
    Integrate a discrete-time dynamical system forward.
    
    Solves the difference equation: s_{k+1} = f(s_k)
    
    Parameters
    ----------
    s0 : np.ndarray
        Initial state vector of shape (r,).
    n_steps : int
        Number of time steps to integrate.
    f : callable
        State transition function f: R^r -> R^r.
    
    Returns
    -------
    is_nan : bool
        True if NaN values were encountered during integration.
    s : np.ndarray
        State trajectory of shape (r, n_steps).
    
    Examples
    --------
    >>> A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    >>> f = lambda x: A @ x
    >>> s0 = np.array([1.0, 0.0])
    >>> is_nan, trajectory = solve_opinf_difference_model(s0, 100, f)
    """
    r = np.size(s0)
    s = np.zeros((r, n_steps))
    is_nan = False

    s[:, 0] = s0
    for i in range(n_steps - 1):
        s[:, i + 1] = f(s[:, i])

        if np.any(np.isnan(s[:, i + 1])):
            print(f"NaN encountered at iteration {i + 1}")
            is_nan = True
            break

    return is_nan, s


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration container for the OpInf pipeline.
    
    Attributes are organized by category matching the YAML structure.
    """
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
    svd_save: int = 100
    
    # Truncation
    truncation_enabled: bool = False
    truncation_method: str = "time"
    truncation_snapshots: Optional[int] = None
    truncation_time: Optional[float] = None
    
    # Training
    training_end: int = 5000
    n_steps: int = 16001
    
    # Regularization
    state_lin: np.ndarray = field(default_factory=lambda: np.array([]))
    state_quad: np.ndarray = field(default_factory=lambda: np.array([]))
    output_lin: np.ndarray = field(default_factory=lambda: np.array([]))
    output_quad: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Model selection
    selection_method: str = "threshold"
    num_top_models: int = 20
    threshold_mean: float = 0.05
    threshold_std: float = 0.30
    
    # Evaluation
    compute_full_state: bool = False
    save_predictions: bool = True
    generate_plots: bool = True
    
    # Execution
    verbose: bool = True
    log_level: str = "INFO"
    engine: str = "h5netcdf"


def _build_reg_array(reg_config: dict) -> np.ndarray:
    """Build regularization parameter array from config dict."""
    scale = reg_config.get("scale", "linear")
    
    # Convert to float to handle string inputs from YAML
    min_val = float(reg_config["min"])
    max_val = float(reg_config["max"])
    num_val = int(reg_config["num"])
    
    if scale == "log":
        return np.logspace(
            np.log10(min_val),
            np.log10(max_val),
            num_val
        )
    else:
        return np.linspace(
            min_val,
            max_val,
            num_val
        )


def load_config(config_path: str) -> PipelineConfig:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    
    Returns
    -------
    PipelineConfig
        Populated configuration object.
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    cfg = PipelineConfig()
    
    # Run identification
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
    cfg.svd_save = pod.get("svd_save", 100)
    
    # Truncation
    trunc = raw.get("truncation", {})
    cfg.truncation_enabled = trunc.get("enabled", False)
    cfg.truncation_method = trunc.get("method", "time")
    cfg.truncation_snapshots = trunc.get("snapshots")
    cfg.truncation_time = trunc.get("time")
    
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
    cfg.selection_method = selection.get("method", "threshold")
    cfg.num_top_models = selection.get("num_top_models", 20)
    cfg.threshold_mean = selection.get("threshold_mean", 0.05)
    cfg.threshold_std = selection.get("threshold_std", 0.30)
    
    # Evaluation
    evaluation = raw.get("evaluation", {})
    cfg.compute_full_state = evaluation.get("compute_full_state", False)
    cfg.save_predictions = evaluation.get("save_predictions", True)
    cfg.generate_plots = evaluation.get("generate_plots", True)
    
    # Execution
    execution = raw.get("execution", {})
    cfg.verbose = execution.get("verbose", True)
    cfg.log_level = execution.get("log_level", "INFO")
    cfg.engine = execution.get("engine", "h5netcdf")
    
    return cfg


def save_config(cfg: PipelineConfig, output_path: str) -> str:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object.
    output_path : str
        Directory to save configuration.
    
    Returns
    -------
    str
        Path to saved configuration file.
    """
    config_dict = {
        "run_name": cfg.run_name,
        "run_dir": cfg.run_dir,
        "paths": {
            "output_base": cfg.output_base,
            "data_dir": cfg.data_dir,
            "training_files": cfg.training_files,
            "test_files": cfg.test_files,
        },
        "physics": {
            "dt": cfg.dt,
            "n_fields": cfg.n_fields,
            "n_x": cfg.n_x,
            "n_y": cfg.n_y,
        },
        "pod": {
            "r": cfg.r,
            "svd_save": cfg.svd_save,
        },
        "truncation": {
            "enabled": cfg.truncation_enabled,
            "method": cfg.truncation_method,
            "snapshots": cfg.truncation_snapshots,
            "time": cfg.truncation_time,
        },
        "training": {
            "training_end": cfg.training_end,
            "n_steps": cfg.n_steps,
        },
        "regularization": {
            "state_lin": cfg.state_lin.tolist(),
            "state_quad": cfg.state_quad.tolist(),
            "output_lin": cfg.output_lin.tolist(),
            "output_quad": cfg.output_quad.tolist(),
        },
        "model_selection": {
            "method": cfg.selection_method,
            "num_top_models": cfg.num_top_models,
            "threshold_mean": cfg.threshold_mean,
            "threshold_std": cfg.threshold_std,
        },
        "evaluation": {
            "compute_full_state": cfg.compute_full_state,
            "save_predictions": cfg.save_predictions,
            "generate_plots": cfg.generate_plots,
        },
        "execution": {
            "verbose": cfg.verbose,
            "log_level": cfg.log_level,
            "engine": cfg.engine,
        },
    }
    
    filepath = os.path.join(output_path, "config.yaml")
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    return filepath


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def create_run_directory(cfg: PipelineConfig) -> str:
    """
    Create a new run directory with timestamp.
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object.
    
    Returns
    -------
    str
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cfg.run_name:
        dir_name = f"{timestamp}_{cfg.run_name}"
    else:
        dir_name = timestamp
    
    run_dir = os.path.join(cfg.output_base, dir_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Update config with run directory
    cfg.run_dir = run_dir
    
    return run_dir


def get_run_directory(cfg: PipelineConfig, run_dir: str = None) -> str:
    """
    Get or create run directory.
    
    If run_dir is provided, use it (for continuing from previous step).
    Otherwise, create a new timestamped directory.
    
    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object.
    run_dir : str, optional
        Existing run directory to use.
    
    Returns
    -------
    str
        Path to run directory.
    """
    if run_dir and os.path.isdir(run_dir):
        cfg.run_dir = run_dir
        return run_dir
    else:
        return create_run_directory(cfg)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(
    name: str,
    run_dir: str,
    log_level: str = "INFO",
    rank: int = 0
) -> logging.Logger:
    """
    Set up logging for a pipeline step.
    
    Parameters
    ----------
    name : str
        Logger name (usually step name).
    run_dir : str
        Run directory for log file.
    log_level : str
        Logging level.
    rank : int
        MPI rank (for parallel execution).
    
    Returns
    -------
    logging.Logger
        Configured logger.
    """
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
    
    # File handler (only rank 0 in parallel)
    if rank == 0 and run_dir:
        log_file = os.path.join(run_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# STEP STATUS TRACKING
# =============================================================================

def save_step_status(run_dir: str, step: str, status: str, metadata: dict = None):
    """
    Save step completion status.
    
    Parameters
    ----------
    run_dir : str
        Run directory.
    step : str
        Step name (e.g., "step_1", "step_2").
    status : str
        Status ("completed", "failed", "running").
    metadata : dict, optional
        Additional metadata to save.
    """
    status_file = os.path.join(run_dir, "pipeline_status.yaml")
    
    # Load existing status or create new
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = yaml.safe_load(f) or {}
    else:
        status_data = {}
    
    # Update status
    status_data[step] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        status_data[step].update(metadata)
    
    # Save
    with open(status_file, 'w') as f:
        yaml.dump(status_data, f, default_flow_style=False)


def load_step_status(run_dir: str) -> dict:
    """
    Load pipeline status.
    
    Parameters
    ----------
    run_dir : str
        Run directory.
    
    Returns
    -------
    dict
        Status dictionary.
    """
    status_file = os.path.join(run_dir, "pipeline_status.yaml")
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def check_step_completed(run_dir: str, step: str) -> bool:
    """Check if a step has completed successfully."""
    status = load_step_status(run_dir)
    return status.get(step, {}).get("status") == "completed"


# =============================================================================
# DATA FILE PATHS
# =============================================================================

def get_output_paths(run_dir: str) -> dict:
    """
    Get standard output file paths for a run.
    
    Parameters
    ----------
    run_dir : str
        Run directory.
    
    Returns
    -------
    dict
        Dictionary of output file paths.
    """
    return {
        # Step 1 outputs
        "pod_file": os.path.join(run_dir, "POD.npz"),
        "xhat_train": os.path.join(run_dir, "X_hat_train.npy"),
        "xhat_test": os.path.join(run_dir, "X_hat_test.npy"),
        "boundaries": os.path.join(run_dir, "data_boundaries.npz"),
        "initial_conditions": os.path.join(run_dir, "initial_conditions.npz"),
        "gamma_ref": os.path.join(run_dir, "gamma_reference.npz"),
        "learning_matrices": os.path.join(run_dir, "learning_matrices.npz"),
        
        # Step 2 outputs
        "ensemble_models": os.path.join(run_dir, "ensemble_models.npz"),
        "sweep_results": os.path.join(run_dir, "sweep_results.npz"),
        
        # Step 3 outputs
        "predictions": os.path.join(run_dir, "ensemble_predictions.npz"),
        "metrics": os.path.join(run_dir, "evaluation_metrics.yaml"),
        "figures_dir": os.path.join(run_dir, "figures"),
    }


# =============================================================================
# CONSOLE OUTPUT HELPERS
# =============================================================================

def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_config_summary(cfg: PipelineConfig):
    """Print a summary of the configuration."""
    print_header("CONFIGURATION SUMMARY")
    print(f"  Run name: {cfg.run_name or '(auto)'}")
    print(f"  Output base: {cfg.output_base}")
    print(f"  Training files: {len(cfg.training_files)}")
    print(f"  Test files: {len(cfg.test_files)}")
    print(f"  POD modes (r): {cfg.r}")
    print(f"  Truncation: {'enabled' if cfg.truncation_enabled else 'disabled'}")
    if cfg.truncation_enabled:
        if cfg.truncation_method == "time":
            print(f"    Time: {cfg.truncation_time} units")
        else:
            print(f"    Snapshots: {cfg.truncation_snapshots}")
    print(f"  Selection method: {cfg.selection_method}")
    print(f"  Regularization grid: {len(cfg.state_lin)}x{len(cfg.state_quad)}x{len(cfg.output_lin)}x{len(cfg.output_quad)}")
    total = len(cfg.state_lin) * len(cfg.state_quad) * len(cfg.output_lin) * len(cfg.output_quad)
    print(f"  Total combinations: {total:,}")
    print("=" * 70 + "\n")
