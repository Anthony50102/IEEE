"""
Parametric-OpInf pipeline configuration loader.

Loads the multi-alpha YAML used by B2 (affine-mu pOpInf). The structure
intentionally differs from the per-alpha B1 config: a single B2 run
sweeps multiple training alphas plus a held-out sentinel.

Example YAML structure:

    run_name: "b2_alpha_p015_r75"
    paths:
      output_base: "/scratch/.../output/"
    physics: {dt: 0.25, n_fields: 2, n_x: 512, n_y: 512}
    reduction: {method: "linear", r: 75, target_energy: 0.99}
    preprocessing: {centering: true, scaling: false}
    parametric:
      alphas:
        0.1: {data_dir: "...", training_file: "trajectory.h5",
              train_start: 6000, train_end: 11000,
              test_start: 11000, test_end: 12001}
        1.0: {...}
        5.0: {...}
      sentinel_alpha: 1.5
      sentinel_data_dir: "..."           # optional; for stability IC + G3 eval
      sentinel_training_file: "trajectory.h5"
      sentinel_eval_start: 6000
      sentinel_eval_end: 7000
    regularization_parametric:
      lambda1_sq: {min: 1.0e-4, max: 1.0e8, num: 6, scale: log}
      lambda2_sq: {min: 1.0e-4, max: 1.0e10, num: 6, scale: log}
    stability:
      sentinel_max_norm: 1.0e5
      sentinel_steps: 200

Author: Anthony Poole
"""

from __future__ import annotations

import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AlphaSpec:
    alpha: float
    data_dir: str
    training_file: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def trajectory_path(self) -> str:
        return os.path.join(self.data_dir, self.training_file)


@dataclass
class ParametricConfig:
    run_name: str = ""
    run_dir: str = ""

    # Paths
    output_base: str = ""

    # Physics (shared across alphas)
    pde: str = "hw2d"
    dt: float = 0.25
    n_fields: int = 2
    n_x: int = 512
    n_y: int = 512
    engine: str = "h5netcdf"

    # Reduction
    reduction_method: str = "linear"
    r: int = 75
    target_energy: float = 0.99

    # Preprocessing
    centering_enabled: bool = True
    scaling_enabled: bool = False

    # Parametric alphas
    training_alphas: List[AlphaSpec] = field(default_factory=list)
    sentinel_alpha: Optional[float] = None
    sentinel_data_dir: str = ""
    sentinel_training_file: str = "trajectory.h5"
    sentinel_eval_start: int = 0
    sentinel_eval_end: int = 0

    # 2D regularization grid (already squared, per opinf convention)
    lambda1_sq: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda2_sq: np.ndarray = field(default_factory=lambda: np.array([]))

    # Stability
    sentinel_max_norm: float = 1.0e5
    sentinel_steps: int = 200

    # Execution
    verbose: bool = True
    log_level: str = "INFO"

    @property
    def training_alpha_values(self) -> List[float]:
        return [a.alpha for a in self.training_alphas]


def _build_reg_array(reg_config: dict) -> np.ndarray:
    scale = reg_config.get("scale", "log")
    lo = float(reg_config["min"])
    hi = float(reg_config["max"])
    n = int(reg_config["num"])
    if scale == "log":
        return np.logspace(np.log10(lo), np.log10(hi), n)
    return np.linspace(lo, hi, n)


def load_parametric_config(config_path: str) -> ParametricConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = ParametricConfig()
    cfg.run_name = raw.get("run_name", "")

    paths = raw.get("paths", {})
    cfg.output_base = paths.get("output_base", "")

    physics = raw.get("physics", {})
    cfg.pde = raw.get("pde", "hw2d")
    cfg.dt = physics.get("dt", 0.25)
    cfg.n_fields = physics.get("n_fields", 2)
    cfg.n_x = physics.get("n_x", 512)
    cfg.n_y = physics.get("n_y", 512)

    reduction = raw.get("reduction", {})
    cfg.reduction_method = reduction.get("method", "linear")
    cfg.r = int(reduction.get("r", 75))
    cfg.target_energy = float(reduction.get("target_energy", 0.99))

    preproc = raw.get("preprocessing", {})
    cfg.centering_enabled = preproc.get("centering", True)
    cfg.scaling_enabled = preproc.get("scaling", False)

    parametric = raw.get("parametric", {})
    alphas_raw = parametric.get("alphas", {})
    training_alphas = []
    for k, v in alphas_raw.items():
        training_alphas.append(AlphaSpec(
            alpha=float(k),
            data_dir=v["data_dir"],
            training_file=v.get("training_file", "trajectory.h5"),
            train_start=int(v["train_start"]),
            train_end=int(v["train_end"]),
            test_start=int(v["test_start"]),
            test_end=int(v["test_end"]),
        ))
    training_alphas.sort(key=lambda a: a.alpha)
    cfg.training_alphas = training_alphas

    if "sentinel_alpha" in parametric:
        cfg.sentinel_alpha = float(parametric["sentinel_alpha"])
        cfg.sentinel_data_dir = parametric.get("sentinel_data_dir", "")
        cfg.sentinel_training_file = parametric.get(
            "sentinel_training_file", "trajectory.h5"
        )
        cfg.sentinel_eval_start = int(parametric.get("sentinel_eval_start", 0))
        cfg.sentinel_eval_end = int(parametric.get("sentinel_eval_end", 0))

    reg = raw.get("regularization_parametric", {})
    if "lambda1_sq" in reg:
        cfg.lambda1_sq = _build_reg_array(reg["lambda1_sq"])
    if "lambda2_sq" in reg:
        cfg.lambda2_sq = _build_reg_array(reg["lambda2_sq"])

    stab = raw.get("stability", {})
    cfg.sentinel_max_norm = float(stab.get("sentinel_max_norm", 1.0e5))
    cfg.sentinel_steps = int(stab.get("sentinel_steps", 200))

    execution = raw.get("execution", {})
    cfg.verbose = execution.get("verbose", True)
    cfg.log_level = execution.get("log_level", "INFO")
    cfg.engine = execution.get("engine", "h5netcdf")

    return cfg


def sentinel_trajectory_path(cfg: ParametricConfig) -> Optional[str]:
    if cfg.sentinel_alpha is None or not cfg.sentinel_data_dir:
        return None
    return os.path.join(cfg.sentinel_data_dir, cfg.sentinel_training_file)
