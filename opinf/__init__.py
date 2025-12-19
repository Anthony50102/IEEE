"""
OpInf ROM Pipeline for Hasegawa-Wakatani Equations.

This package provides a modular, configurable pipeline for training and
evaluating Operator Inference (OpInf) reduced-order models.

Modules
-------
step_1_preprocess
    Data loading, POD computation, and learning matrix preparation.
step_2_train_rom
    Hyperparameter sweep for ROM training (supports MPI parallelization).
step_3_evaluate
    Ensemble prediction and evaluation.
run_pipeline
    Orchestrator for running full or partial pipeline.
utils
    Shared utilities for configuration, logging, and file management.

Usage
-----
See README.md for detailed usage instructions.
"""

from .utils import (
    load_config,
    save_config,
    PipelineConfig,
    create_run_directory,
    get_run_directory,
    setup_logging,
    get_output_paths,
)

__all__ = [
    'load_config',
    'save_config',
    'PipelineConfig',
    'create_run_directory',
    'get_run_directory',
    'setup_logging',
    'get_output_paths',
]
