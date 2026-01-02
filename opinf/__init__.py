"""
Discrete-Time Operator Inference ROM Pipeline.

This package implements a three-step pipeline for learning reduced-order models
using Operator Inference (OpInf):

    Step 1: Data preprocessing and POD computation (MPI parallel)
    Step 2: ROM training via regularization hyperparameter sweep (MPI parallel)
    Step 3: Evaluation and prediction (serial)

The method learns discrete-time operators A, F such that:
    x_{k+1} = A @ x_k + F @ x_k^{(2)}

where x_k are reduced coordinates and x_k^{(2)} are non-redundant quadratic terms.

Modules:
    core        - Core mathematical operations (quadratic terms, solve OpInf)
    utils       - Configuration, logging, MPI utilities
    data        - Data loading and I/O
    pod         - POD computation and projection
    training    - Hyperparameter sweep and model selection
    evaluation  - Prediction and metrics
    plotting    - Visualization utilities

References:
    - Peherstorfer & Willcox (2016). Data-driven operator inference for 
      nonintrusive projection-based model reduction.
    - Qian et al. (2020). Lift & Learn: Physics-informed machine learning
      for large-scale nonlinear dynamical systems.

Author: Anthony Poole
"""

from .core import (
    get_quadratic_terms,
    solve_difference_model,
    solve_opinf_operators,
    build_data_matrix,
)

from .utils import (
    OpInfConfig,
    load_config,
    save_config,
    get_run_directory,
    get_output_paths,
    setup_logging,
    print_header,
    print_config_summary,
)

__version__ = "1.0.0"
__author__ = "Anthony Poole"
