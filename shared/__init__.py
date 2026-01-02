"""
Shared utilities for ROM methods.

This package contains common functionality used across all ROM methods
(OpInf, Manifold OpInf, DMD):

    data_io     - HDF5 data loading and snapshot management
    plotting    - Standardized visualization functions
    physics     - Physical quantity computations (Gamma_n, Gamma_c)
    mpi_utils   - MPI communication utilities (import separately to avoid MPI init)

Author: Anthony Poole
"""

from .data_io import (
    load_dataset,
    load_hw2d_snapshot,
    load_hw2d_timeseries,
    get_file_metadata,
)

from .physics import (
    compute_gamma_n,
    compute_gamma_c,
    periodic_gradient,
)

# NOTE: mpi_utils not imported here to avoid MPI initialization on import.
# Import directly when needed: from shared.mpi_utils import distribute_indices
