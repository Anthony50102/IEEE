"""
Shared utilities for ROM methods.

This package contains common functionality used across all ROM methods
(OpInf, Manifold OpInf, DMD):

    data_io     - HDF5 data loading and snapshot management
    plotting    - Standardized visualization functions
    physics     - Physical quantity computations (Gamma_n, Gamma_c)
    mpi_utils   - MPI communication utilities

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

from .mpi_utils import (
    distribute_indices,
    chunked_bcast,
    create_shared_array,
)
