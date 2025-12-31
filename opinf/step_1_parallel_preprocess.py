"""
Step 1: Parallel Data Preprocessing and POD Computation.

This script handles (in parallel via MPI):
1. Distributed loading of raw simulation data from HDF5 files
2. Computing POD basis via distributed Gram matrix eigendecomposition
3. Projecting training and test data onto POD basis
4. Preparing learning matrices for ROM training
5. Saving all intermediate data

Usage:
    mpirun -np 4 python step_1_parallel_preprocess.py --config config.yaml
    mpirun -np 4 python step_1_parallel_preprocess.py --config config.yaml --save-pod-energy

Author: Anthony Poole
"""

import argparse
import gc
import time
import numpy as np
import xarray as xr
import os

from mpi4py import MPI

from utils import (
    load_config,
    save_config,
    get_run_directory,
    setup_logging,
    save_step_status,
    get_output_paths,
    print_header,
    print_config_summary,
    get_memmap_path,
    cleanup_memmap,
    get_x_sq,
    loader,
    compute_truncation_snapshots,
    PipelineConfig,
)


# =============================================================================
# MPI UTILITIES
# =============================================================================

def distribute_spatial_dof(rank: int, n_spatial: int, size: int) -> tuple:
    """
    Distribute spatial DOF across MPI ranks.
    
    Parameters
    ----------
    rank : int
        MPI rank (0, 1, ..., size-1).
    n_spatial : int
        Total number of spatial DOF.
    size : int
        Number of MPI ranks.
    
    Returns
    -------
    tuple
        (start_idx, end_idx, n_local) - start index, end index, local count.
    """
    n_local_equal = n_spatial // size
    
    start_idx = rank * n_local_equal
    end_idx = (rank + 1) * n_local_equal
    
    # Last rank handles remainder
    if rank == size - 1 and end_idx != n_spatial:
        end_idx += n_spatial - size * n_local_equal
    
    n_local = end_idx - start_idx
    
    return start_idx, end_idx, n_local


# =============================================================================
# DATA LOADING (DISTRIBUTED)
# =============================================================================

def get_file_metadata(
    file_path: str,
    engine: str,
    truncation_enabled: bool,
    truncation_snapshots: int,
    truncation_time: float,
    dt: float,
) -> tuple:
    """
    Get metadata from a single file without loading full data.
    
    Returns
    -------
    tuple
        (n_spatial, n_time, max_snaps)
    """
    with xr.open_dataset(file_path, engine=engine, phony_dims="sort") as fh:
        n_time_original = fh["density"].shape[0]
        if fh["density"].ndim == 3:
            n_y, n_x = fh["density"].shape[1], fh["density"].shape[2]
            n_spatial = 2 * n_y * n_x  # 2 fields: density and phi
        else:
            n_spatial = 2 * fh["density"].shape[1]
    
    if truncation_enabled:
        max_snaps = compute_truncation_snapshots(
            file_path, truncation_snapshots, truncation_time, dt
        )
        n_time = min(n_time_original, max_snaps) if max_snaps else n_time_original
    else:
        n_time = n_time_original
        max_snaps = None
    
    return n_spatial, n_time, max_snaps


def load_distributed_snapshots(
    file_path: str,
    start_idx: int,
    end_idx: int,
    n_local: int,
    engine: str,
    max_snapshots: int,
    rank: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load a portion of snapshots corresponding to this rank's spatial DOF.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    start_idx : int
        Start index of spatial DOF for this rank.
    end_idx : int
        End index of spatial DOF for this rank.
    n_local : int
        Number of local spatial DOF.
    engine : str
        HDF5 engine.
    max_snapshots : int
        Maximum snapshots to load.
    rank : int
        MPI rank.
    verbose : bool
        Print progress.
    
    Returns
    -------
    np.ndarray
        Local data array of shape (n_local, n_time).
    """
    t0 = time.time()
    
    with xr.open_dataset(file_path, engine=engine, phony_dims="sort") as fh:
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
    
    # Extract local portion
    Q_local = Q_full[start_idx:end_idx, :].copy()
    
    # DIAGNOSTIC: Check the loaded data
    if rank == 0 and verbose:
        print(f"    [DIAGNOSTIC] File: {os.path.basename(file_path)}")
        print(f"    [DIAGNOSTIC] Q_full shape: {Q_full.shape}")
        print(f"    [DIAGNOSTIC] Q_local shape: {Q_local.shape}")
        print(f"    [DIAGNOSTIC] Q_local range: [{Q_local.min():.2e}, {Q_local.max():.2e}]")
        print(f"    [DIAGNOSTIC] Q_local std: {Q_local.std():.2e}")
        print(f"    [DIAGNOSTIC] Q_local has variation: {Q_local.std() > 1e-10}")
    
    del Q_full
    
    if verbose and rank == 0:
        print(f"    [TIMING] Distributed load: {time.time() - t0:.1f}s")
    
    return Q_local


def load_all_data_distributed(
    cfg: PipelineConfig,
    run_dir: str,
    comm,
    rank: int,
    size: int,
    logger,
) -> tuple:
    """
    Load all training and test data in distributed fashion.
    
    Each rank loads only its portion of the spatial DOF.
    
    Returns
    -------
    tuple
        (Q_train_local, Q_test_local, train_boundaries, test_boundaries,
         n_spatial, n_local, start_idx, end_idx)
    """
    if rank == 0:
        logger.info("Loading simulation data (distributed)...")
    
    # First pass: determine shapes (rank 0 only, then broadcast)
    if rank == 0:
        train_timesteps = []
        test_timesteps = []
        train_truncations = []
        test_truncations = []
        n_spatial = None
        
        for file_path in cfg.training_files:
            ns, nt, max_snaps = get_file_metadata(
                file_path, cfg.engine, cfg.truncation_enabled,
                cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            if n_spatial is None:
                n_spatial = ns
            train_timesteps.append(nt)
            train_truncations.append(max_snaps)
        
        for file_path in cfg.test_files:
            ns, nt, max_snaps = get_file_metadata(
                file_path, cfg.engine, cfg.truncation_enabled,
                cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            test_timesteps.append(nt)
            test_truncations.append(max_snaps)
        
        metadata = {
            'n_spatial': n_spatial,
            'train_timesteps': train_timesteps,
            'test_timesteps': test_timesteps,
            'train_truncations': train_truncations,
            'test_truncations': test_truncations,
        }
    else:
        metadata = None
    
    # Broadcast metadata
    metadata = comm.bcast(metadata, root=0)
    n_spatial = metadata['n_spatial']
    train_timesteps = metadata['train_timesteps']
    test_timesteps = metadata['test_timesteps']
    train_truncations = metadata['train_truncations']
    test_truncations = metadata['test_truncations']
    
    total_train = sum(train_timesteps)
    total_test = sum(test_timesteps)
    
    # Distribute spatial DOF
    start_idx, end_idx, n_local = distribute_spatial_dof(rank, n_spatial, size)
    
    if rank == 0:
        logger.info(f"  Spatial DOF: {n_spatial:,}")
        logger.info(f"  Total train snapshots: {total_train:,}")
        logger.info(f"  Total test snapshots: {total_test:,}")
        logger.info(f"  MPI ranks: {size}")
        logger.info(f"  Local DOF per rank: ~{n_spatial // size:,}")
    
    # Allocate local arrays
    Q_train_local = np.zeros((n_local, total_train), dtype=np.float64)
    Q_test_local = np.zeros((n_local, total_test), dtype=np.float64)
    
    # Compute boundaries
    train_boundaries = [0] + list(np.cumsum(train_timesteps))
    test_boundaries = [0] + list(np.cumsum(test_timesteps))
    
    # Load training data (distributed)
    if rank == 0:
        logger.info("Loading training trajectories (distributed)...")
    
    for i, file_path in enumerate(cfg.training_files):
        t_start = time.time()
        Q_ic_local = load_distributed_snapshots(
            file_path, start_idx, end_idx, n_local,
            cfg.engine, train_truncations[i], rank, cfg.verbose
        )
        Q_train_local[:, train_boundaries[i]:train_boundaries[i + 1]] = Q_ic_local
        del Q_ic_local
        gc.collect()
        
        if rank == 0 and cfg.verbose:
            print(f"  File {i + 1}/{len(cfg.training_files)}: {time.time() - t_start:.1f}s")
    
    # Load test data (distributed)
    if rank == 0:
        logger.info("Loading test trajectories (distributed)...")
    
    for i, file_path in enumerate(cfg.test_files):
        Q_ic_local = load_distributed_snapshots(
            file_path, start_idx, end_idx, n_local,
            cfg.engine, test_truncations[i], rank, cfg.verbose
        )
        Q_test_local[:, test_boundaries[i]:test_boundaries[i + 1]] = Q_ic_local
        del Q_ic_local
        gc.collect()
    
    return (Q_train_local, Q_test_local, train_boundaries, test_boundaries,
            n_spatial, n_local, start_idx, end_idx)


# =============================================================================
# DATA PREPROCESSING (DISTRIBUTED)
# =============================================================================

def center_data_distributed(
    Q_local: np.ndarray,
    comm,
    rank: int,
    logger,
) -> tuple:
    """
    Center data by subtracting temporal mean at each spatial location.
    
    This is CRITICAL for POD - without centering, the Gram matrix is
    dominated by the mean and produces numerically unstable eigenvalues.
    
    Parameters
    ----------
    Q_local : np.ndarray
        Local data (n_local, n_time).
    comm : MPI.Comm
        MPI communicator.
    rank : int
        MPI rank.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Q_centered, temporal_mean_local) - centered data and the mean that was subtracted.
    """
    if rank == 0:
        logger.info("Centering data (subtracting temporal mean)...")
    
    # DIAGNOSTIC: Check raw data before centering
    if rank == 0:
        logger.info(f"  [DIAGNOSTIC] Raw data shape: {Q_local.shape}")
        logger.info(f"  [DIAGNOSTIC] Raw data range: [{Q_local.min():.2e}, {Q_local.max():.2e}]")
        logger.info(f"  [DIAGNOSTIC] Raw data mean: {Q_local.mean():.2e}")
        logger.info(f"  [DIAGNOSTIC] Raw data std: {Q_local.std():.2e}")
        logger.info(f"  [DIAGNOSTIC] Any NaN: {np.any(np.isnan(Q_local))}")
        logger.info(f"  [DIAGNOSTIC] Any Inf: {np.any(np.isinf(Q_local))}")
    
    # Compute temporal mean at each spatial location (on this rank)
    temporal_mean_local = np.mean(Q_local, axis=1, keepdims=True)
    
    # Center the data
    Q_centered = Q_local - temporal_mean_local
    
    if rank == 0:
        logger.info(f"  Local mean range: [{temporal_mean_local.min():.2e}, {temporal_mean_local.max():.2e}]")
        logger.info(f"  Centered data range: [{Q_centered.min():.2e}, {Q_centered.max():.2e}]")
        logger.info(f"  [DIAGNOSTIC] Centered std: {Q_centered.std():.2e}")
        
        # Check variance per spatial point
        var_per_point = np.var(Q_local, axis=1)
        logger.info(f"  [DIAGNOSTIC] Variance per point - min: {var_per_point.min():.2e}, max: {var_per_point.max():.2e}")
        logger.info(f"  [DIAGNOSTIC] Points with zero variance: {np.sum(var_per_point == 0)}")
    
    return Q_centered, temporal_mean_local.squeeze()


def scale_data_distributed(
    Q_local: np.ndarray,
    n_fields: int,
    n_local_per_field: int,
    comm,
    rank: int,
    logger,
) -> tuple:
    """
    Scale data so each field's values are in [-1, 1].
    
    Uses global max absolute value across all ranks for each field.
    
    Parameters
    ----------
    Q_local : np.ndarray
        Local data (n_local, n_time). Should already be centered.
    n_fields : int
        Number of state variables (e.g., 2 for density and phi).
    n_local_per_field : int
        Local spatial DOF per field.
    comm : MPI.Comm
        MPI communicator.
    rank : int
        MPI rank.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Q_scaled, scaling_factors) - scaled data and the factors used.
    """
    if rank == 0:
        logger.info("Scaling data (normalizing each field to [-1, 1])...")
    
    scaling_factors = np.zeros(n_fields)
    Q_scaled = Q_local.copy()
    
    for j in range(n_fields):
        start = j * n_local_per_field
        end = (j + 1) * n_local_per_field
        
        # Local max absolute value
        local_max = np.max(np.abs(Q_local[start:end, :]))
        
        # Global max via Allreduce
        global_max = np.zeros(1)
        comm.Allreduce(np.array([local_max]), global_max, op=MPI.MAX)
        global_max = global_max[0]
        
        # Scale
        if global_max > 0:
            Q_scaled[start:end, :] /= global_max
            scaling_factors[j] = global_max
        else:
            scaling_factors[j] = 1.0
    
    if rank == 0:
        logger.info(f"  Scaling factors: {scaling_factors}")
        logger.info(f"  Scaled data range: [{Q_scaled.min():.2e}, {Q_scaled.max():.2e}]")
    
    return Q_scaled, scaling_factors


# =============================================================================
# POD COMPUTATION (DISTRIBUTED)
# =============================================================================

def compute_pod_distributed(
    Q_train_local: np.ndarray,
    comm,
    rank: int,
    size: int,
    logger,
    target_energy=0.9999,
) -> tuple:
    """
    Compute POD basis via distributed Gram matrix eigendecomposition.
    
    Uses the method of snapshots (Sirovich):
    1. Compute local Gram matrices D_local = Q_local.T @ Q_local
    2. Allreduce to get global Gram matrix D_global (in chunks if needed)
    3. Eigendecomposition of D_global
    4. Compute transformation matrix Tr = V @ diag(sqrt(1/eigs))
    
    NOTE: Input data should already be centered (and optionally scaled)!
    
    Parameters
    ----------
    Q_train_local : np.ndarray
        Local training data (n_local, n_time). Should be centered!
    comm : MPI.Comm
        MPI communicator.
    rank : int
        MPI rank.
    size : int
        Number of MPI ranks.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (eigs, eigv, D_global) - eigenvalues, eigenvectors, Gram matrix.
    """
    if rank == 0:
        logger.info("Computing POD basis via distributed Gram matrix...")
    
    start_time = MPI.Wtime()
    
    # Compute local Gram matrix
    t_matmul = MPI.Wtime()
    D_local = np.matmul(Q_train_local.T, Q_train_local)
    if rank == 0:
        logger.info(f"  Local Gram matrix compute: {MPI.Wtime() - t_matmul:.2f}s")
    
    # DIAGNOSTIC: Check local Gram matrix before Allreduce
    if rank == 0:
        logger.info(f"  [DIAGNOSTIC] D_local shape: {D_local.shape}, dtype: {D_local.dtype}")
        logger.info(f"  [DIAGNOSTIC] D_local range: [{D_local.min():.2e}, {D_local.max():.2e}]")
        logger.info(f"  [DIAGNOSTIC] D_local diagonal sum: {np.trace(D_local):.2e}")
        logger.info(f"  [DIAGNOSTIC] D_local[0,0]: {D_local[0,0]:.2e}")
        logger.info(f"  [DIAGNOSTIC] D_local is C-contiguous: {D_local.flags['C_CONTIGUOUS']}")
    
    # Allreduce to get global Gram matrix (chunked to avoid overflow)
    t_reduce = MPI.Wtime()
    D_global = np.zeros_like(D_local)
    
    # MPI count limit is ~2^31 - 1 (~2.1 billion elements)
    # Chunk the reduction if matrix is too large
    total_elements = D_local.size
    max_chunk_size = 2**30  # ~1 billion elements per chunk (safe margin)
    
    if total_elements > max_chunk_size:
        if rank == 0:
            logger.info(f"  Large matrix detected ({total_elements:,} elements), using chunked Allreduce...")
        
        n_time = D_local.shape[0]
        # Reduce row by row or in row chunks
        rows_per_chunk = max(1, max_chunk_size // n_time)
        
        for start_row in range(0, n_time, rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, n_time)
            # CRITICAL: Ensure contiguous arrays for MPI
            send_buf = np.ascontiguousarray(D_local[start_row:end_row, :])
            recv_buf = np.zeros_like(send_buf)
            comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
            D_global[start_row:end_row, :] = recv_buf
        
        if rank == 0:
            logger.info(f"  Chunked Allreduce complete: {(n_time + rows_per_chunk - 1) // rows_per_chunk} chunks")
    else:
        # Small enough for single Allreduce
        comm.Allreduce(D_local, D_global, op=MPI.SUM)
    
    # DIAGNOSTIC: Check global Gram matrix after Allreduce
    if rank == 0:
        logger.info(f"  [DIAGNOSTIC] D_global range: [{D_global.min():.2e}, {D_global.max():.2e}]")
        logger.info(f"  [DIAGNOSTIC] D_global diagonal sum: {np.trace(D_global):.2e}")
        logger.info(f"  [DIAGNOSTIC] D_global[0,0]: {D_global[0,0]:.2e}")
        
        # Check for NaN/Inf
        logger.info(f"  [DIAGNOSTIC] D_global has NaN: {np.any(np.isnan(D_global))}")
        logger.info(f"  [DIAGNOSTIC] D_global has Inf: {np.any(np.isinf(D_global))}")
        
        # Check symmetry - CRITICAL for eigh
        symmetry_error = np.max(np.abs(D_global - D_global.T))
        logger.info(f"  [DIAGNOSTIC] D_global symmetry error: {symmetry_error:.2e}")
        
        # Force symmetry if needed (numerical fix)
        if symmetry_error > 1e-10:
            logger.warning(f"  WARNING: Matrix not symmetric! NOT forcing symmetry...")
            # D_global = (D_global + D_global.T) / 2.0
            # logger.info(f"  [DIAGNOSTIC] After symmetrization, error: {np.max(np.abs(D_global - D_global.T)):.2e}")

    # Broadcast the (possibly symmetrized) D_global to all ranks
    # Use chunked communication to avoid MPI size limits (~2GB per message)
    n_time = D_global.shape[0]
    max_rows_per_chunk = max(1, (2**30) // (n_time * 8))  # 8 bytes per float64
    
    if rank == 0:
        logger.info(f"  Broadcasting D_global in chunks ({n_time} rows, {max_rows_per_chunk} rows/chunk)...")
    
    for start_row in range(0, n_time, max_rows_per_chunk):
        end_row = min(start_row + max_rows_per_chunk, n_time)
        if rank == 0:
            chunk = np.ascontiguousarray(D_global[start_row:end_row, :])
        else:
            chunk = np.empty((end_row - start_row, n_time), dtype=np.float64)
        comm.Bcast(chunk, root=0)
        if rank != 0:
            D_global[start_row:end_row, :] = chunk
    
    del D_local
    gc.collect()
    
    # Eigendecomposition of positive semi-definite Gram matrix
    # Only rank 0 computes this to save memory - D_global is 12.8GB for 40k snapshots!
    t_eig = MPI.Wtime()
    
    n_time = D_global.shape[0]
    
    if rank == 0:
        logger.info(f"  [DIAGNOSTIC] D_global dtype: {D_global.dtype}")
        logger.info(f"  [DIAGNOSTIC] D_global is C-contiguous: {D_global.flags['C_CONTIGUOUS']}")
        logger.info(f"  [DIAGNOSTIC] D_global Frobenius norm: {np.linalg.norm(D_global, 'fro'):.2e}")
        
        # Expected: sum of eigenvalues = trace
        expected_eig_sum = np.trace(D_global)
        logger.info(f"  [DIAGNOSTIC] Expected eigenvalue sum (trace): {expected_eig_sum:.2e}")
        
        # Make a contiguous copy to ensure no memory aliasing issues
        D_for_eig = np.ascontiguousarray(D_global.copy())
        logger.info(f"  [DIAGNOSTIC] D_for_eig copy created, is C-contiguous: {D_for_eig.flags['C_CONTIGUOUS']}")
        
        # Use scipy for more robust eigendecomposition
        try:
            from scipy.linalg import eigh as scipy_eigh
            logger.info("  Using scipy.linalg.eigh for eigendecomposition...")
            eigs, eigv = scipy_eigh(D_for_eig, overwrite_a = True, overwrite_b = True)
        except ImportError:
            logger.info("  Using numpy.linalg.eigh for eigendecomposition...")
            eigs, eigv = np.linalg.eigh(D_for_eig)
        #eigs, eigv = np.linalg.eigh(D_for_eig)
        
        del D_for_eig
        
        # Verify eigenvalue sum matches trace
        actual_eig_sum = np.sum(eigs)
        logger.info(f"  [DIAGNOSTIC] Actual eigenvalue sum: {actual_eig_sum:.2e}")
        logger.info(f"  [DIAGNOSTIC] Trace vs eig sum difference: {abs(expected_eig_sum - actual_eig_sum):.2e}")
        
        # If still broken, try alternative approach with SVD
        if abs(actual_eig_sum - expected_eig_sum) / max(abs(expected_eig_sum), 1) > 0.01:
            logger.warning("  WARNING: Eigenvalue sum doesn't match trace! Trying SVD instead...")
            # For symmetric positive semi-definite matrix, SVD gives same result
            # but may be more numerically stable
            U, s, Vt = np.linalg.svd(D_global, full_matrices=True)
            eigs = s  # Singular values = eigenvalues for symmetric PSD
            eigv = U  # Left singular vectors = eigenvectors
            actual_eig_sum = np.sum(eigs)
            logger.info(f"  [DIAGNOSTIC] SVD singular value sum: {actual_eig_sum:.2e}")
    else:
        eigs = np.empty(n_time, dtype=np.float64)
        eigv = np.empty((n_time, n_time), dtype=np.float64)
    
    # Broadcast eigenvalues (small array - OK as single message)
    comm.Bcast(eigs, root=0)
    
    # Broadcast eigenvectors in chunks to avoid MPI 32-bit count overflow
    # eigv is (n_time, n_time) which can exceed 2^31 elements
    n_time = eigv.shape[0]
    total_eigv_elements = n_time * n_time
    max_elements_per_chunk = 2**30  # ~1 billion elements per chunk
    
    if total_eigv_elements > max_elements_per_chunk:
        max_rows_per_chunk = max(1, max_elements_per_chunk // n_time)
        
        if rank == 0:
            logger.info(f"  Large eigenvector matrix ({total_eigv_elements:,} elements), "
                       f"broadcasting in {(n_time + max_rows_per_chunk - 1) // max_rows_per_chunk} chunks...")
        
        for start_row in range(0, n_time, max_rows_per_chunk):
            end_row = min(start_row + max_rows_per_chunk, n_time)
            if rank == 0:
                chunk = np.ascontiguousarray(eigv[start_row:end_row, :])
            else:
                chunk = np.empty((end_row - start_row, n_time), dtype=np.float64)
            comm.Bcast(chunk, root=0)
            if rank != 0:
                eigv[start_row:end_row, :] = chunk
    else:
        # Small enough for single broadcast
        comm.Bcast(eigv, root=0)
    
    # Sort by decreasing eigenvalue
    sorted_indices = np.argsort(eigs)[::-1]
    eigs = eigs[sorted_indices]
    eigv = eigv[:, sorted_indices]
    
    if rank == 0:
        logger.info(f"  Eigendecomposition: {MPI.Wtime() - t_eig:.2f}s")
        
        # Diagnostic info for debugging
        n_negative = np.sum(eigs < 0)
        n_zero = np.sum(np.abs(eigs) < 1e-14)
        n_positive = np.sum(eigs > 1e-14)
        
        logger.info(f"  [DIAGNOSTIC] Eigenvalue summary:")
        logger.info(f"    Positive (>1e-14): {n_positive}")
        logger.info(f"    Near-zero: {n_zero}")
        logger.info(f"    Negative: {n_negative}")
        
        if n_negative > 0:
            logger.warning(f"  WARNING: {n_negative} negative eigenvalues detected (numerical noise)")
            logger.warning(f"  Most negative eigenvalue: {eigs[eigs < 0].min():.2e}")
        
        logger.info(f"  Eigenvalue range: [{eigs.min():.2e}, {eigs.max():.2e}]")
        logger.info(f"  Top 10 eigenvalues: {eigs[:10]}")
        logger.info(f"  Last 5 eigenvalues: {eigs[-5:]}")
        
        # Compute effective rank (eigenvalues above threshold)
        threshold = 1e-10 * eigs.max() if eigs.max() > 0 else 1e-14
        effective_rank = np.sum(eigs > threshold)
        logger.info(f"  Effective rank (eig > {threshold:.2e}): {effective_rank}")
    
    elapsed = MPI.Wtime() - start_time
    if rank == 0:
        logger.info(f"  POD completed in {elapsed:.1f} seconds")
        logger.info(f"  Eigenvalues shape: {eigs.shape}")
    
    # Only consider positive eigenvalues for energy calculation
    eigs_positive = np.maximum(eigs, 0)
    total_energy = np.sum(eigs_positive)
    
    # Compute retained energy
    ret_energy = np.cumsum(eigs_positive) / total_energy
    
    # Find r that captures target energy
    r_energy = np.argmax(ret_energy >= target_energy) + 1
    
    if rank == 0:
        logger.info(f"  Eigenvalue analysis:")
        logger.info(f"    Total eigenvalues: {len(eigs)}")
        logger.info(f"    Positive eigenvalues: {np.sum(eigs > 1e-10)}")
        logger.info(f"    r for {target_energy*100:.2f}% energy: {r_energy}")
        logger.info(f"    Top 10 eigenvalues: {eigs[:10]}")
        logger.info(f"    Eigenvalue at r={r_energy}: {eigs[r_energy-1]:.2e}")
    
    return eigs, eigv, D_global, r_energy  # Return the computed r!


def compute_retained_energy(eigs: np.ndarray) -> np.ndarray:
    """
    Compute cumulative retained energy from eigenvalues.
    
    Parameters
    ----------
    eigs : np.ndarray
        Eigenvalues (sorted descending).
    
    Returns
    -------
    np.ndarray
        Cumulative retained energy.
    """
    return np.cumsum(eigs) / np.sum(eigs)


"""
Improved POD energy plotting function with adaptive zooming and inset plots.
"""

import os
import numpy as np


def save_pod_energy_plot(
    eigs: np.ndarray,
    r: int,
    run_dir: str,
    logger,
):
    """
    Save POD energy plot with adaptive zooming for better visualization.
    
    Creates a 2-row layout:
    - Top row: Full view of all modes (context)
    - Bottom row: Zoomed view around truncation rank (detail)
    
    Parameters
    ----------
    eigs : np.ndarray
        Eigenvalues.
    r : int
        Truncation rank.
    run_dir : str
        Output directory.
    logger : logging.Logger
        Logger instance.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # Only use positive eigenvalues for energy calculations
    eigs_positive = np.maximum(eigs, 0)
    total_energy = np.sum(eigs_positive)
    
    if total_energy <= 0:
        logger.error("Total energy is zero or negative - cannot compute retained energy!")
        return
    
    ret_energy = np.cumsum(eigs_positive) / total_energy
    singular_values = np.sqrt(eigs_positive)
    mode_energy = eigs_positive / total_energy * 100
    
    # Filter to positive values for plotting
    sv_positive = singular_values[singular_values > 0]
    n_modes = len(sv_positive)
    
    # Compute adaptive zoom range based on r and data characteristics
    zoom_info = _compute_zoom_range(r, n_modes, ret_energy)
    zoom_max = zoom_info['zoom_max']
    energy_at_r = ret_energy[r - 1] * 100 if r > 0 and r <= len(ret_energy) else 0
    
    # Determine layout based on whether zooming is needed
    needs_zoom = zoom_info['needs_zoom']
    
    if needs_zoom:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes_full = axes[0]
        axes_zoom = axes[1]
    else:
        fig, axes_full = plt.subplots(1, 3, figsize=(15, 4.5))
        axes_zoom = None
    
    # Color scheme
    color_main = '#1f77b4'
    color_truncation = '#d62728'
    color_fill = '#1f77b4'
    
    # =========================================================================
    # PLOT 1: Singular Value Decay
    # =========================================================================
    ax = axes_full[0]
    ax.semilogy(range(n_modes), sv_positive, color=color_main, linewidth=1.5)
    if r < n_modes:
        ax.axvline(x=r, color=color_truncation, linestyle='--', linewidth=1.5,
                   label=f'r = {r}')
        # Add marker at truncation point
        ax.plot(r, sv_positive[r], 'o', color=color_truncation, markersize=8, zorder=5)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Singular value')
    ax.set_title('Singular Value Decay (Full)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Zoomed version
    if needs_zoom:
        ax = axes_zoom[0]
        zoom_end = min(zoom_max, n_modes)
        ax.semilogy(range(zoom_end), sv_positive[:zoom_end], color=color_main, linewidth=1.5)
        if r < zoom_end:
            ax.axvline(x=r, color=color_truncation, linestyle='--', linewidth=1.5,
                       label=f'r = {r}')
            ax.plot(r, sv_positive[r], 'o', color=color_truncation, markersize=8, zorder=5)
            # Annotate the singular value at r
            ax.annotate(f'σ_{r} = {sv_positive[r]:.2e}',
                        xy=(r, sv_positive[r]), xytext=(r + zoom_max*0.1, sv_positive[r]),
                        fontsize=9, color=color_truncation,
                        arrowprops=dict(arrowstyle='->', color=color_truncation, lw=1))
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Singular value')
        ax.set_title(f'Singular Value Decay (Zoomed: 0-{zoom_end})', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-zoom_max*0.02, zoom_end)
    
    # =========================================================================
    # PLOT 2: Cumulative Retained Energy
    # =========================================================================
    ax = axes_full[1]
    ax.plot(ret_energy * 100, color=color_main, linewidth=1.5)
    ax.fill_between(range(len(ret_energy)), ret_energy * 100, alpha=0.2, color=color_fill)
    if r > 0 and r <= len(ret_energy):
        ax.axhline(y=energy_at_r, color=color_truncation, linestyle='--', linewidth=1,
                   alpha=0.7)
        ax.axvline(x=r, color=color_truncation, linestyle='--', linewidth=1.5,
                   label=f'{energy_at_r:.2f}% at r={r}')
        ax.plot(r, energy_at_r, 'o', color=color_truncation, markersize=8, zorder=5)
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Retained energy (%)')
    ax.set_title('Cumulative Retained Energy (Full)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Zoomed version
    if needs_zoom:
        ax = axes_zoom[1]
        zoom_end = min(zoom_max, len(ret_energy))
        ax.plot(range(zoom_end), ret_energy[:zoom_end] * 100, color=color_main, linewidth=1.5)
        ax.fill_between(range(zoom_end), ret_energy[:zoom_end] * 100, alpha=0.2, color=color_fill)
        if r > 0 and r <= zoom_end:
            ax.axhline(y=energy_at_r, color=color_truncation, linestyle='--', linewidth=1,
                       alpha=0.7)
            ax.axvline(x=r, color=color_truncation, linestyle='--', linewidth=1.5,
                       label=f'{energy_at_r:.2f}% at r={r}')
            ax.plot(r, energy_at_r, 'o', color=color_truncation, markersize=8, zorder=5)
        
        # Adaptive y-limits for zoomed view
        y_min_zoom = max(0, ret_energy[0] * 100 - 5) if len(ret_energy) > 0 else 0
        y_max_zoom = min(105, energy_at_r + 10) if r > 0 else 105
        ax.set_ylim([y_min_zoom, y_max_zoom])
        ax.set_xlim(-zoom_max*0.02, zoom_end)
        
        ax.set_xlabel('Number of modes')
        ax.set_ylabel('Retained energy (%)')
        ax.set_title(f'Cumulative Retained Energy (Zoomed: 0-{zoom_end})', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 3: Energy per Mode
    # =========================================================================
    ax = axes_full[2]
    mode_energy_positive = mode_energy[mode_energy > 0]
    n_plot_full = len(mode_energy_positive)
    if n_plot_full > 0:
        ax.semilogy(range(n_plot_full), mode_energy_positive, color=color_main, linewidth=1.5)
        if r < n_plot_full:
            ax.axvline(x=r, color=color_truncation, linestyle='--', linewidth=1.5,
                       label=f'r = {r}')
            ax.plot(r, mode_energy_positive[r], 'o', color=color_truncation, markersize=8, zorder=5)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Energy contribution (%)')
    ax.set_title('Energy per Mode (Full)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Zoomed version
    if needs_zoom:
        ax = axes_zoom[2]
        zoom_end = min(zoom_max, n_plot_full)
        mode_energy_zoom = mode_energy_positive[:zoom_end]
        if len(mode_energy_zoom) > 0:
            ax.semilogy(range(len(mode_energy_zoom)), mode_energy_zoom, 
                        color=color_main, linewidth=1.5)
            if r < len(mode_energy_zoom):
                ax.axvline(x=r, color=color_truncation, linestyle='--', linewidth=1.5,
                           label=f'r = {r}')
                ax.plot(r, mode_energy_zoom[r], 'o', color=color_truncation, markersize=8, zorder=5)
                # Annotate
                ax.annotate(f'{mode_energy_zoom[r]:.2e}%',
                            xy=(r, mode_energy_zoom[r]), 
                            xytext=(r + zoom_max*0.1, mode_energy_zoom[r]),
                            fontsize=9, color=color_truncation,
                            arrowprops=dict(arrowstyle='->', color=color_truncation, lw=1))
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Energy contribution (%)')
        ax.set_title(f'Energy per Mode (Zoomed: 0-{zoom_end})', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-zoom_max*0.02, zoom_end)
    
    # =========================================================================
    # Add summary statistics text box
    # =========================================================================
    stats_text = _generate_stats_text(eigs_positive, r, ret_energy, singular_values)
    
    # Position the text box
    if needs_zoom:
        fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=10,
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        fig.text(0.5, -0.02, stats_text, ha='center', va='top', fontsize=10,
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    # Save plot
    plot_path = os.path.join(run_dir, "pod_energy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Also save the data
    data_path = os.path.join(run_dir, "pod_energy_data.npz")
    np.savez(
        data_path,
        eigenvalues=eigs,
        singular_values=singular_values,
        retained_energy=ret_energy,
        truncation_rank=r,
    )
    
    logger.info(f"Saved POD energy plot to {plot_path}")
    logger.info(f"Saved POD energy data to {data_path}")


def _compute_zoom_range(r: int, n_modes: int, ret_energy: np.ndarray) -> dict:
    """
    Compute adaptive zoom range based on truncation rank and data.
    
    Returns a zoom range that shows meaningful context around r.
    """
    # If r is already a large fraction of n_modes, no zoom needed
    if n_modes <= 200 or r >= n_modes * 0.4:
        return {'zoom_max': n_modes, 'needs_zoom': False}
    
    # Find where 99% energy is reached
    idx_99 = np.searchsorted(ret_energy, 0.99) + 1
    
    # Zoom to show at least 2x truncation rank, or up to 99% energy point
    # But cap at reasonable values
    zoom_max = max(
        int(r * 2.5),  # At least 2.5x truncation rank
        min(idx_99, int(r * 5)),  # Up to 99% energy or 5x truncation
        50  # Minimum zoom window
    )
    
    # Don't zoom if it wouldn't help much
    needs_zoom = zoom_max < n_modes * 0.7
    
    return {'zoom_max': zoom_max, 'needs_zoom': needs_zoom}


def _generate_stats_text(eigs_positive, r, ret_energy, singular_values):
    """Generate summary statistics text."""
    n_modes = len(eigs_positive[eigs_positive > 0])
    energy_at_r = ret_energy[r - 1] * 100 if r > 0 and r <= len(ret_energy) else 0
    
    # Find modes needed for various energy thresholds
    thresholds = [0.90, 0.95, 0.99, 0.999]
    modes_for_threshold = {}
    for thresh in thresholds:
        idx = np.searchsorted(ret_energy, thresh)
        modes_for_threshold[thresh] = idx + 1 if idx < len(ret_energy) else '>'+str(len(ret_energy))
    
    stats_lines = [
        f"Total modes: {n_modes}",
        f"Truncation rank: r = {r}",
        f"Energy at r={r}: {energy_at_r:.4f}%",
        f"Modes for 90%: {modes_for_threshold[0.90]}",
        f"Modes for 95%: {modes_for_threshold[0.95]}",
        f"Modes for 99%: {modes_for_threshold[0.99]}",
        f"Modes for 99.9%: {modes_for_threshold[0.999]}",
    ]
    
    return "  |  ".join(stats_lines)


# =============================================================================
# PROJECTION (DISTRIBUTED)
# =============================================================================

def project_data_distributed(
    Q_train_local: np.ndarray,
    Q_test_local: np.ndarray,
    eigv: np.ndarray,
    eigs: np.ndarray,
    r: int,
    D_global: np.ndarray,
    comm,
    rank: int,
    logger,
) -> tuple:
    """
    Project data onto POD basis using distributed computation.
    
    The reduced coordinates are computed as:
        Xhat = Tr.T @ (Q.T @ Q) = Tr.T @ D
    
    But we need to handle test data differently since we don't have
    Q_test in the Gram matrix.
    
    For training data: Xhat_train = Q_train.T @ Ur
    where Ur = Q_train @ Tr (the POD modes)
    
    Parameters
    ----------
    Q_train_local, Q_test_local : np.ndarray
        Local data matrices.
    eigv : np.ndarray
        Eigenvectors of Gram matrix.
    eigs : np.ndarray
        Eigenvalues.
    r : int
        Number of modes.
    D_global : np.ndarray
        Global Gram matrix.
    comm : MPI.Comm
        MPI communicator.
    rank : int
        MPI rank.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    tuple
        (Xhat_train, Xhat_test, Ur_local)
    """
    if rank == 0:
        logger.info(f"Projecting data onto {r} POD modes (distributed)...")
    
    # Transformation matrix: Tr = V_r @ diag(1/sqrt(eigs_r))
    # This maps temporal modes to reduced coordinates
    eigs_r = eigs[:r]
    eigv_r = eigv[:, :r]
    
    # Check for problematic eigenvalues
    if rank == 0:
        n_problematic = np.sum(eigs_r <= 0)
        if n_problematic > 0:
            logger.warning(f"  WARNING: {n_problematic} of first {r} eigenvalues are <= 0!")
            logger.warning(f"  Problematic eigenvalues: {eigs_r[eigs_r <= 0]}")
    
    # Only use positive eigenvalues - replace negative/zero with a small positive value
    # This is more robust than just clipping
    eigs_r_safe = np.where(eigs_r > 1e-14, eigs_r, 1e-14)
    
    # Compute transformation matrix
    Tr = eigv_r @ np.diag(eigs_r_safe ** (-0.5))
    
    # =================================================================
    # DIAGNOSTIC: Verify Tr is identical across all ranks
    # =================================================================
    Tr_checksum_local = np.array([np.sum(Tr), np.sum(Tr**2), Tr[0,0], Tr[-1,-1]])
    Tr_checksums = comm.gather(Tr_checksum_local, root=0)
    
    if rank == 0:
        logger.info(f"  Tr matrix: shape={Tr.shape}, has_nan={np.any(np.isnan(Tr))}, has_inf={np.any(np.isinf(Tr))}")
        
        # Check if all ranks have the same Tr
        Tr_checksums = np.array(Tr_checksums)
        Tr_variation = np.max(np.abs(Tr_checksums - Tr_checksums[0]), axis=0)
        logger.info(f"  [DIAGNOSTIC] Tr checksum variation across ranks: {Tr_variation}")
        if np.any(Tr_variation > 1e-10):
            logger.error(f"  CRITICAL: Tr differs across MPI ranks!")
        else:
            logger.info(f"  [DIAGNOSTIC] Tr is identical across all ranks ✓")
        
        # =================================================================
        # DIAGNOSTIC: Verify eigenvector orthonormality
        # This should be ~machine precision if eigh worked correctly
        # =================================================================
        VtV = eigv_r.T @ eigv_r
        eigv_ortho_error = np.linalg.norm(VtV - np.eye(r))
        logger.info(f"  [DIAGNOSTIC] Eigenvector orthonormality ||V.T @ V - I||: {eigv_ortho_error:.6e}")
        if eigv_ortho_error > 1e-10:
            logger.warning(f"  WARNING: Eigenvectors are NOT orthonormal!")
            logger.info(f"  Eigenvector diagonal range: [{np.diag(VtV).min():.6f}, {np.diag(VtV).max():.6f}]")
        
        # =================================================================
        # DIAGNOSTIC: Verify eigenvalue/eigenvector relationship
        # D @ V should equal V @ Lambda
        # =================================================================
        DV = D_global @ eigv_r
        V_Lambda = eigv_r @ np.diag(eigs_r)
        eigen_error = np.linalg.norm(DV - V_Lambda) / np.linalg.norm(V_Lambda)
        logger.info(f"  [DIAGNOSTIC] Eigen equation relative error ||DV - VΛ||/||VΛ||: {eigen_error:.6e}")
        
        # =================================================================
        # DIAGNOSTIC: Verify Tr.T @ D @ Tr should equal I
        # This is the mathematical requirement for Ur to be orthonormal
        # =================================================================
        TrTDTr = Tr.T @ D_global @ Tr
        TrTDTr_error = np.linalg.norm(TrTDTr - np.eye(r))
        logger.info(f"  [DIAGNOSTIC] ||Tr.T @ D @ Tr - I||: {TrTDTr_error:.6e}")
        logger.info(f"  [DIAGNOSTIC] Tr.T @ D @ Tr diagonal: {np.diag(TrTDTr)[:5]}")
        if TrTDTr_error > 1e-6:
            logger.error(f"  CRITICAL: Tr.T @ D @ Tr != I, POD math is broken!")
    
    # Reduced training coordinates: Xhat_train = Tr.T @ D_global
    # Shape: (r, n_time) -> transpose to (n_time, r)
    Xhat_train = (Tr.T @ D_global).T
    
    # For test data, we need the POD modes Ur
    # Ur = Q_train @ Tr (distributed: each rank has partial contribution)
    t_modes = MPI.Wtime()
    Ur_local = Q_train_local @ Tr  # (n_local, r)
    
    if rank == 0:
        logger.info(f"  Local POD modes compute: {MPI.Wtime() - t_modes:.2f}s")
    
    # =================================================================
    # DIAGNOSTIC: Check if distributed Ur computation is correct
    # Ur.T @ Ur = sum_i (Ur_local_i.T @ Ur_local_i) should equal I
    # =================================================================
    UtU_local = Ur_local.T @ Ur_local  # (r, r) local contribution
    UtU_global = np.zeros((r, r), dtype=np.float64)
    comm.Allreduce(UtU_local, UtU_global, op=MPI.SUM)
    
    if rank == 0:
        ortho_error_distributed = np.linalg.norm(UtU_global - np.eye(r))
        diag_dist = np.diag(UtU_global)
        logger.info(f"  [DIAGNOSTIC] Distributed Ur.T @ Ur orthonormality error: {ortho_error_distributed:.6e}")
        logger.info(f"  [DIAGNOSTIC] Distributed diagonal range: [{diag_dist.min():.6f}, {diag_dist.max():.6f}]")
        
        if ortho_error_distributed > 1e-6:
            logger.warning(f"  WARNING: Distributed POD basis is NOT orthonormal!")
            # Show sample diagonal values
            logger.info(f"  [DIAGNOSTIC] Sample diagonal values: {diag_dist[:5]}")
            logger.info(f"  [DIAGNOSTIC] Expected: [1, 1, 1, 1, 1]")

    # Project test data: Xhat_test = Q_test.T @ Ur
    # Need to reduce across ranks
    t_proj = MPI.Wtime()
    Xhat_test_local = Q_test_local.T @ Ur_local  # (n_test_time, r)
    
    Xhat_test = np.zeros_like(Xhat_test_local)
    comm.Allreduce(Xhat_test_local, Xhat_test, op=MPI.SUM)
    
    if rank == 0:
        logger.info(f"  Test projection: {MPI.Wtime() - t_proj:.2f}s")
        logger.info(f"  Xhat_train shape: {Xhat_train.shape}")
        logger.info(f"  Xhat_test shape: {Xhat_test.shape}")
    
    # Gather full POD basis Ur from all ranks for later use
    # This allows projecting arbitrary new trajectories
    t_gather_ur = MPI.Wtime()
    Ur_gathered = comm.gather(Ur_local, root=0)
    
    if rank == 0:
        Ur_full = np.vstack(Ur_gathered)  # (n_spatial, r)
        logger.info(f"  Full POD basis Ur shape: {Ur_full.shape}")
        logger.info(f"  Ur gather time: {MPI.Wtime() - t_gather_ur:.2f}s")
    else:
        Ur_full = None
    
    return Xhat_train, Xhat_test, Ur_local, Ur_full


# =============================================================================
# LEARNING MATRIX PREPARATION
# =============================================================================

def prepare_learning_matrices(
    Xhat_train: np.ndarray,
    train_boundaries: list,
    cfg: PipelineConfig,
    rank: int,
    logger,
) -> dict:
    """
    Prepare matrices for ROM training (same as serial version).
    
    Only rank 0 needs to do this since Xhat_train is global.
    """
    if rank != 0:
        return None
    
    logger.info("Preparing learning matrices...")
    r = cfg.r
    
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
    
    s = int(r * (r + 1) / 2)
    X_state2 = get_x_sq(X_state)
    D_state = np.concatenate((X_state, X_state2), axis=1)
    D_state_2 = D_state.T @ D_state
    
    logger.info(f"  State pairs: {X_state.shape[0]}")
    
    # OUTPUT LEARNING: Use all timesteps
    X_out = Xhat_train
    K = X_out.shape[0]
    E = np.ones((K, 1))
    
    # Check for NaN/Inf in Xhat_train
    if np.any(np.isnan(X_out)) or np.any(np.isinf(X_out)):
        logger.error(f"  ERROR: Xhat_train contains NaN or Inf!")
        logger.error(f"    NaN count: {np.sum(np.isnan(X_out))}")
        logger.error(f"    Inf count: {np.sum(np.isinf(X_out))}")
    
    mean_Xhat = np.mean(X_out, axis=0)
    Xhat_out = X_out - mean_Xhat[np.newaxis, :]
    
    # Robust scaling - handle zero/near-zero case
    scaling_Xhat = np.maximum(np.abs(np.min(X_out)), np.abs(np.max(X_out)))
    if scaling_Xhat < 1e-14 or np.isnan(scaling_Xhat):
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


def load_reference_gamma(
    cfg: PipelineConfig,
    rank: int,
    logger,
) -> dict:
    """
    Load reference Gamma values from training files (rank 0 only).
    """
    if rank != 0:
        return None
    
    logger.info("Loading reference Gamma values...")
    
    Gamma_n_list = []
    Gamma_c_list = []
    
    for file_path in cfg.training_files:
        fh = loader(file_path, ENGINE=cfg.engine)
        
        gamma_n = fh["gamma_n"].data
        gamma_c = fh["gamma_c"].data
        
        if cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            if max_snaps is not None:
                n_time_original = len(gamma_n)
                n_time = min(n_time_original, max_snaps)
                gamma_n = gamma_n[:n_time]
                gamma_c = gamma_c[:n_time]
                logger.info(f"  Truncated gamma for {os.path.basename(file_path)}: "
                           f"{n_time_original} -> {n_time}")
        
        Gamma_n_list.append(gamma_n)
        Gamma_c_list.append(gamma_c)
    
    Gamma_n = np.concatenate(Gamma_n_list)
    Gamma_c = np.concatenate(Gamma_c_list)
    
    Y_Gamma = np.vstack((Gamma_n, Gamma_c))
    
    logger.info(f"  Y_Gamma shape: {Y_Gamma.shape}")
    logger.info(f"  Gamma_n: mean={np.mean(Gamma_n):.4e}, std={np.std(Gamma_n, ddof=1):.4e}")
    logger.info(f"  Gamma_c: mean={np.mean(Gamma_c):.4e}, std={np.std(Gamma_c, ddof=1):.4e}")
    
    return {
        'Y_Gamma': Y_Gamma,
        'mean_Gamma_n': np.mean(Gamma_n),
        'std_Gamma_n': np.std(Gamma_n, ddof=1),
        'mean_Gamma_c': np.mean(Gamma_c),
        'std_Gamma_c': np.std(Gamma_c, ddof=1),
    }


# =============================================================================
# INITIAL CONDITIONS (DISTRIBUTED)
# =============================================================================

def gather_initial_conditions(
    Q_train_local: np.ndarray,
    Q_test_local: np.ndarray,
    Xhat_train: np.ndarray,
    Xhat_test: np.ndarray,
    train_boundaries: list,
    test_boundaries: list,
    n_train_files: int,
    n_test_files: int,
    n_spatial: int,
    comm,
    rank: int,
    size: int,
    logger,
) -> dict:
    """
    Gather initial conditions from all ranks.
    
    Returns
    -------
    dict
        Initial conditions (only valid on rank 0).
    """
    if rank == 0:
        logger.info("Gathering initial conditions...")
    
    # Extract local ICs
    train_ICs_local = [Q_train_local[:, train_boundaries[i]] for i in range(n_train_files)]
    test_ICs_local = [Q_test_local[:, test_boundaries[i]] for i in range(n_test_files)]
    
    # Gather from all ranks
    train_ICs_gathered = []
    test_ICs_gathered = []
    
    for i in range(n_train_files):
        ic_local = train_ICs_local[i]
        ic_gathered = comm.gather(ic_local, root=0)
        if rank == 0:
            train_ICs_gathered.append(np.concatenate(ic_gathered))
    
    for i in range(n_test_files):
        ic_local = test_ICs_local[i]
        ic_gathered = comm.gather(ic_local, root=0)
        if rank == 0:
            test_ICs_gathered.append(np.concatenate(ic_gathered))
    
    if rank == 0:
        train_ICs = np.array(train_ICs_gathered)
        test_ICs = np.array(test_ICs_gathered)
        train_ICs_reduced = np.array([Xhat_train[train_boundaries[i], :]
                                      for i in range(n_train_files)])
        test_ICs_reduced = np.array([Xhat_test[test_boundaries[i], :]
                                     for i in range(n_test_files)])
        
        return {
            'train_ICs': train_ICs,
            'test_ICs': test_ICs,
            'train_ICs_reduced': train_ICs_reduced,
            'test_ICs_reduced': test_ICs_reduced,
        }
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for parallel Step 1."""
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser(
        description="Step 1 (Parallel): Data Preprocessing and POD Computation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run directory (creates new if not specified)"
    )
    parser.add_argument(
        "--save-pod-energy", action="store_true",
        help="Save POD energy plot and data"
    )
    # Note: centering and scaling are now controlled via config.yaml preprocessing section
    args = parser.parse_args()
    
    # Load configuration (all ranks)
    cfg = load_config(args.config)
    
    # Get/create run directory (rank 0 only, then broadcast)
    if rank == 0:
        run_dir = get_run_directory(cfg, args.run_dir)
    else:
        run_dir = None
    run_dir = comm.bcast(run_dir, root=0)
    
    # Set up logging (rank 0 only)
    if rank == 0:
        logger = setup_logging("step_1_parallel", run_dir, cfg.log_level)
        
        print_header("STEP 1 (PARALLEL): DATA PREPROCESSING AND POD COMPUTATION")
        print(f"  Run directory: {run_dir}")
        print(f"  Data directory: {cfg.data_dir}")
        print(f"  MPI ranks: {size}")
        print_config_summary(cfg)
        
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"MPI ranks: {size}")
        save_step_status(run_dir, "step_1", "running")
        save_config(cfg, run_dir, step_name="step_1")
        logger.info("Configuration saved to run directory")
    else:
        logger = None
    
    # Create a dummy logger for non-root ranks
    class DummyLogger:
        def info(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    
    if logger is None:
        logger = DummyLogger()
    
    paths = get_output_paths(run_dir) if rank == 0 else None
    paths = comm.bcast(paths, root=0)
    
    start_time_global = MPI.Wtime()
    
    try:
        # 1. Load data (distributed)
        t_load = MPI.Wtime()
        (Q_train_local, Q_test_local, train_boundaries, test_boundaries,
         n_spatial, n_local, start_idx, end_idx) = load_all_data_distributed(
            cfg, run_dir, comm, rank, size, logger
        )
        
        if rank == 0:
            logger.info(f"Data loading time: {MPI.Wtime() - t_load:.1f}s")
            
            # Save boundaries
            np.savez(
                paths["boundaries"],
                train_boundaries=train_boundaries,
                test_boundaries=test_boundaries,
                n_spatial=n_spatial,
            )
            logger.info(f"Saved boundaries to {paths['boundaries']}")
        
        # DIAGNOSTIC: Check data variation across all ranks
        local_var = np.var(Q_train_local)
        local_mean = np.mean(Q_train_local)
        local_std = np.std(Q_train_local)
        
        # Gather statistics
        all_vars = comm.gather(local_var, root=0)
        all_means = comm.gather(local_mean, root=0)
        all_stds = comm.gather(local_std, root=0)
        
        if rank == 0:
            logger.info(f"  [DIAGNOSTIC] Q_train global stats:")
            logger.info(f"    Mean across ranks: {np.mean(all_means):.2e}")
            logger.info(f"    Std across ranks: {np.mean(all_stds):.2e}")
            logger.info(f"    Var across ranks: {np.mean(all_vars):.2e}")
            logger.info(f"    Train boundaries: {train_boundaries}")
        
        comm.Barrier()
        
        # 2. Center data (CRITICAL for POD numerical stability)
        if not cfg.centering_enabled:
            if rank == 0:
                logger.info("SKIPPING centering (disabled in config)")
            Q_train_centered = Q_train_local
            Q_test_centered = Q_test_local
            train_temporal_mean = np.zeros(n_local)
            test_temporal_mean = np.zeros(n_local)
        else:
            t_center = MPI.Wtime()
            Q_train_centered, train_temporal_mean = center_data_distributed(
                Q_train_local, comm, rank, logger
            )
            Q_test_centered, test_temporal_mean = center_data_distributed(
                Q_test_local, comm, rank, logger
            )
            
            if rank == 0:
                logger.info(f"Centering time: {MPI.Wtime() - t_center:.1f}s")
        
        # 2b. Scale data (optional, normalizes each field to [-1, 1])
        n_local_per_field = n_local // cfg.n_fields
        scaling_factors = None
        if cfg.scaling_enabled:
            t_scale = MPI.Wtime()
            Q_train_centered, scaling_factors = scale_data_distributed(
                Q_train_centered, cfg.n_fields, n_local_per_field,
                comm, rank, logger
            )
            Q_test_centered, _ = scale_data_distributed(
                Q_test_centered, cfg.n_fields, n_local_per_field,
                comm, rank, logger
            )
            if rank == 0:
                logger.info(f"Scaling time: {MPI.Wtime() - t_scale:.1f}s")
        else:
            if rank == 0:
                logger.info("SKIPPING scaling (disabled in config)")
        
        # 3. Compute POD on CENTERED data (distributed)
        t_pod = MPI.Wtime()
        eigs, eigv, D_global, r_from_energy = compute_pod_distributed(
            Q_train_centered, comm, rank, size, logger
        )
        # Use the smaller of config r and energy-based r
        r_actual = min(cfg.r, r_from_energy)

        if rank == 0:
            logger.info(f"  Config r: {cfg.r}, Energy-based r: {r_from_energy}, Using: {r_actual}")
        
        if rank == 0:
            logger.info(f"POD computation time: {MPI.Wtime() - t_pod:.1f}s")
            
            # Save eigenvalues (these are sigma^2, singular values are sqrt)
            singular_values = np.sqrt(np.maximum(eigs, 0))
            np.savez(paths["pod_file"], S=singular_values, eigs=eigs, eigv=eigv)
            logger.info(f"Saved POD to {paths['pod_file']}")
            
            # Optionally save energy plot
            if args.save_pod_energy:
                save_pod_energy_plot(eigs, r_actual, run_dir, logger)
        
        # 4. Project CENTERED data (distributed)
        t_proj = MPI.Wtime()
        Xhat_train, Xhat_test, Ur_local, Ur_full = project_data_distributed(
            Q_train_centered, Q_test_centered, eigv, eigs, r_actual, D_global,
            comm, rank, logger
        )
        
        if rank == 0:
            logger.info(f"Projection time: {MPI.Wtime() - t_proj:.1f}s")
            
            np.save(paths["xhat_train"], Xhat_train)
            np.save(paths["xhat_test"], Xhat_test)
            logger.info("Saved projected data")
            
            # Save POD basis for projecting arbitrary new trajectories
            np.save(os.path.join(run_dir, "POD_basis_Ur.npy"), Ur_full)
            logger.info(f"Saved POD basis Ur to POD_basis_Ur.npy (shape: {Ur_full.shape})")
        
        # 5. Gather and save initial conditions (use ORIGINAL data, not centered)
        t_ic = MPI.Wtime()
        ics = gather_initial_conditions(
            Q_train_local, Q_test_local, Xhat_train, Xhat_test,
            train_boundaries, test_boundaries,
            len(cfg.training_files), len(cfg.test_files),
            n_spatial, comm, rank, size, logger
        )
        
        # Also save the temporal means for reconstruction
        train_means_gathered = comm.gather(train_temporal_mean, root=0)
        test_means_gathered = comm.gather(test_temporal_mean, root=0)
        
        if rank == 0:
            # Reconstruct full temporal means
            train_temporal_mean_full = np.concatenate(train_means_gathered)
            test_temporal_mean_full = np.concatenate(test_means_gathered)
            
            np.savez(
                paths["initial_conditions"],
                train_ICs=ics['train_ICs'],
                test_ICs=ics['test_ICs'],
                train_ICs_reduced=ics['train_ICs_reduced'],
                test_ICs_reduced=ics['test_ICs_reduced'],
                train_temporal_mean=train_temporal_mean_full,
                test_temporal_mean=test_temporal_mean_full,
            )
            logger.info("Saved initial conditions and temporal means")
            logger.info(f"IC gathering time: {MPI.Wtime() - t_ic:.1f}s")
        
        # 6. Prepare learning matrices (rank 0 only)
        t_learn = MPI.Wtime()
        learning = prepare_learning_matrices(
            Xhat_train, train_boundaries, cfg, rank, logger
        )
        
        # 7. Load reference Gamma (rank 0 only)
        gamma_ref = load_reference_gamma(cfg, rank, logger)
        
        if rank == 0:
            # Save learning matrices
            np.savez(
                paths["learning_matrices"],
                X_state=learning['X_state'],
                Y_state=learning['Y_state'],
                D_state=learning['D_state'],
                D_state_2=learning['D_state_2'],
                D_out=learning['D_out'],
                D_out_2=learning['D_out_2'],
                mean_Xhat=learning['mean_Xhat'],
                scaling_Xhat=learning['scaling_Xhat'],
            )
            logger.info(f"Saved learning matrices to {paths['learning_matrices']}")
            
            # Save gamma reference
            np.savez(
                paths["gamma_ref"],
                Y_Gamma=gamma_ref['Y_Gamma'],
                mean_Gamma_n=gamma_ref['mean_Gamma_n'],
                std_Gamma_n=gamma_ref['std_Gamma_n'],
                mean_Gamma_c=gamma_ref['mean_Gamma_c'],
                std_Gamma_c=gamma_ref['std_Gamma_c'],
            )
            logger.info(f"Saved gamma reference to {paths['gamma_ref']}")
            
            # Save preprocessing info for consistency in later steps
            # This ensures centering/scaling choices are carried through
            preproc_data = {
                'centering_applied': cfg.centering_enabled,
                'scaling_applied': cfg.scaling_enabled,
                'r_actual': r_actual,
                'r_config': cfg.r,
                'r_from_energy': r_from_energy,
                'n_spatial': n_spatial,
                'n_fields': cfg.n_fields,
                'n_x': cfg.n_x,
                'n_y': cfg.n_y,
                'dt': cfg.dt,
            }
            # Add scaling factors if scaling was applied
            if scaling_factors is not None:
                preproc_data['scaling_factors'] = scaling_factors
            
            np.savez(paths["preprocessing_info"], **preproc_data)
            logger.info(f"Saved preprocessing info to {paths['preprocessing_info']}")
            logger.info(f"  Centering applied: {cfg.centering_enabled}")
            logger.info(f"  Scaling applied: {cfg.scaling_enabled}")
            
            logger.info(f"Learning matrix prep time: {MPI.Wtime() - t_learn:.1f}s")
        
        # Cleanup
        del Q_train_local, Q_test_local, Q_train_centered, Q_test_centered
        gc.collect()
        
        # Final timing and status
        total_time = MPI.Wtime() - start_time_global
        
        if rank == 0:
            save_step_status(run_dir, "step_1", "completed", {
                "n_spatial": int(n_spatial),
                "n_train_snapshots": int(train_boundaries[-1]),
                "n_test_snapshots": int(test_boundaries[-1]),
                "mpi_ranks": size,
                "total_time_seconds": total_time,
            })
            
            print_header("STEP 1 (PARALLEL) COMPLETE")
            print(f"  Output directory: {run_dir}")
            print(f"  Total runtime: {total_time:.1f} seconds")
            print(f"  MPI ranks used: {size}")
            logger.info(f"Step 1 completed successfully in {total_time:.1f}s")
        
    except Exception as e:
        if rank == 0:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            save_step_status(run_dir, "step_1", "failed", {"error": str(e)})
        raise
    
    finally:
        # MPI finalize is handled automatically by mpi4py
        pass


if __name__ == "__main__":
    main()
