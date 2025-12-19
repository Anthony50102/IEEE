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
# POD COMPUTATION (DISTRIBUTED)
# =============================================================================

def compute_pod_distributed(
    Q_train_local: np.ndarray,
    comm,
    rank: int,
    size: int,
    logger,
) -> tuple:
    """
    Compute POD basis via distributed Gram matrix eigendecomposition.
    
    Uses the method of snapshots (Sirovich):
    1. Compute local Gram matrices D_local = Q_local.T @ Q_local
    2. Allreduce to get global Gram matrix D_global (in chunks if needed)
    3. Eigendecomposition of D_global
    4. Compute transformation matrix Tr = V @ diag(sqrt(1/eigs))
    
    Parameters
    ----------
    Q_train_local : np.ndarray
        Local training data (n_local, n_time).
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
        (eigs, eigv, Tr_global) - eigenvalues, eigenvectors, transformation matrix.
    """
    if rank == 0:
        logger.info("Computing POD basis via distributed Gram matrix...")
    
    start_time = MPI.Wtime()
    
    # Compute local Gram matrix
    t_matmul = MPI.Wtime()
    D_local = np.matmul(Q_train_local.T, Q_train_local)
    if rank == 0:
        logger.info(f"  Local Gram matrix compute: {MPI.Wtime() - t_matmul:.2f}s")
    
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
            comm.Allreduce(
                D_local[start_row:end_row, :],
                D_global[start_row:end_row, :],
                op=MPI.SUM
            )
        
        if rank == 0:
            logger.info(f"  Chunked Allreduce complete: {(n_time + rows_per_chunk - 1) // rows_per_chunk} chunks")
    else:
        # Small enough for single Allreduce
        comm.Allreduce(D_local, D_global, op=MPI.SUM)
    
    if rank == 0:
        logger.info(f"  Allreduce: {MPI.Wtime() - t_reduce:.2f}s")
    
    del D_local
    gc.collect()
    
    # Eigendecomposition of positive semi-definite Gram matrix
    t_eig = MPI.Wtime()
    eigs, eigv = np.linalg.eigh(D_global)
    
    # Sort by decreasing eigenvalue
    sorted_indices = np.argsort(eigs)[::-1]
    eigs = eigs[sorted_indices]
    eigv = eigv[:, sorted_indices]
    
    if rank == 0:
        logger.info(f"  Eigendecomposition: {MPI.Wtime() - t_eig:.2f}s")
    
    elapsed = MPI.Wtime() - start_time
    if rank == 0:
        logger.info(f"  POD completed in {elapsed:.1f} seconds")
        logger.info(f"  Eigenvalues shape: {eigs.shape}")
    
    return eigs, eigv, D_global


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


def save_pod_energy_plot(
    eigs: np.ndarray,
    r: int,
    run_dir: str,
    logger,
):
    """
    Save POD energy plot.
    
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
    
    ret_energy = compute_retained_energy(eigs)
    singular_values = np.sqrt(np.maximum(eigs, 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Singular value decay
    ax = axes[0]
    ax.semilogy(singular_values, 'b-', linewidth=1.5)
    ax.axvline(x=r, color='r', linestyle='--', label=f'r = {r}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Singular value')
    ax.set_title('Singular Value Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Retained energy
    ax = axes[1]
    ax.plot(ret_energy * 100, 'b-', linewidth=1.5)
    ax.axhline(y=ret_energy[r - 1] * 100, color='r', linestyle='--',
               label=f'{ret_energy[r - 1] * 100:.4f}% at r={r}')
    ax.axvline(x=r, color='r', linestyle='--')
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Retained energy (%)')
    ax.set_title('Cumulative Retained Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy per mode
    ax = axes[2]
    mode_energy = eigs / np.sum(eigs) * 100
    ax.semilogy(mode_energy[:min(100, len(mode_energy))], 'b-', linewidth=1.5)
    ax.axvline(x=r, color='r', linestyle='--', label=f'r = {r}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Energy contribution (%)')
    ax.set_title('Energy per Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
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
    
    # Avoid division by zero for small eigenvalues
    eigs_r_safe = np.maximum(eigs_r, 1e-14)
    Tr = eigv_r @ np.diag(eigs_r_safe ** (-0.5))
    
    # Reduced training coordinates: Xhat_train = Tr.T @ D_global
    # Shape: (r, n_time) -> transpose to (n_time, r)
    Xhat_train = (Tr.T @ D_global).T
    
    # For test data, we need the POD modes Ur
    # Ur = Q_train @ Tr (distributed: each rank has partial contribution)
    t_modes = MPI.Wtime()
    Ur_local = Q_train_local @ Tr  # (n_local, r)
    
    if rank == 0:
        logger.info(f"  Local POD modes compute: {MPI.Wtime() - t_modes:.2f}s")
    
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
    
    return Xhat_train, Xhat_test, Ur_local


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
    
    mean_Xhat = np.mean(X_out, axis=0)
    Xhat_out = X_out - mean_Xhat[np.newaxis, :]
    
    scaling_Xhat = np.maximum(np.abs(np.min(X_out)), np.abs(np.max(X_out)))
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
        save_config(cfg, run_dir)
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
        
        comm.Barrier()
        
        # 2. Compute POD (distributed)
        t_pod = MPI.Wtime()
        eigs, eigv, D_global = compute_pod_distributed(
            Q_train_local, comm, rank, size, logger
        )
        
        if rank == 0:
            logger.info(f"POD computation time: {MPI.Wtime() - t_pod:.1f}s")
            
            # Save eigenvalues (these are sigma^2, singular values are sqrt)
            singular_values = np.sqrt(np.maximum(eigs, 0))
            np.savez(paths["pod_file"], S=singular_values, eigs=eigs, eigv=eigv)
            logger.info(f"Saved POD to {paths['pod_file']}")
            
            # Optionally save energy plot
            if args.save_pod_energy:
                save_pod_energy_plot(eigs, cfg.r, run_dir, logger)
        
        # 3. Project data (distributed)
        t_proj = MPI.Wtime()
        Xhat_train, Xhat_test, Ur_local = project_data_distributed(
            Q_train_local, Q_test_local, eigv, eigs, cfg.r, D_global,
            comm, rank, logger
        )
        
        if rank == 0:
            logger.info(f"Projection time: {MPI.Wtime() - t_proj:.1f}s")
            
            np.save(paths["xhat_train"], Xhat_train)
            np.save(paths["xhat_test"], Xhat_test)
            logger.info("Saved projected data")
        
        # 4. Gather and save initial conditions
        t_ic = MPI.Wtime()
        ics = gather_initial_conditions(
            Q_train_local, Q_test_local, Xhat_train, Xhat_test,
            train_boundaries, test_boundaries,
            len(cfg.training_files), len(cfg.test_files),
            n_spatial, comm, rank, size, logger
        )
        
        if rank == 0:
            np.savez(
                paths["initial_conditions"],
                train_ICs=ics['train_ICs'],
                test_ICs=ics['test_ICs'],
                train_ICs_reduced=ics['train_ICs_reduced'],
                test_ICs_reduced=ics['test_ICs_reduced'],
            )
            logger.info("Saved initial conditions")
            logger.info(f"IC gathering time: {MPI.Wtime() - t_ic:.1f}s")
        
        # 5. Prepare learning matrices (rank 0 only)
        t_learn = MPI.Wtime()
        learning = prepare_learning_matrices(
            Xhat_train, train_boundaries, cfg, rank, logger
        )
        
        # 6. Load reference Gamma (rank 0 only)
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
            logger.info(f"Learning matrix prep time: {MPI.Wtime() - t_learn:.1f}s")
        
        # Cleanup
        del Q_train_local, Q_test_local
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
