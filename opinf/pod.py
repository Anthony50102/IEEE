"""
POD computation and projection utilities.

This module handles:
- Distributed Gram matrix eigendecomposition (Sirovich method)
- Data projection onto POD basis
- POD mode/energy computation

Author: Anthony Poole
"""

import gc
import numpy as np
from mpi4py import MPI


# =============================================================================
# DISTRIBUTED POD COMPUTATION
# =============================================================================

def compute_pod_distributed(Q_train_local, comm, rank, size, logger, target_energy=0.9999) -> tuple:
    """
    Compute POD basis via distributed Gram matrix eigendecomposition.
    
    Uses the method of snapshots (Sirovich):
    1. Compute local Gram matrices D_local = Q_local.T @ Q_local
    2. Allreduce to get global Gram matrix D_global
    3. Eigendecomposition of D_global
    """
    if rank == 0:
        logger.info("Computing POD basis via distributed Gram matrix...")
    
    t0 = MPI.Wtime()
    
    # Compute local Gram matrix
    D_local = Q_train_local.T @ Q_train_local
    
    if rank == 0:
        logger.debug(f"  [DIAG] D_local shape: {D_local.shape}")
        logger.debug(f"  [DIAG] D_local diagonal sum: {np.trace(D_local):.2e}")
    
    # Allreduce to get global Gram matrix (chunked if needed)
    D_global = np.zeros_like(D_local)
    total_elements = D_local.size
    max_chunk = 2**30
    
    if total_elements > max_chunk:
        n_time = D_local.shape[0]
        rows_per_chunk = max(1, max_chunk // n_time)
        
        for start_row in range(0, n_time, rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, n_time)
            send_buf = np.ascontiguousarray(D_local[start_row:end_row, :])
            recv_buf = np.zeros_like(send_buf)
            comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
            D_global[start_row:end_row, :] = recv_buf
    else:
        comm.Allreduce(D_local, D_global, op=MPI.SUM)
    
    del D_local
    gc.collect()
    
    # Eigendecomposition (rank 0 only to save memory)
    n_time = D_global.shape[0]
    
    if rank == 0:
        logger.debug(f"  [DIAG] D_global trace: {np.trace(D_global):.2e}")
        
        try:
            from scipy.linalg import eigh as scipy_eigh
            logger.info("  Using scipy.linalg.eigh...")
            eigs, eigv = scipy_eigh(D_global.copy())
        except ImportError:
            logger.info("  Using numpy.linalg.eigh...")
            eigs, eigv = np.linalg.eigh(D_global.copy())
        
        # Sort by decreasing eigenvalue
        idx = np.argsort(eigs)[::-1]
        eigs, eigv = eigs[idx], eigv[:, idx]
        
        logger.debug(f"  [DIAG] Eigenvalue sum: {np.sum(eigs):.2e}")
        logger.debug(f"  [DIAG] Top 5 eigenvalues: {eigs[:5]}")
    else:
        eigs = np.empty(n_time, dtype=np.float64)
        eigv = np.zeros((n_time, n_time), dtype=np.float64)
    
    # Broadcast eigenvalues/eigenvectors
    comm.Bcast(eigs, root=0)
    
    # Chunked broadcast for eigenvectors if large
    max_rows = max(1, 2**30 // (n_time * 8))
    for start in range(0, n_time, max_rows):
        end = min(start + max_rows, n_time)
        if rank == 0:
            chunk = np.ascontiguousarray(eigv[start:end, :])
        else:
            chunk = np.empty((end - start, n_time), dtype=np.float64)
        comm.Bcast(chunk, root=0)
        if rank != 0:
            eigv[start:end, :] = chunk
    
    comm.Barrier()
    
    # Compute retained energy
    eigs_positive = np.maximum(eigs, 0)
    ret_energy = np.cumsum(eigs_positive) / np.sum(eigs_positive)
    r_energy = np.argmax(ret_energy >= target_energy) + 1
    
    if rank == 0:
        logger.info(f"  POD computed in {MPI.Wtime() - t0:.1f}s")
        logger.info(f"  r for {target_energy*100:.2f}% energy: {r_energy}")
    
    return eigs, eigv, D_global, r_energy


def project_data_distributed(
    Q_train_local, Q_test_local, eigv, eigs, r, D_global, comm, rank, logger
) -> tuple:
    """Project data onto POD basis using distributed computation."""
    if rank == 0:
        logger.info(f"Projecting data onto {r} POD modes...")
    
    # Transformation matrix: Tr = V_r @ diag(1/sqrt(eigs_r))
    eigs_r = eigs[:r]
    eigv_r = eigv[:, :r]
    
    # Handle problematic eigenvalues
    eigs_r_safe = np.where(eigs_r > 1e-14, eigs_r, 1e-14)
    Tr = eigv_r @ np.diag(eigs_r_safe ** (-0.5))
    
    if rank == 0:
        logger.debug(f"  [DIAG] Tr shape: {Tr.shape}")
        logger.debug(f"  [DIAG] Tr has NaN: {np.any(np.isnan(Tr))}")
        
        # Verify Tr.T @ D @ Tr ≈ I
        TrTDTr_err = np.linalg.norm(Tr.T @ D_global @ Tr - np.eye(r))
        logger.debug(f"  [DIAG] ||Tr.T @ D @ Tr - I||: {TrTDTr_err:.2e}")
    
    # Reduced training coordinates: Xhat_train = Tr.T @ D_global
    Xhat_train = (Tr.T @ D_global).T
    
    # POD modes: Ur = Q_train @ Tr
    Ur_local = Q_train_local @ Tr
    
    # Verify orthonormality: Ur.T @ Ur ≈ I
    UtU_local = Ur_local.T @ Ur_local
    UtU_global = np.zeros((r, r), dtype=np.float64)
    comm.Allreduce(UtU_local, UtU_global, op=MPI.SUM)
    
    if rank == 0:
        logger.debug(f"  [DIAG] ||Ur.T @ Ur - I||: {np.linalg.norm(UtU_global - np.eye(r)):.2e}")
    
    # Project test data: Xhat_test = Q_test.T @ Ur
    Xhat_test_local = Q_test_local.T @ Ur_local
    Xhat_test = np.zeros_like(Xhat_test_local)
    comm.Allreduce(Xhat_test_local, Xhat_test, op=MPI.SUM)
    
    # Gather full Ur
    Ur_gathered = comm.gather(Ur_local, root=0)
    Ur_full = np.vstack(Ur_gathered) if rank == 0 else None
    
    if rank == 0:
        logger.info(f"  Xhat_train shape: {Xhat_train.shape}")
        logger.info(f"  Xhat_test shape: {Xhat_test.shape}")
    
    return Xhat_train, Xhat_test, Ur_local, Ur_full
