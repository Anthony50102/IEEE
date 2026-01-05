"""
Dimensionality reduction: POD and Quadratic Manifold.

This module handles:
- Distributed Gram matrix eigendecomposition (Sirovich method) for linear POD
- Quadratic manifold via greedy mode selection
- Data projection/lifting for both methods
- Unified interface via BasisData container

Author: Anthony Poole
"""

import gc
import time
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from typing import Optional, Callable


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class BasisData:
    """Container for dimensionality reduction basis (works for both methods)."""
    method: str                    # "linear" or "manifold"
    V: np.ndarray                  # Linear basis (n_spatial, r)
    W: Optional[np.ndarray]        # Quadratic coefficients (manifold only)
    shift: np.ndarray              # Mean shift vector
    r: int                         # Number of modes
    eigs: np.ndarray               # Eigenvalues/singular values squared
    selected_indices: Optional[np.ndarray] = None  # For manifold: which modes selected


def save_basis(basis: BasisData, filepath: str):
    """Save basis to npz file."""
    np.savez(
        filepath,
        method=basis.method,
        V=basis.V,
        W=basis.W if basis.W is not None else np.array([]),
        shift=basis.shift,
        r=basis.r,
        eigs=basis.eigs,
        selected_indices=basis.selected_indices if basis.selected_indices is not None else np.array([]),
    )


def load_basis(filepath: str) -> BasisData:
    """Load basis from npz file."""
    d = np.load(filepath, allow_pickle=True)
    return BasisData(
        method=str(d['method']),
        V=d['V'],
        W=d['W'] if d['W'].size > 0 else None,
        shift=d['shift'],
        r=int(d['r']),
        eigs=d['eigs'],
        selected_indices=d['selected_indices'] if d['selected_indices'].size > 0 else None,
    )


# =============================================================================
# PROJECTION AND LIFTING (unified interface)
# =============================================================================

def _quadratic_features(z: np.ndarray) -> np.ndarray:
    """Compute quadratic features: z_i * z_j for j <= i."""
    r = z.shape[0]
    return np.concatenate([z[i:i+1] * z[:i+1] for i in range(r)], axis=0)


def encode(data: np.ndarray, basis: BasisData) -> np.ndarray:
    """Project full state to reduced coordinates: z = V^T (x - shift)."""
    return basis.V.T @ (data - basis.shift[:, None])


def decode(z: np.ndarray, basis: BasisData) -> np.ndarray:
    """Lift reduced coordinates to full state."""
    if basis.method == "linear":
        return basis.V @ z + basis.shift[:, None]
    else:  # manifold
        return basis.V @ z + basis.W @ _quadratic_features(z) + basis.shift[:, None]


def reconstruction_error(data: np.ndarray, basis: BasisData) -> tuple:
    """Compute reconstruction error: ||x - decode(encode(x))||."""
    z = encode(data, basis)
    recon = decode(z, basis)
    abs_err = np.linalg.norm(recon - data, 'fro')
    rel_err = abs_err / np.linalg.norm(data, 'fro')
    return abs_err, rel_err


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
    
    # Verify Q_train_local shape consistency across ranks
    local_shape = Q_train_local.shape
    all_shapes = comm.gather(local_shape, root=0)
    if rank == 0:
        n_times = set(s[1] for s in all_shapes)
        if len(n_times) > 1:
            logger.error(f"  [ERROR] Inconsistent n_time across ranks: {all_shapes}")
            raise ValueError(f"Q_train_local has inconsistent n_time across ranks: {all_shapes}")
    
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
    # IMPORTANT: Broadcast n_time to ensure all ranks agree on matrix dimensions
    n_time = D_global.shape[0]
    n_time = comm.bcast(n_time, root=0)
    
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


# =============================================================================
# QUADRATIC MANIFOLD (greedy algorithm)
# =============================================================================

def _lstsq_reg(A: np.ndarray, B: np.ndarray, reg: float = 1e-6) -> tuple:
    """Tikhonov-regularized least squares via SVD."""
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s_inv = s / (s**2 + reg**2)
    x = (VT.T * s_inv) @ (U.T @ B)
    residual = np.linalg.norm(B - A @ x, 'fro')
    return x, residual


def _greedy_error(idx_in, idx_out, sigma, VT, reg):
    """Reconstruction error for a mode partition."""
    z = sigma[idx_in, None] * VT[idx_in]
    target = VT[idx_out].T * sigma[idx_out]
    H = _quadratic_features(z)
    _, residual = _lstsq_reg(H.T, target, reg)
    return residual


def compute_manifold_greedy(Q_full: np.ndarray, r: int, n_check: int, reg: float, 
                            logger) -> BasisData:
    """
    Compute quadratic manifold using greedy mode selection.
    
    The manifold approximates: x ≈ V·z + W·h(z) + μ
    where h(z) are quadratic features z_i·z_j.
    
    Parameters
    ----------
    Q_full : ndarray (n_spatial, n_time)
        Full snapshot matrix (must be gathered on calling rank).
    r : int
        Number of modes.
    n_check : int
        Max candidates to evaluate per greedy step.
    reg : float
        Tikhonov regularization.
    logger : Logger
    
    Returns
    -------
    BasisData with method="manifold"
    """
    logger.info(f"Computing quadratic manifold: r={r}, n_check={n_check}, reg={reg:.1e}")
    
    # Center and SVD
    shift = np.mean(Q_full, axis=1)
    Q_centered = Q_full - shift[:, None]
    
    logger.info("  Computing SVD...")
    t0 = time.time()
    U, sigma, VT = np.linalg.svd(Q_centered, full_matrices=False)
    logger.info(f"  SVD done in {time.time()-t0:.1f}s, rank={len(sigma)}")
    
    # Cumulative energy
    eigs = sigma**2
    energy = np.cumsum(eigs) / np.sum(eigs)
    logger.info(f"  Energy: r=10 → {energy[9]*100:.2f}%, r=50 → {energy[49]*100:.2f}%, r=100 → {energy[min(99, len(energy)-1)]*100:.2f}%")
    
    # Greedy selection
    idx_in = np.array([0], dtype=np.int64)
    idx_out = np.arange(1, len(sigma), dtype=np.int64)
    
    while len(idx_in) < r:
        t0 = time.time()
        n_consider = min(n_check, len(idx_out))
        errors = np.zeros(n_consider)
        
        for i in range(n_consider):
            trial_in = np.append(idx_in, idx_out[i])
            trial_out = np.delete(idx_out, i)
            errors[i] = _greedy_error(trial_in, trial_out, sigma, VT, reg)
        
        best = np.argmin(errors)
        idx_in = np.append(idx_in, idx_out[best])
        idx_out = np.delete(idx_out, best)
        
        if len(idx_in) % 10 == 0 or len(idx_in) == r:
            logger.info(f"  Step {len(idx_in)}: mode {idx_in[-1]}, t={time.time()-t0:.1f}s")
    
    # Compute W matrix
    logger.info("  Computing quadratic coefficients...")
    V = U[:, idx_in]
    z = sigma[idx_in, None] * VT[idx_in]
    target = VT[idx_out].T * sigma[idx_out]
    H = _quadratic_features(z)
    W_coeffs, residual = _lstsq_reg(H.T, target, reg)
    W = U[:, idx_out] @ W_coeffs.T
    
    logger.info(f"  Final residual: {residual:.6e}")
    
    return BasisData(
        method="manifold",
        V=V,
        W=W,
        shift=shift,
        r=r,
        eigs=eigs,
        selected_indices=idx_in,
    )

