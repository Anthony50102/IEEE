"""
Quadratic Manifold via Greedy Algorithm.

Implements nonlinear dimensionality reduction where:
    x ≈ V @ z + W @ h(z) + μ

where V is the linear basis (n×r), W contains quadratic coefficients,
h(z) are quadratic features z_i·z_j, and μ is the data mean.

The greedy algorithm selects which SVD modes form V by minimizing
the reconstruction error when remaining modes are approximated
via quadratic features of the selected modes.

Reference: [Add paper citation here]

Author: [Your name]
"""

import numpy as np
import gc
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import time


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ShiftedSVD:
    """SVD of mean-centered data."""
    U: np.ndarray       # Left singular vectors (n_spatial, k)
    S: np.ndarray       # Singular values (k,)
    VT: np.ndarray      # Right singular vectors (k, n_snapshots)
    shift: np.ndarray   # Mean vector (n_spatial,)


@dataclass
class QuadraticManifold:
    """Quadratic manifold representation."""
    V: np.ndarray               # Linear basis (n_spatial, r)
    W: np.ndarray               # Quadratic coefficients (n_spatial, n_quad)
    shift: np.ndarray           # Mean shift (n_spatial,)
    selected_indices: np.ndarray
    singular_values: np.ndarray
    r: int


# =============================================================================
# Core Algorithm
# =============================================================================

def default_feature_map(z: np.ndarray) -> np.ndarray:
    """
    Compute quadratic features from reduced coordinates.
    
    For r-dimensional z, produces r(r+1)/2 features:
        [z_0·z_0, z_1·z_0, z_1·z_1, z_2·z_0, ...]
    
    Parameters
    ----------
    z : ndarray, shape (r, n_snapshots)
    
    Returns
    -------
    h : ndarray, shape (r(r+1)/2, n_snapshots)
    """
    r = z.shape[0]
    return np.concatenate([z[i:i+1] * z[:i+1] for i in range(r)], axis=0)


def lstsq_l2(A: np.ndarray, B: np.ndarray, reg: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Tikhonov-regularized least squares: min ||Ax - B||² + reg²||x||²
    
    Solution via SVD: x = V·diag(s/(s² + reg²))·Uᵀ·B
    """
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s_inv = s / (s**2 + reg**2)
    x = (VT.T * s_inv) @ (U.T @ B)
    residual = np.linalg.norm(B - A @ x, 'fro')
    return x, residual


def compute_error(idx_in: np.ndarray, idx_out: np.ndarray,
                  sigma: np.ndarray, VT: np.ndarray,
                  feature_map: Callable, reg: float) -> float:
    """Compute reconstruction error for a given mode partition."""
    sigma_in, sigma_out = sigma[idx_in], sigma[idx_out]
    VT_in, VT_out = VT[idx_in], VT[idx_out]
    
    # Embedded snapshots: diag(σ_in) @ VT_in
    z = sigma_in[:, None] * VT_in
    
    # Target: VT_out.T @ diag(σ_out)
    target = VT_out.T * sigma_out
    
    # Quadratic features and solve
    H = feature_map(z)
    _, residual = lstsq_l2(H.T, target, reg)
    return residual


def greedy_step(idx_in: np.ndarray, idx_out: np.ndarray,
                sigma: np.ndarray, VT: np.ndarray,
                n_check: int, feature_map: Callable, reg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add the best mode from idx_out to idx_in."""
    n_consider = min(n_check, len(idx_out))
    errors = np.zeros(n_consider)
    
    for i in range(n_consider):
        idx_in_trial = np.append(idx_in, idx_out[i])
        idx_out_trial = np.concatenate([idx_out[:i], idx_out[i+1:]])
        errors[i] = compute_error(idx_in_trial, idx_out_trial, sigma, VT, feature_map, reg)
    
    best = np.argmin(errors)
    return np.append(idx_in, idx_out[best]), np.delete(idx_out, best)


def quadmani_greedy_from_svd(
    svd: ShiftedSVD,
    r: int,
    n_vectors_to_check: int = 200,
    reg_magnitude: float = 1e-6,
    idx_in_initial: Optional[np.ndarray] = None,
    feature_map: Callable = default_feature_map,
    verbose: bool = True,
    logger = None,
) -> QuadraticManifold:
    """
    Compute quadratic manifold from pre-computed SVD.
    
    This is useful when exploring different r values without
    recomputing the expensive SVD.
    
    Parameters
    ----------
    svd : ShiftedSVD
        Pre-computed SVD of centered data.
    r : int
        Target reduced dimension.
    n_vectors_to_check : int
        Max candidates to evaluate per greedy step.
    reg_magnitude : float
        Tikhonov regularization parameter.
    idx_in_initial : ndarray, optional
        Initial mode indices (warm start).
    feature_map : callable
        Function computing quadratic features.
    verbose : bool
        Print progress.
    logger : Logger, optional
        Logging instance.
    
    Returns
    -------
    QuadraticManifold
    """
    def log(msg):
        if logger: logger.info(msg)
        elif verbose: print(msg)
    
    U, sigma, VT, shift = svd.U, svd.S, svd.VT, svd.shift
    n_modes = len(sigma)
    
    # Initialize
    if idx_in_initial is None or len(idx_in_initial) == 0:
        idx_in = np.array([0], dtype=np.int64)
    else:
        idx_in = np.asarray(idx_in_initial, dtype=np.int64)
    idx_out = np.setdiff1d(np.arange(n_modes), idx_in)
    
    log(f"Greedy quadratic manifold: r={r}, n_check={n_vectors_to_check}, reg={reg_magnitude:.1e}")
    
    # Greedy iteration
    step = 1
    while len(idx_in) < r:
        t0 = time.time()
        idx_in, idx_out = greedy_step(idx_in, idx_out, sigma, VT,
                                       n_vectors_to_check, feature_map, reg_magnitude)
        log(f"  Step {step}: selected mode {idx_in[-1]}, t={time.time()-t0:.1f}s")
        step += 1
    
    # Compute final W matrix
    log("Computing quadratic coefficients...")
    
    V = U[:, idx_in]
    sigma_in, sigma_out = sigma[idx_in], sigma[idx_out]
    VT_in, VT_out = VT[idx_in], VT[idx_out]
    
    z = sigma_in[:, None] * VT_in
    target = VT_out.T * sigma_out
    H = feature_map(z)
    W_coeffs, residual = lstsq_l2(H.T, target, reg_magnitude)
    W = U[:, idx_out] @ W_coeffs.T
    
    log(f"  Final residual: {residual:.6e}")
    
    return QuadraticManifold(
        V=V, W=W, shift=shift,
        selected_indices=idx_in,
        singular_values=sigma,
        r=r
    )


def quadmani_greedy(
    data: np.ndarray,
    r: int,
    n_vectors_to_check: int = 200,
    reg_magnitude: float = 1e-6,
    idx_in_initial: Optional[np.ndarray] = None,
    feature_map: Callable = default_feature_map,
    verbose: bool = True,
    logger = None,
) -> QuadraticManifold:
    """
    Compute quadratic manifold using greedy algorithm.
    
    Parameters
    ----------
    data : ndarray, shape (n_spatial, n_snapshots)
        Snapshot matrix.
    r : int
        Target reduced dimension.
    n_vectors_to_check : int
        Max candidates per greedy step.
    reg_magnitude : float
        Tikhonov regularization.
    idx_in_initial : ndarray, optional
        Initial mode indices.
    feature_map : callable
        Quadratic feature map.
    verbose : bool
        Print progress.
    logger : Logger, optional
    
    Returns
    -------
    QuadraticManifold
    """
    def log(msg):
        if logger: logger.info(msg)
        elif verbose: print(msg)
    
    log("Computing SVD...")
    t0 = time.time()
    
    shift = np.mean(data, axis=1)
    data_centered = data - shift[:, None]
    U, s, VT = np.linalg.svd(data_centered, full_matrices=False)
    
    log(f"  SVD complete: {time.time()-t0:.1f}s, shape=({U.shape[0]}, {len(s)})")
    
    svd = ShiftedSVD(U=U, S=s, VT=VT, shift=shift)
    return quadmani_greedy_from_svd(
        svd, r, n_vectors_to_check, reg_magnitude,
        idx_in_initial, feature_map, verbose, logger
    )


def compute_shifted_svd(data: np.ndarray, copy: bool = True) -> ShiftedSVD:
    """
    Compute and return the shifted SVD for later use with from_svd.
    
    Useful when you want to try multiple r values without recomputing SVD.
    
    Parameters
    ----------
    data : ndarray
        Snapshot matrix (n_spatial, n_snapshots).
    copy : bool
        If False, modifies data in-place to save memory.
    """
    shift = np.mean(data, axis=1)
    
    if copy:
        data_centered = data - shift[:, None]
    else:
        # In-place subtraction to save memory
        data -= shift[:, None]
        data_centered = data
    
    U, s, VT = np.linalg.svd(data_centered, full_matrices=False)
    
    # Clean up intermediate arrays
    if not copy:
        del data_centered
    gc.collect()
    
    return ShiftedSVD(U=U, S=s, VT=VT, shift=shift)


# =============================================================================
# Projection and Lifting
# =============================================================================

def linear_reduce(qm: QuadraticManifold, data: np.ndarray) -> np.ndarray:
    """Project to reduced coordinates: z = Vᵀ(x - μ)"""
    return qm.V.T @ (data - qm.shift[:, None])


def lift_quadratic(qm: QuadraticManifold, z: np.ndarray,
                   feature_map: Callable = default_feature_map) -> np.ndarray:
    """Reconstruct: x ≈ V·z + W·h(z) + μ"""
    return qm.V @ z + qm.W @ feature_map(z) + qm.shift[:, None]


def reconstruction_error(qm: QuadraticManifold, data: np.ndarray,
                         feature_map: Callable = default_feature_map) -> Tuple[float, float]:
    """Compute absolute and relative reconstruction error."""
    z = linear_reduce(qm, data)
    recon = lift_quadratic(qm, z, feature_map)
    abs_err = np.linalg.norm(recon - data, 'fro')
    rel_err = abs_err / np.linalg.norm(data, 'fro')
    return abs_err, rel_err


# =============================================================================
# Comparison and I/O
# =============================================================================

def compute_energy_metrics(qm: QuadraticManifold, verbose: bool = True, 
                           logger=None) -> dict:
    """
    Compute energy conservation metrics for the quadratic manifold.
    
    Uses pre-computed singular values from the manifold to avoid
    expensive recomputation.
    
    Parameters
    ----------
    qm : QuadraticManifold
        Computed quadratic manifold.
    verbose : bool
        Print progress.
    logger : Logger, optional
    
    Returns
    -------
    dict
        Energy metrics including total, captured, and percentage.
    """
    def log(msg):
        if logger: logger.info(msg)
        elif verbose: print(msg)
    
    s = qm.singular_values
    total_energy = np.sum(s**2)
    
    # Energy from selected modes (linear part)
    selected_energy = np.sum(s[qm.selected_indices]**2)
    linear_energy_pct = selected_energy / total_energy * 100
    
    # Energy from first r modes (standard POD would capture this)
    top_r_energy = np.sum(s[:qm.r]**2)
    top_r_energy_pct = top_r_energy / total_energy * 100
    
    log(f"Energy conservation (r={qm.r}):")
    log(f"  Selected modes energy: {linear_energy_pct:.4f}%")
    log(f"  Top-{qm.r} modes energy: {top_r_energy_pct:.4f}%")
    log(f"  Selected indices: {qm.selected_indices[:10]}..." if len(qm.selected_indices) > 10 
        else f"  Selected indices: {qm.selected_indices}")
    
    return {
        'total_energy': float(total_energy),
        'selected_modes_energy': float(selected_energy),
        'selected_modes_energy_pct': float(linear_energy_pct),
        'top_r_modes_energy': float(top_r_energy),
        'top_r_modes_energy_pct': float(top_r_energy_pct),
        'n_modes': qm.r,
        'selected_indices': qm.selected_indices.tolist()
    }


def save_quadratic_manifold(qm: QuadraticManifold, filepath: str):
    """Save quadratic manifold to npz."""
    np.savez(filepath, V=qm.V, W=qm.W, shift=qm.shift,
             selected_indices=qm.selected_indices,
             singular_values=qm.singular_values, r=qm.r)


def load_quadratic_manifold(filepath: str) -> QuadraticManifold:
    """Load quadratic manifold from npz."""
    d = np.load(filepath)
    return QuadraticManifold(
        V=d['V'], W=d['W'], shift=d['shift'],
        selected_indices=d['selected_indices'],
        singular_values=d['singular_values'], r=int(d['r'])
    )


def save_shifted_svd(svd: ShiftedSVD, filepath: str):
    """Save pre-computed SVD for reuse."""
    np.savez(filepath, U=svd.U, S=svd.S, VT=svd.VT, shift=svd.shift)


def load_shifted_svd(filepath: str) -> ShiftedSVD:
    """Load pre-computed SVD."""
    d = np.load(filepath)
    return ShiftedSVD(U=d['U'], S=d['S'], VT=d['VT'], shift=d['shift'])


def get_num_quadratic_features(r: int) -> int:
    """Number of quadratic features for dimension r."""
    return r * (r + 1) // 2