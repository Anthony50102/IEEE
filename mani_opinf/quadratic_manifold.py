"""
Quadratic Manifold Computation using Greedy Algorithm.

This module implements the greedy quadratic manifold method for nonlinear
dimensionality reduction. Instead of using only linear POD modes, it augments
the reconstruction with quadratic terms to better capture nonlinear dynamics.

The method:
1. Computes SVD of centered snapshot data (like standard POD)
2. Greedily selects which modes form the "linear" basis V
3. Approximates remaining modes via quadratic features of selected modes
4. Results in: x ≈ V @ z + W @ h(z) + shift
   where h(z) contains quadratic terms z_i * z_j

Based on: Geelen et al., "Operator inference for non-intrusive model reduction 
with quadratic manifolds" (2022)

Usage:
    python quadratic_manifold.py --config config.yaml
    python quadratic_manifold.py --config config.yaml --n-vectors-check 200

Author: Adapted from JAX implementation for HPC NumPy/SciPy environment
"""

import argparse
import gc
import time
import numpy as np
from scipy import linalg
import os
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import from the existing utils module
from opinf.utils import (
    load_config,
    save_config,
    get_run_directory,
    setup_logging,
    save_step_status,
    get_output_paths,
    print_header,
    print_config_summary,
    loader,
    compute_truncation_snapshots,
    PipelineConfig,
)
HAS_UTILS = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ShiftedSVD:
    """Container for SVD results with data shift (mean)."""
    U: np.ndarray      # Left singular vectors (spatial modes), shape (n_spatial, n_snapshots)
    S: np.ndarray      # Singular values, shape (min(n_spatial, n_snapshots),)
    VT: np.ndarray     # Right singular vectors transposed (temporal modes), shape (n_snapshots, n_snapshots)
    shift: np.ndarray  # Mean shift vector, shape (n_spatial,)


@dataclass
class QuadraticManifold:
    """Container for quadratic manifold results."""
    V: np.ndarray           # Linear basis, shape (n_spatial, r)
    W: np.ndarray           # Quadratic coefficient matrix, shape (n_spatial, n_quad_features)
    shift: np.ndarray       # Mean shift, shape (n_spatial,)
    selected_indices: np.ndarray  # Indices of selected modes
    singular_values: np.ndarray   # All singular values from SVD
    r: int                  # Reduced dimension
    

# =============================================================================
# FEATURE MAP
# =============================================================================

def default_feature_map(reduced_data: np.ndarray) -> np.ndarray:
    """
    Compute quadratic features from reduced coordinates.
    
    For r-dimensional input, creates r*(r+1)/2 quadratic features:
    [x_0*x_0, x_1*x_0, x_1*x_1, x_2*x_0, x_2*x_1, x_2*x_2, ...]
    
    Parameters
    ----------
    reduced_data : np.ndarray
        Reduced coordinates, shape (r, n_snapshots).
    
    Returns
    -------
    np.ndarray
        Quadratic features, shape (r*(r+1)/2, n_snapshots).
    """
    r = reduced_data.shape[0]
    features = []
    for i in range(r):
        # x_i * x_j for j = 0, 1, ..., i
        features.append(reduced_data[i:i+1, :] * reduced_data[:i+1, :])
    return np.concatenate(features, axis=0)


def get_num_quadratic_features(r: int) -> int:
    """Get number of quadratic features for dimension r."""
    return r * (r + 1) // 2


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def shift_data(data: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Shift data by subtracting the shift vector.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix, shape (n_spatial, n_snapshots).
    shift : np.ndarray
        Shift vector (typically mean), shape (n_spatial,).
    
    Returns
    -------
    np.ndarray
        Shifted data, shape (n_spatial, n_snapshots).
    """
    return data - shift[:, np.newaxis]


def lstsq_l2(
    A: np.ndarray, 
    B: np.ndarray, 
    reg_magnitude: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """
    Solve regularized least squares: min ||Ax - B||^2 + reg^2 ||x||^2
    
    Uses SVD-based solution for numerical stability:
    x = V @ diag(s/(s^2 + reg^2)) @ U^T @ B
    
    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape (m, n).
    B : np.ndarray
        Target matrix, shape (m, k).
    reg_magnitude : float
        Regularization parameter (Tikhonov regularization).
    
    Returns
    -------
    Tuple[np.ndarray, float]
        (x, residual) where x is shape (n, k) and residual is Frobenius norm.
    """
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    
    # Regularized inverse of singular values
    s_inv = s / (s**2 + reg_magnitude**2)
    
    # Compute solution: x = V @ diag(s_inv) @ U^T @ B
    x = (VT.T * s_inv) @ (U.T @ B)
    
    # Compute residual
    B_estimate = A @ x
    residual = np.linalg.norm(B - B_estimate, 'fro')
    
    return x, residual


# =============================================================================
# GREEDY ALGORITHM
# =============================================================================

def compute_error_for_selection(
    idx_in: np.ndarray,
    idx_out: np.ndarray,
    sigma: np.ndarray,
    VT: np.ndarray,
    feature_map: Callable,
    reg_magnitude: float,
) -> float:
    """
    Compute reconstruction error for a given mode selection.
    
    Parameters
    ----------
    idx_in : np.ndarray
        Indices of selected (linear) modes.
    idx_out : np.ndarray
        Indices of remaining (quadratic-approximated) modes.
    sigma : np.ndarray
        Singular values.
    VT : np.ndarray
        Right singular vectors (temporal modes), shape (n_modes, n_snapshots).
    feature_map : Callable
        Function to compute quadratic features.
    reg_magnitude : float
        Regularization for least squares.
    
    Returns
    -------
    float
        Residual error (Frobenius norm).
    """
    # Extract singular values and temporal modes for in/out sets
    sigma_in = sigma[idx_in]
    sigma_out = sigma[idx_out]
    VT_in = VT[idx_in, :]
    VT_out = VT[idx_out, :]
    
    # Embedded snapshots in reduced space: diag(sigma_in) @ VT_in
    # This is the "z" coordinates
    embedded_snapshots = sigma_in[:, np.newaxis] * VT_in
    
    # Target: VT_out^T @ diag(sigma_out) = V2 @ S2
    # We want to approximate this with quadratic features of embedded_snapshots
    V2S2 = VT_out.T * sigma_out  # shape (n_snapshots, n_out)
    
    # Quadratic features
    H = feature_map(embedded_snapshots)  # shape (n_features, n_snapshots)
    
    # Solve: H^T @ W^T ≈ V2S2
    # i.e., min ||H^T @ W^T - V2S2||
    _, residual = lstsq_l2(H.T, V2S2, reg_magnitude)
    
    return residual


def greedy_step(
    idx_in: np.ndarray,
    idx_out: np.ndarray,
    sigma: np.ndarray,
    VT: np.ndarray,
    n_vectors_to_check: int,
    feature_map: Callable,
    reg_magnitude: float,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one greedy step: add the best mode from idx_out to idx_in.
    
    Parameters
    ----------
    idx_in : np.ndarray
        Current selected indices.
    idx_out : np.ndarray
        Remaining candidate indices.
    sigma : np.ndarray
        Singular values.
    VT : np.ndarray
        Right singular vectors.
    n_vectors_to_check : int
        Maximum number of candidates to evaluate (for efficiency).
    feature_map : Callable
        Quadratic feature map function.
    reg_magnitude : float
        Regularization parameter.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Updated (idx_in, idx_out).
    """
    n_consider = min(n_vectors_to_check, len(idx_out))
    idx_consider = idx_out[:n_consider]
    
    errors = np.zeros(n_consider)
    
    for i in range(n_consider):
        # Trial: add idx_consider[i] to idx_in
        idx_in_trial = np.append(idx_in, idx_consider[i])
        idx_out_trial = np.delete(idx_consider, i)
        # Also need to include the rest of idx_out not in idx_consider
        if n_consider < len(idx_out):
            idx_out_trial = np.concatenate([idx_out_trial, idx_out[n_consider:]])
        
        errors[i] = compute_error_for_selection(
            idx_in_trial, idx_out_trial, sigma, VT, feature_map, reg_magnitude
        )
        
        if verbose and (i + 1) % 50 == 0:
            print(f"    Checked {i + 1}/{n_consider} candidates...")
    
    # Select the best (minimum error)
    best_idx = np.argmin(errors)
    
    # Update indices
    idx_in_next = np.append(idx_in, idx_out[best_idx])
    idx_out_next = np.delete(idx_out, best_idx)
    
    return idx_in_next, idx_out_next


def quadmani_greedy_from_svd(
    shifted_svd: ShiftedSVD,
    r: int,
    n_vectors_to_check: int = 200,
    reg_magnitude: float = 1e-6,
    idx_in_initial: Optional[np.ndarray] = None,
    feature_map: Callable = default_feature_map,
    verbose: bool = True,
    logger = None,
) -> QuadraticManifold:
    """
    Compute quadratic manifold using greedy algorithm from pre-computed SVD.
    
    Parameters
    ----------
    shifted_svd : ShiftedSVD
        Pre-computed SVD with shift.
    r : int
        Target reduced dimension (number of linear modes).
    n_vectors_to_check : int
        Max candidates per greedy step.
    reg_magnitude : float
        Tikhonov regularization parameter.
    idx_in_initial : np.ndarray, optional
        Initial indices to include (warm start).
    feature_map : Callable
        Function to compute quadratic features.
    verbose : bool
        Print progress.
    logger : logging.Logger, optional
        Logger instance.
    
    Returns
    -------
    QuadraticManifold
        Computed quadratic manifold.
    """
    U, sigma, VT, shift = shifted_svd.U, shifted_svd.S, shifted_svd.VT, shifted_svd.shift
    
    def log_info(msg):
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)
    
    # Initialize indices
    if idx_in_initial is None or len(idx_in_initial) == 0:
        # Start with the first mode (highest energy)
        idx_in = np.array([0], dtype=np.int64)
    else:
        idx_in = np.asarray(idx_in_initial, dtype=np.int64)
    
    n_modes = len(sigma)
    idx_out = np.setdiff1d(np.arange(n_modes), idx_in)
    
    log_info(f"Starting greedy quadratic manifold computation")
    log_info(f"  Target dimension r = {r}")
    log_info(f"  Total modes available = {n_modes}")
    log_info(f"  Initial selected modes = {len(idx_in)}")
    log_info(f"  Candidates to check per step = {n_vectors_to_check}")
    log_info(f"  Regularization = {reg_magnitude}")
    
    # Greedy iteration
    step = 1
    while len(idx_in) < r:
        t_step = time.time()
        
        idx_in, idx_out = greedy_step(
            idx_in, idx_out, sigma, VT,
            n_vectors_to_check, feature_map, reg_magnitude,
            verbose=verbose
        )
        
        log_info(f"  Step {step}: selected mode {idx_in[-1]}, "
                f"total selected = {len(idx_in)}, time = {time.time() - t_step:.1f}s")
        step += 1
    
    # Extract final basis
    log_info("Computing final quadratic coefficients...")
    
    V = U[:, idx_in]  # Linear basis (n_spatial, r)
    
    sigma_in = sigma[idx_in]
    sigma_out = sigma[idx_out]
    VT_in = VT[idx_in, :]
    VT_out = VT[idx_out, :]
    
    # Embedded snapshots
    embedded_snapshots = sigma_in[:, np.newaxis] * VT_in
    
    # Target for quadratic approximation
    V2S2 = VT_out.T * sigma_out
    
    # Quadratic features
    H = feature_map(embedded_snapshots)
    
    # Solve for W: H^T @ W^T ≈ V2S2
    W_coeffs, final_residual = lstsq_l2(H.T, V2S2, reg_magnitude)
    
    # W maps quadratic features to the "out" subspace
    # Full W is U_out @ W_coeffs^T
    W = U[:, idx_out] @ W_coeffs.T
    
    log_info(f"  Final residual = {final_residual:.6e}")
    log_info(f"  V shape = {V.shape}")
    log_info(f"  W shape = {W.shape}")
    
    return QuadraticManifold(
        V=V,
        W=W,
        shift=shift,
        selected_indices=idx_in,
        singular_values=sigma,
        r=r,
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
    
    This is the main entry point. It:
    1. Centers the data (subtracts mean)
    2. Computes SVD
    3. Runs greedy selection
    4. Computes quadratic coefficients
    
    Parameters
    ----------
    data : np.ndarray
        Snapshot matrix, shape (n_spatial, n_snapshots).
    r : int
        Target reduced dimension.
    n_vectors_to_check : int
        Max candidates per greedy step.
    reg_magnitude : float
        Tikhonov regularization.
    idx_in_initial : np.ndarray, optional
        Initial mode indices.
    feature_map : Callable
        Quadratic feature map.
    verbose : bool
        Print progress.
    logger : logging.Logger, optional
        Logger instance.
    
    Returns
    -------
    QuadraticManifold
        Computed quadratic manifold.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)
    
    log_info("Computing shifted SVD...")
    t_svd = time.time()
    
    # Center data
    shift = np.mean(data, axis=1)
    data_centered = shift_data(data, shift)
    
    # SVD
    U, s, VT = np.linalg.svd(data_centered, full_matrices=False)
    
    log_info(f"  SVD completed in {time.time() - t_svd:.1f}s")
    log_info(f"  U shape = {U.shape}, S shape = {s.shape}, VT shape = {VT.shape}")
    
    shifted_svd = ShiftedSVD(U=U, S=s, VT=VT, shift=shift)
    
    return quadmani_greedy_from_svd(
        shifted_svd, r, n_vectors_to_check, reg_magnitude,
        idx_in_initial, feature_map, verbose, logger
    )


# =============================================================================
# PROJECTION AND LIFTING
# =============================================================================

def linear_reduce(
    qm: QuadraticManifold,
    data: np.ndarray,
) -> np.ndarray:
    """
    Project data onto the linear reduced basis.
    
    Parameters
    ----------
    qm : QuadraticManifold
        Computed quadratic manifold.
    data : np.ndarray
        Full-order data, shape (n_spatial, n_snapshots).
    
    Returns
    -------
    np.ndarray
        Reduced coordinates, shape (r, n_snapshots).
    """
    data_shifted = shift_data(data, qm.shift)
    return qm.V.T @ data_shifted


def lift_quadratic(
    qm: QuadraticManifold,
    reduced_data: np.ndarray,
    feature_map: Callable = default_feature_map,
) -> np.ndarray:
    """
    Lift reduced coordinates back to full space using quadratic manifold.
    
    x ≈ V @ z + W @ h(z) + shift
    
    Parameters
    ----------
    qm : QuadraticManifold
        Computed quadratic manifold.
    reduced_data : np.ndarray
        Reduced coordinates, shape (r, n_snapshots).
    feature_map : Callable
        Quadratic feature map (must match training).
    
    Returns
    -------
    np.ndarray
        Reconstructed full-order data, shape (n_spatial, n_snapshots).
    """
    linear_part = qm.V @ reduced_data
    quadratic_features = feature_map(reduced_data)
    quadratic_part = qm.W @ quadratic_features
    
    # Add back the shift
    return linear_part + quadratic_part + qm.shift[:, np.newaxis]


def compute_reconstruction_error(
    qm: QuadraticManifold,
    data: np.ndarray,
    feature_map: Callable = default_feature_map,
) -> Tuple[float, float]:
    """
    Compute reconstruction error for given data.
    
    Parameters
    ----------
    qm : QuadraticManifold
        Computed quadratic manifold.
    data : np.ndarray
        Test data, shape (n_spatial, n_snapshots).
    feature_map : Callable
        Quadratic feature map.
    
    Returns
    -------
    Tuple[float, float]
        (absolute_error, relative_error)
    """
    reduced = linear_reduce(qm, data)
    reconstructed = lift_quadratic(qm, reduced, feature_map)
    
    abs_error = np.linalg.norm(reconstructed - data, 'fro')
    rel_error = abs_error / np.linalg.norm(data, 'fro')
    
    return abs_error, rel_error


# =============================================================================
# COMPARISON WITH LINEAR POD
# =============================================================================

def compare_with_linear_pod(
    qm: QuadraticManifold,
    data: np.ndarray,
    logger = None,
    verbose: bool = True,
) -> dict:
    """
    Compare quadratic manifold reconstruction with standard linear POD.
    
    Parameters
    ----------
    qm : QuadraticManifold
        Computed quadratic manifold.
    data : np.ndarray
        Test data.
    logger : logging.Logger, optional
        Logger.
    verbose : bool
        Print results.
    
    Returns
    -------
    dict
        Comparison metrics.
    """
    def log_info(msg):
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)
    
    r = qm.r
    
    # Quadratic manifold reconstruction
    qm_abs, qm_rel = compute_reconstruction_error(qm, data)
    
    # Linear POD reconstruction (using same r modes, but just the first r by energy)
    shift = np.mean(data, axis=1)
    data_centered = shift_data(data, shift)
    U, s, VT = np.linalg.svd(data_centered, full_matrices=False)
    
    # Standard POD with r modes
    V_pod = U[:, :r]
    reduced_pod = V_pod.T @ data_centered
    reconstructed_pod = V_pod @ reduced_pod + shift[:, np.newaxis]
    
    pod_abs = np.linalg.norm(reconstructed_pod - data, 'fro')
    pod_rel = pod_abs / np.linalg.norm(data, 'fro')
    
    # Compute energy captured
    total_energy = np.sum(s**2)
    pod_energy = np.sum(s[:r]**2) / total_energy
    qm_energy = np.sum(s[qm.selected_indices]**2) / total_energy
    
    log_info(f"\nComparison (r = {r}):")
    log_info(f"  Linear POD:")
    log_info(f"    Relative error: {pod_rel:.6e}")
    log_info(f"    Energy captured: {pod_energy*100:.4f}%")
    log_info(f"  Quadratic Manifold:")
    log_info(f"    Relative error: {qm_rel:.6e}")
    log_info(f"    Linear energy: {qm_energy*100:.4f}%")
    log_info(f"  Improvement: {(pod_rel - qm_rel) / pod_rel * 100:.2f}%")
    
    return {
        'pod_rel_error': pod_rel,
        'pod_abs_error': pod_abs,
        'pod_energy': pod_energy,
        'qm_rel_error': qm_rel,
        'qm_abs_error': qm_abs,
        'qm_linear_energy': qm_energy,
        'improvement_percent': (pod_rel - qm_rel) / pod_rel * 100,
    }


# =============================================================================
# SAVE/LOAD
# =============================================================================

def save_quadratic_manifold(qm: QuadraticManifold, filepath: str):
    """Save quadratic manifold to npz file."""
    np.savez(
        filepath,
        V=qm.V,
        W=qm.W,
        shift=qm.shift,
        selected_indices=qm.selected_indices,
        singular_values=qm.singular_values,
        r=qm.r,
    )


def load_quadratic_manifold(filepath: str) -> QuadraticManifold:
    """Load quadratic manifold from npz file."""
    data = np.load(filepath)
    return QuadraticManifold(
        V=data['V'],
        W=data['W'],
        shift=data['shift'],
        selected_indices=data['selected_indices'],
        singular_values=data['singular_values'],
        r=int(data['r']),
    )


# =============================================================================
# STANDALONE DEMO
# =============================================================================

def generate_demo_data(n_spatial: int = 1000, n_snapshots: int = 500) -> np.ndarray:
    """
    Generate demo data: advecting pulse with nonlinear dynamics.
    
    This mimics the traveling pulse from the original JAX implementation.
    """
    x = np.linspace(0, 2*np.pi, n_spatial)
    t = np.linspace(0, 4*np.pi, n_snapshots)
    
    data = np.zeros((n_spatial, n_snapshots))
    for i, ti in enumerate(t):
        # Traveling Gaussian pulse with some nonlinearity
        center = ti % (2*np.pi)
        pulse = np.exp(-10 * (x - center)**2)
        # Add secondary reflection
        pulse += 0.3 * np.exp(-10 * (x - (2*np.pi - center))**2)
        data[:, i] = pulse
    
    return data


def main_demo():
    """Demo showing quadratic manifold vs linear POD."""
    print("=" * 60)
    print("Quadratic Manifold Demo")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating demo data (advecting pulse)...")
    data = generate_demo_data(n_spatial=500, n_snapshots=200)
    
    # Split train/test
    train_data = data[:, ::2]
    test_data = data[:, 1::2]
    
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Test data shape: {test_data.shape}")
    
    # Compute quadratic manifold
    print("\nComputing quadratic manifold...")
    r = 10
    qm = quadmani_greedy(
        train_data,
        r=r,
        n_vectors_to_check=50,
        reg_magnitude=1e-6,
        verbose=True,
    )
    
    print(f"\nSelected mode indices: {qm.selected_indices}")
    
    # Compare with linear POD on test data
    print("\nEvaluating on test data...")
    comparison = compare_with_linear_pod(qm, test_data, verbose=True)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT (for use with config file)
# =============================================================================

def main():
    """Main entry point compatible with the POD pipeline."""
    parser = argparse.ArgumentParser(
        description="Quadratic Manifold Computation using Greedy Algorithm"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration YAML file (optional for demo)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic data"
    )
    parser.add_argument(
        "--r", type=int, default=20,
        help="Reduced dimension"
    )
    parser.add_argument(
        "--n-vectors-check", type=int, default=200,
        help="Number of vectors to check per greedy step"
    )
    parser.add_argument(
        "--reg", type=float, default=1e-6,
        help="Regularization magnitude"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input data file (.npy or .npz with 'data' key)"
    )
    parser.add_argument(
        "--output", type=str, default="quadratic_manifold.npz",
        help="Output file for quadratic manifold"
    )
    args = parser.parse_args()
    
    if args.demo:
        main_demo()
        return
    
    if args.input is None and args.config is None:
        print("Error: Either --input, --config, or --demo required")
        print("Run with --demo for a demonstration")
        return
    
    # Load data
    if args.input is not None:
        print(f"Loading data from {args.input}...")
        if args.input.endswith('.npy'):
            data = np.load(args.input)
        else:
            loaded = np.load(args.input)
            if 'data' in loaded:
                data = loaded['data']
            elif 'Q' in loaded:
                data = loaded['Q']
            else:
                # Try first array
                data = loaded[list(loaded.keys())[0]]
        
        print(f"  Data shape: {data.shape}")
        
        # Ensure (n_spatial, n_snapshots) format
        if data.shape[0] < data.shape[1]:
            print(f"  Note: Assuming shape is (n_snapshots, n_spatial), transposing...")
            data = data.T
        
        # Compute quadratic manifold
        print("\nComputing quadratic manifold...")
        qm = quadmani_greedy(
            data,
            r=args.r,
            n_vectors_to_check=args.n_vectors_check,
            reg_magnitude=args.reg,
            verbose=True,
        )
        
        # Save results
        print(f"\nSaving to {args.output}...")
        save_quadratic_manifold(qm, args.output)
        
        # Show comparison
        print("\nComparing with linear POD...")
        compare_with_linear_pod(qm, data, verbose=True)
        
        print("\nDone!")


if __name__ == "__main__":
    main()