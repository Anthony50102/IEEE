"""
Parallel Manifold Regularization Experiment
============================================
Run manifold OpInf at different regularization values in parallel.
Only one job (--with-pod flag) computes POD for comparison.

Outputs are compatible with the OpInf pipeline (step 2 + 3).

Usage:
    python mani_parallel_experiment.py --data-file /path/to/data.h5 \
        --output-base /path/to/output --run-name my_experiment \
        --reg 1e7 --r 84 --with-pod

Outputs per regularization value:
    output_base/run_name_reg{value}/
        POD_basis_Ur.npy          # Linear basis V (n_spatial, r)
        POD_basis_Ur_basis.npz    # Full BasisData (method, V, W, shift, etc.)
        X_hat_train.npy           # Projected training data (n_train, r)
        X_hat_test.npy            # Projected test data (n_test, r)
        learning_matrices.npz     # For step 2
        initial_conditions.npz    # For step 3
        preprocessing_info.npz    # Metadata
        data_boundaries.npz       # Trajectory boundaries
        gamma_reference.npz       # Reference Gamma values
        reconstruction_results.npz # Train/test errors for this reg

Author: Anthony Poole
"""

import argparse
import gc
import os
import time
import numpy as np
import xarray as xr
import yaml
from datetime import datetime


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    """Simple logger that writes to both console and file."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.fh = open(filepath, 'w')
    
    def __call__(self, msg=""):
        print(msg)
        print(msg, file=self.fh, flush=True)
    
    def info(self, msg):
        self(msg)
    
    def close(self):
        self.fh.close()


# =============================================================================
# QUADRATIC FEATURES
# =============================================================================

def quadratic_features(z):
    """Compute quadratic features: z_i * z_j for j <= i."""
    r = z.shape[0]
    return np.concatenate([z[i:i+1] * z[:i+1] for i in range(r)], axis=0)


def lstsq_reg(A, B, reg):
    """Tikhonov-regularized least squares via SVD."""
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s_inv = s / (s**2 + reg**2)
    x = (VT.T * s_inv) @ (U.T @ B)
    residual = np.linalg.norm(B - A @ x, 'fro')
    return x, residual


def greedy_error(idx_in, idx_out, sigma, VT, reg):
    """Reconstruction error for a mode partition."""
    z = sigma[idx_in, None] * VT[idx_in]
    target = VT[idx_out].T * sigma[idx_out]
    H = quadratic_features(z)
    _, residual = lstsq_reg(H.T, target, reg)
    return residual


# =============================================================================
# POD COMPUTATION (Gram matrix method)
# =============================================================================

def compute_pod_basis(Q_train, r, log):
    """Compute POD basis via Gram matrix eigendecomposition."""
    log("Computing POD basis via Gram matrix...")
    t0 = time.time()
    
    n_spatial, n_time = Q_train.shape
    
    # Gram matrix: D = Q^T @ Q
    D = Q_train.T @ Q_train
    log(f"  Gram matrix shape: {D.shape}")
    
    # Eigendecomposition
    eigs, eigv = np.linalg.eigh(D)
    
    # Sort descending
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    eigv = eigv[:, idx]
    
    # Cumulative energy
    eigs_pos = np.maximum(eigs, 0)
    total_energy = np.sum(eigs_pos)
    cum_energy = np.cumsum(eigs_pos) / total_energy
    
    log(f"  Energy: r=10 → {cum_energy[9]*100:.4f}%, r=50 → {cum_energy[49]*100:.4f}%, r=100 → {cum_energy[min(99, len(cum_energy)-1)]*100:.4f}%")
    
    # Compute POD basis
    eigs_r = eigs[:r]
    eigv_r = eigv[:, :r]
    eigs_safe = np.where(eigs_r > 1e-14, eigs_r, 1e-14)
    Tr = eigv_r @ np.diag(eigs_safe ** (-0.5))
    V_pod = Q_train @ Tr  # (n_spatial, r)
    
    # Verify orthonormality
    ortho_err = np.linalg.norm(V_pod.T @ V_pod - np.eye(r))
    log(f"  Orthonormality error: {ortho_err:.6e}")
    log(f"  POD computed in {time.time()-t0:.1f}s")
    
    return V_pod, eigs, cum_energy


def compute_pod_reconstruction_error(Q, V):
    """Compute POD reconstruction error."""
    z = V.T @ Q
    Q_rec = V @ z
    abs_err = np.linalg.norm(Q - Q_rec, 'fro')
    rel_err = abs_err / np.linalg.norm(Q, 'fro')
    return abs_err, rel_err


# =============================================================================
# MANIFOLD COMPUTATION
# =============================================================================

def compute_manifold(Q_train, r, n_check, reg, log):
    """
    Compute quadratic manifold using greedy mode selection.
    
    Returns V, W, shift, selected_indices, eigs
    """
    log(f"Computing quadratic manifold: r={r}, n_check={n_check}, reg={reg:.1e}")
    t0_total = time.time()
    
    n_spatial, n_time = Q_train.shape
    
    # Compute SVD via Gram matrix
    log("  Computing SVD via Gram matrix...")
    t0 = time.time()
    
    D = Q_train.T @ Q_train
    eigs_raw, eigv = np.linalg.eigh(D)
    
    # Sort descending
    idx_sort = np.argsort(eigs_raw)[::-1]
    eigs_raw = eigs_raw[idx_sort]
    eigv = eigv[:, idx_sort]
    
    # Singular values
    eigs_positive = np.maximum(eigs_raw, 0)
    sigma = np.sqrt(eigs_positive)
    
    # U = Q @ V @ Sigma^{-1}
    sigma_inv = np.where(sigma > 1e-14, 1.0 / sigma, 0.0)
    U = Q_train @ (eigv * sigma_inv[None, :])
    VT = eigv.T
    
    log(f"  SVD done in {time.time()-t0:.1f}s")
    
    # Cumulative energy
    eigs = eigs_positive
    energy = np.cumsum(eigs) / np.sum(eigs)
    log(f"  Energy: r=10 → {energy[9]*100:.2f}%, r=50 → {energy[49]*100:.2f}%")
    
    # Greedy selection
    log("  Starting greedy selection...")
    idx_in = np.array([0], dtype=np.int64)
    idx_out = np.arange(1, len(sigma), dtype=np.int64)
    
    while len(idx_in) < r:
        t0 = time.time()
        n_consider = min(n_check, len(idx_out))
        errors = np.zeros(n_consider)
        
        for i in range(n_consider):
            trial_in = np.append(idx_in, idx_out[i])
            trial_out = np.delete(idx_out, i)
            errors[i] = greedy_error(trial_in, trial_out, sigma, VT, reg)
        
        best = np.argmin(errors)
        idx_in = np.append(idx_in, idx_out[best])
        idx_out = np.delete(idx_out, best)
        
        if len(idx_in) % 10 == 0 or len(idx_in) == r:
            log(f"    Step {len(idx_in)}/{r}: mode {idx_in[-1]}, t={time.time()-t0:.1f}s")
    
    # Compute W matrix
    log("  Computing quadratic coefficients...")
    V = U[:, idx_in]
    z = sigma[idx_in, None] * VT[idx_in]
    target = VT[idx_out].T * sigma[idx_out]
    H = quadratic_features(z)
    W_coeffs, residual = lstsq_reg(H.T, target, reg)
    W = U[:, idx_out] @ W_coeffs.T
    
    log(f"  Final residual: {residual:.6e}")
    log(f"  Total manifold time: {time.time()-t0_total:.1f}s")
    
    return V, W, idx_in, eigs


def compute_manifold_reconstruction_error(Q, V, W):
    """Compute manifold reconstruction error."""
    n_time = Q.shape[1]
    z = V.T @ Q
    Q_rec = V @ z
    for t in range(n_time):
        h = quadratic_features(z[:, t:t+1]).squeeze()
        Q_rec[:, t] += W @ h
    abs_err = np.linalg.norm(Q - Q_rec, 'fro')
    rel_err = abs_err / np.linalg.norm(Q, 'fro')
    return abs_err, rel_err


# =============================================================================
# PREPARE LEARNING MATRICES (for OpInf step 2 compatibility)
# =============================================================================

def prepare_learning_matrices(Xhat_train, dt, training_end, log):
    """Prepare learning matrices for OpInf step 2."""
    log("Preparing learning matrices...")
    
    n_time, r = Xhat_train.shape
    
    # Use training_end snapshots for fitting
    n_fit = min(training_end, n_time - 1)
    
    # State matrix and time derivative
    X = Xhat_train[:n_fit].T  # (r, n_fit)
    Xdot = (Xhat_train[1:n_fit+1] - Xhat_train[:n_fit]).T / dt  # (r, n_fit)
    
    log(f"  X shape: {X.shape}, Xdot shape: {Xdot.shape}")
    
    return {
        'X': X,
        'Xdot': Xdot,
        'dt': dt,
        'n_fit': n_fit,
    }


# =============================================================================
# SAVE OUTPUTS (OpInf compatible)
# =============================================================================

def save_outputs(output_dir, args, V, W, shift, selected_indices, eigs,
                 Xhat_train, Xhat_test, learning_matrices,
                 train_err, test_err, n_spatial, log):
    """Save all outputs in OpInf-compatible format."""
    os.makedirs(output_dir, exist_ok=True)
    log(f"Saving outputs to: {output_dir}")
    
    # 1. POD basis (linear part) - for compatibility
    np.save(os.path.join(output_dir, "POD_basis_Ur.npy"), V)
    
    # 2. Full basis data (BasisData format)
    np.savez(
        os.path.join(output_dir, "POD_basis_Ur_basis.npz"),
        method="manifold",
        V=V,
        W=W if W is not None else np.array([]),
        shift=shift,
        r=V.shape[1],
        eigs=eigs,
        selected_indices=selected_indices if selected_indices is not None else np.array([]),
    )
    
    # 3. Projected data
    np.save(os.path.join(output_dir, "X_hat_train.npy"), Xhat_train)
    np.save(os.path.join(output_dir, "X_hat_test.npy"), Xhat_test)
    
    # 4. Learning matrices
    np.savez(os.path.join(output_dir, "learning_matrices.npz"), **learning_matrices)
    
    # 5. Data boundaries (single trajectory for temporal split)
    n_train = Xhat_train.shape[0]
    n_test = Xhat_test.shape[0]
    np.savez(
        os.path.join(output_dir, "data_boundaries.npz"),
        train_boundaries=np.array([0, n_train]),
        test_boundaries=np.array([0, n_test]),
        n_spatial=n_spatial,
    )
    
    # 6. Initial conditions
    np.savez(
        os.path.join(output_dir, "initial_conditions.npz"),
        train_ic_full=np.zeros((1, n_spatial)),  # Placeholder
        train_ic_hat=Xhat_train[0:1],
        test_ic_full=np.zeros((1, n_spatial)),
        test_ic_hat=Xhat_test[0:1],
        train_temporal_mean=shift,
        test_temporal_mean=shift,
    )
    
    # 7. Preprocessing info
    np.savez(
        os.path.join(output_dir, "preprocessing_info.npz"),
        reduction_method="manifold",
        centering_applied=args.centering,
        scaling_applied=False,
        r_actual=V.shape[1],
        r_config=args.r,
        r_from_energy=args.r,
        n_spatial=n_spatial,
        n_fields=args.n_fields,
        n_x=args.nx,
        n_y=args.ny,
        dt=args.dt,
    )
    
    # 8. Gamma reference (placeholder)
    np.savez(
        os.path.join(output_dir, "gamma_reference.npz"),
        gamma_train=np.array([]),
        gamma_test=np.array([]),
    )
    
    # 9. Reconstruction results
    np.savez(
        os.path.join(output_dir, "reconstruction_results.npz"),
        train_error_rel=train_err,
        test_error_rel=test_err,
        r=V.shape[1],
    )
    
    # 10. Pipeline status
    with open(os.path.join(output_dir, "pipeline_status.yaml"), 'w') as f:
        yaml.dump({
            "step_1": {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "reduction_method": "manifold",
                "n_spatial": int(n_spatial),
                "r": int(V.shape[1]),
            }
        }, f)
    
    log(f"  Saved all OpInf-compatible outputs")


def save_pod_outputs(output_dir, args, V_pod, shift, eigs,
                     Xhat_train, Xhat_test, learning_matrices,
                     train_err, test_err, n_spatial, log):
    """Save POD outputs in OpInf-compatible format."""
    os.makedirs(output_dir, exist_ok=True)
    log(f"Saving POD outputs to: {output_dir}")
    
    # 1. POD basis
    np.save(os.path.join(output_dir, "POD_basis_Ur.npy"), V_pod)
    
    # 2. Full basis data (BasisData format - linear)
    np.savez(
        os.path.join(output_dir, "POD_basis_Ur_basis.npz"),
        method="linear",
        V=V_pod,
        W=np.array([]),
        shift=shift,
        r=V_pod.shape[1],
        eigs=eigs,
        selected_indices=np.arange(V_pod.shape[1]),
    )
    
    # 3. Projected data
    np.save(os.path.join(output_dir, "X_hat_train.npy"), Xhat_train)
    np.save(os.path.join(output_dir, "X_hat_test.npy"), Xhat_test)
    
    # 4. Learning matrices
    np.savez(os.path.join(output_dir, "learning_matrices.npz"), **learning_matrices)
    
    # 5. Data boundaries
    n_train = Xhat_train.shape[0]
    n_test = Xhat_test.shape[0]
    np.savez(
        os.path.join(output_dir, "data_boundaries.npz"),
        train_boundaries=np.array([0, n_train]),
        test_boundaries=np.array([0, n_test]),
        n_spatial=n_spatial,
    )
    
    # 6. Initial conditions
    np.savez(
        os.path.join(output_dir, "initial_conditions.npz"),
        train_ic_full=np.zeros((1, n_spatial)),
        train_ic_hat=Xhat_train[0:1],
        test_ic_full=np.zeros((1, n_spatial)),
        test_ic_hat=Xhat_test[0:1],
        train_temporal_mean=shift,
        test_temporal_mean=shift,
    )
    
    # 7. Preprocessing info
    np.savez(
        os.path.join(output_dir, "preprocessing_info.npz"),
        reduction_method="linear",
        centering_applied=args.centering,
        scaling_applied=False,
        r_actual=V_pod.shape[1],
        r_config=args.r,
        r_from_energy=args.r,
        n_spatial=n_spatial,
        n_fields=args.n_fields,
        n_x=args.nx,
        n_y=args.ny,
        dt=args.dt,
    )
    
    # 8. Gamma reference
    np.savez(
        os.path.join(output_dir, "gamma_reference.npz"),
        gamma_train=np.array([]),
        gamma_test=np.array([]),
    )
    
    # 9. Reconstruction results
    np.savez(
        os.path.join(output_dir, "reconstruction_results.npz"),
        train_error_rel=train_err,
        test_error_rel=test_err,
        r=V_pod.shape[1],
    )
    
    # 10. POD eigenvalues
    np.savez(
        os.path.join(output_dir, "POD.npz"),
        S=np.sqrt(np.maximum(eigs, 0)),
        eigs=eigs,
    )
    
    # 11. Pipeline status
    with open(os.path.join(output_dir, "pipeline_status.yaml"), 'w') as f:
        yaml.dump({
            "step_1": {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "reduction_method": "linear",
                "n_spatial": int(n_spatial),
                "r": int(V_pod.shape[1]),
            }
        }, f)
    
    log(f"  Saved all POD outputs")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel Manifold Experiment")
    
    # Required arguments
    parser.add_argument("--data-file", type=str, required=True, help="Path to HDF5 data file")
    parser.add_argument("--output-base", type=str, required=True, help="Base output directory")
    parser.add_argument("--run-name", type=str, required=True, help="Run name prefix")
    parser.add_argument("--reg", type=float, required=True, help="Regularization value")
    
    # Optional arguments with defaults
    parser.add_argument("--r", type=int, default=84, help="Number of modes (default: 84)")
    parser.add_argument("--n-check", type=int, default=200, help="Candidates per greedy step (default: 200)")
    parser.add_argument("--train-start", type=int, default=8000, help="Train start snapshot (default: 8000)")
    parser.add_argument("--train-end", type=int, default=16000, help="Train end snapshot (default: 16000)")
    parser.add_argument("--test-start", type=int, default=16000, help="Test start snapshot (default: 16000)")
    parser.add_argument("--test-end", type=int, default=20000, help="Test end snapshot (default: 20000)")
    parser.add_argument("--training-end", type=int, default=8000, help="Snapshots for OpInf training (default: 8000)")
    parser.add_argument("--dt", type=float, default=0.025, help="Time step (default: 0.025)")
    parser.add_argument("--n-fields", type=int, default=2, help="Number of fields (default: 2)")
    parser.add_argument("--nx", type=int, default=512, help="Grid size x (default: 512)")
    parser.add_argument("--ny", type=int, default=512, help="Grid size y (default: 512)")
    parser.add_argument("--centering", action="store_true", help="Apply centering")
    parser.add_argument("--with-pod", action="store_true", help="Also compute POD baseline")
    
    args = parser.parse_args()
    
    reg = args.reg
    
    # Output directory for this regularization value (use scientific notation in name)
    reg_str = f"{reg:.0e}".replace("+", "")
    output_dir = os.path.join(args.output_base, f"{args.run_name}_reg{reg_str}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log = Logger(log_file)
    
    log("=" * 70)
    log("PARALLEL MANIFOLD EXPERIMENT")
    log("=" * 70)
    log(f"Data file: {args.data_file}")
    log(f"Output base: {args.output_base}")
    log(f"Run name: {args.run_name}")
    log(f"Regularization: {reg:.2e}")
    log(f"With POD: {args.with_pod}")
    log(f"Output: {output_dir}")
    log(f"Log: {log_file}")
    log("")
    
    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    log("=" * 70)
    log("STEP 1: Loading data")
    log("=" * 70)
    log(f"Data file: {args.data_file}")
    log(f"Train range: [{args.train_start}, {args.train_end})")
    log(f"Test range:  [{args.test_start}, {args.test_end})")
    
    t0 = time.time()
    
    with xr.open_dataset(args.data_file, engine="h5netcdf", phony_dims="sort") as fh:
        train_density = fh["density"].values[args.train_start:args.train_end]
        train_phi = fh["phi"].values[args.train_start:args.train_end]
        test_density = fh["density"].values[args.test_start:args.test_end]
        test_phi = fh["phi"].values[args.test_start:args.test_end]
    
    n_train = train_density.shape[0]
    n_test = test_density.shape[0]
    ny, nx = train_density.shape[1], train_density.shape[2]
    n_spatial = 2 * ny * nx
    
    log(f"Grid: {ny} x {nx}, n_spatial: {n_spatial}")
    log(f"Train snapshots: {n_train}, Test snapshots: {n_test}")
    
    # Stack into Q matrices: (n_spatial, n_time)
    Q_train = np.vstack([
        train_density.reshape(n_train, -1).T,
        train_phi.reshape(n_train, -1).T
    ]).astype(np.float64)
    
    Q_test = np.vstack([
        test_density.reshape(n_test, -1).T,
        test_phi.reshape(n_test, -1).T
    ]).astype(np.float64)
    
    del train_density, train_phi, test_density, test_phi
    gc.collect()
    
    log(f"Q_train: {Q_train.shape}, {Q_train.nbytes/1e9:.2f} GB")
    log(f"Q_test:  {Q_test.shape}, {Q_test.nbytes/1e9:.2f} GB")
    log(f"Load time: {time.time()-t0:.1f}s")
    
    # =========================================================================
    # STEP 2: Center data
    # =========================================================================
    log("")
    log("=" * 70)
    log("STEP 2: Centering data")
    log("=" * 70)
    
    train_mean = np.mean(Q_train, axis=1, keepdims=True)
    Q_train_centered = Q_train - train_mean
    Q_test_centered = Q_test - train_mean
    shift = train_mean.squeeze()
    
    if args.centering:
        log(f"Centering enabled. Train mean norm: {np.linalg.norm(train_mean):.6e}")
    else:
        log("Centering for manifold computation (data will use centered form)")
    
    # =========================================================================
    # STEP 3: Compute POD baseline (if requested)
    # =========================================================================
    if args.with_pod:
        log("")
        log("=" * 70)
        log("STEP 3a: Computing POD baseline")
        log("=" * 70)
        
        V_pod, eigs_pod, cum_energy = compute_pod_basis(Q_train_centered, args.r, log)
        
        # POD reconstruction errors
        pod_train_err_abs, pod_train_err = compute_pod_reconstruction_error(Q_train_centered, V_pod)
        pod_test_err_abs, pod_test_err = compute_pod_reconstruction_error(Q_test_centered, V_pod)
        
        log(f"POD Train error: {pod_train_err*100:.4f}%")
        log(f"POD Test error:  {pod_test_err*100:.4f}%")
        
        # Project data for POD
        Xhat_train_pod = (V_pod.T @ Q_train_centered).T  # (n_train, r)
        Xhat_test_pod = (V_pod.T @ Q_test_centered).T
        
        # Learning matrices for POD
        learning_pod = prepare_learning_matrices(Xhat_train_pod, args.dt, args.training_end, log)
        
        # Save POD outputs
        pod_output_dir = os.path.join(args.output_base, f"{args.run_name}_POD")
        save_pod_outputs(
            pod_output_dir, args, V_pod, shift, eigs_pod,
            Xhat_train_pod, Xhat_test_pod, learning_pod,
            pod_train_err, pod_test_err, n_spatial, log
        )
        
        del V_pod, Xhat_train_pod, Xhat_test_pod
        gc.collect()
    
    # =========================================================================
    # STEP 4: Compute Manifold
    # =========================================================================
    log("")
    log("=" * 70)
    log(f"STEP 4: Computing Manifold (reg={reg:.2e})")
    log("=" * 70)
    
    V_mani, W_mani, selected_idx, eigs_mani = compute_manifold(
        Q_train_centered, args.r, args.n_check, reg, log
    )
    
    # Manifold reconstruction errors
    mani_train_err_abs, mani_train_err = compute_manifold_reconstruction_error(Q_train_centered, V_mani, W_mani)
    mani_test_err_abs, mani_test_err = compute_manifold_reconstruction_error(Q_test_centered, V_mani, W_mani)
    
    log(f"Manifold Train error: {mani_train_err*100:.4f}%")
    log(f"Manifold Test error:  {mani_test_err*100:.4f}%")
    
    # Project data for manifold (linear projection only - quadratic handled at decode)
    Xhat_train_mani = (V_mani.T @ Q_train_centered).T  # (n_train, r)
    Xhat_test_mani = (V_mani.T @ Q_test_centered).T
    
    # Learning matrices for manifold
    learning_mani = prepare_learning_matrices(Xhat_train_mani, args.dt, args.training_end, log)
    
    # Save manifold outputs
    save_outputs(
        output_dir, args, V_mani, W_mani, shift, selected_idx, eigs_mani,
        Xhat_train_mani, Xhat_test_mani, learning_mani,
        mani_train_err, mani_test_err, n_spatial, log
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Regularization: {reg:.2e}")
    log(f"Modes: r={args.r}")
    log(f"Manifold Train Error: {mani_train_err*100:.4f}%")
    log(f"Manifold Test Error:  {mani_test_err*100:.4f}%")
    if args.with_pod:
        log(f"POD Train Error:      {pod_train_err*100:.4f}%")
        log(f"POD Test Error:       {pod_test_err*100:.4f}%")
        log(f"Improvement (train):  {(pod_train_err - mani_train_err)*100:+.4f}%")
        log(f"Improvement (test):   {(pod_test_err - mani_test_err)*100:+.4f}%")
    log("")
    log(f"Output directory: {output_dir}")
    if args.with_pod:
        log(f"POD output: {pod_output_dir}")
    log("")
    log("DONE - Ready for OpInf step 2 + 3")
    
    log.close()


if __name__ == "__main__":
    main()
