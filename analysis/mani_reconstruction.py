"""
POD vs Quadratic Manifold Reconstruction Comparison
====================================================
Compare reconstruction quality between linear POD and quadratic manifold
at various numbers of modes and regularization parameters.

Metrics:
- Reconstruction error (train & test)
- POD retained energy

Memory-conscious: Uses Gram matrix for POD, processes data in chunks.

Configuration:
    Edit the CONFIGURATION section below to customize analysis.

Outputs:
    - mani_reconstruction_YYYYMMDD_HHMMSS.log: Detailed analysis log
    - pod_vs_manifold.npz: Results file

Author: Anthony Poole
"""

import gc
import time
import numpy as np
import xarray as xr
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to HW2D simulation HDF5 file
DATA_FILE = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d512x512_striped/test_nu5e-9.h5"

# Snapshot ranges
TRAIN_START = 8000
TRAIN_END = 12000
TEST_START = 12000
TEST_END = 16000

# POD mode counts to test
R_VALUES = [25, 50, 100, 150, 200]

# Manifold regularization values to test
REG_VALUES = [1e-8, 1e-6, 1e-4, 1e-2]

# Greedy algorithm settings
N_CHECK = 200  # Candidates per greedy step

# Output file
OUTPUT_FILE = "pod_vs_manifold.npz"

# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = f"mani_reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class Logger:
    """Simple logger that writes to both console and file."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.fh = open(filepath, 'w')
    
    def __call__(self, msg=""):
        print(msg)
        print(msg, file=self.fh, flush=True)
    
    def close(self):
        self.fh.close()


log = Logger(LOG_FILE)

# =============================================================================
# HELPER: QUADRATIC FEATURES
# =============================================================================

def quadratic_features(z):
    """Compute quadratic features: z_i * z_j for j <= i."""
    r = z.shape[0]
    return np.concatenate([z[i:i+1] * z[:i+1] for i in range(r)], axis=0)


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

log("=" * 60)
log("STEP 1: Loading data")
log("=" * 60)
log(f"Log file: {LOG_FILE}")
log(f"Data file: {DATA_FILE}")
log(f"Train range: [{TRAIN_START}, {TRAIN_END})")
log(f"Test range:  [{TEST_START}, {TEST_END})")

t0 = time.time()

with xr.open_dataset(DATA_FILE, engine="h5netcdf", phony_dims="sort") as fh:
    # Load training data
    train_density = fh["density"].values[TRAIN_START:TRAIN_END]
    train_phi = fh["phi"].values[TRAIN_START:TRAIN_END]
    
    # Load test data
    test_density = fh["density"].values[TEST_START:TEST_END]
    test_phi = fh["phi"].values[TEST_START:TEST_END]

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

# =============================================================================
# STEP 2: CENTER DATA (for POD)
# =============================================================================

log("")
log("=" * 60)
log("STEP 2: Centering data")
log("=" * 60)

train_mean = np.mean(Q_train, axis=1, keepdims=True)
Q_train_centered = Q_train - train_mean
Q_test_centered = Q_test - train_mean

log(f"Train mean norm: {np.linalg.norm(train_mean):.6e}")

# =============================================================================
# STEP 3: COMPUTE POD BASIS VIA GRAM MATRIX
# =============================================================================

log("")
log("=" * 60)
log("STEP 3: Computing POD basis via Gram matrix")
log("=" * 60)

t0 = time.time()

# Gram matrix: D = Q^T @ Q
D = Q_train_centered.T @ Q_train_centered
log(f"Gram matrix shape: {D.shape}")

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

log(f"POD computed in {time.time()-t0:.1f}s")
log(f"Energy: r=10 → {cum_energy[9]*100:.4f}%, r=50 → {cum_energy[49]*100:.4f}%, r=100 → {cum_energy[99]*100:.4f}%")

# Compute full POD basis for max r
r_max = max(R_VALUES)
eigs_r = eigs[:r_max]
eigv_r = eigv[:, :r_max]
eigs_safe = np.where(eigs_r > 1e-14, eigs_r, 1e-14)
Tr = eigv_r @ np.diag(eigs_safe ** (-0.5))
V_pod = Q_train_centered @ Tr  # (n_spatial, r_max)

log(f"POD basis shape: {V_pod.shape}")

# Verify orthonormality
VtV = V_pod.T @ V_pod
ortho_err = np.linalg.norm(VtV - np.eye(r_max))
log(f"Orthonormality error: {ortho_err:.6e}")

del D, eigv, Tr
gc.collect()

# =============================================================================
# STEP 4: EVALUATE POD RECONSTRUCTION
# =============================================================================

log("")
log("=" * 60)
log("STEP 4: POD reconstruction errors")
log("=" * 60)

pod_results = {
    'train_error_rel': np.zeros(len(R_VALUES)),
    'test_error_rel': np.zeros(len(R_VALUES)),
    'energy_retained': np.zeros(len(R_VALUES)),
}

Q_train_norm = np.linalg.norm(Q_train_centered, 'fro')
Q_test_norm = np.linalg.norm(Q_test_centered, 'fro')

log(f"{'r':>6} | {'Train Err%':>10} | {'Test Err%':>10} | {'Energy%':>10}")
log("-" * 50)

for i, r in enumerate(R_VALUES):
    V_r = V_pod[:, :r]
    
    # Train reconstruction: Q_rec = V @ V^T @ Q
    z_train = V_r.T @ Q_train_centered
    Q_train_rec = V_r @ z_train
    train_err = np.linalg.norm(Q_train_centered - Q_train_rec, 'fro') / Q_train_norm
    
    # Test reconstruction
    z_test = V_r.T @ Q_test_centered
    Q_test_rec = V_r @ z_test
    test_err = np.linalg.norm(Q_test_centered - Q_test_rec, 'fro') / Q_test_norm
    
    # Energy
    energy = np.sum(eigs_pos[:r]) / total_energy
    
    pod_results['train_error_rel'][i] = train_err
    pod_results['test_error_rel'][i] = test_err
    pod_results['energy_retained'][i] = energy
    
    log(f"{r:6d} | {train_err*100:10.4f} | {test_err*100:10.4f} | {energy*100:10.4f}")
    
    del z_train, Q_train_rec, z_test, Q_test_rec
    gc.collect()

# Free POD basis
del V_pod
gc.collect()

# =============================================================================
# STEP 5: COMPUTE MANIFOLD AND EVALUATE
# =============================================================================

log("")
log("=" * 60)
log("STEP 5: Quadratic manifold reconstruction")
log("=" * 60)

mani_results = {
    'train_error_rel': np.zeros((len(R_VALUES), len(REG_VALUES))),
    'test_error_rel': np.zeros((len(R_VALUES), len(REG_VALUES))),
}

# For manifold we need SVD of centered training data
log("Computing SVD for manifold...")
t0 = time.time()

# Use training mean as shift (same centering)
shift = train_mean.squeeze()
U, sigma, VT = np.linalg.svd(Q_train_centered, full_matrices=False)
mani_eigs = sigma**2

log(f"SVD done in {time.time()-t0:.1f}s, rank={len(sigma)}")

# Precompute norms for error computation
Q_train_norm_orig = np.linalg.norm(Q_train, 'fro')
Q_test_norm_orig = np.linalg.norm(Q_test, 'fro')

for i, r in enumerate(R_VALUES):
    for j, reg in enumerate(REG_VALUES):
        log(f"\n--- r={r}, reg={reg:.1e} ---")
        t0 = time.time()
        
        # Greedy mode selection
        idx_in = np.array([0], dtype=np.int64)
        idx_out = np.arange(1, len(sigma), dtype=np.int64)
        
        while len(idx_in) < r:
            n_consider = min(N_CHECK, len(idx_out))
            errors = np.zeros(n_consider)
            
            for k in range(n_consider):
                trial_in = np.append(idx_in, idx_out[k])
                trial_out = np.delete(idx_out, k)
                
                # Compute greedy error
                z = sigma[trial_in, None] * VT[trial_in]
                target = VT[trial_out].T * sigma[trial_out]
                H = quadratic_features(z)
                
                # Regularized least squares
                U_ls, s_ls, VT_ls = np.linalg.svd(H.T, full_matrices=False)
                s_inv = s_ls / (s_ls**2 + reg**2)
                x = (VT_ls.T * s_inv) @ (U_ls.T @ target)
                errors[k] = np.linalg.norm(target - H.T @ x, 'fro')
            
            best = np.argmin(errors)
            idx_in = np.append(idx_in, idx_out[best])
            idx_out = np.delete(idx_out, best)
            
            if len(idx_in) % 50 == 0:
                log(f"  Greedy step {len(idx_in)}/{r}")
        
        log(f"  Greedy done, selected modes: {idx_in[:5]}...")
        
        # Compute W matrix
        V_mani = U[:, idx_in]
        z = sigma[idx_in, None] * VT[idx_in]
        target = VT[idx_out].T * sigma[idx_out]
        H = quadratic_features(z)
        
        U_ls, s_ls, VT_ls = np.linalg.svd(H.T, full_matrices=False)
        s_inv = s_ls / (s_ls**2 + reg**2)
        W_coeffs = (VT_ls.T * s_inv) @ (U_ls.T @ target)
        W = U[:, idx_out] @ W_coeffs.T
        
        # Training reconstruction error
        z_train = V_mani.T @ Q_train_centered  # (r, n_train)
        Q_train_rec = V_mani @ z_train + shift[:, None]
        for t in range(n_train):
            h = quadratic_features(z_train[:, t:t+1]).squeeze()
            Q_train_rec[:, t] += W @ h
        train_err = np.linalg.norm(Q_train - Q_train_rec, 'fro') / Q_train_norm_orig
        
        del z_train, Q_train_rec
        gc.collect()
        
        # Test reconstruction error
        z_test = V_mani.T @ Q_test_centered  # (r, n_test)
        Q_test_rec = V_mani @ z_test + shift[:, None]
        for t in range(n_test):
            h = quadratic_features(z_test[:, t:t+1]).squeeze()
            Q_test_rec[:, t] += W @ h
        test_err = np.linalg.norm(Q_test - Q_test_rec, 'fro') / Q_test_norm_orig
        
        del z_test, Q_test_rec, V_mani, W
        gc.collect()
        
        mani_results['train_error_rel'][i, j] = train_err
        mani_results['test_error_rel'][i, j] = test_err
        
        log(f"  Train err: {train_err*100:.4f}%, Test err: {test_err*100:.4f}%, time: {time.time()-t0:.1f}s")

del U, VT, sigma
gc.collect()

# =============================================================================
# STEP 6: SUMMARY
# =============================================================================

log("")
log("=" * 80)
log("SUMMARY")
log("=" * 80)

log("\n--- POD RESULTS ---")
log(f"{'r':>6} | {'Train Err%':>10} | {'Test Err%':>10} | {'Energy%':>10}")
log("-" * 50)
for i, r in enumerate(R_VALUES):
    log(f"{r:6d} | {pod_results['train_error_rel'][i]*100:10.4f} | "
        f"{pod_results['test_error_rel'][i]*100:10.4f} | "
        f"{pod_results['energy_retained'][i]*100:10.4f}")

log("\n--- MANIFOLD RESULTS ---")
for j, reg in enumerate(REG_VALUES):
    log(f"\nReg = {reg:.1e}")
    log(f"{'r':>6} | {'Train Err%':>10} | {'Test Err%':>10}")
    log("-" * 35)
    for i, r in enumerate(R_VALUES):
        log(f"{r:6d} | {mani_results['train_error_rel'][i,j]*100:10.4f} | "
            f"{mani_results['test_error_rel'][i,j]*100:10.4f}")

log("\n--- COMPARISON (Manifold - POD) ---")
log("Negative = Manifold better")
for j, reg in enumerate(REG_VALUES):
    log(f"\nReg = {reg:.1e}")
    log(f"{'r':>6} | {'Train Δ%':>10} | {'Test Δ%':>10}")
    log("-" * 35)
    for i, r in enumerate(R_VALUES):
        train_diff = (mani_results['train_error_rel'][i,j] - pod_results['train_error_rel'][i]) * 100
        test_diff = (mani_results['test_error_rel'][i,j] - pod_results['test_error_rel'][i]) * 100
        log(f"{r:6d} | {train_diff:+10.4f} | {test_diff:+10.4f}")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

log("")
log("=" * 60)
log("STEP 7: Saving results")
log("=" * 60)

np.savez(
    OUTPUT_FILE,
    # Config
    r_values=np.array(R_VALUES),
    reg_values=np.array(REG_VALUES),
    n_spatial=n_spatial,
    n_train=n_train,
    n_test=n_test,
    data_file=DATA_FILE,
    train_range=(TRAIN_START, TRAIN_END),
    test_range=(TEST_START, TEST_END),
    # POD
    pod_train_error_rel=pod_results['train_error_rel'],
    pod_test_error_rel=pod_results['test_error_rel'],
    pod_energy_retained=pod_results['energy_retained'],
    pod_eigs=eigs,
    # Manifold
    mani_train_error_rel=mani_results['train_error_rel'],
    mani_test_error_rel=mani_results['test_error_rel'],
)

log(f"Saved: {OUTPUT_FILE}")
log(f"Log:   {LOG_FILE}")
log("")
log("DONE")

log.close()
