"""
DMD Reconstruction Analysis: POD-DMD vs Direct DMD
===================================================
Compare DMD reconstruction and forecasting quality between:
1. POD-DMD: POD preprocessing then DMD in reduced space
2. Direct DMD: DMD directly on full state (with internal SVD)

Uses BOPDMD (optimized DMD) to match the main training pipeline.

Metrics:
- Train reconstruction error (fitting quality)
- Test forecast error (generalization)

Configuration:
    Edit the CONFIGURATION section below to customize analysis.

Outputs:
    - dmd_reconstruction_YYYYMMDD_HHMMSS.log: Detailed analysis log
    - dmd_comparison.npz: Results file

Author: Anthony Poole
"""

import gc
import time
import numpy as np
import xarray as xr
from datetime import datetime

try:
    from pydmd import BOPDMD
except ImportError:
    raise ImportError("pydmd required. Install with: pip install pydmd")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to HW2D simulation HDF5 file
DATA_FILE = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d512x512_striped/test_nu5e-9.h5"

# Snapshot ranges
TRAIN_START = 8000
TRAIN_END = 10000    # Keep small for tractability
TEST_START = 10000
TEST_END = 11000

# DMD/POD rank values to test
R_VALUES = [25, 50, 75, 100]

# Physical parameters
DT = 0.025

# BOPDMD settings
NUM_TRIALS = 0       # Bagging trials (0 = no bagging, faster)
EIG_CONSTRAINTS = {"stable"}  # Force stable eigenvalues

# Output file
OUTPUT_FILE = "dmd_comparison.npz"

# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = f"dmd_reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


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
# STEP 1: LOAD DATA
# =============================================================================

log("=" * 60)
log("DMD RECONSTRUCTION COMPARISON")
log("POD-DMD vs Direct DMD")
log("=" * 60)
log(f"Log file: {LOG_FILE}")
log(f"Data file: {DATA_FILE}")
log(f"Train range: [{TRAIN_START}, {TRAIN_END})")
log(f"Test range:  [{TEST_START}, {TEST_END})")
log(f"R values: {R_VALUES}")
log("")

t0 = time.time()

with xr.open_dataset(DATA_FILE, engine="h5netcdf", phony_dims="sort") as fh:
    train_density = fh["density"].values[TRAIN_START:TRAIN_END]
    train_phi = fh["phi"].values[TRAIN_START:TRAIN_END]
    test_density = fh["density"].values[TEST_START:TEST_END]
    test_phi = fh["phi"].values[TEST_START:TEST_END]

n_train = train_density.shape[0]
n_test = test_density.shape[0]
ny, nx = train_density.shape[1], train_density.shape[2]
n_spatial = 2 * ny * nx

log(f"Grid: {ny} x {nx}, n_spatial: {n_spatial}")
log(f"Train snapshots: {n_train}, Test snapshots: {n_test}")
log(f"Estimated memory per Q matrix: {n_spatial * max(n_train, n_test) * 8 / 1e9:.2f} GB")

# Stack into Q matrices: (n_spatial, n_time)
Q_train = np.vstack([
    train_density.reshape(n_train, -1).T,
    train_phi.reshape(n_train, -1).T
]).astype(np.float64)

del train_density, train_phi
gc.collect()

Q_test = np.vstack([
    test_density.reshape(n_test, -1).T,
    test_phi.reshape(n_test, -1).T
]).astype(np.float64)

del test_density, test_phi
gc.collect()

log(f"Q_train: {Q_train.shape}, {Q_train.nbytes/1e9:.2f} GB")
log(f"Q_test:  {Q_test.shape}, {Q_test.nbytes/1e9:.2f} GB")
log(f"Load time: {time.time()-t0:.1f}s")

# =============================================================================
# STEP 2: CENTER DATA
# =============================================================================

log("")
log("=" * 60)
log("STEP 2: Centering data")
log("=" * 60)

train_mean = np.mean(Q_train, axis=1, keepdims=True)

# Center in place to save memory
Q_train -= train_mean
Q_test -= train_mean

log(f"Train mean norm: {np.linalg.norm(train_mean):.6e}")

# Keep references (same arrays, just renamed for clarity)
Q_train_c = Q_train
Q_test_c = Q_test

# =============================================================================
# STEP 3: COMPUTE POD BASIS
# =============================================================================

log("")
log("=" * 60)
log("STEP 3: Computing POD basis")
log("=" * 60)

t0 = time.time()

# Gram matrix approach
D = Q_train_c.T @ Q_train_c
eigs, eigv = np.linalg.eigh(D)

# Sort descending
idx = np.argsort(eigs)[::-1]
eigs = eigs[idx]
eigv = eigv[:, idx]

# Compute POD basis for max r
r_max = max(R_VALUES)
eigs_r = eigs[:r_max]
eigv_r = eigv[:, :r_max]
eigs_safe = np.where(eigs_r > 1e-14, eigs_r, 1e-14)
Tr = eigv_r @ np.diag(eigs_safe ** (-0.5))
V_pod = Q_train_c @ Tr  # (n_spatial, r_max)

log(f"POD basis shape: {V_pod.shape}, {V_pod.nbytes/1e9:.2f} GB")
log(f"POD time: {time.time()-t0:.1f}s")

# Energy captured
total_energy = np.sum(np.maximum(eigs, 0))
for r in R_VALUES:
    energy = np.sum(np.maximum(eigs[:r], 0)) / total_energy
    log(f"  r={r}: {energy*100:.4f}% energy")

del D, eigv, Tr, eigs_r, eigs
gc.collect()

# =============================================================================
# STEP 4: DMD HELPER FUNCTIONS
# =============================================================================

def fit_bopdmd(X, r, dt, logger_fn):
    """
    Fit BOPDMD model to data matrix X.
    
    Parameters
    ----------
    X : ndarray, shape (n_features, n_time)
        Snapshot matrix.
    r : int
        DMD rank.
    dt : float
        Time step.
    logger_fn : callable
        Logging function.
    
    Returns
    -------
    dict with 'eigs', 'modes', 'amplitudes', 'model'
    """
    n_features, n_time = X.shape
    t_vec = np.arange(n_time) * dt
    
    # Fit BOPDMD
    dmd_model = BOPDMD(
        svd_rank=r,
        num_trials=NUM_TRIALS,
        eig_sort="real",
        eig_constraints=EIG_CONSTRAINTS,
    )
    dmd_model.fit(X, t=t_vec)
    
    return {
        'eigs': dmd_model.eigs,
        'modes': dmd_model.modes,
        'amplitudes': dmd_model._b,
        'model': dmd_model,
    }


def dmd_reconstruct(model_dict, t_vec):
    """
    Reconstruct/forecast using DMD model.
    
    Parameters
    ----------
    model_dict : dict
        DMD model from fit_bopdmd.
    t_vec : ndarray
        Time vector.
    
    Returns
    -------
    X_rec : ndarray, shape (n_features, len(t_vec))
    """
    eigs = model_dict['eigs']
    Phi = model_dict['modes']
    b = model_dict['amplitudes']
    
    # Time dynamics: exp(eig * t) for each mode
    dynamics = np.exp(np.outer(eigs, t_vec))  # (r, n_time)
    
    # Reconstruct: Phi @ diag(b) @ dynamics
    X_rec = Phi @ (b[:, None] * dynamics)
    
    return np.real(X_rec)


# =============================================================================
# STEP 5: RUN COMPARISON
# =============================================================================

log("")
log("=" * 60)
log("STEP 5: BOPDMD Comparison")
log("=" * 60)
log(f"Using BOPDMD with num_trials={NUM_TRIALS}, eig_constraints={EIG_CONSTRAINTS}")

# Time vectors
t_train = np.arange(n_train) * DT
t_test = np.arange(n_test) * DT + n_train * DT  # Continuation

# Norms for relative error
Q_train_norm = np.linalg.norm(Q_train_c, 'fro')
Q_test_norm = np.linalg.norm(Q_test_c, 'fro')

# Results storage
results = {
    'r_values': np.array(R_VALUES),
    'pod_dmd_train_err': np.zeros(len(R_VALUES)),
    'pod_dmd_test_err': np.zeros(len(R_VALUES)),
    'direct_dmd_train_err': np.zeros(len(R_VALUES)),
    'direct_dmd_test_err': np.zeros(len(R_VALUES)),
    'pod_dmd_time': np.zeros(len(R_VALUES)),
    'direct_dmd_time': np.zeros(len(R_VALUES)),
}

log("")
log(f"{'r':>6} | {'POD-DMD Train%':>14} | {'POD-DMD Test%':>14} | {'Direct Train%':>14} | {'Direct Test%':>14}")
log("-" * 80)

for i, r in enumerate(R_VALUES):
    log(f"\n--- Processing r={r} ---")
    
    # -------------------------------------------------------------------------
    # POD-DMD: Project to POD space, fit DMD, reconstruct
    # -------------------------------------------------------------------------
    V_r = V_pod[:, :r]
    
    # Project training data
    Z_train = V_r.T @ Q_train_c  # (r, n_train)
    
    # Fit BOPDMD in reduced space
    t0 = time.time()
    pod_model = fit_bopdmd(Z_train, r, DT, log)
    pod_dmd_time = time.time() - t0
    
    del Z_train
    gc.collect()
    
    # Reconstruct training (in reduced space, then lift)
    # Compute error in chunks to avoid huge reconstruction matrix
    Z_train_rec = dmd_reconstruct(pod_model, t_train)
    Q_train_rec_pod = V_r @ Z_train_rec
    pod_train_err = np.linalg.norm(Q_train_c - Q_train_rec_pod, 'fro') / Q_train_norm
    
    del Z_train_rec, Q_train_rec_pod
    gc.collect()
    
    # Forecast test (continue from end of training)
    Z_test_rec = dmd_reconstruct(pod_model, t_test)
    Q_test_rec_pod = V_r @ Z_test_rec
    pod_test_err = np.linalg.norm(Q_test_c - Q_test_rec_pod, 'fro') / Q_test_norm
    
    del Z_test_rec, Q_test_rec_pod, pod_model, V_r
    gc.collect()
    
    results['pod_dmd_train_err'][i] = pod_train_err
    results['pod_dmd_test_err'][i] = pod_test_err
    results['pod_dmd_time'][i] = pod_dmd_time
    
    # -------------------------------------------------------------------------
    # Direct DMD: Fit DMD directly on full data
    # -------------------------------------------------------------------------
    t0 = time.time()
    direct_model = fit_bopdmd(Q_train_c, r, DT, log)
    direct_dmd_time = time.time() - t0
    
    # Reconstruct training
    Q_train_rec_direct = dmd_reconstruct(direct_model, t_train)
    direct_train_err = np.linalg.norm(Q_train_c - Q_train_rec_direct, 'fro') / Q_train_norm
    
    del Q_train_rec_direct
    gc.collect()
    
    # Forecast test
    Q_test_rec_direct = dmd_reconstruct(direct_model, t_test)
    direct_test_err = np.linalg.norm(Q_test_c - Q_test_rec_direct, 'fro') / Q_test_norm
    
    del Q_test_rec_direct, direct_model
    gc.collect()
    
    results['direct_dmd_train_err'][i] = direct_train_err
    results['direct_dmd_test_err'][i] = direct_test_err
    results['direct_dmd_time'][i] = direct_dmd_time
    
    # Log results
    log(f"{r:>6} | {pod_train_err*100:>14.4f} | {pod_test_err*100:>14.4f} | "
        f"{direct_train_err*100:>14.4f} | {direct_test_err*100:>14.4f}")

# Clean up large arrays no longer needed
del Q_train_c, Q_test_c, Q_train, Q_test, V_pod
gc.collect()

# =============================================================================
# STEP 6: SUMMARY
# =============================================================================

log("")
log("=" * 60)
log("SUMMARY")
log("=" * 60)

log("")
log("POD-DMD (POD preprocessing then BOPDMD in reduced space):")
for i, r in enumerate(R_VALUES):
    log(f"  r={r:3d}: train={results['pod_dmd_train_err'][i]*100:.4f}%, "
        f"test={results['pod_dmd_test_err'][i]*100:.4f}%, time={results['pod_dmd_time'][i]:.1f}s")

log("")
log("Direct DMD (BOPDMD on full state with rank-r SVD truncation):")
for i, r in enumerate(R_VALUES):
    log(f"  r={r:3d}: train={results['direct_dmd_train_err'][i]*100:.4f}%, "
        f"test={results['direct_dmd_test_err'][i]*100:.4f}%, time={results['direct_dmd_time'][i]:.1f}s")

log("")
log("Comparison (Direct - POD-DMD, negative = Direct better):")
log(f"{'r':>6} | {'Train Diff%':>12} | {'Test Diff%':>12}")
log("-" * 35)
for i, r in enumerate(R_VALUES):
    train_diff = (results['direct_dmd_train_err'][i] - results['pod_dmd_train_err'][i]) * 100
    test_diff = (results['direct_dmd_test_err'][i] - results['pod_dmd_test_err'][i]) * 100
    log(f"{r:>6} | {train_diff:>+12.4f} | {test_diff:>+12.4f}")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

log("")
log("=" * 60)
log("Saving results")
log("=" * 60)

np.savez(OUTPUT_FILE, **results)
log(f"Results saved to: {OUTPUT_FILE}")

log.close()
print(f"\nDone. Log saved to: {LOG_FILE}")
