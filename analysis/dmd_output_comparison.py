"""
DMD Output Operator Comparison
==============================
Compare DMD Gamma predictions with and without learned output operators.

This script:
1. Loads HW2D simulation data
2. Computes POD basis
3. Fits BOPDMD model
4. Trains learned output operators on training data
5. Predicts test Gamma two ways:
   - Physics-based: reconstruct full state, compute Gamma from fields
   - Learned: use quadratic output operators directly on reduced state
6. Produces comparison plot

Configuration:
    Edit the CONFIGURATION section below.

Outputs:
    - dmd_output_comparison_YYYYMMDD_HHMMSS.log: Analysis log
    - dmd_output_comparison.png: Comparison plot

Author: Anthony Poole
"""

import time
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
TRAIN_END = 10000
TEST_START = 10000
TEST_END = 12000

# POD/DMD rank
R = 250

# Physical parameters
DT = 0.025
K0 = 0.15
C1 = 1.0

# Output operator regularization (single values for reference)
ALPHA_LIN = 1e-4
ALPHA_QUAD = 1e-2

# Regularization search grid
ALPHA_LIN_GRID = np.logspace(-6, 0, 13)   # 1e-6 to 1e0
ALPHA_QUAD_GRID = np.logspace(-6, 0, 13)  # 1e-6 to 1e0

# Thresholds for "good" predictions
MEAN_ERR_THRESH = 0.05   # 5%
STD_ERR_THRESH = 0.30    # 30%

# Whether to center data before POD
CENTERING = False

# Output plot
OUTPUT_PLOT = "dmd_output_comparison.png"

# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = f"dmd_output_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class Logger:
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
# HELPER FUNCTIONS
# =============================================================================

def periodic_gradient(f, dx, axis=-2):
    """Central difference with periodic BCs."""
    pad_width = [(0, 0)] * f.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(f, pad_width, mode="wrap")
    slices_p = [slice(None)] * padded.ndim
    slices_m = [slice(None)] * padded.ndim
    slices_p[axis] = slice(2, None)
    slices_m[axis] = slice(None, -2)
    return (padded[tuple(slices_p)] - padded[tuple(slices_m)]) / (2 * dx)


def compute_gamma_from_fields(n, phi, dx, c1):
    """Compute Gamma_n and Gamma_c from density and phi fields."""
    dy_phi = periodic_gradient(phi, dx, axis=-2)
    gamma_n = -np.mean(n * dy_phi, axis=(-1, -2))
    gamma_c = c1 * np.mean((n - phi) ** 2, axis=(-1, -2))
    return gamma_n, gamma_c


def get_quadratic_terms(X):
    """Compute non-redundant quadratic terms."""
    K, r = X.shape
    prods = [X[:, i:i+1] * X[:, i:] for i in range(r)]
    return np.concatenate(prods, axis=1)


def fit_output_operators(X_train, Y_Gamma, alpha_lin, alpha_quad):
    """Fit quadratic output model: Y = C @ X + G @ X2 + c."""
    K, r = X_train.shape
    s = r * (r + 1) // 2
    
    mean_X = np.mean(X_train, axis=0)
    X_centered = X_train - mean_X
    scaling_X = max(np.abs(X_centered).max(), 1e-14)
    X_scaled = X_centered / scaling_X
    
    X2 = get_quadratic_terms(X_scaled)
    D_out = np.concatenate([X_scaled, X2, np.ones((K, 1))], axis=1)
    
    d_out = r + s + 1
    reg = np.zeros(d_out)
    reg[:r] = alpha_lin
    reg[r:r+s] = alpha_quad
    reg[r+s:] = alpha_lin
    
    DtD = D_out.T @ D_out + np.diag(reg)
    DtY = D_out.T @ Y_Gamma
    O = np.linalg.solve(DtD, DtY).T
    
    return {
        'C': O[:, :r],
        'G': O[:, r:r+s],
        'c': O[:, r+s],
        'mean_X': mean_X,
        'scaling_X': scaling_X,
    }


def predict_gamma_learned(X_pred, output_model):
    """Predict Gamma using learned output operators."""
    C, G, c = output_model['C'], output_model['G'], output_model['c']
    mean_X, scaling_X = output_model['mean_X'], output_model['scaling_X']
    
    if X_pred.shape[0] == C.shape[1]:
        X_pred = X_pred.T
    
    X_scaled = (X_pred - mean_X) / scaling_X
    X2 = get_quadratic_terms(X_scaled)
    Y = C @ X_scaled.T + G @ X2.T + c[:, np.newaxis]
    return Y[0, :], Y[1, :]


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

log("=" * 60)
log("DMD OUTPUT OPERATOR COMPARISON")
log("=" * 60)
log(f"Log file: {LOG_FILE}")
log(f"Data file: {DATA_FILE}")
log(f"Train: [{TRAIN_START}, {TRAIN_END}), Test: [{TEST_START}, {TEST_END})")
log(f"POD/DMD rank: {R}")
log("")

t0 = time.time()
log("STEP 1: Loading data")
log("-" * 40)

with xr.open_dataset(DATA_FILE, engine="h5netcdf", phony_dims="sort") as fh:
    train_n = fh["density"].values[TRAIN_START:TRAIN_END]
    train_phi = fh["phi"].values[TRAIN_START:TRAIN_END]
    test_n = fh["density"].values[TEST_START:TEST_END]
    test_phi = fh["phi"].values[TEST_START:TEST_END]
    train_gamma_n = fh["gamma_n"].values[TRAIN_START:TRAIN_END]
    train_gamma_c = fh["gamma_c"].values[TRAIN_START:TRAIN_END]
    gt_gamma_n = fh["gamma_n"].values[TEST_START:TEST_END]
    gt_gamma_c = fh["gamma_c"].values[TEST_START:TEST_END]

n_train, ny, nx = train_n.shape
n_test = test_n.shape[0]
n_spatial = 2 * ny * nx
dx = 2 * np.pi / K0

log(f"Grid: {ny} x {nx}, n_spatial: {n_spatial}")
log(f"Train: {n_train} snapshots, Test: {n_test} snapshots")

# Stack into Q matrices: (n_spatial, n_time)
Q_train = np.vstack([train_n.reshape(n_train, -1).T, train_phi.reshape(n_train, -1).T])
Q_test = np.vstack([test_n.reshape(n_test, -1).T, test_phi.reshape(n_test, -1).T])

log(f"Load time: {time.time()-t0:.1f}s")

# =============================================================================
# STEP 2: CENTER DATA AND COMPUTE POD
# =============================================================================

log("")
log("STEP 2: Computing POD basis")
log("-" * 40)

if CENTERING:
    log("Centering data...")
    train_mean = np.mean(Q_train, axis=1, keepdims=True)
    Q_train_c = Q_train - train_mean
    Q_test_c = Q_test - train_mean
else:
    log("Skipping centering (CENTERING=False)")
    train_mean = np.zeros((Q_train.shape[0], 1))
    Q_train_c = Q_train
    Q_test_c = Q_test

log("Computing Gram matrix...")
G = Q_train_c.T @ Q_train_c
eigs, eigv = np.linalg.eigh(G)
idx = np.argsort(eigs)[::-1]
eigs = np.maximum(eigs[idx], 1e-14)
eigv = eigv[:, idx]
S = np.sqrt(eigs)

U_r = Q_train_c @ eigv[:, :R] @ np.diag(1.0 / S[:R])
energy = np.cumsum(S**2) / np.sum(S**2)
log(f"POD rank {R} captures {energy[R-1]*100:.2f}% energy")

# Project to reduced space
Xhat_train = (U_r.T @ Q_train_c).T  # (n_train, r)
Xhat_test = (U_r.T @ Q_test_c).T    # (n_test, r)

log(f"Xhat_train: {Xhat_train.shape}, Xhat_test: {Xhat_test.shape}")

# =============================================================================
# STEP 3: FIT BOPDMD
# =============================================================================

log("")
log("STEP 3: Fitting BOPDMD")
log("-" * 40)

t_train = np.arange(n_train) * DT
t_test = np.arange(n_test) * DT

dmd = BOPDMD(
    svd_rank=R,
    num_trials=0,
    proj_basis=None,
    use_proj=False,
    eig_sort="real",
    eig_constraints={"stable"},
)

t1 = time.time()
dmd.fit(Xhat_train.T.astype(np.float64), t=t_train.astype(np.float64))
log(f"BOPDMD fit time: {time.time()-t1:.1f}s")

eigs_dmd = dmd.eigs
modes_dmd = dmd.modes
amps_dmd = dmd._b

n_stable = np.sum(eigs_dmd.real < 0)
log(f"Eigenvalues: {len(eigs_dmd)} ({n_stable} stable)")

# =============================================================================
# STEP 4: TRAIN OUTPUT OPERATORS + REGULARIZATION SEARCH
# =============================================================================

log("")
log("STEP 4: Training output operators + regularization search")
log("-" * 40)

Y_Gamma_train = np.column_stack([train_gamma_n, train_gamma_c])

# Single model for reference
output_model = fit_output_operators(Xhat_train, Y_Gamma_train, ALPHA_LIN, ALPHA_QUAD)
log(f"Output operators: C {output_model['C'].shape}, G {output_model['G'].shape}")

# =============================================================================
# STEP 5: FORECAST TEST DATA
# =============================================================================

log("")
log("STEP 5: Forecasting test data")
log("-" * 40)

# Get test IC in reduced space
x0_test = Xhat_test[0, :]

# Compute amplitudes for test IC
b_test = np.linalg.lstsq(modes_dmd, x0_test, rcond=None)[0]

# Forecast: X_hat(t) = W @ diag(b) @ exp(eigs * t)
Et = np.exp(np.outer(eigs_dmd, t_test))
X_hat_pred = modes_dmd @ (b_test[:, None] * Et)
X_hat_pred = np.real(X_hat_pred)  # (r, n_test)

log(f"Forecast shape: {X_hat_pred.shape}")

# =============================================================================
# STEP 6: COMPUTE GAMMA - PHYSICS-BASED
# =============================================================================

log("")
log("STEP 6: Computing Gamma (physics-based)")
log("-" * 40)

# Reconstruct full state
Q_pred = U_r @ X_hat_pred + train_mean  # (n_spatial, n_test)

# Extract n and phi fields
n_pred = Q_pred[:ny*nx, :].T.reshape(n_test, ny, nx)
phi_pred = Q_pred[ny*nx:, :].T.reshape(n_test, ny, nx)

gamma_n_physics, gamma_c_physics = compute_gamma_from_fields(n_pred, phi_pred, dx, C1)

log(f"Physics Gamma_n: mean={gamma_n_physics.mean():.6f}, std={gamma_n_physics.std():.6f}")
log(f"Physics Gamma_c: mean={gamma_c_physics.mean():.6f}, std={gamma_c_physics.std():.6f}")

# =============================================================================
# STEP 7: COMPUTE GAMMA - LEARNED OPERATORS
# =============================================================================

log("")
log("STEP 7: Computing Gamma (learned operators)")
log("-" * 40)

gamma_n_learned, gamma_c_learned = predict_gamma_learned(X_hat_pred, output_model)

log(f"Learned Gamma_n: mean={gamma_n_learned.mean():.6f}, std={gamma_n_learned.std():.6f}")
log(f"Learned Gamma_c: mean={gamma_c_learned.mean():.6f}, std={gamma_c_learned.std():.6f}")

# =============================================================================
# STEP 7b: REGULARIZATION SEARCH
# =============================================================================

log("")
log("STEP 7b: Regularization search")
log("-" * 40)

# Store results: (alpha_lin, alpha_quad, mean_err_n, std_err_n, mean_err_c, std_err_c, gamma_n, gamma_c)
search_results = []

for a_lin in ALPHA_LIN_GRID:
    for a_quad in ALPHA_QUAD_GRID:
        om = fit_output_operators(Xhat_train, Y_Gamma_train, a_lin, a_quad)
        gn, gc = predict_gamma_learned(X_hat_pred, om)
        
        me_n = np.abs(gt_gamma_n.mean() - gn.mean()) / np.abs(gt_gamma_n.mean())
        se_n = np.abs(gt_gamma_n.std() - gn.std()) / gt_gamma_n.std()
        me_c = np.abs(gt_gamma_c.mean() - gc.mean()) / np.abs(gt_gamma_c.mean())
        se_c = np.abs(gt_gamma_c.std() - gc.std()) / gt_gamma_c.std()
        
        search_results.append((a_lin, a_quad, me_n, se_n, me_c, se_c, gn, gc))

log(f"Searched {len(search_results)} (alpha_lin, alpha_quad) combinations")

# Filter passing results
passing_n = [(r[0], r[1], r[2], r[3], r[6]) for r in search_results 
             if r[2] < MEAN_ERR_THRESH and r[3] < STD_ERR_THRESH]
passing_c = [(r[0], r[1], r[4], r[5], r[7]) for r in search_results 
             if r[4] < MEAN_ERR_THRESH and r[5] < STD_ERR_THRESH]

log(f"Gamma_n: {len(passing_n)} pass (mean_err<{MEAN_ERR_THRESH*100:.0f}%, std_err<{STD_ERR_THRESH*100:.0f}%)")
log(f"Gamma_c: {len(passing_c)} pass (mean_err<{MEAN_ERR_THRESH*100:.0f}%, std_err<{STD_ERR_THRESH*100:.0f}%)")

# =============================================================================
# STEP 8: COMPUTE ERRORS
# =============================================================================

log("")
log("STEP 8: Computing errors")
log("-" * 40)

# Mean errors
err_mean_n_phys = np.abs(gt_gamma_n.mean() - gamma_n_physics.mean()) / np.abs(gt_gamma_n.mean())
err_mean_c_phys = np.abs(gt_gamma_c.mean() - gamma_c_physics.mean()) / np.abs(gt_gamma_c.mean())
err_mean_n_learn = np.abs(gt_gamma_n.mean() - gamma_n_learned.mean()) / np.abs(gt_gamma_n.mean())
err_mean_c_learn = np.abs(gt_gamma_c.mean() - gamma_c_learned.mean()) / np.abs(gt_gamma_c.mean())

# Std errors
err_std_n_phys = np.abs(gt_gamma_n.std() - gamma_n_physics.std()) / gt_gamma_n.std()
err_std_c_phys = np.abs(gt_gamma_c.std() - gamma_c_physics.std()) / gt_gamma_c.std()
err_std_n_learn = np.abs(gt_gamma_n.std() - gamma_n_learned.std()) / gt_gamma_n.std()
err_std_c_learn = np.abs(gt_gamma_c.std() - gamma_c_learned.std()) / gt_gamma_c.std()

log("Ground truth:")
log(f"  Gamma_n: mean={gt_gamma_n.mean():.6f}, std={gt_gamma_n.std():.6f}")
log(f"  Gamma_c: mean={gt_gamma_c.mean():.6f}, std={gt_gamma_c.std():.6f}")
log("")
log("Physics-based:")
log(f"  Gamma_n: mean_err={err_mean_n_phys:.4f}, std_err={err_std_n_phys:.4f}")
log(f"  Gamma_c: mean_err={err_mean_c_phys:.4f}, std_err={err_std_c_phys:.4f}")
log("")
log("Learned operators:")
log(f"  Gamma_n: mean_err={err_mean_n_learn:.4f}, std_err={err_std_n_learn:.4f}")
log(f"  Gamma_c: mean_err={err_mean_c_learn:.4f}, std_err={err_std_c_learn:.4f}")

# =============================================================================
# STEP 9: GENERATE PLOT
# =============================================================================

log("")
log("STEP 9: Generating plot")
log("-" * 40)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Gamma_n
ax1 = axes[0]
ax1.plot(t_test, gt_gamma_n, 'k-', label='Ground Truth', linewidth=1.5, alpha=0.8)
ax1.plot(t_test, gamma_n_physics, 'b--', label='DMD (physics)', linewidth=1.2, alpha=0.7)
ax1.plot(t_test, gamma_n_learned, 'r:', label='DMD (learned)', linewidth=1.2, alpha=0.7)
ax1.set_ylabel(r'$\Gamma_n$', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title(r'Particle Flux $\Gamma_n$', fontsize=12)
ax1.grid(True, alpha=0.3)

textstr_n = f'Physics: mean_err={err_mean_n_phys:.3f}, std_err={err_std_n_phys:.3f}\n'
textstr_n += f'Learned: mean_err={err_mean_n_learn:.3f}, std_err={err_std_n_learn:.3f}'
ax1.text(0.02, 0.98, textstr_n, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Gamma_c
ax2 = axes[1]
ax2.plot(t_test, gt_gamma_c, 'k-', label='Ground Truth', linewidth=1.5, alpha=0.8)
ax2.plot(t_test, gamma_c_physics, 'b--', label='DMD (physics)', linewidth=1.2, alpha=0.7)
ax2.plot(t_test, gamma_c_learned, 'r:', label='DMD (learned)', linewidth=1.2, alpha=0.7)
ax2.set_ylabel(r'$\Gamma_c$', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_title(r'Conductive Flux $\Gamma_c$', fontsize=12)
ax2.grid(True, alpha=0.3)

textstr_c = f'Physics: mean_err={err_mean_c_phys:.3f}, std_err={err_std_c_phys:.3f}\n'
textstr_c += f'Learned: mean_err={err_mean_c_learn:.3f}, std_err={err_std_c_learn:.3f}'
ax2.text(0.02, 0.98, textstr_c, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add passing ensemble as light lines
for r in passing_n:
    ax1.plot(t_test, r[4], 'g-', linewidth=0.3, alpha=0.3)
for r in passing_c:
    ax2.plot(t_test, r[4], 'g-', linewidth=0.3, alpha=0.3)

# Add legend entry for ensemble
if passing_n:
    ax1.plot([], [], 'g-', linewidth=1, alpha=0.5, label=f'Passing ensemble ({len(passing_n)})')
    ax1.legend(loc='upper right', fontsize=10)
if passing_c:
    ax2.plot([], [], 'g-', linewidth=1, alpha=0.5, label=f'Passing ensemble ({len(passing_c)})')
    ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
log(f"Saved plot to {OUTPUT_PLOT}")

# =============================================================================
# STEP 10: SUMMARY TABLE
# =============================================================================

log("")
log("STEP 10: Best regularization summary")
log("-" * 40)

# Find best for each metric
results_arr = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in search_results])

best_mean_n_idx = np.argmin(results_arr[:, 2])
best_std_n_idx = np.argmin(results_arr[:, 3])
best_mean_c_idx = np.argmin(results_arr[:, 4])
best_std_c_idx = np.argmin(results_arr[:, 5])

log("")
log("BEST REGULARIZATION VALUES:")
log(f"{'Metric':<20} {'alpha_lin':<12} {'alpha_quad':<12} {'Error':<10}")
log("-" * 54)
log(f"{'Gamma_n mean':<20} {results_arr[best_mean_n_idx, 0]:<12.2e} {results_arr[best_mean_n_idx, 1]:<12.2e} {results_arr[best_mean_n_idx, 2]:<10.4f}")
log(f"{'Gamma_n std':<20} {results_arr[best_std_n_idx, 0]:<12.2e} {results_arr[best_std_n_idx, 1]:<12.2e} {results_arr[best_std_n_idx, 3]:<10.4f}")
log(f"{'Gamma_c mean':<20} {results_arr[best_mean_c_idx, 0]:<12.2e} {results_arr[best_mean_c_idx, 1]:<12.2e} {results_arr[best_mean_c_idx, 4]:<10.4f}")
log(f"{'Gamma_c std':<20} {results_arr[best_std_c_idx, 0]:<12.2e} {results_arr[best_std_c_idx, 1]:<12.2e} {results_arr[best_std_c_idx, 5]:<10.4f}")

# Range of passing values
if passing_n:
    pn_arr = np.array([(p[0], p[1]) for p in passing_n])
    log(f"")
    log(f"Gamma_n passing range: alpha_lin=[{pn_arr[:,0].min():.2e}, {pn_arr[:,0].max():.2e}], alpha_quad=[{pn_arr[:,1].min():.2e}, {pn_arr[:,1].max():.2e}]")
if passing_c:
    pc_arr = np.array([(p[0], p[1]) for p in passing_c])
    log(f"Gamma_c passing range: alpha_lin=[{pc_arr[:,0].min():.2e}, {pc_arr[:,0].max():.2e}], alpha_quad=[{pc_arr[:,1].min():.2e}, {pc_arr[:,1].max():.2e}]")

# =============================================================================
# DONE
# =============================================================================

log("")
log("=" * 60)
log(f"ANALYSIS COMPLETE (total time: {time.time()-t0:.1f}s)")
log("=" * 60)
log.close()
