"""
POD Reconstruction Analysis
===========================
Analyzes POD reconstruction quality as a function of the number of modes.

This script:
1. Loads HW2D simulation data
2. Verifies Gamma_n, Gamma_c computations match saved values
3. Computes POD basis via Gram matrix eigendecomposition
4. Analyzes reconstruction error vs number of modes (r)
5. Compares gamma estimation accuracy from reconstructed states

Configuration:
    Edit DATA_FILE, R_VALUES, and NT_MAX below to customize analysis.

Outputs:
    - pod_reconstruction_YYYYMMDD_HHMMSS.log: Detailed analysis log
    - pod_error_vs_r.png: Error metrics vs number of modes
    - pod_gamma_timeseries.png: Time series comparison
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
print("importing numpy"); import numpy as np
print("importing xarray"); import xarray as xr
print("importing matplotlib"); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
print("importing datetime"); from datetime import datetime
# Import directly from module to avoid MPI init from shared/__init__.py
print("importing physics"); from shared.physics import periodic_gradient, compute_gamma_n, compute_gamma_c
print("done imports")
print("DEBUG: FINISHED THE IMPORTS")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to HW2D simulation HDF5 file
DATA_FILE = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d512x512/test_nu5e-9.h5"
TEST_FILE = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d512x512/test_nu5e-9_2.h5" 

# Physical parameters
k0 = 0.15
c1 = 1.0
Lx = 2 * np.pi / k0
nx = 256
dx = Lx / nx

# Range of r values to test
R_VALUES = [10, 25, 50, 75, 100, 150, 200]

# Number of timesteps to use (None = use all)
NT_MAX = 4000

# =============================================================================
# LOGGING SETUP
# =============================================================================
LOG_FILE = f"pod_reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


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
print("DEBUG: LOAD THE DATA")
log("=" * 60)
log("STEP 1: Loading data")
log("=" * 60)
log(f"Log file: {LOG_FILE}")
log(f"Data file: {DATA_FILE}")

with xr.open_dataset(DATA_FILE, engine="h5netcdf", phony_dims="sort") as fh:
    # Load density and phi
    density = fh["density"].values[:NT_MAX]  # (nt, ny, nx)
    phi = fh["phi"].values[:NT_MAX]          # (nt, ny, nx)
    
    # Ground truth gamma values from file
    gt_gamma_n = fh["gamma_n"].values[:NT_MAX]
    gt_gamma_c = fh["gamma_c"].values[:NT_MAX]

nt = density.shape[0]
ny, nx_grid = density.shape[1], density.shape[2]
log(f"Loaded {nt} timesteps, grid size: {ny} x {nx_grid}")

# Stack into Q matrix: shape (2*nx*ny, nt)
# Each column is a flattened state [n_flat; phi_flat]
Q = np.vstack([
    density.reshape(nt, -1).T,  # (nx*ny, nt)
    phi.reshape(nt, -1).T       # (nx*ny, nt)
])
log(f"Q matrix shape: {Q.shape}")

# =============================================================================
# STEP 2: VERIFY GAMMA COMPUTATIONS
# =============================================================================
log("")
log("=" * 60)
log("STEP 2: Verifying Gamma computations")
log("=" * 60)

recomputed_gamma_n = np.zeros(nt)
recomputed_gamma_c = np.zeros(nt)

for t in range(nt):
    n_t = density[t]    # (ny, nx)
    phi_t = phi[t]      # (ny, nx)
    recomputed_gamma_n[t] = compute_gamma_n(n_t, phi_t, dx)
    recomputed_gamma_c[t] = compute_gamma_c(n_t, phi_t, c1)

# Compare with ground truth
gamma_n_error = np.abs(recomputed_gamma_n - gt_gamma_n).max()
gamma_c_error = np.abs(recomputed_gamma_c - gt_gamma_c).max()

log(f"Max error in Gamma_n: {gamma_n_error:.6e}")
log(f"Max error in Gamma_c: {gamma_c_error:.6e}")

if gamma_n_error < 1e-10 and gamma_c_error < 1e-10:
    log("✓ Gamma computations match saved values!")
else:
    log("⚠ Warning: Gamma computations differ from saved values")

# =============================================================================
# STEP 3: COMPUTE POD BASIS
# =============================================================================
log("")
log("=" * 60)
log("STEP 3: Computing POD basis via Gram matrix")
log("=" * 60)

# Using Gram matrix approach (more efficient when nx*ny >> nt)
D = Q.T @ Q  # Gram matrix, shape (nt, nt)
log(f"Gram matrix shape: {D.shape}")

# Eigendecomposition
eigs, eigv = np.linalg.eigh(D)

# Sort by decreasing eigenvalue
sorted_idx = np.argsort(eigs)[::-1]
eigs = eigs[sorted_idx]
eigv = eigv[:, sorted_idx]

# Compute cumulative energy
total_energy = np.sum(eigs)
cumulative_energy = np.cumsum(eigs) / total_energy

log(f"Total energy (sum of eigenvalues): {total_energy:.6e}")
log(f"Energy captured by first 10 modes:  {cumulative_energy[9]*100:>8.4f}%")
log(f"Energy captured by first 50 modes:  {cumulative_energy[49]*100:>8.4f}%")
log(f"Energy captured by first 100 modes: {cumulative_energy[99]*100:>8.4f}%")

# =============================================================================
# STEP 4: RECONSTRUCTION FOR DIFFERENT r VALUES
# =============================================================================
log("")
log("=" * 60)
log("STEP 4: Reconstruction analysis for different r values")
log("=" * 60)

results = {}

for r in R_VALUES:
    log(f"\n--- r = {r} modes ---")
    
    # Compute POD basis Vr
    # Vr = Q @ eigv[:, :r] @ diag(1/sqrt(eigs[:r]))
    Tr = eigv[:, :r] @ np.diag(eigs[:r] ** (-0.5))
    Vr = Q @ Tr  # shape (2*nx*ny, r)
    
    # Verify orthonormality: Vr.T @ Vr should be identity
    VtV = Vr.T @ Vr
    ortho_error = np.linalg.norm(VtV - np.eye(r))
    log(f"  Orthonormality error: {ortho_error:.6e}")
    
    # Reconstruct: Q_approx = Vr @ Vr.T @ Q
    Q_approx = Vr @ (Vr.T @ Q)
    
    # Reconstruction error
    rel_error = np.linalg.norm(Q - Q_approx) / np.linalg.norm(Q)
    log(f"  Relative reconstruction error: {rel_error:.6e} ({rel_error*100:.4f}%)")
    log(f"  Energy retained: {cumulative_energy[r-1]*100:.4f}%")
    
    # Compute Gamma from reconstructed data
    approx_gamma_n = np.zeros(nt)
    approx_gamma_c = np.zeros(nt)
    
    n_size = ny * nx_grid
    for t in range(nt):
        state = Q_approx[:, t]
        n_approx = state[:n_size].reshape(ny, nx_grid)
        phi_approx = state[n_size:].reshape(ny, nx_grid)
        approx_gamma_n[t] = compute_gamma_n(n_approx, phi_approx, dx)
        approx_gamma_c[t] = compute_gamma_c(n_approx, phi_approx, c1)
    
    # Errors in gamma
    gamma_n_rel_error = np.linalg.norm(approx_gamma_n - gt_gamma_n) / np.linalg.norm(gt_gamma_n)
    gamma_c_rel_error = np.linalg.norm(approx_gamma_c - gt_gamma_c) / np.linalg.norm(gt_gamma_c)
    
    log(f"  Gamma_n relative error: {gamma_n_rel_error:.6e} ({gamma_n_rel_error*100:.4f}%)")
    log(f"  Gamma_c relative error: {gamma_c_rel_error:.6e} ({gamma_c_rel_error*100:.4f}%)")
    
    results[r] = {
        'rel_error': rel_error,
        'energy': cumulative_energy[r-1],
        'gamma_n': approx_gamma_n,
        'gamma_c': approx_gamma_c,
        'gamma_n_error': gamma_n_rel_error,
        'gamma_c_error': gamma_c_rel_error,
    }

# =============================================================================
# STEP 4b: SVD-BASED RECONSTRUCTION (verification)
# =============================================================================
log("")
log("=" * 60)
log("STEP 4b: SVD-based reconstruction (verification)")
log("=" * 60)

U, S, _ = np.linalg.svd(Q, full_matrices=False)
svd_energy = np.cumsum(S**2) / np.sum(S**2)

log(f"{'r':>4} | {'||Q-Qr||²/||Q||²':>18} | {'1 - retained':>18} | {'sum(S[r:]²)/sum(S²)':>20} | {'match?':>6}")
log("-" * 75)

for r in R_VALUES:
    Vr_svd = U[:, :r]
    Q_approx_svd = Vr_svd @ (Vr_svd.T @ Q)
    
    recon_err_sq = np.linalg.norm(Q - Q_approx_svd)**2 / np.linalg.norm(Q)**2
    one_minus_energy = 1 - svd_energy[r-1]
    tail_energy = np.sum(S[r:]**2) / np.sum(S**2)
    
    match = np.allclose(recon_err_sq, tail_energy, rtol=1e-10)
    log(f"{r:>4} | {recon_err_sq:>18.10e} | {one_minus_energy:>18.10e} | {tail_energy:>20.10e} | {'✓' if match else '✗':>6}")

log("")
log("Comparing Gram eigendecomp vs SVD energy retention:")
log(f"{'r':>4} | {'Gram energy':>14} | {'SVD energy':>14} | {'diff':>12}")
log("-" * 55)
for r in R_VALUES:
    diff = abs(cumulative_energy[r-1] - svd_energy[r-1])
    log(f"{r:>4} | {cumulative_energy[r-1]*100:>13.6f}% | {svd_energy[r-1]*100:>13.6f}% | {diff:.2e}")

# =============================================================================
# STEP 6: TEST INITIAL CONDITIONS
# =============================================================================
log("")
log("=" * 60)
log("STEP 6: Using POD basis to reconstruct initial condition from test set")
log("Using SVD-based POD for this step")
log("=" * 60)
with xr.open_dataset(TEST_FILE, engine="h5netcdf", phony_dims="sort") as fh:
    # Load density and phi
    test_density = fh["density"].values[:NT_MAX]  # (nt, ny, nx)
    test_phi = fh["phi"].values[:NT_MAX]          # (nt, ny, nx)
    
    # Ground truth gamma values from file
    test_gt_gamma_n = fh["gamma_n"].values[:NT_MAX]
    test_gt_gamma_c = fh["gamma_c"].values[:NT_MAX]

test_nt = test_density.shape[0]
test_ny, test_nx_grid = test_density.shape[1], test_density.shape[2]
log(f"Loaded {test_nt} timesteps, grid size: {test_ny} x {test_nx_grid}")

# Stack into Q_test matrix: shape (2*nx*ny, nt)
# Each column is a flattened state [n_flat; phi_flat]
Q_test = np.vstack([
    test_density.reshape(test_nt, -1).T,  # (nx*ny, nt)
    test_phi.reshape(test_nt, -1).T       # (nx*ny, nt)
])
log(f"Q_test matrix shape: {Q_test.shape}")

test_results = {}

log(f"\n{'r':>4} | {'||Q-Qr||²/||Q||²':>18} | {'1 - retained':>18} | {'sum(S[r:]²)/sum(S²)':>20}")
log("-" * 75)

for r in R_VALUES:
    Vr_svd = U[:, :r]
    Q_approx_svd = Vr_svd @ (Vr_svd.T @ Q_test)
    
    recon_err_sq = np.linalg.norm(Q_test - Q_approx_svd)**2 / np.linalg.norm(Q_test)**2
    one_minus_energy = 1 - svd_energy[r-1]
    tail_energy = np.sum(S[r:]**2) / np.sum(S**2)
    
    log(f"{r:>4} | {recon_err_sq:>18.10e} | {one_minus_energy:>18.10e} | {tail_energy:>20.10e}")

    # Compute Gamma from reconstructed data
    approx_gamma_n = np.zeros(test_nt)
    approx_gamma_c = np.zeros(test_nt)
    
    n_size = test_ny * test_nx_grid
    for t in range(test_nt):
        state = Q_approx_svd[:, t]
        n_approx = state[:n_size].reshape(test_ny, test_nx_grid)
        phi_approx = state[n_size:].reshape(test_ny, test_nx_grid)
        approx_gamma_n[t] = compute_gamma_n(n_approx, phi_approx, dx)
        approx_gamma_c[t] = compute_gamma_c(n_approx, phi_approx, c1)
    
    # Errors in gamma
    gamma_n_rel_error = np.linalg.norm(approx_gamma_n - test_gt_gamma_n) / np.linalg.norm(test_gt_gamma_n)
    gamma_c_rel_error = np.linalg.norm(approx_gamma_c - test_gt_gamma_c) / np.linalg.norm(test_gt_gamma_c)
    
    log(f"  Gamma_n relative error: {gamma_n_rel_error:.6e} ({gamma_n_rel_error*100:.4f}%)")
    log(f"  Gamma_c relative error: {gamma_c_rel_error:.6e} ({gamma_c_rel_error*100:.4f}%)")
    
    test_results[r] = {
        'rel_error': np.sqrt(recon_err_sq),
        'gamma_n': approx_gamma_n,
        'gamma_c': approx_gamma_c,
        'gamma_n_error': gamma_n_rel_error,
        'gamma_c_error': gamma_c_rel_error,
    }

# =============================================================================
# STEP 5: PLOTTING
# =============================================================================
log("")
log("=" * 60)
log("STEP 5: Generating plots")
log("=" * 60)

# Plot 1: Error vs r (training data)
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

r_vals = list(results.keys())
recon_errors = [results[r]['rel_error'] * 100 for r in r_vals]
gamma_n_errors = [results[r]['gamma_n_error'] * 100 for r in r_vals]
gamma_c_errors = [results[r]['gamma_c_error'] * 100 for r in r_vals]
energies = [results[r]['energy'] * 100 for r in r_vals]

axes1[0].semilogy(r_vals, recon_errors, 'o-', linewidth=2, markersize=8)
axes1[0].set_xlabel('Number of modes (r)')
axes1[0].set_ylabel('Reconstruction error (%)')
axes1[0].set_title('State Reconstruction Error')
axes1[0].grid(True, alpha=0.3)

axes1[1].semilogy(r_vals, gamma_n_errors, 'o-', label=r'$\Gamma_n$', linewidth=2, markersize=8)
axes1[1].semilogy(r_vals, gamma_c_errors, 's-', label=r'$\Gamma_c$', linewidth=2, markersize=8)
axes1[1].set_xlabel('Number of modes (r)')
axes1[1].set_ylabel('Relative error (%)')
axes1[1].set_title('Gamma Estimation Error')
axes1[1].legend()
axes1[1].grid(True, alpha=0.3)

axes1[2].plot(r_vals, energies, 'o-', linewidth=2, markersize=8)
axes1[2].set_xlabel('Number of modes (r)')
axes1[2].set_ylabel('Energy retained (%)')
axes1[2].set_title('Cumulative Energy')
axes1[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pod_error_vs_r.png", dpi=150)
log("Saved: pod_error_vs_r.png")

# Plot 2: Time series comparison for select r values (training data)
r_to_plot = [R_VALUES[0], R_VALUES[len(R_VALUES)//2], R_VALUES[-1]]

fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

axes2[0].plot(gt_gamma_n, 'k-', label='Ground truth', linewidth=1.5, alpha=0.8)
for r in r_to_plot:
    axes2[0].plot(results[r]['gamma_n'], '--', label=f'r={r}', alpha=0.8)
axes2[0].set_ylabel(r'$\Gamma_n$')
axes2[0].set_title(r'Particle Flux $\Gamma_n$ (Training Data)')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(gt_gamma_c, 'k-', label='Ground truth', linewidth=1.5, alpha=0.8)
for r in r_to_plot:
    axes2[1].plot(results[r]['gamma_c'], '--', label=f'r={r}', alpha=0.8)
axes2[1].set_ylabel(r'$\Gamma_c$')
axes2[1].set_xlabel('Time step')
axes2[1].set_title(r'Conductive Flux $\Gamma_c$ (Training Data)')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pod_gamma_timeseries.png", dpi=150)
log("Saved: pod_gamma_timeseries.png")

# Plot 3: Error vs r comparison (training vs test)
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

test_recon_errors = [test_results[r]['rel_error'] * 100 for r in r_vals]
test_gamma_n_errors = [test_results[r]['gamma_n_error'] * 100 for r in r_vals]
test_gamma_c_errors = [test_results[r]['gamma_c_error'] * 100 for r in r_vals]

axes3[0].semilogy(r_vals, recon_errors, 'o-', linewidth=2, markersize=8, label='Training')
axes3[0].semilogy(r_vals, test_recon_errors, 's--', linewidth=2, markersize=8, label='Test')
axes3[0].set_xlabel('Number of modes (r)')
axes3[0].set_ylabel('Reconstruction error (%)')
axes3[0].set_title('State Reconstruction Error')
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)

axes3[1].semilogy(r_vals, gamma_n_errors, 'o-', linewidth=2, markersize=8, label=r'$\Gamma_n$ (Train)')
axes3[1].semilogy(r_vals, test_gamma_n_errors, 'o--', linewidth=2, markersize=8, label=r'$\Gamma_n$ (Test)')
axes3[1].semilogy(r_vals, gamma_c_errors, 's-', linewidth=2, markersize=8, label=r'$\Gamma_c$ (Train)')
axes3[1].semilogy(r_vals, test_gamma_c_errors, 's--', linewidth=2, markersize=8, label=r'$\Gamma_c$ (Test)')
axes3[1].set_xlabel('Number of modes (r)')
axes3[1].set_ylabel('Relative error (%)')
axes3[1].set_title('Gamma Estimation Error')
axes3[1].legend()
axes3[1].grid(True, alpha=0.3)

axes3[2].plot(r_vals, energies, 'o-', linewidth=2, markersize=8, label='Energy retained')
axes3[2].set_xlabel('Number of modes (r)')
axes3[2].set_ylabel('Energy retained (%)')
axes3[2].set_title('Cumulative Energy (from Training)')
axes3[2].legend()
axes3[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pod_error_vs_r_comparison.png", dpi=150)
log("Saved: pod_error_vs_r_comparison.png")

# Plot 4: Time series comparison for test data
fig4, axes4 = plt.subplots(2, 1, figsize=(12, 8))

axes4[0].plot(test_gt_gamma_n, 'k-', label='Ground truth', linewidth=1.5, alpha=0.8)
for r in r_to_plot:
    axes4[0].plot(test_results[r]['gamma_n'], '--', label=f'r={r}', alpha=0.8)
axes4[0].set_ylabel(r'$\Gamma_n$')
axes4[0].set_title(r'Particle Flux $\Gamma_n$ (Test Data)')
axes4[0].legend()
axes4[0].grid(True, alpha=0.3)

axes4[1].plot(test_gt_gamma_c, 'k-', label='Ground truth', linewidth=1.5, alpha=0.8)
for r in r_to_plot:
    axes4[1].plot(test_results[r]['gamma_c'], '--', label=f'r={r}', alpha=0.8)
axes4[1].set_ylabel(r'$\Gamma_c$')
axes4[1].set_xlabel('Time step')
axes4[1].set_title(r'Conductive Flux $\Gamma_c$ (Test Data)')
axes4[1].legend()
axes4[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pod_gamma_timeseries_test.png", dpi=150)
log("Saved: pod_gamma_timeseries_test.png")

plt.show()

# =============================================================================
# CLEANUP
# =============================================================================
log("")
log("=" * 60)
log("DONE")
log("=" * 60)
log(f"Full log saved to: {LOG_FILE}")

log.close()
