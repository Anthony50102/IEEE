"""
Simplified POD Reconstruction Analysis
======================================
Bare-bones implementation following the workflow:
1. Load data (Q)
2. Compute Gamma_n, Gamma_c and verify against saved values
3. Compute POD basis (Vr)
4. Compute Q_approx = Vr @ Vr.T @ Q for different r values
5. Estimate Gamma_n, Gamma_c from Q_approx
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d256x256_striped/hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250315142044_11702_0.h5" 

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
# HELPER FUNCTIONS: Gradient computations
# =============================================================================
def periodic_gradient(field, dx, axis):
    """Compute gradient with periodic boundary conditions."""
    if axis == -1:  # x-direction
        padded = np.pad(field, ((0, 0), (1, 1)), mode='wrap')
        return (padded[:, 2:] - padded[:, :-2]) / (2 * dx)
    elif axis == -2:  # y-direction
        padded = np.pad(field, ((1, 1), (0, 0)), mode='wrap')
        return (padded[2:, :] - padded[:-2, :]) / (2 * dx)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def compute_gamma_n(n, phi, dx):
    """Compute Gamma_n = -<n * d(phi)/dy>."""
    dphi_dy = periodic_gradient(phi, dx, axis=-2)
    return -np.mean(n * dphi_dy)


def compute_gamma_c(n, phi, c1):
    """Compute flux Gamma_c = c1 * <(n - phi)^2>."""
    return c1 * np.mean((n - phi) ** 2)


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

with xr.open_dataset(DATA_FILE, engine="h5netcdf", phony_dims="sort") as fh:
    # Load density and phi, reshape to (nx*ny, nt) and stack
    density = fh["density"].values[:NT_MAX]  # (nt, ny, nx)
    phi = fh["phi"].values[:NT_MAX]          # (nt, ny, nx)
    
    # Ground truth gamma values from file
    gt_gamma_n = fh["gamma_n"].values[:NT_MAX]
    gt_gamma_c = fh["gamma_c"].values[:NT_MAX]

nt = density.shape[0]
ny, nx_grid = density.shape[1], density.shape[2]
print(f"Loaded {nt} timesteps, grid size: {ny} x {nx_grid}")

# Stack into Q matrix: shape (2*nx*ny, nt)
# Each column is a flattened state [n_flat; phi_flat]
Q = np.vstack([
    density.reshape(nt, -1).T,  # (nx*ny, nt)
    phi.reshape(nt, -1).T       # (nx*ny, nt)
])
print(f"Q matrix shape: {Q.shape}")

# =============================================================================
# STEP 2: VERIFY GAMMA COMPUTATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Verifying Gamma computations")
print("=" * 60)

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

print(f"Max error in Gamma_n: {gamma_n_error:.6e}")
print(f"Max error in Gamma_c: {gamma_c_error:.6e}")

if gamma_n_error < 1e-10 and gamma_c_error < 1e-10:
    print("Gamma computations match saved values!")
else:
    print("Warning: Gamma computations differ from saved values")

# =============================================================================
# STEP 3: COMPUTE POD BASIS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Computing POD basis")
print("=" * 60)

# Method: SVD of Q directly, or eigendecomposition of Q^T Q (Gram matrix)
# Using Gram matrix approach (more efficient when nx*ny >> nt)

D = Q.T @ Q  # Gram matrix, shape (nt, nt)
print(f"Gram matrix shape: {D.shape}")

# Eigendecomposition
eigs, eigv = np.linalg.eigh(D)

# Sort by decreasing eigenvalue
sorted_idx = np.argsort(eigs)[::-1]
eigs = eigs[sorted_idx]
eigv = eigv[:, sorted_idx]

# Compute cumulative energy
total_energy = np.sum(eigs)
cumulative_energy = np.cumsum(eigs) / total_energy

print(f"Total energy (sum of eigenvalues): {total_energy:.6e}")
print(f"Energy captured by first 10 modes: {cumulative_energy[9]*100:.2f}%")
print(f"Energy captured by first 50 modes: {cumulative_energy[49]*100:.2f}%")
print(f"Energy captured by first 100 modes: {cumulative_energy[99]*100:.2f}%")

# =============================================================================
# STEP 4: RECONSTRUCTION FOR DIFFERENT r VALUES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Reconstruction analysis for different r values")
print("=" * 60)

results = {}

for r in R_VALUES:
    print(f"\n--- r = {r} modes ---")
    
    # Compute POD basis Vr
    # Vr = Q @ eigv[:, :r] @ diag(1/sqrt(eigs[:r]))
    Tr = eigv[:, :r] @ np.diag(eigs[:r] ** (-0.5))
    Vr = Q @ Tr  # shape (2*nx*ny, r)
    
    # Verify orthonormality: Vr.T @ Vr should be identity
    VtV = Vr.T @ Vr
    ortho_error = np.linalg.norm(VtV - np.eye(r))
    print(f"  Orthonormality error: {ortho_error:.6e}")
    
    # Reconstruct: Q_approx = Vr @ Vr.T @ Q
    Q_approx = Vr @ (Vr.T @ Q)
    
    # Reconstruction error
    rel_error = np.linalg.norm(Q - Q_approx) / np.linalg.norm(Q)
    print(f"  Relative reconstruction error: {rel_error:.6e} ({rel_error*100:.4f}%)")
    print(f"  Energy retained: {cumulative_energy[r-1]*100:.4f}%")
    
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
    
    print(f"  Gamma_n relative error: {gamma_n_rel_error:.6e} ({gamma_n_rel_error*100:.4f}%)")
    print(f"  Gamma_c relative error: {gamma_c_rel_error:.6e} ({gamma_c_rel_error*100:.4f}%)")
    
    results[r] = {
        'rel_error': rel_error,
        'energy': cumulative_energy[r-1],
        'gamma_n': approx_gamma_n,
        'gamma_c': approx_gamma_c,
        'gamma_n_error': gamma_n_rel_error,
        'gamma_c_error': gamma_c_rel_error,
    }

# =============================================================================
# STEP 5: PLOTTING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Generating plots")
print("=" * 60)

# Plot 1: Error vs r
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

axes1[1].semilogy(r_vals, gamma_n_errors, 'o-', label='Gamma_n', linewidth=2, markersize=8)
axes1[1].semilogy(r_vals, gamma_c_errors, 's-', label='Gamma_c', linewidth=2, markersize=8)
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
print("Saved: pod_error_vs_r.png")

# Plot 2: Time series comparison for select r values
r_to_plot = [R_VALUES[0], R_VALUES[len(R_VALUES)//2], R_VALUES[-1]]

fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

axes2[0].plot(gt_gamma_n, 'k-', label='Ground truth', linewidth=1.5, alpha=0.8)
for r in r_to_plot:
    axes2[0].plot(results[r]['gamma_n'], '--', label=f'r={r}', alpha=0.8)
axes2[0].set_ylabel('Gamma_n')
axes2[0].set_title('Gamma_n')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(gt_gamma_c, 'k-', label='Ground truth', linewidth=1.5, alpha=0.8)
for r in r_to_plot:
    axes2[1].plot(results[r]['gamma_c'], '--', label=f'r={r}', alpha=0.8)
axes2[1].set_ylabel('Gamma_c')
axes2[1].set_xlabel('Time step')
axes2[1].set_title('Gamma_c')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pod_gamma_timeseries.png", dpi=150)
print("Saved: pod_gamma_timeseries.png")

plt.show()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
