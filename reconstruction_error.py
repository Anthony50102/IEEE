"""Test POD reconstruction error on a new trajectory."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================================
# DIAGNOSTIC: Check POD basis orthonormality and reconstruction
# ============================================================================
def diagnose_pod_basis(Ur, Q_test, mean, centered):
    """Diagnose POD basis issues."""
    print("\n" + "="*60)
    print("POD BASIS DIAGNOSTICS")
    print("="*60)
    
    # Check orthonormality: Ur.T @ Ur should be identity
    UtU = Ur.T @ Ur
    orthonormality_error = np.linalg.norm(UtU - np.eye(UtU.shape[0]))
    print(f"\n1. Orthonormality check (||Ur.T @ Ur - I||): {orthonormality_error:.6e}")
    print(f"   Diagonal values: min={np.diag(UtU).min():.6f}, max={np.diag(UtU).max():.6f}")
    
    # Check POD mode magnitudes
    mode_norms = np.linalg.norm(Ur, axis=0)
    print(f"\n2. POD mode norms: min={mode_norms.min():.6e}, max={mode_norms.max():.6e}")
    
    # Check mean statistics
    print(f"\n3. Mean statistics:")
    print(f"   Mean array shape: {mean.shape}")
    print(f"   Mean range: [{mean.min():.6e}, {mean.max():.6e}]")
    print(f"   Mean is all zeros: {np.allclose(mean, 0)}")
    
    # Check test data statistics
    print(f"\n4. Test data statistics:")
    print(f"   Q_test range: [{Q_test.min():.6e}, {Q_test.max():.6e}]")
    print(f"   Q_test mean: {Q_test.mean():.6e}")
    
    # Check what happens with reconstruction
    print(f"\n5. Reconstruction analysis:")
    if centered:
        Q_proc = Q_test - mean[:, None]
    else:
        Q_proc = Q_test
    
    # Project and reconstruct WITHOUT adding mean
    Xhat = Ur.T @ Q_proc
    Q_recon_no_mean = Ur @ Xhat
    
    # Project and reconstruct WITH mean
    Q_recon_with_mean = Q_recon_no_mean + mean[:, None]
    
    print(f"   Xhat (reduced coords) range: [{Xhat.min():.6e}, {Xhat.max():.6e}]")
    print(f"   Q_recon (no mean) range: [{Q_recon_no_mean.min():.6e}, {Q_recon_no_mean.max():.6e}]")
    print(f"   Q_recon (with mean) range: [{Q_recon_with_mean.min():.6e}, {Q_recon_with_mean.max():.6e}]")
    
    # Reconstruction error without mean
    err_no_mean = np.linalg.norm(Q_test - Q_recon_no_mean) / np.linalg.norm(Q_test)
    err_with_mean = np.linalg.norm(Q_test - Q_recon_with_mean) / np.linalg.norm(Q_test)
    print(f"\n6. Reconstruction errors:")
    print(f"   Error WITHOUT adding mean: {err_no_mean:.6e} ({err_no_mean*100:.4f}%)")
    print(f"   Error WITH adding mean:    {err_with_mean:.6e} ({err_with_mean*100:.4f}%)")
    
    print("="*60 + "\n")
    
    return Q_recon_no_mean, Q_recon_with_mean

# --- Functions ----
def gradient(padded: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    if axis == 0:
        return (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / (2 * dx)
    elif axis == 1:
        return (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / (2 * dx)
    elif axis == -2:
        return (padded[..., 2:, 1:-1] - padded[..., 0:-2, 1:-1]) / (2 * dx)
    elif axis == -1:
        return (padded[..., 1:-1, 2:] - padded[..., 1:-1, 0:-2]) / (2 * dx)

def periodic_gradient(input_field: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    if axis < 0:
        pad_size = [(0, 0) for _ in range(len(input_field.shape))]
        pad_size[-1] = (1, 1)
        pad_size[-2] = (1, 1)
    else:
        pad_size = 1
    padded = np.pad(input_field, pad_width=pad_size, mode="wrap")
    return gradient(padded, dx, axis=axis)

def get_gamma_n(n: np.ndarray, p: np.ndarray, dx: float, dy_p=None) -> float:
    if dy_p is None:
        dy_p = periodic_gradient(p, dx=dx, axis=-2)  # gradient in y
    gamma_n = -np.mean((n * dy_p), axis=(-1, -2))  # mean over y & x
    return gamma_n


def get_gamma_c(n: np.ndarray, p: np.ndarray, c1: float, dx: float) -> float:
    gamma_c = c1 * np.mean((n - p) ** 2, axis=(-1, -2))  # mean over y & x
    return gamma_c


# === CONFIGURE THESE ===
run_dir = "/scratch2/10407/anthony50102/IEEE/output/20251230_200218_1train_5test"
test_file = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d256x256_striped/hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250315142044_11702_0.h5"
r = 75  # number of modes to use
k0 = .15
c1 = 1
dx = 2 * np.pi / k0
dx = dx / 256

# SET THESE TO MATCH YOUR POD TRAINING CONFIG!
centered = True   # <-- Set to True if POD was trained with centering enabled
scaled = False

# Load saved POD
pod = np.load(f"{run_dir}/POD.npz")
eigs, eigv = pod["eigs"], pod["eigv"]

# Load preprocessing info
ics = np.load(f"{run_dir}/initial_conditions.npz")
mean = ics["train_temporal_mean"]

# Load test trajectory
with xr.open_dataset(test_file, engine="h5netcdf", phony_dims="sort") as fh:
        Q_test = np.vstack([fh["density"].values.reshape(fh["density"].shape[0], -1).T,
            fh["phi"].values.reshape(fh["phi"].shape[0], -1).T])[:,:8000]
        gt_gamma_n = fh["gamma_n"].values[:8000]
        gt_gamma_c = fh["gamma_c"].values[:8000]

# NOTE: Ur (spatial modes) isn't saved by your script - you'd need to add that
# For now, load Xhat_train and recompute or add this line to step_1:
#   np.save(paths["pod_modes"], Ur)  # after computing Ur_local and gathering

Ur = np.load(f"{run_dir}/POD_basis_Ur.npy")[:, :r]

# ============================================================================
# RUN DIAGNOSTICS FIRST
# ============================================================================
Q_recon_no_mean, Q_recon_with_mean = diagnose_pod_basis(Ur, Q_test, mean, centered)

# ============================================================================
# CORRECTED RECONSTRUCTION LOGIC
# ============================================================================
# The reconstruction should depend on how POD was trained:
# - If POD trained on centered data: subtract mean, project, reconstruct, add mean back
# - If POD trained on raw data: just project and reconstruct (NO mean operations)

if centered:
    # POD trained on centered data
    Q_processed = Q_test - mean[:, None]
    Xhat = Ur.T @ Q_processed
    Q_recon = Ur @ Xhat + mean[:, None]
elif scaled:
    # Handle scaling case
    Q_processed = Q_test
    Xhat = Ur.T @ Q_processed
    Q_recon = Ur @ Xhat
else:
    # POD trained on raw data - NO mean operations
    Q_processed = Q_test
    Xhat = Ur.T @ Q_processed
    Q_recon = Ur @ Xhat  # <-- FIXED: removed "+ mean[:, None]"

# Error
rel_error = np.linalg.norm(Q_test - Q_recon) / np.linalg.norm(Q_test)
print(f"Relative reconstruction error: {rel_error:.6e} ({rel_error*100:.4f}%)")

print(Q_recon.shape)

gamma_ns = []
gamma_cs = []
gamma_n_recomps = []
gamma_c_recomps = []
for i in range(Q_recon.shape[1]):
    state = Q_recon[:,i].reshape(2 , 256, 256)
    state_gt = Q_processed[:,i].reshape(2,256,256)
    gamma_n = get_gamma_n(state[0], state[1], dx=dx)
    gamma_c = get_gamma_c(state[0], state[1], c1=c1, dx=dx)
    gamma_n_recomps.append(get_gamma_n(state_gt[0], state_gt[1], dx = dx))
    gamma_c_recomps.append(get_gamma_c(state_gt[0], state_gt[1], c1=c1, dx=dx)) 
    gamma_ns.append(gamma_n)
    gamma_cs.append(gamma_c)

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

ax[0].plot(np.array(gamma_ns), label='Reconstructed', alpha=0.8)
ax[0].plot(gt_gamma_n, label='Ground truth (file)', alpha=0.8)
ax[0].plot(np.array(gamma_n_recomps), label='Recomputed from test', alpha=0.8)
ax[0].set_ylabel('gamma_n')
ax[0].set_title(f'Gamma_n - r={r} modes, centered={centered}')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].plot(np.array(gamma_cs), label='Reconstructed', alpha=0.8)
ax[1].plot(gt_gamma_c, label='Ground truth (file)', alpha=0.8)
ax[1].plot(np.array(gamma_c_recomps), label='Recomputed from test', alpha=0.8)
ax[1].set_ylabel('gamma_c')
ax[1].set_xlabel('Time step')
ax[1].set_title(f'Gamma_c - Reconstruction error: {rel_error*100:.4f}%')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reconstruction_error.png", dpi=150)
print(f"\nSaved reconstruction_error.png")

# Also create a visual comparison of fields at frame 50
frame_idx = 50
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

state_orig = Q_test[:, frame_idx].reshape(2, 256, 256)
state_recon = Q_recon[:, frame_idx].reshape(2, 256, 256)

for field_idx, field_name in enumerate(['density (n)', 'potential (phi)']):
    vmin = state_orig[field_idx].min()
    vmax = state_orig[field_idx].max()
    
    im0 = axes[field_idx, 0].imshow(state_orig[field_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[field_idx, 0].set_title(f'Original {field_name}')
    plt.colorbar(im0, ax=axes[field_idx, 0])
    
    im1 = axes[field_idx, 1].imshow(state_recon[field_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[field_idx, 1].set_title(f'Reconstructed {field_name}')
    plt.colorbar(im1, ax=axes[field_idx, 1])
    
    error = state_orig[field_idx] - state_recon[field_idx]
    im2 = axes[field_idx, 2].imshow(error, cmap='RdBu_r')
    axes[field_idx, 2].set_title(f'Error {field_name}')
    plt.colorbar(im2, ax=axes[field_idx, 2])

plt.suptitle(f'Frame {frame_idx} - r={r} modes, centered={centered}', fontsize=14)
plt.tight_layout()
plt.savefig("reconstruction_fields.png", dpi=150)
print(f"Saved reconstruction_fields.png")

plt.show()
