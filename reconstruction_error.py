"""Test POD reconstruction error on a new trajectory."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
run_dir = "/scratch2/10407/anthony50102/IEEE/output/20251229_144913_1train_5test"
test_file = "/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d256x256_striped/hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250315165458_9483_0.h5"
r = 75  # number of modes to use
k0 = .15
c1 = 1
dx = 2 * np.pi / k0
dx = dx / 256
centered = False
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

# Project and reconstruct
if centered:
    Q_processed = Q_test - mean[:, None]
elif scaled:
    pass
else:
    Q_processed = Q_test

Xhat = Ur.T @ Q_processed
Q_recon = Ur @ Xhat + mean[:, None]

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

fig, ax = plt.subplots(2,1)

ax[0].plot(np.array(gamma_ns))
ax[0].plot(gt_gamma_n)
ax[0].plot(np.array(gamma_n_recomps))
ax[1].plot(np.array(gamma_cs))
ax[1].plot(gt_gamma_c)
ax[1].plot(np.array(gamma_c_recomps))

plt.savefig("reconstruction_error.png")
plt.cfg()

#fig, ax = plt.subplots(2,2)

#state_8000 = Q_recon[:,i].reshape(2 , 256, 256)
#state = Q_processed[:,i].reshape(2,256,256)

#ax[0,0].imshow(
