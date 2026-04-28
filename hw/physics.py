"""HW physics: fluxes, gradients, grid utilities.

Author: Anthony Poole

This is the *canonical* implementation of Gamma_n and Gamma_c for our paper.
Every method that reports these QoIs (B1 OpInf, B2 affine OpInf, B4 ROM,
DISCO-lite) routes through it. The implementation is a verbatim port of the
HW-relevant functions from the prior `shared/physics.py` (now in archive
context); the rest of `shared/physics.py` (KS, NS) is dead code per the
refactor.

Conventions:
  - Fields are 2D numpy arrays of shape (n_y, n_x). Periodic in both axes.
  - Time series of fields are 3D, shape (n_t, n_y, n_x).
  - `dx == dy` in our setup (square uniform grid).
  - Spatial average <.> is `np.mean(.)` over the grid axes; the unit-box
    convention of hw2d is preserved (Lx = Ly = 2*pi/k0 normalized to 1).

Related:
  - Reference table: `hw/reference_gammas.yaml`
  - hw2d-native equivalents (for cross-check):
    `hw2d.physical_properties.numpy_properties.get_gamma_n / get_gamma_c`.
"""

from __future__ import annotations

import numpy as np


# === gradients ===

def periodic_gradient(field: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Central-difference gradient with periodic BCs.

    Parameters
    ----------
    field : np.ndarray
        2D array, shape (n_y, n_x).
    dx : float
        Grid spacing (assumed isotropic).
    axis : int
        -1 for d/dx, -2 for d/dy.
    """
    if axis == -1:
        padded = np.pad(field, ((0, 0), (1, 1)), mode="wrap")
        return (padded[:, 2:] - padded[:, :-2]) / (2 * dx)
    if axis == -2:
        padded = np.pad(field, ((1, 1), (0, 0)), mode="wrap")
        return (padded[2:, :] - padded[:-2, :]) / (2 * dx)
    raise ValueError(f"Unsupported axis {axis}; use -1 (x) or -2 (y).")


def periodic_gradient_batch(field: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Same as :func:`periodic_gradient` but for batched 3D arrays (n_t, n_y, n_x)."""
    if axis == -1:
        padded = np.pad(field, ((0, 0), (0, 0), (1, 1)), mode="wrap")
        return (padded[:, :, 2:] - padded[:, :, :-2]) / (2 * dx)
    if axis == -2:
        padded = np.pad(field, ((0, 0), (1, 1), (0, 0)), mode="wrap")
        return (padded[:, 2:, :] - padded[:, :-2, :]) / (2 * dx)
    raise ValueError(f"Unsupported axis {axis}; use -1 (x) or -2 (y).")


# === fluxes (single snapshot) ===

def gamma_n(n: np.ndarray, phi: np.ndarray, dx: float) -> float:
    """Particle flux Gamma_n = -<n * d(phi)/dy> for a single snapshot."""
    dphi_dy = periodic_gradient(phi, dx, axis=-2)
    return float(-np.mean(n * dphi_dy))


def gamma_c(n: np.ndarray, phi: np.ndarray, c1: float = 1.0) -> float:
    """Conductive flux Gamma_c = c1 * <(n - phi)^2> for a single snapshot."""
    return float(c1 * np.mean((n - phi) ** 2))


# === fluxes (time series) ===

def gamma_timeseries(
    density: np.ndarray,
    phi: np.ndarray,
    dx: float,
    c1: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (gamma_n, gamma_c) at every timestep.

    Parameters
    ----------
    density, phi : np.ndarray
        Shape (n_t, n_y, n_x).
    dx : float
        Grid spacing.
    c1 : float
        Adiabaticity parameter alpha (= c1 in hw2d).

    Returns
    -------
    g_n, g_c : np.ndarray
        Each of shape (n_t,).
    """
    dphi_dy = periodic_gradient_batch(phi, dx, axis=-2)
    g_n = -np.mean(density * dphi_dy, axis=(-2, -1))
    g_c = c1 * np.mean((density - phi) ** 2, axis=(-2, -1))
    return g_n, g_c


# === grid utilities ===

def grid(k0: float = 0.15, nx: int = 256) -> dict:
    """Standard HW2D grid parameters (square periodic box).

    Returns
    -------
    dict
        Keys: Lx, Ly, dx, dy, k0, nx, ny.
    """
    Lx = 2 * np.pi / k0
    dx = Lx / nx
    return {"Lx": Lx, "Ly": Lx, "dx": dx, "dy": dx, "k0": k0, "nx": nx, "ny": nx}


# === sanity check ===

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g = grid(k0=0.15, nx=128)
    n_field = rng.standard_normal((g["ny"], g["nx"]))
    phi_field = rng.standard_normal((g["ny"], g["nx"]))
    gn = gamma_n(n_field, phi_field, g["dx"])
    gc = gamma_c(n_field, phi_field, c1=1.0)
    print(f"random fields: gamma_n = {gn:+.6e}, gamma_c = {gc:+.6e}")

    n_const = np.ones((g["ny"], g["nx"]))
    phi_const = np.ones((g["ny"], g["nx"]))
    assert abs(gamma_n(n_const, phi_const, g["dx"])) < 1e-12
    assert abs(gamma_c(n_const, phi_const, c1=1.0)) < 1e-12
    print("constant-field sanity check passed.")
