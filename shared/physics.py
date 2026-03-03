"""
Physical quantity computations for Hasegawa-Wakatani simulations.

Provides functions for computing:
- Gamma_n (particle flux)
- Gamma_c (conductive flux)
- Gradient operators with periodic boundaries

Author: Anthony Poole
"""

import numpy as np


# =============================================================================
# GRADIENT OPERATORS
# =============================================================================

def periodic_gradient(field: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """
    Compute gradient with periodic boundary conditions using central differences.
    
    Args:
        field: 2D array of shape (n_y, n_x)
        dx: Grid spacing
        axis: -1 for x-direction, -2 for y-direction
    
    Returns:
        Gradient array of same shape as input
    """
    if axis == -1:  # x-direction
        padded = np.pad(field, ((0, 0), (1, 1)), mode='wrap')
        return (padded[:, 2:] - padded[:, :-2]) / (2 * dx)
    elif axis == -2:  # y-direction
        padded = np.pad(field, ((1, 1), (0, 0)), mode='wrap')
        return (padded[2:, :] - padded[:-2, :]) / (2 * dx)
    else:
        raise ValueError(f"Unsupported axis: {axis}. Use -1 (x) or -2 (y).")


def periodic_gradient_vectorized(field: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """
    Compute gradient for a batch of 2D fields with periodic boundary conditions.
    
    Args:
        field: 3D array of shape (n_time, n_y, n_x)
        dx: Grid spacing
        axis: -1 for x-direction, -2 for y-direction
    
    Returns:
        Gradient array of same shape as input
    """
    if axis == -1:  # x-direction
        padded = np.pad(field, ((0, 0), (0, 0), (1, 1)), mode='wrap')
        return (padded[:, :, 2:] - padded[:, :, :-2]) / (2 * dx)
    elif axis == -2:  # y-direction
        padded = np.pad(field, ((0, 0), (1, 1), (0, 0)), mode='wrap')
        return (padded[:, 2:, :] - padded[:, :-2, :]) / (2 * dx)
    else:
        raise ValueError(f"Unsupported axis: {axis}. Use -1 (x) or -2 (y).")


# =============================================================================
# FLUX COMPUTATIONS
# =============================================================================

def compute_gamma_n(n: np.ndarray, phi: np.ndarray, dx: float) -> float:
    """
    Compute particle flux Gamma_n = -<n * d(phi)/dy>.
    
    Args:
        n: Density field, shape (n_y, n_x)
        phi: Potential field, shape (n_y, n_x)
        dx: Grid spacing
    
    Returns:
        Scalar particle flux value
    """
    dphi_dy = periodic_gradient(phi, dx, axis=-2)
    return -np.mean(n * dphi_dy)


def compute_gamma_c(n: np.ndarray, phi: np.ndarray, c1: float = 1.0) -> float:
    """
    Compute conductive flux Gamma_c = c1 * <(n - phi)^2>.
    
    Args:
        n: Density field, shape (n_y, n_x)
        phi: Potential field, shape (n_y, n_x)
        c1: Adiabaticity parameter (default 1.0)
    
    Returns:
        Scalar conductive flux value
    """
    return c1 * np.mean((n - phi) ** 2)


def compute_gamma_timeseries(
    density: np.ndarray, 
    phi: np.ndarray, 
    dx: float, 
    c1: float = 1.0
) -> tuple:
    """
    Compute Gamma_n and Gamma_c time series from full simulation data.
    
    Args:
        density: Density field, shape (n_time, n_y, n_x)
        phi: Potential field, shape (n_time, n_y, n_x)
        dx: Grid spacing
        c1: Adiabaticity parameter
    
    Returns:
        Tuple of (gamma_n, gamma_c), each shape (n_time,)
    """
    n_time = density.shape[0]
    gamma_n = np.zeros(n_time)
    gamma_c = np.zeros(n_time)
    
    for t in range(n_time):
        gamma_n[t] = compute_gamma_n(density[t], phi[t], dx)
        gamma_c[t] = compute_gamma_c(density[t], phi[t], c1)
    
    return gamma_n, gamma_c


def compute_gamma_from_state_vector(
    state: np.ndarray, 
    n_y: int, 
    n_x: int, 
    dx: float, 
    c1: float = 1.0
) -> tuple:
    """
    Compute Gamma_n and Gamma_c from a flattened state vector.
    
    The state vector is assumed to be [n_flat; phi_flat].
    
    Args:
        state: Flattened state vector, shape (2 * n_y * n_x,)
        n_y: Number of grid points in y
        n_x: Number of grid points in x
        dx: Grid spacing
        c1: Adiabaticity parameter
    
    Returns:
        Tuple of (gamma_n, gamma_c) scalar values
    """
    n_spatial = n_y * n_x
    n = state[:n_spatial].reshape(n_y, n_x)
    phi = state[n_spatial:].reshape(n_y, n_x)
    
    return compute_gamma_n(n, phi, dx), compute_gamma_c(n, phi, c1)


# =============================================================================
# GRID UTILITIES
# =============================================================================

def get_hw2d_grid_params(k0: float = 0.15, nx: int = 256) -> dict:
    """
    Get standard HW2D grid parameters.
    
    Args:
        k0: Fundamental wavenumber
        nx: Number of grid points (assumes square grid)
    
    Returns:
        Dictionary with Lx, dx, k0, nx, ny
    """
    Lx = 2 * np.pi / k0
    dx = Lx / nx
    
    return {
        'Lx': Lx,
        'Ly': Lx,  # Square domain
        'dx': dx,
        'dy': dx,
        'k0': k0,
        'nx': nx,
        'ny': nx,
    }


# =============================================================================
# KURAMOTO-SIVASHINSKY PHYSICS
# =============================================================================

def get_ks_grid_params(L: float = 100.0, N: int = 200) -> dict:
    """
    Get standard KS grid parameters.

    Parameters
    ----------
    L : float
        Domain length.
    N : int
        Number of spatial grid points.

    Returns
    -------
    dict
        Dictionary with L, dx, N.
    """
    dx = L / N
    return {
        'L': L,
        'dx': dx,
        'N': N,
    }


def compute_ks_energy(u: np.ndarray) -> float:
    """
    Compute spatial-mean kinetic energy for KS: E = <u^2> / 2.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        KS field at a single timestep.

    Returns
    -------
    float
        Scalar energy value.
    """
    return 0.5 * np.mean(u ** 2)


def compute_ks_enstrophy(u: np.ndarray, dx: float) -> float:
    """
    Compute spatial-mean enstrophy (energy production) for KS: P = <u_x^2>.

    Uses central differences with periodic boundary conditions.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        KS field at a single timestep.
    dx : float
        Grid spacing.

    Returns
    -------
    float
        Scalar enstrophy value.
    """
    u_padded = np.pad(u, (1, 1), mode='wrap')
    u_x = (u_padded[2:] - u_padded[:-2]) / (2 * dx)
    return np.mean(u_x ** 2)


def compute_ks_qoi_timeseries(u: np.ndarray, dx: float) -> tuple:
    """
    Compute energy and enstrophy time series for KS data.

    Parameters
    ----------
    u : np.ndarray, shape (n_time, N)
        KS field evolution.
    dx : float
        Grid spacing.

    Returns
    -------
    tuple
        (energy, enstrophy), each shape (n_time,).
    """
    energy = 0.5 * np.mean(u ** 2, axis=-1)

    u_padded = np.pad(u, ((0, 0), (1, 1)), mode='wrap')
    u_x = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * dx)
    enstrophy = np.mean(u_x ** 2, axis=-1)

    return energy, enstrophy


def compute_ks_qoi_from_state_vector(
    state: np.ndarray,
    N: int,
    dx: float,
) -> tuple:
    """
    Compute KS energy and enstrophy from a flattened state vector.

    The state vector is assumed to be [u_flat].

    Parameters
    ----------
    state : np.ndarray, shape (N,) or (N, n_time)
        Flattened state vector(s).
    N : int
        Number of spatial points.
    dx : float
        Grid spacing.

    Returns
    -------
    tuple
        (energy, enstrophy) — scalars if input is 1D, arrays if 2D.
    """
    if state.ndim == 1:
        u = state[:N]
        return compute_ks_energy(u), compute_ks_enstrophy(u, dx)
    else:
        # state is (N, n_time), transpose to (n_time, N) for vectorized computation
        u = state[:N, :].T
        return compute_ks_qoi_timeseries(u, dx)
