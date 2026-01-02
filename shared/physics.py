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
