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


# =============================================================================
# KS PHYSICS-PRESERVATION METRICS
# =============================================================================

def compute_ks_psd(u: np.ndarray, dx: float) -> tuple:
    """
    Compute time-averaged spatial power spectral density.

    Parameters
    ----------
    u : np.ndarray, shape (n_time, N)
        KS field evolution.
    dx : float
        Grid spacing.

    Returns
    -------
    k : np.ndarray, shape (N//2 + 1,)
        Angular wavenumbers.
    psd : np.ndarray, shape (N//2 + 1,)
        Time-averaged power spectral density |û(k)|².
    """
    N = u.shape[1]
    freqs = np.fft.rfftfreq(N, d=dx)
    k = 2 * np.pi * freqs
    U_hat = np.fft.rfft(u, axis=1)
    psd = np.mean(np.abs(U_hat) ** 2, axis=0)
    return k, psd


def compute_ks_field_pdf(u: np.ndarray, n_bins: int = 100) -> tuple:
    """
    Compute the probability density function (histogram) of field values.

    Approximates the invariant measure of the KS attractor by pooling
    all spatial and temporal samples.

    Parameters
    ----------
    u : np.ndarray, shape (n_time, N)
        KS field evolution.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    bin_centres : np.ndarray, shape (n_bins,)
        Bin centre values.
    density : np.ndarray, shape (n_bins,)
        Normalised probability density.
    """
    counts, bin_edges = np.histogram(u.ravel(), bins=n_bins, density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centres, counts


def compute_ks_spatial_autocorrelation(u: np.ndarray, dx: float) -> tuple:
    """
    Compute time-averaged spatial autocorrelation function C(Δx).

    Uses the Wiener-Khinchin theorem: autocorrelation is the inverse FFT
    of the power spectrum.  Result is normalised so C(0) = 1.

    Parameters
    ----------
    u : np.ndarray, shape (n_time, N)
        KS field evolution.
    dx : float
        Grid spacing.

    Returns
    -------
    lags : np.ndarray, shape (N,)
        Spatial lag values Δx = [0, dx, 2dx, ...].
    C : np.ndarray, shape (N,)
        Normalised autocorrelation (C[0] = 1).
    """
    N = u.shape[1]
    # Zero-mean each snapshot before computing autocorrelation
    u_zm = u - u.mean(axis=1, keepdims=True)
    U_hat = np.fft.fft(u_zm, axis=1)
    power = np.abs(U_hat) ** 2
    acf = np.fft.ifft(power, axis=1).real
    # Time-average, then normalise
    acf_mean = np.mean(acf, axis=0)
    C = acf_mean / acf_mean[0] if acf_mean[0] > 0 else acf_mean
    lags = np.arange(N) * dx
    return lags, C


def compute_ks_energy_rate(
    u: np.ndarray,
    dx: float,
    dt: float,
) -> dict:
    """
    Compute energy budget terms for the KS equation.

    For u_t = -u_xx - u_xxxx - u u_x with periodic BCs the energy
    equation is::

        dE/dt = <u u_xx> + <u u_xxxx>

    (The nonlinear term ∫ u² u_x dx vanishes by periodicity.)

    All spatial derivatives use spectral (FFT) differentiation for
    accuracy.

    Parameters
    ----------
    u : np.ndarray, shape (n_time, N)
        KS field evolution.
    dx : float
        Grid spacing.
    dt : float
        Time step.

    Returns
    -------
    dict
        Keys:

        - ``dEdt_fd`` — dE/dt from finite differences of E(t), shape (n_time - 2,)
        - ``dEdt_pde`` — dE/dt from PDE terms <u u_xx> + <u u_xxxx>, shape (n_time,)
        - ``dissipation`` — <u u_xx> (energy injection from negative diffusion)
        - ``hyperdissipation`` — <u u_xxxx> (energy removal from hyper-diffusion)
        - ``residual`` — dEdt_fd - dEdt_pde[1:-1], shape (n_time - 2,)
    """
    N = u.shape[1]
    L = N * dx

    # Spectral wavenumbers
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # (N,)

    U_hat = np.fft.fft(u, axis=1)  # (n_time, N)

    # u_xx via spectral differentiation
    u_xx_hat = -(k ** 2) * U_hat
    u_xx = np.fft.ifft(u_xx_hat, axis=1).real

    # u_xxxx via spectral differentiation
    u_xxxx_hat = (k ** 4) * U_hat
    u_xxxx = np.fft.ifft(u_xxxx_hat, axis=1).real

    # PDE energy budget: dE/dt = <u * u_xx> + <u * u_xxxx>
    #   <u u_xx>   = negative-diffusion contribution (injection, generally > 0)
    #   <u u_xxxx> = hyper-diffusion contribution (dissipation, generally < 0)
    dissipation = np.mean(u * u_xx, axis=1)
    hyperdissipation = np.mean(u * u_xxxx, axis=1)
    dEdt_pde = dissipation + hyperdissipation

    # Finite-difference dE/dt from E(t) = <u²>/2
    energy = 0.5 * np.mean(u ** 2, axis=1)
    dEdt_fd = (energy[2:] - energy[:-2]) / (2 * dt)

    return {
        "dEdt_fd": dEdt_fd,
        "dEdt_pde": dEdt_pde,
        "dissipation": dissipation,
        "hyperdissipation": hyperdissipation,
        "residual": dEdt_fd - dEdt_pde[1:-1],
    }


def compute_ks_statistical_moments(u: np.ndarray) -> dict:
    """
    Compute spatial statistical moments of the KS field at each timestep.

    Parameters
    ----------
    u : np.ndarray, shape (n_time, N)
        KS field evolution.

    Returns
    -------
    dict
        Keys: ``mean``, ``variance``, ``skewness``, ``kurtosis``,
        each shape (n_time,).  Kurtosis is excess kurtosis (normal = 0).
    """
    mu = np.mean(u, axis=1)
    var = np.var(u, axis=1)
    std = np.sqrt(var)
    std_safe = np.where(std < 1e-12, 1e-12, std)

    u_centered = u - mu[:, None]
    skew = np.mean((u_centered / std_safe[:, None]) ** 3, axis=1)
    kurt = np.mean((u_centered / std_safe[:, None]) ** 4, axis=1) - 3.0

    return {
        "mean": mu,
        "variance": var,
        "skewness": skew,
        "kurtosis": kurt,
    }


# =============================================================================
# 2D NAVIER-STOKES PHYSICS (ω-ψ FORMULATION)
# =============================================================================

def get_ns_grid_params(
    Re: float = 100.0,
    nx: int = 256,
    ny: int = 256,
    Lx: float = 2 * np.pi,
    Ly: float = None,
) -> dict:
    """
    Get standard 2D NS grid parameters.

    Parameters
    ----------
    Re : float
        Reynolds number.
    nx, ny : int
        Number of grid points.
    Lx : float
        Domain length in x.
    Ly : float, optional
        Domain length in y (defaults to Lx).

    Returns
    -------
    dict
        Dictionary with Re, nu, Lx, Ly, dx, dy, nx, ny.
    """
    if Ly is None:
        Ly = Lx
    dx = Lx / nx
    dy = Ly / ny
    return {
        'Re': Re,
        'nu': 1.0 / Re,
        'Lx': Lx,
        'Ly': Ly,
        'dx': dx,
        'dy': dy,
        'nx': nx,
        'ny': ny,
    }


def compute_ns_enstrophy(omega: np.ndarray) -> float:
    """
    Compute enstrophy from vorticity: Z = ½⟨ω²⟩.

    Parameters
    ----------
    omega : np.ndarray, shape (ny, nx)
        Vorticity field at a single timestep.

    Returns
    -------
    float
        Scalar enstrophy value.
    """
    return 0.5 * np.mean(omega ** 2)


def compute_ns_kinetic_energy(omega: np.ndarray, Lx: float, Ly: float = None) -> float:
    """
    Compute kinetic energy from vorticity via Poisson solve in Fourier space.

    E = ½⟨|u|²⟩ = ½⟨|∇ψ|²⟩, where ∇²ψ = -ω.

    Parameters
    ----------
    omega : np.ndarray, shape (ny, nx)
        Vorticity field at a single timestep.
    Lx : float
        Domain length in x.
    Ly : float, optional
        Domain length in y (defaults to Lx).

    Returns
    -------
    float
        Scalar kinetic energy value.
    """
    if Ly is None:
        Ly = Lx
    ny, nx = omega.shape

    # Wavenumber arrays
    kx_1d = np.fft.fftfreq(nx, d=Lx / (2 * np.pi * nx))
    ky_1d = np.fft.fftfreq(ny, d=Ly / (2 * np.pi * ny))
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    ksq = kx**2 + ky**2

    # Poisson solve: ψ̂ = ω̂ / k²
    omega_hat = np.fft.fft2(omega)
    psi_hat = np.zeros_like(omega_hat)
    nonzero = ksq > 0
    psi_hat[nonzero] = omega_hat[nonzero] / ksq[nonzero]

    # Velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real

    return 0.5 * np.mean(u**2 + v**2)


def compute_ns_qoi_timeseries(
    omega: np.ndarray,
    Lx: float,
    Ly: float = None,
) -> tuple:
    """
    Compute energy and enstrophy time series for 2D NS data.

    Parameters
    ----------
    omega : np.ndarray, shape (n_time, ny, nx)
        Vorticity field evolution.
    Lx : float
        Domain length in x.
    Ly : float, optional
        Domain length in y (defaults to Lx).

    Returns
    -------
    tuple
        (energy, enstrophy), each shape (n_time,).
    """
    n_time = omega.shape[0]
    energy = np.zeros(n_time)
    enstrophy = np.zeros(n_time)

    for t in range(n_time):
        energy[t] = compute_ns_kinetic_energy(omega[t], Lx, Ly)
        enstrophy[t] = compute_ns_enstrophy(omega[t])

    return energy, enstrophy


def compute_ns_qoi_from_state_vector(
    state: np.ndarray,
    ny: int,
    nx: int,
    Lx: float,
    Ly: float = None,
) -> tuple:
    """
    Compute NS energy and enstrophy from a flattened state vector.

    The state vector is assumed to be [omega_flat].

    Parameters
    ----------
    state : np.ndarray, shape (ny*nx,) or (ny*nx, n_time)
        Flattened state vector(s).
    ny, nx : int
        Grid dimensions.
    Lx : float
        Domain length in x.
    Ly : float, optional
        Domain length in y (defaults to Lx).

    Returns
    -------
    tuple
        (energy, enstrophy) — scalars if input is 1D, arrays if 2D.
    """
    n_spatial = ny * nx
    if state.ndim == 1:
        omega = state[:n_spatial].reshape(ny, nx)
        return (
            compute_ns_kinetic_energy(omega, Lx, Ly),
            compute_ns_enstrophy(omega),
        )
    else:
        # state is (ny*nx, n_time)
        omega_series = state[:n_spatial, :].T.reshape(-1, ny, nx)
        return compute_ns_qoi_timeseries(omega_series, Lx, Ly)


def compute_ns_energy_spectrum(omega: np.ndarray, Lx: float, Ly: float = None) -> tuple:
    """
    Compute isotropic kinetic energy spectrum E(k) from vorticity.

    The spectrum is shell-averaged: E(k) = Σ_{|k'| ∈ [k-Δk/2, k+Δk/2)} ½|û(k')|².

    Parameters
    ----------
    omega : np.ndarray, shape (ny, nx) or (n_time, ny, nx)
        Vorticity field(s). If 3D, time-averaged spectrum is returned.
    Lx : float
        Domain length in x.
    Ly : float, optional
        Domain length in y (defaults to Lx).

    Returns
    -------
    k_bins : np.ndarray
        Wavenumber bin centres.
    E_k : np.ndarray
        Energy spectrum E(k).
    """
    if Ly is None:
        Ly = Lx

    if omega.ndim == 2:
        omega = omega[np.newaxis, ...]

    n_time, ny, nx = omega.shape

    kx_1d = np.fft.fftfreq(nx, d=Lx / (2 * np.pi * nx))
    ky_1d = np.fft.fftfreq(ny, d=Ly / (2 * np.pi * ny))
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    ksq = kx**2 + ky**2
    k_mag = np.sqrt(ksq)

    # Velocity spectrum from vorticity: |û|² = |ω̂|²/k² (for k≠0)
    dk = max(2 * np.pi / Lx, 2 * np.pi / Ly)
    k_max = np.sqrt((np.pi * nx / Lx)**2 + (np.pi * ny / Ly)**2)
    k_bins = np.arange(dk, k_max, dk)

    E_k = np.zeros(len(k_bins))

    for t in range(n_time):
        omega_hat = np.fft.fft2(omega[t])
        vel_sq_hat = np.zeros_like(ksq)
        nonzero = ksq > 0
        vel_sq_hat[nonzero] = np.abs(omega_hat[nonzero])**2 / ksq[nonzero]

        for i, k_c in enumerate(k_bins):
            shell = (k_mag >= k_c - dk / 2) & (k_mag < k_c + dk / 2)
            E_k[i] += 0.5 * np.sum(vel_sq_hat[shell]) / (nx * ny)**2

    E_k /= n_time

    return k_bins, E_k


def compute_ns_statistical_moments(omega: np.ndarray) -> dict:
    """
    Compute spatial statistical moments of vorticity at each timestep.

    Parameters
    ----------
    omega : np.ndarray, shape (n_time, ny, nx)
        Vorticity field evolution.

    Returns
    -------
    dict
        Keys: mean, variance, skewness, kurtosis, each shape (n_time,).
        Kurtosis is excess kurtosis (normal = 0).
    """
    # Flatten spatial dims
    n_time = omega.shape[0]
    omega_flat = omega.reshape(n_time, -1)

    mu = np.mean(omega_flat, axis=1)
    var = np.var(omega_flat, axis=1)
    std = np.sqrt(var)
    std_safe = np.where(std < 1e-12, 1e-12, std)

    centered = omega_flat - mu[:, None]
    skew = np.mean((centered / std_safe[:, None]) ** 3, axis=1)
    kurt = np.mean((centered / std_safe[:, None]) ** 4, axis=1) - 3.0

    return {
        "mean": mu,
        "variance": var,
        "skewness": skew,
        "kurtosis": kurt,
    }
