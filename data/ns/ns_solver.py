"""
2D Incompressible Navier-Stokes Solver (Vorticity-Streamfunction Formulation).

Pseudospectral solver for 2D decaying turbulence on a periodic domain [0, L)^2.

PDE:
    ∂ω/∂t + J(ψ, ω) = ν ∇²ω

    where ∇²ψ = -ω  (Poisson equation for streamfunction)
    J(ψ, ω) = ∂ψ/∂x · ∂ω/∂y − ∂ψ/∂y · ∂ω/∂x  (Jacobian / advection)
    u = ∂ψ/∂y, v = −∂ψ/∂x  (velocity from streamfunction)
    ν = 1/Re  (kinematic viscosity)

Spatial discretization: Fourier pseudospectral with 2/3 dealiasing
Time integration: RK4 (classical 4th-order Runge-Kutta)

Author: Anthony Poole
"""

import numpy as np
import h5py
from typing import Optional, Tuple


# =============================================================================
# SPECTRAL UTILITIES
# =============================================================================

def build_wavenumbers(nx: int, ny: int, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D wavenumber arrays for spectral operations.

    Parameters
    ----------
    nx : int
        Number of grid points in x.
    ny : int
        Number of grid points in y.
    Lx : float
        Domain length in x.
    Ly : float
        Domain length in y.

    Returns
    -------
    kx : np.ndarray, shape (ny, nx)
        Wavenumber grid in x.
    ky : np.ndarray, shape (ny, nx)
        Wavenumber grid in y.
    ksq : np.ndarray, shape (ny, nx)
        |k|^2 = kx^2 + ky^2.
    """
    kx_1d = np.fft.fftfreq(nx, d=Lx / (2 * np.pi * nx))
    ky_1d = np.fft.fftfreq(ny, d=Ly / (2 * np.pi * ny))
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    ksq = kx**2 + ky**2
    return kx, ky, ksq


def build_dealias_mask(nx: int, ny: int) -> np.ndarray:
    """
    Build 2/3-rule dealiasing mask in Fourier space.

    Parameters
    ----------
    nx : int
        Number of grid points in x.
    ny : int
        Number of grid points in y.

    Returns
    -------
    mask : np.ndarray, shape (ny, nx), dtype bool
        True for retained modes, False for aliased modes.
    """
    kx_1d = np.fft.fftfreq(nx, d=1.0 / nx)
    ky_1d = np.fft.fftfreq(ny, d=1.0 / ny)
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    kmax_x = nx // 3
    kmax_y = ny // 3
    mask = (np.abs(kx) <= kmax_x) & (np.abs(ky) <= kmax_y)
    return mask


# =============================================================================
# POISSON SOLVER & SPECTRAL DERIVATIVES
# =============================================================================

def poisson_solve(omega_hat: np.ndarray, ksq: np.ndarray) -> np.ndarray:
    """
    Solve ∇²ψ = -ω in Fourier space: ψ̂ = ω̂ / k².

    The k=0 mode is set to zero (zero-mean streamfunction).

    Parameters
    ----------
    omega_hat : np.ndarray
        Fourier transform of vorticity.
    ksq : np.ndarray
        |k|^2 wavenumber array.

    Returns
    -------
    psi_hat : np.ndarray
        Fourier transform of streamfunction.
    """
    psi_hat = np.zeros_like(omega_hat)
    nonzero = ksq > 0
    psi_hat[nonzero] = omega_hat[nonzero] / ksq[nonzero]
    return psi_hat


# =============================================================================
# RHS COMPUTATION
# =============================================================================

def compute_rhs(
    omega_hat: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    ksq: np.ndarray,
    nu: float,
    dealias_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute the RHS of the vorticity equation in Fourier space.

    ∂ω/∂t = -J(ψ, ω) + ν ∇²ω

    The Jacobian J(ψ, ω) is computed in physical space (pseudospectral)
    with 2/3 dealiasing applied to the product terms.

    Parameters
    ----------
    omega_hat : np.ndarray
        Fourier transform of vorticity.
    kx, ky : np.ndarray
        Wavenumber arrays.
    ksq : np.ndarray
        |k|^2 array.
    nu : float
        Kinematic viscosity (1/Re).
    dealias_mask : np.ndarray
        Dealiasing mask (True = keep).

    Returns
    -------
    rhs_hat : np.ndarray
        RHS in Fourier space.
    """
    # Poisson solve: ψ̂ = ω̂ / k²
    psi_hat = poisson_solve(omega_hat, ksq)

    # Spectral derivatives for velocity and vorticity gradients
    # u = ∂ψ/∂y → û = i·ky·ψ̂
    # v = -∂ψ/∂x → v̂ = -i·kx·ψ̂
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    # ∂ω/∂x, ∂ω/∂y
    domega_dx_hat = 1j * kx * omega_hat
    domega_dy_hat = 1j * ky * omega_hat

    # Transform to physical space
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real
    domega_dx = np.fft.ifft2(domega_dx_hat).real
    domega_dy = np.fft.ifft2(domega_dy_hat).real

    # Nonlinear term: J(ψ, ω) = u · ∂ω/∂x + v · ∂ω/∂y
    # (equivalent to ∂ψ/∂y · ∂ω/∂x − ∂ψ/∂x · ∂ω/∂y)
    jacobian = u * domega_dx + v * domega_dy

    # Dealias the nonlinear product
    jacobian_hat = np.fft.fft2(jacobian)
    jacobian_hat[~dealias_mask] = 0.0

    # Diffusion: ν ∇²ω → -ν k² ω̂
    diffusion_hat = -nu * ksq * omega_hat

    # RHS = -J(ψ, ω) + ν ∇²ω
    rhs_hat = -jacobian_hat + diffusion_hat

    return rhs_hat


# =============================================================================
# TIME INTEGRATION
# =============================================================================

def rk4_step(
    omega_hat: np.ndarray,
    dt: float,
    kx: np.ndarray,
    ky: np.ndarray,
    ksq: np.ndarray,
    nu: float,
    dealias_mask: np.ndarray,
) -> np.ndarray:
    """
    Advance vorticity one timestep using classical RK4.

    Parameters
    ----------
    omega_hat : np.ndarray
        Current vorticity in Fourier space.
    dt : float
        Time step.
    kx, ky, ksq : np.ndarray
        Wavenumber arrays.
    nu : float
        Kinematic viscosity.
    dealias_mask : np.ndarray
        Dealiasing mask.

    Returns
    -------
    omega_hat_new : np.ndarray
        Updated vorticity in Fourier space.
    """
    k1 = compute_rhs(omega_hat, kx, ky, ksq, nu, dealias_mask)
    k2 = compute_rhs(omega_hat + 0.5 * dt * k1, kx, ky, ksq, nu, dealias_mask)
    k3 = compute_rhs(omega_hat + 0.5 * dt * k2, kx, ky, ksq, nu, dealias_mask)
    k4 = compute_rhs(omega_hat + dt * k3, kx, ky, ksq, nu, dealias_mask)

    omega_hat_new = omega_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return omega_hat_new


# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

def random_vorticity_ic(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    k_peak: float = 4.0,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate random vorticity IC with energy concentrated near k_peak.

    The energy spectrum is E(k) ∝ k^4 exp(-2(k/k_peak)^2), which peaks
    at k = k_peak and produces smooth, large-scale vortical structures
    suitable for decaying turbulence studies.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size.
    k_peak : float
        Peak wavenumber for the energy spectrum.
    amplitude : float
        Overall amplitude scaling.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    omega : np.ndarray, shape (ny, nx)
        Initial vorticity field in physical space.
    """
    rng = np.random.default_rng(seed)

    kx, ky, ksq = build_wavenumbers(nx, ny, Lx, Ly)
    k_mag = np.sqrt(ksq)

    # Energy spectrum envelope: E(k) ∝ k^4 exp(-2(k/k_peak)^2)
    envelope = np.where(
        k_mag > 0,
        k_mag**4 * np.exp(-2.0 * (k_mag / k_peak) ** 2),
        0.0,
    )

    # Amplitude per mode: |ω̂(k)| ∝ sqrt(E(k) / k) ∝ k^(3/2) exp(-(k/k_peak)^2)
    # (E(k) ~ k |ω̂|^2 in 2D)
    amp = np.sqrt(envelope / np.where(k_mag > 0, k_mag, 1.0))

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=(ny, nx))
    omega_hat = amp * np.exp(1j * phases)

    # Enforce reality: ω̂(-k) = conj(ω̂(k))
    # Easiest: take real part of ifft2
    omega = np.fft.ifft2(omega_hat).real

    # Normalize to desired amplitude
    omega *= amplitude / (np.std(omega) + 1e-12)

    return omega


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def compute_diagnostics(
    omega: np.ndarray,
    psi: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    dx: float,
    dy: float,
) -> dict:
    """
    Compute kinetic energy and enstrophy from vorticity and streamfunction.

    Parameters
    ----------
    omega : np.ndarray, shape (ny, nx)
        Vorticity field.
    psi : np.ndarray, shape (ny, nx)
        Streamfunction field.
    kx, ky : np.ndarray
        Wavenumber arrays.
    dx, dy : float
        Grid spacings.

    Returns
    -------
    dict
        energy: E = ½⟨|u|²⟩ = ½⟨|∇ψ|²⟩
        enstrophy: Z = ½⟨ω²⟩
    """
    # Velocity from streamfunction in Fourier space
    psi_hat = np.fft.fft2(psi)
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real

    energy = 0.5 * np.mean(u**2 + v**2)
    enstrophy = 0.5 * np.mean(omega**2)

    return {"energy": energy, "enstrophy": enstrophy}


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_ns2d(
    Re: float = 100.0,
    nx: int = 256,
    ny: int = 256,
    Lx: float = 2 * np.pi,
    Ly: Optional[float] = None,
    dt: float = 1e-3,
    n_steps: int = 10000,
    save_every: int = 10,
    k_peak: float = 4.0,
    amplitude: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run 2D Navier-Stokes simulation of decaying turbulence.

    Parameters
    ----------
    Re : float
        Reynolds number (ν = 1/Re).
    nx, ny : int
        Grid resolution.
    Lx : float
        Domain size in x.
    Ly : float, optional
        Domain size in y (defaults to Lx for square domain).
    dt : float
        Time step for RK4.
    n_steps : int
        Total number of time steps.
    save_every : int
        Save every N-th timestep.
    k_peak : float
        Peak wavenumber for IC energy spectrum.
    amplitude : float
        IC amplitude scaling.
    seed : int
        Random seed for IC.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        omega: shape (n_saved, ny, nx) — vorticity snapshots
        psi: shape (n_saved, ny, nx) — streamfunction snapshots
        u: shape (n_saved, ny, nx) — x-velocity snapshots
        v: shape (n_saved, ny, nx) — y-velocity snapshots
        energy: shape (n_saved,) — kinetic energy time series
        enstrophy: shape (n_saved,) — enstrophy time series
        time: shape (n_saved,) — time values
        params: dict of simulation parameters
    """
    if Ly is None:
        Ly = Lx
    nu = 1.0 / Re
    dx = Lx / nx
    dy = Ly / ny

    # Build spectral operators
    kx, ky, ksq = build_wavenumbers(nx, ny, Lx, Ly)
    dealias_mask = build_dealias_mask(nx, ny)

    # Initial condition
    omega = random_vorticity_ic(nx, ny, Lx, Ly, k_peak, amplitude, seed)
    omega_hat = np.fft.fft2(omega)
    omega_hat[~dealias_mask] = 0.0
    omega = np.fft.ifft2(omega_hat).real

    # Storage
    n_saved = n_steps // save_every + 1
    omega_save = np.zeros((n_saved, ny, nx))
    psi_save = np.zeros((n_saved, ny, nx))
    u_save = np.zeros((n_saved, ny, nx))
    v_save = np.zeros((n_saved, ny, nx))
    energy_save = np.zeros(n_saved)
    enstrophy_save = np.zeros(n_saved)
    time_save = np.zeros(n_saved)

    # Save IC
    psi_hat = poisson_solve(omega_hat, ksq)
    psi = np.fft.ifft2(psi_hat).real
    u_hat = 1j * ky * psi_hat
    v_hat_field = -1j * kx * psi_hat
    u_field = np.fft.ifft2(u_hat).real
    v_field = np.fft.ifft2(v_hat_field).real

    diag = compute_diagnostics(omega, psi, kx, ky, dx, dy)
    omega_save[0] = omega
    psi_save[0] = psi
    u_save[0] = u_field
    v_save[0] = v_field
    energy_save[0] = diag["energy"]
    enstrophy_save[0] = diag["enstrophy"]
    time_save[0] = 0.0

    save_idx = 1

    # Time integration
    for step in range(1, n_steps + 1):
        omega_hat = rk4_step(omega_hat, dt, kx, ky, ksq, nu, dealias_mask)

        if step % save_every == 0:
            omega = np.fft.ifft2(omega_hat).real
            psi_hat_cur = poisson_solve(omega_hat, ksq)
            psi = np.fft.ifft2(psi_hat_cur).real
            u_hat_cur = 1j * ky * psi_hat_cur
            v_hat_cur = -1j * kx * psi_hat_cur
            u_field = np.fft.ifft2(u_hat_cur).real
            v_field = np.fft.ifft2(v_hat_cur).real

            diag = compute_diagnostics(omega, psi, kx, ky, dx, dy)
            omega_save[save_idx] = omega
            psi_save[save_idx] = psi
            u_save[save_idx] = u_field
            v_save[save_idx] = v_field
            energy_save[save_idx] = diag["energy"]
            enstrophy_save[save_idx] = diag["enstrophy"]
            time_save[save_idx] = step * dt

            save_idx += 1

            if verbose and step % (save_every * 100) == 0:
                print(
                    f"Step {step:6d}/{n_steps}  "
                    f"t={step * dt:.4f}  "
                    f"E={diag['energy']:.6e}  "
                    f"Z={diag['enstrophy']:.6e}"
                )

    # Trim if n_steps is not exactly divisible by save_every
    omega_save = omega_save[:save_idx]
    psi_save = psi_save[:save_idx]
    u_save = u_save[:save_idx]
    v_save = v_save[:save_idx]
    energy_save = energy_save[:save_idx]
    enstrophy_save = enstrophy_save[:save_idx]
    time_save = time_save[:save_idx]

    params = {
        "Re": Re,
        "nu": nu,
        "nx": nx,
        "ny": ny,
        "Lx": Lx,
        "Ly": Ly,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "dt_save": dt * save_every,
        "n_steps": n_steps,
        "save_every": save_every,
        "k_peak": k_peak,
        "amplitude": amplitude,
        "seed": seed,
    }

    return {
        "omega": omega_save,
        "psi": psi_save,
        "u": u_save,
        "v": v_save,
        "energy": energy_save,
        "enstrophy": enstrophy_save,
        "time": time_save,
        "params": params,
    }


# =============================================================================
# HDF5 I/O
# =============================================================================

def save_to_hdf5(result: dict, file_path: str) -> None:
    """
    Save simulation results to HDF5 file.

    File structure:
        omega:     (n_time, ny, nx)  — vorticity
        psi:       (n_time, ny, nx)  — streamfunction
        u:         (n_time, ny, nx)  — x-velocity
        v:         (n_time, ny, nx)  — y-velocity
        energy:    (n_time,)         — kinetic energy
        enstrophy: (n_time,)         — enstrophy
        time:      (n_time,)         — time values
        Attributes: Re, nu, nx, ny, Lx, Ly, dx, dy, dt, dt_save, seed, ...

    Parameters
    ----------
    result : dict
        Output from solve_ns2d().
    file_path : str
        Output HDF5 file path.
    """
    with h5py.File(file_path, "w") as f:
        f.create_dataset("omega", data=result["omega"], compression="gzip")
        f.create_dataset("psi", data=result["psi"], compression="gzip")
        f.create_dataset("u", data=result["u"], compression="gzip")
        f.create_dataset("v", data=result["v"], compression="gzip")
        f.create_dataset("energy", data=result["energy"])
        f.create_dataset("enstrophy", data=result["enstrophy"])
        f.create_dataset("time", data=result["time"])

        for key, val in result["params"].items():
            f.attrs[key] = val

    print(f"Saved to {file_path}")
    print(f"  Shape: {result['omega'].shape}")
    print(f"  Time range: [{result['time'][0]:.4f}, {result['time'][-1]:.4f}]")
    print(f"  Energy: {result['energy'][0]:.6e} → {result['energy'][-1]:.6e}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="2D Navier-Stokes pseudospectral solver (ω-ψ formulation)"
    )
    parser.add_argument("--Re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--nx", type=int, default=256, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=256, help="Grid points in y")
    parser.add_argument("--Lx", type=float, default=2 * np.pi, help="Domain size")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step")
    parser.add_argument("--n-steps", type=int, default=10000, help="Total steps")
    parser.add_argument("--save-every", type=int, default=10, help="Save interval")
    parser.add_argument("--k-peak", type=float, default=4.0, help="IC peak wavenumber")
    parser.add_argument("--amplitude", type=float, default=1.0, help="IC amplitude")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    result = solve_ns2d(
        Re=args.Re,
        nx=args.nx,
        ny=args.ny,
        Lx=args.Lx,
        dt=args.dt,
        n_steps=args.n_steps,
        save_every=args.save_every,
        k_peak=args.k_peak,
        amplitude=args.amplitude,
        seed=args.seed,
    )

    filename = (
        f"ns2d_re{int(args.Re)}"
        f"_{args.nx}x{args.ny}"
        f"_dt{args.dt}"
        f"_steps{args.n_steps}"
        f"_seed{args.seed}.h5"
    )
    filepath = os.path.join(args.output_dir, filename)
    save_to_hdf5(result, filepath)
