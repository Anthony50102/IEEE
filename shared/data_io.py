"""
Data I/O utilities for HW2D simulations.

Provides functions for loading HDF5 simulation data in various formats:
- Single snapshots
- Full time series
- Distributed loading across MPI ranks

Author: Anthony Poole
"""

import os
import numpy as np
import h5py
import xarray as xr
from typing import Optional, Tuple, List


# =============================================================================
# BASIC DATA LOADING
# =============================================================================

def load_dataset(path: str, engine: str = "h5netcdf"):
    """
    Load xarray dataset from HDF5 file.
    
    Args:
        path: Path to HDF5 file
        engine: xarray engine ('h5netcdf' or 'netcdf4')
    
    Returns:
        xarray.Dataset
    """
    try:
        return xr.open_dataset(path, engine=engine)
    except Exception:
        return xr.open_dataset(path, engine=engine, phony_dims="sort")


def get_dt_from_file(file_path: str, default: float = 0.025) -> float:
    """
    Extract time step (dt) from HDF5 file attributes.
    
    Args:
        file_path: Path to HDF5 file
        default: Default dt if not found
    
    Returns:
        Time step value
    """
    try:
        with h5py.File(file_path, 'r') as f:
            for loc in [f.attrs, f]:
                if 'dt' in loc:
                    return float(f['dt'][()] if 'dt' in f else f.attrs['dt'])
            for grp_name in ['params', 'metadata', 'parameters']:
                if grp_name in f:
                    grp = f[grp_name]
                    if 'dt' in grp.attrs:
                        return float(grp.attrs['dt'])
                    if 'dt' in grp:
                        return float(grp['dt'][()])
    except Exception:
        pass
    return default


def get_file_metadata(file_path: str, engine: str = "h5netcdf") -> dict:
    """
    Get metadata from simulation file without loading full data.
    
    Args:
        file_path: Path to HDF5 file
        engine: xarray engine
    
    Returns:
        Dictionary with n_time, n_y, n_x, n_spatial, dt
    """
    with load_dataset(file_path, engine) as fh:
        n_time = fh["density"].shape[0]
        if fh["density"].ndim == 3:
            n_y, n_x = fh["density"].shape[1], fh["density"].shape[2]
        else:
            n_y = n_x = int(np.sqrt(fh["density"].shape[1]))
    
    dt = get_dt_from_file(file_path)
    
    return {
        'n_time': n_time,
        'n_y': n_y,
        'n_x': n_x,
        'n_spatial': 2 * n_y * n_x,  # density + phi
        'dt': dt,
    }


# =============================================================================
# SNAPSHOT LOADING
# =============================================================================

def load_hw2d_snapshot(file_path: str, timestep: int, engine: str = "h5netcdf") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single timestep from HW2D simulation.
    
    Args:
        file_path: Path to HDF5 file
        timestep: Time index to load
        engine: xarray engine
    
    Returns:
        Tuple of (density, phi) arrays, each shape (n_y, n_x)
    """
    with load_dataset(file_path, engine) as fh:
        density = fh["density"].values[timestep]
        phi = fh["phi"].values[timestep]
    
    # Reshape if flattened
    if density.ndim == 1:
        grid_size = int(np.sqrt(density.shape[0]))
        density = density.reshape(grid_size, grid_size)
        phi = phi.reshape(grid_size, grid_size)
    
    return density, phi


def load_hw2d_timeseries(
    file_path: str, 
    engine: str = "h5netcdf",
    max_timesteps: Optional[int] = None,
    start_timestep: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load full time series from HW2D simulation.
    
    Args:
        file_path: Path to HDF5 file
        engine: xarray engine
        max_timesteps: Maximum number of timesteps to load (None = all)
        start_timestep: Starting timestep index
    
    Returns:
        Tuple of (density, phi, gamma_n, gamma_c)
        - density: shape (n_time, n_y, n_x)
        - phi: shape (n_time, n_y, n_x)
        - gamma_n: shape (n_time,)
        - gamma_c: shape (n_time,)
    """
    with load_dataset(file_path, engine) as fh:
        end_idx = start_timestep + max_timesteps if max_timesteps else None
        
        density = fh["density"].values[start_timestep:end_idx]
        phi = fh["phi"].values[start_timestep:end_idx]
        gamma_n = fh["gamma_n"].values[start_timestep:end_idx]
        gamma_c = fh["gamma_c"].values[start_timestep:end_idx]
    
    n_time = density.shape[0]
    
    # Reshape if flattened
    if density.ndim == 2:
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    return density, phi, gamma_n, gamma_c


def load_stacked_state_matrix(
    file_path: str,
    engine: str = "h5netcdf",
    max_timesteps: Optional[int] = None,
    fields: List[str] = ["density", "phi"],
) -> np.ndarray:
    """
    Load data as stacked state matrix Q with shape (n_spatial, n_time).
    
    This is the standard format for POD/ROM methods where each column
    is a flattened state vector [field1_flat; field2_flat; ...].
    
    Args:
        file_path: Path to HDF5 file
        engine: xarray engine
        max_timesteps: Maximum number of timesteps
        fields: List of field names to stack
    
    Returns:
        Q matrix with shape (n_fields * n_y * n_x, n_time)
    """
    with load_dataset(file_path, engine) as fh:
        field_data = []
        for field_name in fields:
            data = fh[field_name].values[:max_timesteps]
            n_time = data.shape[0]
            
            # Reshape to (n_time, n_y, n_x) if needed
            if data.ndim == 2:
                grid_size = int(np.sqrt(data.shape[1]))
                data = data.reshape(n_time, grid_size, grid_size)
            
            # Flatten spatial dimensions: (n_time, n_spatial_per_field)
            data = data.reshape(n_time, -1)
            field_data.append(data)
    
    # Stack fields and transpose: (n_time, n_total_spatial) -> (n_total_spatial, n_time)
    Q = np.hstack(field_data).T
    
    return Q


# =============================================================================
# TRUNCATION UTILITIES
# =============================================================================

def compute_truncation_snapshots(
    file_path: str,
    truncate_snapshots: Optional[int] = None,
    truncate_time: Optional[float] = None,
    default_dt: float = 0.025,
) -> Optional[int]:
    """
    Compute number of snapshots to keep based on truncation settings.
    
    Args:
        file_path: Path to HDF5 file (for dt extraction)
        truncate_snapshots: Direct snapshot count limit
        truncate_time: Time-based limit (uses dt from file)
        default_dt: Default dt if not found in file
    
    Returns:
        Number of snapshots to keep, or None if no truncation
    """
    if truncate_snapshots is not None:
        return truncate_snapshots
    elif truncate_time is not None:
        dt = get_dt_from_file(file_path, default_dt)
        return int(truncate_time / dt)
    return None


# =============================================================================
# STATE LOADING AND RECONSTRUCTION
# =============================================================================

def load_reference_states(
    ref_file: str, 
    engine: str, 
    n_steps: int, 
    ref_offset: int = 0
) -> Tuple[np.ndarray, int, int]:
    """
    Load reference full states from simulation file.
    
    Returns states in shape (n_spatial, n_steps) with [density; phi] stacking.
    
    Parameters
    ----------
    ref_file : str
        Path to HDF5 simulation file.
    engine : str
        xarray engine for loading data.
    n_steps : int
        Number of timesteps to load.
    ref_offset : int
        Starting offset into the file (for temporal_split mode).
    
    Returns
    -------
    Q : np.ndarray, shape (n_spatial, n_steps)
        Stacked state matrix [density; phi].
    n_y : int
        Grid size in y.
    n_x : int
        Grid size in x.
    """
    fh = load_dataset(ref_file, engine=engine)
    density = fh["density"].values[ref_offset:ref_offset + n_steps]
    phi = fh["phi"].values[ref_offset:ref_offset + n_steps]
    
    n_time = density.shape[0]
    
    # Handle flattened vs 3D arrays
    if density.ndim == 2:
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    n_y, n_x = density.shape[1], density.shape[2]
    
    # Stack fields: (n_fields, n_time, n_y, n_x) -> (n_spatial, n_time)
    Q = np.stack([density, phi], axis=0).transpose(0, 2, 3, 1)  # (2, n_y, n_x, n_time)
    Q = Q.reshape(2 * n_y * n_x, n_time)  # (n_spatial, n_time)
    
    return Q, n_y, n_x


def reconstruct_full_state(
    X_hat: np.ndarray,
    pod_basis: np.ndarray,
    temporal_mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct full state from reduced state using linear POD basis.
    
    Q_reconstructed = U @ X_hat + mean
    
    Parameters
    ----------
    X_hat : np.ndarray, shape (r, n_time) or (n_time, r)
        Reduced state trajectory.
    pod_basis : np.ndarray, shape (n_spatial, r)
        POD basis matrix U.
    temporal_mean : np.ndarray, shape (n_spatial,), optional
        Temporal mean to add back.
    
    Returns
    -------
    Q : np.ndarray, shape (n_spatial, n_time)
        Reconstructed full state.
    """
    # Ensure X_hat is (r, n_time)
    if X_hat.shape[0] != pod_basis.shape[1]:
        X_hat = X_hat.T
    
    # Reconstruct
    Q = pod_basis @ X_hat  # (n_spatial, n_time)
    
    # Add back mean if provided
    if temporal_mean is not None:
        Q = Q + temporal_mean[:, np.newaxis]
    
    return Q


def _quadratic_features(z: np.ndarray) -> np.ndarray:
    """Compute quadratic features: z_i * z_j for j <= i.
    
    Parameters
    ----------
    z : np.ndarray, shape (r,) or (r, n_time)
        Reduced coordinates.
    
    Returns
    -------
    h : np.ndarray, shape (r*(r+1)//2,) or (r*(r+1)//2, n_time)
        Quadratic features.
    """
    if z.ndim == 1:
        r = z.shape[0]
        return np.concatenate([z[i:i+1] * z[:i+1] for i in range(r)], axis=0)
    else:
        r = z.shape[0]
        return np.concatenate([z[i:i+1, :] * z[:i+1, :] for i in range(r)], axis=0)


def reconstruct_full_state_manifold(
    X_hat: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct full state from reduced state using quadratic manifold.
    
    Q_reconstructed = V @ z + W @ h(z) + shift
    
    where h(z) are the quadratic features z_i * z_j for j <= i.
    
    Parameters
    ----------
    X_hat : np.ndarray, shape (r, n_time) or (n_time, r)
        Reduced state trajectory.
    V : np.ndarray, shape (n_spatial, r)
        Linear basis matrix.
    W : np.ndarray, shape (n_spatial, r*(r+1)//2)
        Quadratic coefficient matrix.
    shift : np.ndarray, shape (n_spatial,)
        Mean shift vector.
    
    Returns
    -------
    Q : np.ndarray, shape (n_spatial, n_time)
        Reconstructed full state.
    """
    # Ensure X_hat is (r, n_time)
    if X_hat.shape[0] != V.shape[1]:
        X_hat = X_hat.T
    
    # Compute quadratic features
    H = _quadratic_features(X_hat)  # (r*(r+1)//2, n_time)
    
    # Reconstruct: Q = V @ z + W @ h(z) + shift
    Q = V @ X_hat + W @ H + shift[:, np.newaxis]
    
    return Q


# =============================================================================
# KURAMOTO-SIVASHINSKY DATA LOADING
# =============================================================================

def get_ks_file_metadata(file_path: str) -> dict:
    """
    Get metadata from KS simulation file without loading full data.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.

    Returns
    -------
    dict
        Dictionary with n_time, N, n_spatial, dt, L, dx.
    """
    with h5py.File(file_path, 'r') as f:
        n_time, N = f['u'].shape
        dt = float(f.attrs.get('dt', 0.1))
        L = float(f.attrs.get('L', 100.0))
        dx = float(f.attrs.get('dx', L / N))

    return {
        'n_time': n_time,
        'N': N,
        'n_spatial': N,  # Single field
        'n_fields': 1,
        'dt': dt,
        'L': L,
        'dx': dx,
    }


def load_ks_timeseries(
    file_path: str,
    max_timesteps: Optional[int] = None,
    start_timestep: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load full time series from KS simulation.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    max_timesteps : int, optional
        Maximum number of timesteps to load (None = all).
    start_timestep : int
        Starting timestep index.

    Returns
    -------
    tuple
        (u, energy, enstrophy)
        - u: shape (n_time, N)
        - energy: shape (n_time,)
        - enstrophy: shape (n_time,)
    """
    with h5py.File(file_path, 'r') as f:
        end_idx = start_timestep + max_timesteps if max_timesteps else None
        u = np.array(f['u'][start_timestep:end_idx])
        energy = np.array(f['energy'][start_timestep:end_idx])
        enstrophy = np.array(f['enstrophy'][start_timestep:end_idx])

    return u, energy, enstrophy


def load_ks_stacked_state_matrix(
    file_path: str,
    max_timesteps: Optional[int] = None,
    start_timestep: int = 0,
) -> np.ndarray:
    """
    Load KS data as state matrix Q with shape (N, n_time).

    This is the standard format for POD/ROM methods where each column
    is a flattened state vector. For KS there is only one field (u).

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    max_timesteps : int, optional
        Maximum number of timesteps.
    start_timestep : int
        Starting timestep index.

    Returns
    -------
    np.ndarray
        Q matrix with shape (N, n_time).
    """
    with h5py.File(file_path, 'r') as f:
        end_idx = start_timestep + max_timesteps if max_timesteps else None
        u = np.array(f['u'][start_timestep:end_idx])  # (n_time, N)

    return u.T  # (N, n_time)


def load_ks_reference_states(
    ref_file: str,
    n_steps: int,
    ref_offset: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Load reference KS states from simulation file.

    Returns states in shape (N, n_steps).

    Parameters
    ----------
    ref_file : str
        Path to HDF5 simulation file.
    n_steps : int
        Number of timesteps to load.
    ref_offset : int
        Starting offset into the file.

    Returns
    -------
    Q : np.ndarray, shape (N, n_steps)
        State matrix.
    N : int
        Number of spatial points.
    """
    with h5py.File(ref_file, 'r') as f:
        u = np.array(f['u'][ref_offset:ref_offset + n_steps])  # (n_steps, N)
    N = u.shape[1]
    return u.T, N


# =============================================================================
# 2D NAVIER-STOKES DATA LOADING (ω-ψ FORMULATION)
# =============================================================================

def get_ns_file_metadata(file_path: str) -> dict:
    """
    Get metadata from 2D NS simulation file without loading full data.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.

    Returns
    -------
    dict
        Dictionary with n_time, ny, nx, n_spatial, n_fields, dt, dt_save,
        Re, Lx, Ly, dx, dy.
    """
    with h5py.File(file_path, 'r') as f:
        n_time, ny, nx = f['omega'].shape
        dt = float(f.attrs.get('dt', 1e-3))
        dt_save = float(f.attrs.get('dt_save', dt))
        Re = float(f.attrs.get('Re', 100.0))
        Lx = float(f.attrs.get('Lx', 2 * np.pi))
        Ly = float(f.attrs.get('Ly', Lx))
        dx = float(f.attrs.get('dx', Lx / nx))
        dy = float(f.attrs.get('dy', Ly / ny))

    return {
        'n_time': n_time,
        'ny': ny,
        'nx': nx,
        'n_spatial': ny * nx,   # Single field (omega)
        'n_fields': 1,
        'dt': dt,
        'dt_save': dt_save,
        'Re': Re,
        'Lx': Lx,
        'Ly': Ly,
        'dx': dx,
        'dy': dy,
    }


def load_ns_timeseries(
    file_path: str,
    max_timesteps: Optional[int] = None,
    start_timestep: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load full time series from 2D NS simulation.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    max_timesteps : int, optional
        Maximum number of timesteps to load (None = all).
    start_timestep : int
        Starting timestep index.

    Returns
    -------
    tuple
        (omega, psi, energy, enstrophy)
        - omega: shape (n_time, ny, nx)
        - psi: shape (n_time, ny, nx)
        - energy: shape (n_time,)
        - enstrophy: shape (n_time,)
    """
    with h5py.File(file_path, 'r') as f:
        end_idx = start_timestep + max_timesteps if max_timesteps else None
        omega = np.array(f['omega'][start_timestep:end_idx])
        psi = np.array(f['psi'][start_timestep:end_idx])
        energy = np.array(f['energy'][start_timestep:end_idx])
        enstrophy = np.array(f['enstrophy'][start_timestep:end_idx])

    return omega, psi, energy, enstrophy


def load_ns_stacked_state_matrix(
    file_path: str,
    max_timesteps: Optional[int] = None,
    start_timestep: int = 0,
) -> np.ndarray:
    """
    Load NS data as state matrix Q with shape (ny*nx, n_time).

    Uses vorticity (omega) as the single state field, matching the
    KS pattern where each column is a flattened state vector.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file.
    max_timesteps : int, optional
        Maximum number of timesteps.
    start_timestep : int
        Starting timestep index.

    Returns
    -------
    np.ndarray
        Q matrix with shape (ny*nx, n_time).
    """
    with h5py.File(file_path, 'r') as f:
        end_idx = start_timestep + max_timesteps if max_timesteps else None
        omega = np.array(f['omega'][start_timestep:end_idx])  # (n_time, ny, nx)

    n_time = omega.shape[0]
    return omega.reshape(n_time, -1).T  # (ny*nx, n_time)


def load_ns_reference_states(
    ref_file: str,
    n_steps: int,
    ref_offset: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """
    Load reference NS states from simulation file.

    Returns states in shape (ny*nx, n_steps) using vorticity field.

    Parameters
    ----------
    ref_file : str
        Path to HDF5 simulation file.
    n_steps : int
        Number of timesteps to load.
    ref_offset : int
        Starting offset into the file.

    Returns
    -------
    Q : np.ndarray, shape (ny*nx, n_steps)
        State matrix (flattened vorticity).
    ny : int
        Grid size in y.
    nx : int
        Grid size in x.
    """
    with h5py.File(ref_file, 'r') as f:
        omega = np.array(f['omega'][ref_offset:ref_offset + n_steps])  # (n_steps, ny, nx)
    n_time, ny, nx = omega.shape
    return omega.reshape(n_time, -1).T, ny, nx
