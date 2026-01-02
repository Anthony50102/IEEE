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
