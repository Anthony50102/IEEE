"""
Data loading and preprocessing utilities.

This module handles:
- Distributed data loading from HDF5 files
- Data centering and scaling
- Initial condition extraction
- Reference data loading

Author: Anthony Poole
"""

import gc
import os
import numpy as np
from mpi4py import MPI

from utils import (
    compute_truncation_snapshots, load_dataset, distribute_indices,
    chunked_bcast, create_shared_array,
)


# =============================================================================
# FILE METADATA
# =============================================================================

def get_file_metadata(cfg, file_path: str) -> tuple:
    """Get metadata from a file without loading full data."""
    with load_dataset(file_path, cfg.engine) as fh:
        n_time = fh["density"].shape[0]
        if fh["density"].ndim == 3:
            n_y, n_x = fh["density"].shape[1], fh["density"].shape[2]
        else:
            n_y = n_x = int(np.sqrt(fh["density"].shape[1]))
        n_spatial = cfg.n_fields * n_y * n_x
    
    if cfg.truncation_enabled:
        max_snaps = compute_truncation_snapshots(
            file_path, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
        )
        n_time = min(n_time, max_snaps) if max_snaps else n_time
    else:
        max_snaps = None
    
    return n_spatial, n_time, max_snaps


# =============================================================================
# DISTRIBUTED DATA LOADING
# =============================================================================

def load_distributed_snapshots(
    file_path: str, start_idx: int, end_idx: int, engine: str, max_snapshots: int,
) -> np.ndarray:
    """Load a portion of snapshots corresponding to this rank's spatial DOF."""
    with load_dataset(file_path, engine) as fh:
        density = fh["density"].values
        phi = fh["phi"].values
    
    if max_snapshots is not None and max_snapshots < density.shape[0]:
        density = density[:max_snapshots]
        phi = phi[:max_snapshots]
    
    n_time = density.shape[0]
    
    # Reshape if needed
    if density.ndim == 2:
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    # Stack fields: (n_fields, n_time, n_y, n_x) -> (n_spatial, n_time)
    Q_full = np.stack([density, phi], axis=0).transpose(0, 2, 3, 1)
    n_field, n_y, n_x, n_time = Q_full.shape
    Q_full = Q_full.reshape(n_field * n_y * n_x, n_time)
    
    return Q_full[start_idx:end_idx, :].copy()


def load_all_data_distributed(cfg, run_dir: str, comm, rank: int, size: int, logger) -> tuple:
    """Load all training and test data in distributed fashion."""
    
    # Check for temporal split mode
    if cfg.training_mode == "temporal_split":
        return load_temporal_split_distributed(cfg, run_dir, comm, rank, size, logger)
    
    # Original multi-trajectory mode
    if rank == 0:
        logger.info("Loading simulation data (distributed)...")
        
        # Get metadata on rank 0
        train_timesteps, test_timesteps = [], []
        train_truncations, test_truncations = [], []
        n_spatial = None
        
        for fp in cfg.training_files:
            ns, nt, max_snaps = get_file_metadata(cfg, fp)
            n_spatial = n_spatial or ns
            train_timesteps.append(nt)
            train_truncations.append(max_snaps)
        
        for fp in cfg.test_files:
            _, nt, max_snaps = get_file_metadata(cfg, fp)
            test_timesteps.append(nt)
            test_truncations.append(max_snaps)
        
        metadata = {
            'n_spatial': n_spatial,
            'train_timesteps': train_timesteps,
            'test_timesteps': test_timesteps,
            'train_truncations': train_truncations,
            'test_truncations': test_truncations,
        }
    else:
        metadata = None
    
    metadata = comm.bcast(metadata, root=0)
    n_spatial = metadata['n_spatial']
    
    total_train = sum(metadata['train_timesteps'])
    total_test = sum(metadata['test_timesteps'])
    
    # Distribute spatial DOF
    start_idx, end_idx, n_local = distribute_indices(rank, n_spatial, size)
    
    if rank == 0:
        logger.info(f"  Spatial DOF: {n_spatial:,}, MPI ranks: {size}")
        logger.info(f"  Train snapshots: {total_train:,}, Test snapshots: {total_test:,}")
    
    # Allocate local arrays
    Q_train_local = np.zeros((n_local, total_train), dtype=np.float64)
    Q_test_local = np.zeros((n_local, total_test), dtype=np.float64)
    
    # Compute boundaries
    train_boundaries = [0] + list(np.cumsum(metadata['train_timesteps']))
    test_boundaries = [0] + list(np.cumsum(metadata['test_timesteps']))
    
    # Load data
    for i, fp in enumerate(cfg.training_files):
        Q_local = load_distributed_snapshots(
            fp, start_idx, end_idx, cfg.engine, metadata['train_truncations'][i]
        )
        Q_train_local[:, train_boundaries[i]:train_boundaries[i + 1]] = Q_local
        del Q_local
        gc.collect()
    
    for i, fp in enumerate(cfg.test_files):
        Q_local = load_distributed_snapshots(
            fp, start_idx, end_idx, cfg.engine, metadata['test_truncations'][i]
        )
        Q_test_local[:, test_boundaries[i]:test_boundaries[i + 1]] = Q_local
        del Q_local
        gc.collect()
    
    return (Q_train_local, Q_test_local, train_boundaries, test_boundaries,
            n_spatial, n_local, start_idx, end_idx)


def load_temporal_split_distributed(cfg, run_dir: str, comm, rank: int, size: int, logger) -> tuple:
    """
    Load data for temporal split mode using explicit snapshot ranges.
    
    Uses a single trajectory file with user-specified train/test ranges.
    """
    if rank == 0:
        logger.info("Loading simulation data (temporal split mode)...")
        
        # Get metadata from single training file
        fp = cfg.training_files[0]
        n_spatial, n_time_total, _ = get_file_metadata(cfg, fp)
        
        # Get ranges from config
        train_start, train_end = cfg.train_start, cfg.train_end
        test_start, test_end = cfg.test_start, cfg.test_end
        
        # Validate ranges
        if train_end > n_time_total or test_end > n_time_total:
            raise ValueError(f"Range exceeds file length ({n_time_total} snapshots)")
        if train_start >= train_end:
            raise ValueError(f"Invalid train range: [{train_start}, {train_end})")
        if test_start >= test_end:
            raise ValueError(f"Invalid test range: [{test_start}, {test_end})")
        
        n_train = train_end - train_start
        n_test = test_end - test_start
        
        metadata = {
            'n_spatial': n_spatial,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'n_train': n_train,
            'n_test': n_test,
        }
        logger.info(f"  Train: snapshots [{train_start}, {train_end}) = {n_train} snapshots")
        logger.info(f"  Test:  snapshots [{test_start}, {test_end}) = {n_test} snapshots")
    else:
        metadata = None
    
    metadata = comm.bcast(metadata, root=0)
    n_spatial = metadata['n_spatial']
    train_start = metadata['train_start']
    train_end = metadata['train_end']
    test_start = metadata['test_start']
    test_end = metadata['test_end']
    n_train = metadata['n_train']
    n_test = metadata['n_test']
    
    # Distribute spatial DOF
    start_idx, end_idx, n_local = distribute_indices(rank, n_spatial, size)
    
    if rank == 0:
        logger.info(f"  Spatial DOF: {n_spatial:,}, MPI ranks: {size}")
    
    # Load full trajectory (we need to load enough to cover both ranges)
    max_snap_needed = max(train_end, test_end)
    Q_full_local = load_distributed_snapshots(
        cfg.training_files[0], start_idx, end_idx, cfg.engine, max_snap_needed
    )
    
    # Extract train and test ranges
    Q_train_local = Q_full_local[:, train_start:train_end].copy()
    Q_test_local = Q_full_local[:, test_start:test_end].copy()
    del Q_full_local
    gc.collect()
    
    # Boundaries for single trajectory each
    train_boundaries = [0, n_train]
    test_boundaries = [0, n_test]
    
    return (Q_train_local, Q_test_local, train_boundaries, test_boundaries,
            n_spatial, n_local, start_idx, end_idx)


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def center_data_distributed(Q_local: np.ndarray, comm, rank: int, logger) -> tuple:
    """Center data by subtracting temporal mean at each spatial location."""
    if rank == 0:
        logger.info("Centering data (subtracting temporal mean)...")
        logger.debug(f"  [DIAG] Raw data range: [{Q_local.min():.2e}, {Q_local.max():.2e}]")
    
    temporal_mean = np.mean(Q_local, axis=1, keepdims=True)
    Q_centered = Q_local - temporal_mean
    
    if rank == 0:
        logger.debug(f"  [DIAG] Centered data range: [{Q_centered.min():.2e}, {Q_centered.max():.2e}]")
    
    return Q_centered, temporal_mean.squeeze()


def scale_data_distributed(
    Q_local: np.ndarray, n_fields: int, n_local_per_field: int, comm, rank: int, logger
) -> tuple:
    """Scale data so each field's values are in [-1, 1]."""
    if rank == 0:
        logger.info("Scaling data (normalizing each field to [-1, 1])...")
    
    scaling_factors = np.zeros(n_fields)
    Q_scaled = Q_local.copy()
    
    for j in range(n_fields):
        start, end = j * n_local_per_field, (j + 1) * n_local_per_field
        local_max = np.max(np.abs(Q_local[start:end, :]))
        
        global_max = np.zeros(1)
        comm.Allreduce(np.array([local_max]), global_max, op=MPI.MAX)
        global_max = global_max[0]
        
        if global_max > 0:
            Q_scaled[start:end, :] /= global_max
            scaling_factors[j] = global_max
        else:
            scaling_factors[j] = 1.0
    
    if rank == 0:
        logger.info(f"  Scaling factors: {scaling_factors}")
    
    return Q_scaled, scaling_factors


# =============================================================================
# REFERENCE DATA & INITIAL CONDITIONS
# =============================================================================

def load_reference_gamma(cfg, rank, logger) -> dict:
    """Load reference Gamma values from training files (rank 0 only)."""
    if rank != 0:
        return None
    
    logger.info("Loading reference Gamma values...")
    
    Gamma_n_list, Gamma_c_list = [], []
    
    for fp in cfg.training_files:
        fh = load_dataset(fp, cfg.engine)
        gamma_n, gamma_c = fh["gamma_n"].values, fh["gamma_c"].values
        
        if cfg.truncation_enabled:
            max_snaps = compute_truncation_snapshots(
                fp, cfg.truncation_snapshots, cfg.truncation_time, cfg.dt
            )
            if max_snaps:
                gamma_n = gamma_n[:max_snaps]
                gamma_c = gamma_c[:max_snaps]
        
        Gamma_n_list.append(gamma_n)
        Gamma_c_list.append(gamma_c)
    
    Gamma_n = np.concatenate(Gamma_n_list)
    Gamma_c = np.concatenate(Gamma_c_list)
    Y_Gamma = np.vstack([Gamma_n, Gamma_c])
    
    logger.info(f"  Y_Gamma shape: {Y_Gamma.shape}")
    
    return {
        'Y_Gamma': Y_Gamma,
        'mean_Gamma_n': np.mean(Gamma_n), 'std_Gamma_n': np.std(Gamma_n, ddof=1),
        'mean_Gamma_c': np.mean(Gamma_c), 'std_Gamma_c': np.std(Gamma_c, ddof=1),
    }


def gather_initial_conditions(
    Q_train_local, Q_test_local, Xhat_train, Xhat_test,
    train_boundaries, test_boundaries, n_train, n_test, n_spatial, comm, rank
) -> dict:
    """Gather initial conditions from all ranks."""
    train_ICs_gathered, test_ICs_gathered = [], []
    
    for i in range(n_train):
        ic = comm.gather(Q_train_local[:, train_boundaries[i]], root=0)
        if rank == 0:
            train_ICs_gathered.append(np.concatenate(ic))
    
    for i in range(n_test):
        ic = comm.gather(Q_test_local[:, test_boundaries[i]], root=0)
        if rank == 0:
            test_ICs_gathered.append(np.concatenate(ic))
    
    if rank == 0:
        return {
            'train_ICs': np.array(train_ICs_gathered),
            'test_ICs': np.array(test_ICs_gathered),
            'train_ICs_reduced': np.array([Xhat_train[train_boundaries[i]] for i in range(n_train)]),
            'test_ICs_reduced': np.array([Xhat_test[test_boundaries[i]] for i in range(n_test)]),
        }
    return None


# =============================================================================
# SHARED MEMORY DATA LOADING (STEP 2)
# =============================================================================

def load_data_shared_memory(paths: dict, comm, logger):
    """Load data using MPI shared memory for efficiency."""
    rank = comm.Get_rank()
    
    # Create node-local communicator
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    
    # Rank 0 loads data
    if rank == 0:
        logger.info("Loading pre-computed data...")
        learning = np.load(paths["learning_matrices"])
        gamma_ref = np.load(paths["gamma_ref"])
        
        data_local = {
            'X_state': learning['X_state'].copy(),
            'Y_state': learning['Y_state'].copy(),
            'D_state': learning['D_state'].copy(),
            'D_state_2': learning['D_state_2'].copy(),
            'D_out': learning['D_out'].copy(),
            'D_out_2': learning['D_out_2'].copy(),
            'mean_Xhat': learning['mean_Xhat'].copy(),
            'Y_Gamma': gamma_ref['Y_Gamma'].copy(),
        }
        scalars = {
            'scaling_Xhat': float(learning['scaling_Xhat']),
            'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
            'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
            'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
            'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
        }
        shapes = {k: v.shape for k, v in data_local.items()}
        learning.close()
        gamma_ref.close()
    else:
        data_local, shapes, scalars = None, None, None
    
    # Broadcast shapes and scalars
    shapes = comm.bcast(shapes, root=0)
    scalars = comm.bcast(scalars, root=0)
    
    # Create shared memory arrays
    arrays = {}
    windows = []
    for name, shape in shapes.items():
        arr, win = create_shared_array(node_comm, shape)
        arrays[name] = arr
        windows.append(win)
    
    # Fill shared memory from node-rank-0s
    node_root_comm = comm.Split(color=0 if node_rank == 0 else MPI.UNDEFINED, key=rank)
    
    if node_rank == 0 and node_root_comm != MPI.COMM_NULL:
        node_root_rank = node_root_comm.Get_rank()
        is_source = (node_root_rank == 0)
        
        for name in shapes.keys():
            arrays[name][:] = chunked_bcast(
                node_root_comm,
                data_local[name] if is_source else None,
                root=0
            )
    
    node_comm.Barrier()
    
    if node_root_comm != MPI.COMM_NULL:
        node_root_comm.Free()
    
    if rank == 0:
        del data_local
        gc.collect()
    
    # Build data dict
    data = {**arrays, **scalars}
    
    if rank == 0:
        logger.info("Data loaded with shared memory")
    
    return data, windows


# =============================================================================
# MODEL I/O
# =============================================================================

def load_model_from_file(filepath: str) -> dict:
    """Load a single model from NPZ file."""
    data = np.load(filepath)
    return {
        'A': data['A'], 'F': data['F'], 'C': data['C'], 'G': data['G'], 'c': data['c'],
        'total_error': float(data['total_error']),
        'mean_err_Gamma_n': float(data['mean_err_Gamma_n']),
        'std_err_Gamma_n': float(data['std_err_Gamma_n']),
        'mean_err_Gamma_c': float(data['mean_err_Gamma_c']),
        'std_err_Gamma_c': float(data['std_err_Gamma_c']),
        'alpha_state_lin': float(data['alpha_state_lin']),
        'alpha_state_quad': float(data['alpha_state_quad']),
        'alpha_out_lin': float(data['alpha_out_lin']),
        'alpha_out_quad': float(data['alpha_out_quad']),
    }


def load_ensemble(filepath: str, operators_dir: str, logger) -> list:
    """Load ensemble models from directory or single file."""
    # Try loading from individual files first (new format)
    if operators_dir and os.path.exists(operators_dir):
        model_files = sorted([
            f for f in os.listdir(operators_dir)
            if f.startswith('model_') and f.endswith('.npz')
        ])
        
        if model_files:
            logger.info(f"Loading {len(model_files)} models from {operators_dir}")
            models = []
            for fname in model_files:
                model = load_model_from_file(os.path.join(operators_dir, fname))
                models.append((model['total_error'], model))
            models.sort(key=lambda x: x[0])
            return models
    
    # Fall back to single ensemble file
    logger.info(f"Loading ensemble from {filepath}")
    data = np.load(filepath, allow_pickle=True)
    num_models = int(data['num_models'])
    
    models = []
    for i in range(num_models):
        prefix = f'model_{i}_'
        model = {
            key: data[prefix + key] if key in ['A', 'F', 'C', 'G', 'c'] 
                 else float(data[prefix + key])
            for key in ['A', 'F', 'C', 'G', 'c', 'total_error', 
                       'mean_err_Gamma_n', 'std_err_Gamma_n',
                       'mean_err_Gamma_c', 'std_err_Gamma_c',
                       'alpha_state_lin', 'alpha_state_quad',
                       'alpha_out_lin', 'alpha_out_quad']
        }
        models.append((model['total_error'], model))
    
    logger.info(f"  Loaded {len(models)} models")
    return models


def load_preprocessing_info(filepath: str, logger) -> dict:
    """Load preprocessing information from Step 1."""
    if not os.path.exists(filepath):
        logger.warning("Preprocessing info not found, using defaults")
        return {'centering_applied': True, 'scaling_applied': False}
    
    data = np.load(filepath, allow_pickle=True)
    info = {
        'centering_applied': bool(data['centering_applied']),
        'scaling_applied': bool(data.get('scaling_applied', False)),
        'r_actual': int(data['r_actual']),
        'dt': float(data['dt']),
    }
    
    logger.info(f"Preprocessing: centering={info['centering_applied']}, "
               f"scaling={info['scaling_applied']}, r={info['r_actual']}")
    return info


def save_ensemble(models: list, output_path: str, cfg, logger) -> str:
    """Save ensemble models to NPZ file."""
    ensemble_data = {
        'num_models': len(models),
        'r': cfg.r,
        'threshold_mean': cfg.threshold_mean,
        'threshold_std': cfg.threshold_std,
    }
    
    for i, (score, model) in enumerate(models):
        prefix = f'model_{i}_'
        for key in ['A', 'F', 'C', 'G', 'c', 'alpha_state_lin', 'alpha_state_quad',
                    'alpha_out_lin', 'alpha_out_quad', 'total_error',
                    'mean_err_Gamma_n', 'std_err_Gamma_n', 'mean_err_Gamma_c', 'std_err_Gamma_c']:
            ensemble_data[prefix + key] = model[key]
    
    np.savez(output_path, **ensemble_data)
    logger.info(f"Saved {len(models)} models to {output_path}")
    return output_path
