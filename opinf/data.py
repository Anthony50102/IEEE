"""
Data loading and preprocessing utilities.

This module handles:
- Distributed data loading from HDF5 files
- Data centering and scaling
- Initial condition extraction
- Reference data loading
- Manifold basis loading

Author: Anthony Poole
"""

import gc
import os
import numpy as np

from utils import (
    compute_truncation_snapshots, load_dataset, distribute_indices,
    chunked_bcast, create_shared_array,
)


# =============================================================================
# LAZY MPI IMPORT
# =============================================================================

MPI = None


def _get_mpi():
    """Lazily import MPI only when needed for distributed functions."""
    global MPI
    if MPI is None:
        from mpi4py import MPI as _MPI
        MPI = _MPI
    return MPI


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
    
    # Return contiguous copy in native dtype (float32 or float64)
    return np.ascontiguousarray(Q_full[start_idx:end_idx, :])


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
        logger.debug(f"  [DIAG] Input dtype: {Q_local.dtype}")
    
    # Preserve input dtype (float32 or float64)
    temporal_mean = np.mean(Q_local, axis=1, keepdims=True)
    Q_centered = Q_local - temporal_mean
    
    if rank == 0:
        logger.debug(f"  [DIAG] Centered data range: [{Q_centered.min():.2e}, {Q_centered.max():.2e}]")
    
    return Q_centered, temporal_mean.squeeze()


def scale_data_distributed(
    Q_local: np.ndarray, n_fields: int, n_local_per_field: int, comm, rank: int, logger
) -> tuple:
    """Scale data so each field's values are in [-1, 1]."""
    MPI = _get_mpi()  # Lazy import
    if rank == 0:
        logger.info("Scaling data (normalizing each field to [-1, 1])...")
    
    # Use float64 for scaling factors (small array), preserve input dtype for data
    dtype = Q_local.dtype
    scaling_factors = np.zeros(n_fields, dtype=np.float64)
    Q_scaled = Q_local.copy()
    
    for j in range(n_fields):
        start, end = j * n_local_per_field, (j + 1) * n_local_per_field
        local_max = np.float64(np.max(np.abs(Q_local[start:end, :])))
        
        # Use float64 for the reduction (small scalar)
        global_max = np.zeros(1, dtype=np.float64)
        comm.Allreduce(np.array([local_max], dtype=np.float64), global_max, op=MPI.MAX)
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
        
        # Handle temporal_split mode: use explicit train range
        if cfg.training_mode == "temporal_split":
            train_start, train_end = cfg.train_start, cfg.train_end
            gamma_n = gamma_n[train_start:train_end]
            gamma_c = gamma_c[train_start:train_end]
            logger.info(f"  Temporal split: using gamma[{train_start}:{train_end}]")
        elif cfg.truncation_enabled:
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
    MPI = _get_mpi()  # Lazy import
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


def load_manifold_basis(paths: dict, comm, logger) -> dict:
    """
    Load manifold basis (V, W, shift) for manifold-aware training.
    
    Only loads if manifold_basis file exists. Returns None if not found
    or if method is linear POD.
    
    Uses chunked broadcasts to avoid MPI 32-bit integer overflow for large
    arrays (V and W can be n_spatial × r which easily exceeds 2GB).
    """
    rank = comm.Get_rank()
    
    # First, determine if manifold data exists (small metadata broadcast)
    has_manifold = False
    r_val = 0
    
    if rank == 0:
        if os.path.exists(paths["manifold_basis"]):
            from pod import load_basis
            basis = load_basis(paths["manifold_basis"])
            
            if basis.method == "manifold" and basis.W is not None:
                has_manifold = True
                r_val = basis.r
                # Store locally for chunked broadcast
                V_data = basis.V
                W_data = basis.W
                shift_data = basis.shift
                logger.info(f"Loaded manifold basis: V={V_data.shape}, W={W_data.shape}")
            else:
                logger.info("Basis file found but not manifold method, skipping manifold-aware training")
        else:
            logger.info(f"No manifold basis found at {paths['manifold_basis']}")
    
    # Broadcast metadata first
    has_manifold = comm.bcast(has_manifold, root=0)
    
    if not has_manifold:
        return None
    
    r_val = comm.bcast(r_val, root=0)
    
    # Use chunked_bcast for large arrays to avoid MPI 32-bit overflow
    if rank == 0:
        V = chunked_bcast(comm, V_data, root=0)
        W = chunked_bcast(comm, W_data, root=0)
        shift = chunked_bcast(comm, shift_data, root=0)
    else:
        V = chunked_bcast(comm, None, root=0)
        W = chunked_bcast(comm, None, root=0)
        shift = chunked_bcast(comm, None, root=0)
    
    return {
        'V': V,
        'W': W,
        'shift': shift,
        'r': r_val,
    }


def load_physics_data(paths: dict, cfg, comm, logger) -> dict:
    """
    Load physics data needed for physics-based Gamma computation during training.
    
    This includes:
    - POD basis for reconstructing full state
    - Temporal mean (if centering was applied)
    - Reference Gamma statistics for computing errors
    - Grid and physics parameters
    
    Parameters
    ----------
    paths : dict
        Output paths from get_output_paths.
    cfg : OpInfConfig
        Configuration object.
    comm : MPI.Comm
        MPI communicator.
    logger : Logger
        Logger instance.
    
    Returns
    -------
    dict or None
        Physics data dict, or None if loading failed.
    """
    rank = comm.Get_rank()
    
    # First, load scalar metadata on rank 0
    metadata = None
    if rank == 0:
        try:
            # Load POD basis
            if not os.path.exists(paths["pod_basis"]):
                logger.warning(f"POD basis not found at {paths['pod_basis']}")
                metadata = {'error': 'pod_basis_not_found'}
            elif not os.path.exists(paths["gamma_ref"]):
                logger.warning(f"Gamma reference not found at {paths['gamma_ref']}")
                metadata = {'error': 'gamma_ref_not_found'}
            else:
                pod_basis = np.load(paths["pod_basis"])
                gamma_ref = np.load(paths["gamma_ref"])
                
                # Load temporal mean from initial conditions if centering was applied
                temporal_mean = None
                if os.path.exists(paths["initial_conditions"]):
                    ics = np.load(paths["initial_conditions"])
                    if 'train_temporal_mean' in ics:
                        temporal_mean = ics['train_temporal_mean']
                        logger.info(f"Loaded temporal mean: {temporal_mean.shape}")
                
                # Estimate memory usage for warning
                mem_mb = pod_basis.nbytes / 1e6
                logger.info(f"Physics data: POD basis {pod_basis.shape} ({mem_mb:.1f} MB)")
                if mem_mb > 100:
                    logger.warning(f"Large POD basis will be broadcast to all ranks "
                                  f"(total ~{mem_mb * comm.Get_size() / 1e3:.1f} GB across cluster)")
                
                metadata = {
                    'pod_basis_shape': pod_basis.shape,
                    'temporal_mean_shape': temporal_mean.shape if temporal_mean is not None else None,
                    'n_y': cfg.n_y,
                    'n_x': cfg.n_x,
                    'k0': cfg.k0,
                    'c1': cfg.c1,
                    'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
                    'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
                    'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
                    'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
                }
        except Exception as e:
            logger.warning(f"Failed to load physics data: {e}")
            metadata = {'error': str(e)}
    
    # Broadcast metadata first (small)
    metadata = comm.bcast(metadata, root=0)
    
    if metadata is None or 'error' in metadata:
        return None
    
    # Broadcast large arrays using chunked_bcast to avoid MPI 32-bit overflow
    if rank == 0:
        pod_basis_bcast = chunked_bcast(comm, pod_basis, root=0)
        temporal_mean_bcast = chunked_bcast(comm, temporal_mean, root=0) if temporal_mean is not None else None
    else:
        pod_basis_bcast = chunked_bcast(comm, None, root=0)
        temporal_mean_bcast = chunked_bcast(comm, None, root=0) if metadata['temporal_mean_shape'] is not None else None
    
    physics_data = {
        'pod_basis': pod_basis_bcast,
        'temporal_mean': temporal_mean_bcast,
        'n_y': metadata['n_y'],
        'n_x': metadata['n_x'],
        'k0': metadata['k0'],
        'c1': metadata['c1'],
        'mean_Gamma_n': metadata['mean_Gamma_n'],
        'std_Gamma_n': metadata['std_Gamma_n'],
        'mean_Gamma_c': metadata['mean_Gamma_c'],
        'std_Gamma_c': metadata['std_Gamma_c'],
    }
    
    return physics_data


# =============================================================================
# MODEL I/O
# =============================================================================

def load_model_from_file(filepath: str) -> dict:
    """Load a single model from NPZ file."""
    data = np.load(filepath)
    
    # Base model (always present)
    model = {
        'A': data['A'],
        'F': data['F'],
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
    
    # Output operators (optional - may not exist for state_only or physics_gamma modes)
    has_output = bool(data.get('use_learned_output', 'C' in data.files))
    if has_output and 'C' in data.files:
        model['C'] = data['C']
        model['G'] = data['G']
        model['c'] = data['c']
    
    model['has_output_operators'] = has_output
    return model


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
    
    # Check if models have output operators
    use_learned_output = bool(data.get('use_learned_output', True))
    
    models = []
    for i in range(num_models):
        prefix = f'model_{i}_'
        
        # State operators (always present)
        model = {
            'A': data[prefix + 'A'],
            'F': data[prefix + 'F'],
            'total_error': float(data[prefix + 'total_error']),
            'alpha_state_lin': float(data[prefix + 'alpha_state_lin']),
            'alpha_state_quad': float(data[prefix + 'alpha_state_quad']),
            'alpha_out_lin': float(data[prefix + 'alpha_out_lin']),
            'alpha_out_quad': float(data[prefix + 'alpha_out_quad']),
            'mean_err_Gamma_n': float(data[prefix + 'mean_err_Gamma_n']),
            'std_err_Gamma_n': float(data[prefix + 'std_err_Gamma_n']),
            'mean_err_Gamma_c': float(data[prefix + 'mean_err_Gamma_c']),
            'std_err_Gamma_c': float(data[prefix + 'std_err_Gamma_c']),
        }
        
        # Output operators (only if learned output was used)
        has_output = bool(data.get(prefix + 'has_output_operators', use_learned_output))
        if has_output and (prefix + 'C') in data:
            model['C'] = data[prefix + 'C']
            model['G'] = data[prefix + 'G']
            model['c'] = data[prefix + 'c']
        
        model['has_output_operators'] = has_output
        models.append((model['total_error'], model))
    
    logger.info(f"  Loaded {len(models)} models (has_output_operators={use_learned_output})")
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
        'selection_metric': getattr(cfg, 'selection_metric', 'gamma_learned'),
        'use_learned_output': getattr(cfg, 'use_learned_output', True),
    }
    
    # Determine which keys to save based on whether output operators exist
    state_keys = ['A', 'F', 'alpha_state_lin', 'alpha_state_quad', 'total_error',
                  'mean_err_Gamma_n', 'std_err_Gamma_n', 'mean_err_Gamma_c', 'std_err_Gamma_c',
                  'alpha_out_lin', 'alpha_out_quad']
    output_keys = ['C', 'G', 'c']
    
    for i, (score, model) in enumerate(models):
        prefix = f'model_{i}_'
        
        # Always save state-related keys
        for key in state_keys:
            if key in model:
                ensemble_data[prefix + key] = model[key]
        
        # Save output keys only if they exist (use_learned_output was True)
        for key in output_keys:
            if key in model:
                ensemble_data[prefix + key] = model[key]
        
        # Mark if this model has output operators
        ensemble_data[prefix + 'has_output_operators'] = 'C' in model
    
    np.savez(output_path, **ensemble_data)
    logger.info(f"Saved {len(models)} models to {output_path}")
    return output_path
