"""
MPI communication utilities.

Provides helper functions for distributed computing:
- Index distribution across ranks
- Chunked broadcasts (avoiding 32-bit overflow)
- Shared memory array creation

Author: Anthony Poole
"""

import numpy as np
from mpi4py import MPI


def distribute_indices(rank: int, n_total: int, size: int) -> tuple:
    """
    Distribute indices across MPI ranks.
    
    Args:
        rank: Current MPI rank
        n_total: Total number of items to distribute
        size: Number of MPI ranks
    
    Returns:
        Tuple of (start_idx, end_idx, n_local)
    """
    n_per_rank = n_total // size
    start = rank * n_per_rank
    end = (rank + 1) * n_per_rank
    
    # Last rank handles remainder
    if rank == size - 1 and end != n_total:
        end = n_total
    
    return start, end, end - start


def chunked_bcast(comm, data, root: int = 0, max_bytes: int = 2**30):
    """
    Broadcast a numpy array in chunks to avoid MPI 32-bit integer overflow.
    
    MPI's Bcast uses a 32-bit signed integer for count, limiting messages to ~2GB.
    This function handles larger arrays by broadcasting in chunks.
    
    Args:
        comm: MPI communicator
        data: Numpy array to broadcast (only valid on root)
        root: Root rank for broadcast
        max_bytes: Maximum bytes per chunk (default 1GB)
    
    Returns:
        Broadcast array on all ranks
    """
    rank = comm.Get_rank()
    
    # Broadcast shape and dtype first
    if rank == root:
        shape, dtype = data.shape, data.dtype
    else:
        shape, dtype = None, None
    
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    
    if rank != root:
        data = np.empty(shape, dtype=dtype)
    
    # If small enough, single broadcast
    itemsize = np.dtype(dtype).itemsize
    total_bytes = int(np.prod(shape)) * itemsize
    
    if total_bytes <= max_bytes:
        comm.Bcast(data, root=root)
        return data
    
    # Chunked broadcast for large arrays
    n_rows = shape[0]
    bytes_per_row = total_bytes // n_rows
    rows_per_chunk = max(1, max_bytes // bytes_per_row)
    
    data_flat = data.reshape(n_rows, -1) if len(shape) > 1 else data.reshape(n_rows, 1)
    
    for start_row in range(0, n_rows, rows_per_chunk):
        end_row = min(start_row + rows_per_chunk, n_rows)
        if rank == root:
            chunk = np.ascontiguousarray(data_flat[start_row:end_row, :])
        else:
            chunk = np.empty((end_row - start_row, data_flat.shape[1]), dtype=dtype)
        comm.Bcast(chunk, root=root)
        if rank != root:
            data_flat[start_row:end_row, :] = chunk
    
    return data


def create_shared_array(node_comm, shape, dtype=np.float64):
    """
    Create a numpy array backed by MPI shared memory within a node.
    
    All ranks on the same node share the same physical memory,
    reducing memory usage for read-only data.
    
    Args:
        node_comm: Node-local MPI communicator (from Split_type)
        shape: Shape of the array
        dtype: Data type (default float64)
    
    Returns:
        Tuple of (array, window) - window must be freed when done
    """
    node_rank = node_comm.Get_rank()
    itemsize = np.dtype(dtype).itemsize
    nbytes = int(np.prod(shape)) * itemsize
    
    if node_rank == 0:
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    else:
        win = MPI.Win.Allocate_shared(0, itemsize, comm=node_comm)
    
    buf, _ = win.Shared_query(0)
    arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    
    return arr, win


def gather_to_root(comm, local_data, root: int = 0):
    """
    Gather arrays from all ranks to root.
    
    Args:
        comm: MPI communicator
        local_data: Local numpy array
        root: Root rank to gather to
    
    Returns:
        Concatenated array on root, None on other ranks
    """
    rank = comm.Get_rank()
    gathered = comm.gather(local_data, root=root)
    
    if rank == root:
        return np.concatenate(gathered)
    return None


def allreduce_sum(comm, local_array):
    """
    Allreduce with sum operation.
    
    Args:
        comm: MPI communicator
        local_array: Local numpy array
    
    Returns:
        Sum across all ranks
    """
    global_array = np.zeros_like(local_array)
    comm.Allreduce(local_array, global_array, op=MPI.SUM)
    return global_array
