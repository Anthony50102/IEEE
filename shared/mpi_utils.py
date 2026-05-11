"""
MPI communication utilities -- *safe* wrappers around mpi4py collectives.

WHY THIS MODULE EXISTS
======================
mpi4py exposes two flavors of every collective:

  - Lowercase  (comm.gather, comm.bcast, comm.scatter, comm.allreduce):
        Pickle path. Accepts arbitrary Python objects. Each message is
        pickled, and *each pickled byte string passes through
        PyBytes_FromStringAndSize*, whose `size` is a 32-bit signed int
        -- a hard ~2 GB ceiling **per message**. For gather, root also
        builds a list of all rank payloads, so the limiting payload is
        the *aggregate*, not the per-rank chunk.

  - Capitalized (comm.Gather, comm.Bcast, comm.Allreduce):
        Buffer path. Numpy buffers go through directly. Per-rank count
        is still a 32-bit int (max ~2 GB / itemsize elements), but no
        pickle hop -- so we don't pay the python overhead and we can
        chunk per rank trivially.

Real-world failure mode we hit (job 7706223, r=50 alpha=1):
    524288 spatial DOFs x 2000 train snapshots split across 56 ranks.
    Per-rank payload ~150 MB (well under 2 GB), but root aggregates
    8.4 GB through pickle -> "Negative size passed to
    PyBytes_FromStringAndSize" in mpi4py.MPI.Pickle.allocv.

RULES OF THE ROAD
=================
1. **Never pickle-gather a numpy array bigger than ~200 MB total**
   (collective total, not per-rank). Use `chunked_gather` from this module.

2. **Never pickle-bcast a numpy array bigger than ~200 MB.**
   Use `chunked_bcast` from this module.

3. Pickle-mode collectives are fine for small python objects (dicts,
   paths, scalars, lists of metadata). Reserve them for control-plane
   messages, not bulk data.

4. For per-rank buffer collectives that approach 2 GB per rank, switch
   to Gatherv/Scatterv with chunked counts. (Not implemented here yet --
   we don't have a use case where a *single* rank holds >2 GB.)

5. Always reason about the *root-aggregate* payload for gather, not the
   per-rank tile. This is the gate that this module's `chunked_gather`
   gets right and the bug we tripped over previously got wrong.

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
    Broadcast a numpy array in chunks to avoid the MPI 32-bit count limit.

    Uses the buffer-mode Bcast (no pickle); chunks across rows so that no
    single Bcast exceeds `max_bytes` (default 1 GB, well under the 2 GB
    hard cap). Shape and dtype are exchanged first via small pickle bcasts.

    Args:
        comm: MPI communicator
        data: Numpy array to broadcast (only meaningful on root)
        root: Root rank for broadcast
        max_bytes: Maximum bytes per chunk (default 1 GB)

    Returns:
        Broadcast array on all ranks.
    """
    rank = comm.Get_rank()

    # Broadcast shape and dtype first (small pickle messages)
    if rank == root:
        shape, dtype = data.shape, data.dtype
    else:
        shape, dtype = None, None

    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)

    if rank != root:
        data = np.empty(shape, dtype=dtype)

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


def chunked_gather(comm, local_data, root: int = 0, max_bytes: int = 2**30):
    """
    Gather numpy arrays from all ranks to root, chunked along axis 0.

    Pickle-mode gather aggregates *all* rank payloads into a single
    pickled list at root, hitting the ~2 GB PyBytes limit even when
    each rank's tile is small. This wrapper:

      1. Computes the **total** aggregate payload across ranks.
      2. If it fits under `max_bytes`, falls through to the cheap
         single-call pickle gather.
      3. Otherwise gathers row-blocks via repeated pickle gathers,
         each of which is bounded by `max_bytes / size`.

    Args:
        comm: MPI communicator
        local_data: Local numpy array (must have same dtype and same
            trailing shape across ranks; axis 0 is the partitioned axis).
        root: Root rank to gather to
        max_bytes: Maximum aggregate bytes per gather call (default 1 GB)

    Returns:
        On root: vertically stacked array from all ranks.
        On non-root: None.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_shape = local_data.shape
    dtype = local_data.dtype
    itemsize = np.dtype(dtype).itemsize

    local_bytes = int(np.prod(local_shape)) * itemsize

    # Gate on TOTAL gathered payload, not per-rank max. The 2 GB pickle
    # limit applies to root's aggregated list, so this is the right scale.
    total_bytes = comm.allreduce(local_bytes, op=MPI.SUM)

    if total_bytes <= max_bytes:
        gathered = comm.gather(local_data, root=root)
        if rank == root:
            return np.vstack(gathered)
        return None

    # Chunked path: gather row-blocks. 1D inputs are treated as columns.
    if len(local_shape) == 1:
        local_data = local_data.reshape(-1, 1)
        local_shape = local_data.shape
        was_1d = True
    else:
        was_1d = False

    n_local_rows = local_shape[0]
    bytes_per_row = int(np.prod(local_shape[1:])) * itemsize

    max_bytes_per_row = comm.allreduce(bytes_per_row, op=MPI.MAX)
    rows_per_chunk = max(1, max_bytes // (max_bytes_per_row * size))

    all_n_rows = comm.allgather(n_local_rows)

    if rank == root:
        total_rows = sum(all_n_rows)
        result = np.empty((total_rows,) + local_shape[1:], dtype=dtype)
        row_offsets = [0]
        for i in range(size - 1):
            row_offsets.append(row_offsets[-1] + all_n_rows[i])
    else:
        result = None

    max_rows = max(all_n_rows)
    for chunk_start in range(0, max_rows, rows_per_chunk):
        chunk_end = min(chunk_start + rows_per_chunk, n_local_rows)

        if chunk_start < n_local_rows:
            chunk = np.ascontiguousarray(local_data[chunk_start:chunk_end])
        else:
            chunk = np.empty((0,) + local_shape[1:], dtype=dtype)

        chunks = comm.gather(chunk, root=root)

        if rank == root:
            for r, ch in enumerate(chunks):
                if ch.size > 0:
                    dest_start = row_offsets[r] + chunk_start
                    dest_end = dest_start + ch.shape[0]
                    result[dest_start:dest_end] = ch

    if rank == root and was_1d:
        result = result.ravel()

    return result


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
        Tuple of (array, window) - window must be freed when done.
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
    Gather arrays from all ranks to root, **safely** across the 2 GB limit.

    Thin wrapper over `chunked_gather` that returns a vstacked array
    on root and None elsewhere. Kept as a convenience name; new code
    should prefer `chunked_gather` directly.
    """
    return chunked_gather(comm, local_data, root=root)


def allreduce_sum(comm, local_array):
    """
    Allreduce with sum operation, buffer-mode (safe for any size up to
    ~2 GB per rank; chunk yourself if you exceed that).

    Args:
        comm: MPI communicator
        local_array: Local numpy array

    Returns:
        Sum across all ranks (numpy array, same shape).
    """
    global_array = np.zeros_like(local_array)
    comm.Allreduce(local_array, global_array, op=MPI.SUM)
    return global_array
