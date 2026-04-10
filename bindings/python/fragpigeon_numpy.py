"""
fragpigeon_numpy — NumPy and PyTorch zero-copy helpers for fragmind-pigeon.

Provides tensor-aware wrappers around the LOA pool:
- alloc_tensor: returns a numpy array backed by LOA shared memory
- deref_tensor: returns a read-only numpy view of a tensor in the LOA pool
- write_tensor: writes a numpy/torch tensor via LOA + ring
- read_tensor: reads a tensor from LOA + ring

Usage:
    from fragpigeon_numpy import alloc_tensor, deref_tensor, write_tensor, read_tensor
    import numpy as np

    # Writer: allocate tensor directly in shm
    arr, ref = alloc_tensor(pool, shape=(1024, 1024), dtype=np.float16, owner=1)
    arr[:] = my_weights  # writes directly to shm, no copy
    pool.commit(ref)

    # Reader: zero-copy view
    arr = deref_tensor(pool, ref, shape=(1024, 1024), dtype=np.float16)
    # arr is a numpy view into shm — no copy, no GC pressure

Requires: numpy (for numpy helpers), torch (for torch helpers, optional)
"""

from __future__ import annotations

import struct
from typing import Optional, TYPE_CHECKING

from fragpigeon import (
    Header,
    LOAPool,
    LOARef,
    Ring,
    FLAG_LOA_PTR,
    LOA_REF_SIZE,
    SCHEMA_WEIGHT_SHARD,
    SCHEMA_KV_CACHE,
    SCHEMA_ACTIVATION,
    SCHEMA_GRADIENT,
)

if TYPE_CHECKING:
    import numpy as np
    import torch


# ================================================================
# NumPy helpers
# ================================================================

def alloc_tensor(
    pool: LOAPool,
    shape: tuple[int, ...],
    dtype: "np.dtype | type",
    owner_frag_id: int = 0,
) -> tuple["np.ndarray", LOARef]:
    """Allocate a tensor directly in LOA shared memory.

    Returns a writable numpy array whose buffer IS the shm slot.
    No copy — writes go directly to shared memory.
    Call pool.commit(ref) after filling the array.

    Args:
        pool: LOA pool handle
        shape: tensor shape (e.g., (1024, 1024))
        dtype: numpy dtype (e.g., np.float16, np.bfloat16)
        owner_frag_id: owning fragment ID

    Returns:
        (array, LOARef) — array is writable, backed by shm
    """
    import numpy as np

    dt = np.dtype(dtype)
    nbytes = int(dt.itemsize * _prod(shape))
    buf, ref = pool.alloc(nbytes, owner_frag_id)

    # Create numpy array that shares the memoryview buffer
    arr = np.frombuffer(buf, dtype=dt).reshape(shape)
    return arr, ref


def deref_tensor(
    pool: LOAPool,
    ref: LOARef,
    shape: tuple[int, ...],
    dtype: "np.dtype | type",
) -> "np.ndarray":
    """Get a zero-copy numpy view of a tensor in LOA shared memory.

    The returned array is a VIEW into shm — no copy. The caller must
    call pool.release(ref) when done (after which the array is invalid).

    For a safe copy, use deref_tensor_copy() instead.
    """
    import numpy as np

    dt = np.dtype(dtype)
    view = pool.deref_view(ref)  # memoryview into shm
    arr = np.frombuffer(view, dtype=dt).reshape(shape)
    return arr


def deref_tensor_copy(
    pool: LOAPool,
    ref: LOARef,
    shape: tuple[int, ...],
    dtype: "np.dtype | type",
) -> "np.ndarray":
    """Get a copied numpy array from LOA (safe to use after release)."""
    import numpy as np

    dt = np.dtype(dtype)
    data = pool.deref(ref)  # bytes copy
    return np.frombuffer(data, dtype=dt).reshape(shape).copy()


def write_tensor(
    ring: Ring,
    pool: LOAPool,
    arr: "np.ndarray",
    header: Header,
    owner_frag_id: int = 0,
) -> Optional[LOARef]:
    """Write a numpy tensor via LOA and send pointer over ring.

    Copies arr data into an LOA slot, commits, and sends
    an LOA pointer message over the ring.
    """
    data = arr.tobytes()
    header.length = LOA_REF_SIZE
    header.flags |= FLAG_LOA_PTR

    buf, ref = pool.alloc(len(data), owner_frag_id)
    buf[:] = data
    pool.commit(ref)

    ref_bytes = ref.encode()
    if not ring.try_write(header, ref_bytes):
        pool.release(ref)
        return None
    return ref


def write_tensor_zerocopy(
    ring: Ring,
    pool: LOAPool,
    shape: tuple[int, ...],
    dtype: "np.dtype | type",
    header: Header,
    owner_frag_id: int = 0,
) -> tuple["np.ndarray", LOARef]:
    """Allocate tensor in shm, return writable array.

    Caller fills the array, then calls commit_tensor() to send.
    This is the true zero-copy write path.
    """
    arr, ref = alloc_tensor(pool, shape, dtype, owner_frag_id)
    return arr, ref


def commit_tensor(ring: Ring, pool: LOAPool, ref: LOARef, header: Header) -> bool:
    """Commit an allocated tensor and send pointer over ring."""
    pool.commit(ref)
    header.length = LOA_REF_SIZE
    header.flags |= FLAG_LOA_PTR
    ref_bytes = ref.encode()
    if not ring.try_write(header, ref_bytes):
        pool.release(ref)
        return False
    return True


def read_tensor(
    ring: Ring,
    pool: LOAPool,
    shape: tuple[int, ...],
    dtype: "np.dtype | type",
    timeout_s: float = 0,
    copy: bool = False,
) -> Optional[tuple[Header, "np.ndarray", Optional[LOARef]]]:
    """Read a tensor from ring + LOA.

    Args:
        copy: if True, returns a copy (safe after release).
              if False, returns a view into shm (must release ref when done).

    Returns:
        (header, array, ref) or None if timeout.
        If ref is not None, caller must call pool.release(ref) when done.
    """
    import numpy as np
    from fragpigeon import read_loa

    result = read_loa(ring, pool, timeout_s)
    if result is None:
        return None

    msg, data, ref = result
    dt = np.dtype(dtype)

    if copy or ref is None:
        arr = np.frombuffer(data if isinstance(data, bytes) else bytes(data), dtype=dt).reshape(shape).copy()
        if ref is not None:
            pool.release(ref)
            ref = None
    else:
        # Zero-copy: deref_view for the actual tensor data
        view = pool.deref_view(ref)
        arr = np.frombuffer(view, dtype=dt).reshape(shape)
        # Note: caller gets TWO refs — one from read_loa, one from deref_view.
        # We release the read_loa one; caller releases deref_view's via the returned ref.

    return msg.header, arr, ref


# ================================================================
# PyTorch helpers
# ================================================================

def loa_to_torch(
    pool: LOAPool,
    ref: LOARef,
    shape: tuple[int, ...],
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Create a PyTorch tensor from an LOA slot (zero-copy if possible).

    Uses numpy as bridge. The tensor shares memory with the LOA pool.
    Caller must pool.release(ref) when done with the tensor.
    """
    import torch
    import numpy as np

    torch_to_numpy = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.dtype('V2'),  # bfloat16 via raw view
        torch.int8: np.int8,
        torch.int32: np.int32,
        torch.uint8: np.uint8,
    }
    np_dtype = torch_to_numpy.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported dtype: {dtype}")

    view = pool.deref_view(ref)
    np_arr = np.frombuffer(view, dtype=np_dtype).reshape(shape)

    if dtype == torch.bfloat16:
        # numpy doesn't support bfloat16 natively; view as uint16 and cast
        np_arr = np.frombuffer(view, dtype=np.uint16).reshape(shape)
        t = torch.from_numpy(np_arr.copy()).view(torch.bfloat16)
    else:
        t = torch.from_numpy(np_arr)
    return t


def torch_to_loa(
    pool: LOAPool,
    tensor: "torch.Tensor",
    owner_frag_id: int = 0,
) -> tuple[LOARef, None]:
    """Copy a PyTorch tensor into an LOA slot.

    Returns LOARef. Call pool.commit(ref) after this.
    """
    data = tensor.contiguous().numpy().tobytes()
    buf, ref = pool.alloc(len(data), owner_frag_id)
    buf[:] = data
    return ref, None


# ================================================================
# Utilities
# ================================================================

def _prod(shape: tuple[int, ...]) -> int:
    result = 1
    for s in shape:
        result *= s
    return result
