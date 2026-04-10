"""
Tests for fragpigeon_numpy — NumPy zero-copy tensor helpers.

Run: python -m pytest test_numpy.py -v -s
"""

import os
import struct
import time

import numpy as np
import pytest

from fragpigeon import (
    Header,
    LOAPool,
    Ring,
    FLAG_LOA_PTR,
    KIND_PROCESS,
    LOA_REF_SIZE,
    SCHEMA_WEIGHT_SHARD,
    SCHEMA_ACTIVATION,
)
from fragpigeon_numpy import (
    alloc_tensor,
    commit_tensor,
    deref_tensor,
    deref_tensor_copy,
    read_tensor,
    write_tensor,
    write_tensor_zerocopy,
)
import mmap


def make_ring(tmp_path, cap_slots=64, slot_size=512):
    path = os.path.join(tmp_path, "ring.shm")
    size = 64 + cap_slots * slot_size
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
    struct.pack_into("<Q", mm, 0, cap_slots)
    struct.pack_into("<Q", mm, 8, 0)
    struct.pack_into("<Q", mm, 16, 0)
    struct.pack_into("<I", mm, 24, slot_size)
    struct.pack_into("<Q", mm, 32, 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<Q", mm, 40, 0xFFFFFFFFFFFFFFFF)
    mm.close()
    ring = Ring.from_fd(fd)
    os.close(fd)
    os.unlink(path)
    return ring


class TestAllocTensor:
    def test_alloc_f16(self, tmp_path):
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=8, slot_size=65536)

        arr, ref = alloc_tensor(pool, shape=(32, 64), dtype=np.float16, owner_frag_id=1)
        assert arr.shape == (32, 64)
        assert arr.dtype == np.float16
        assert ref.length == 32 * 64 * 2  # 4096 bytes

        # Write directly into shm
        arr[:] = np.ones((32, 64), dtype=np.float16) * 3.14
        pool.commit(ref)

        # Verify data persists in shm
        got = deref_tensor(pool, ref, (32, 64), np.float16)
        np.testing.assert_allclose(got, 3.14, rtol=1e-2)

        pool.release(ref)
        pool.close()

    def test_alloc_f32(self, tmp_path):
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=8, slot_size=65536)

        arr, ref = alloc_tensor(pool, shape=(100,), dtype=np.float32, owner_frag_id=1)
        assert arr.shape == (100,)
        assert ref.length == 400

        arr[:] = np.arange(100, dtype=np.float32)
        pool.commit(ref)

        got = deref_tensor(pool, ref, (100,), np.float32)
        np.testing.assert_array_equal(got, np.arange(100, dtype=np.float32))

        pool.release(ref)
        pool.close()


class TestDerefTensor:
    def test_deref_copy(self, tmp_path):
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=4, slot_size=8192)

        arr, ref = alloc_tensor(pool, (16, 16), np.float32, 1)
        arr[:] = np.eye(16, dtype=np.float32)
        pool.commit(ref)

        # deref_tensor_copy returns a copy — safe after release
        got = deref_tensor_copy(pool, ref, (16, 16), np.float32)
        pool.release(ref)

        # Array still valid after release (it's a copy)
        np.testing.assert_array_equal(got, np.eye(16, dtype=np.float32))
        pool.close()

    def test_deref_view_zerocopy(self, tmp_path):
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=4, slot_size=8192)

        arr, ref = alloc_tensor(pool, (64,), np.int32, 1)
        arr[:] = np.arange(64, dtype=np.int32)
        pool.commit(ref)

        # deref_tensor returns a zero-copy view
        got = deref_tensor(pool, ref, (64,), np.int32)
        assert got[0] == 0
        assert got[63] == 63

        pool.release(ref)
        pool.close()


class TestWriteReadTensor:
    def test_write_read_f16(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=16, slot_size=256)
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=8, slot_size=65536)

        # Write tensor
        src = np.random.randn(128, 64).astype(np.float16)
        hdr = Header(
            kind=KIND_PROCESS,
            schema_id=SCHEMA_WEIGHT_SHARD,
            concept_id=0x0001000000000000,
            concept_bits=16,
            src_id=1,
            msg_id=1,
        )
        ref = write_tensor(ring, pool, src, hdr, owner_frag_id=1)
        assert ref is not None

        # Read tensor
        result = read_tensor(ring, pool, (128, 64), np.float16, copy=True)
        assert result is not None
        got_hdr, got_arr, got_ref = result
        assert got_hdr.schema_id == SCHEMA_WEIGHT_SHARD
        assert got_hdr.msg_id == 1
        np.testing.assert_array_equal(got_arr, src)
        assert got_ref is None  # already released since copy=True

        ring.close()
        pool.close()

    def test_zerocopy_write_path(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=16, slot_size=256)
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=8, slot_size=65536)

        # Allocate tensor directly in shm
        hdr = Header(
            kind=KIND_PROCESS,
            schema_id=SCHEMA_ACTIVATION,
            concept_id=0x00AA000000000000,
            concept_bits=16,
            src_id=2,
            msg_id=42,
        )
        arr, ref = write_tensor_zerocopy(ring, pool, (256,), np.float32, hdr, owner_frag_id=2)

        # Fill in-place (zero-copy into shm)
        arr[:] = np.linspace(0, 1, 256, dtype=np.float32)

        # Commit and send
        assert commit_tensor(ring, pool, ref, hdr)

        # Read back
        result = read_tensor(ring, pool, (256,), np.float32, copy=True)
        assert result is not None
        _, got_arr, _ = result
        np.testing.assert_allclose(got_arr, np.linspace(0, 1, 256, dtype=np.float32), rtol=1e-6)

        ring.close()
        pool.close()


class TestPerformance:
    def test_numpy_loa_throughput(self, tmp_path):
        """Measure numpy tensor alloc+fill+commit+deref throughput."""
        pool_path = os.path.join(str(tmp_path), "perf.loa")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=64, slot_size=65536)

        shape = (128, 128)  # 32KB at float16
        n = 1000
        template = np.random.randn(*shape).astype(np.float16)

        start = time.monotonic()
        for _ in range(n):
            arr, ref = alloc_tensor(pool, shape, np.float16, 1)
            arr[:] = template  # ~32KB memcpy into shm
            pool.commit(ref)

            got = deref_tensor(pool, ref, shape, np.float16)
            _ = got[0, 0]  # touch it
            pool.release(ref)
        elapsed = time.monotonic() - start

        bytes_per_op = 128 * 128 * 2
        throughput_mb = (n * bytes_per_op) / elapsed / 1e6
        ops_per_sec = n / elapsed
        print(f"\nNumPy LOA: {ops_per_sec:,.0f} ops/s, {throughput_mb:.1f} MB/s "
              f"({shape} float16, {elapsed*1000:.1f}ms)")

        pool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
