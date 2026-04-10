"""
End-to-end tests for fragpigeon Python bindings.

Tests ring, LOA, header, COI table, and cross-language interop with Go.
Run: python -m pytest test_fragpigeon.py -v
"""

import mmap
import os
import struct
import tempfile
import time
import zlib

import pytest

from fragpigeon import (
    COITable,
    Header,
    LOAPool,
    LOARef,
    Msg,
    Ring,
    DTYPE_BF16,
    FLAG_DROP_OK,
    FLAG_LOA_PTR,
    HDR_SIZE,
    KIND_PING,
    KIND_PROCESS,
    LOA_MAGIC,
    LOA_REF_SIZE,
    SCHEMA_ACTIVATION,
    SCHEMA_KV_CACHE,
    SCHEMA_RAW,
    SCHEMA_WEIGHT_SHARD,
    SLOT_FREE,
    SLOT_READY,
    read_loa,
    write_loa,
)


# ================================================================
# Helpers
# ================================================================

def make_ring(tmp_path, cap_slots=64, slot_size=512):
    """Create a temp ring shm file and return a Ring."""
    path = os.path.join(tmp_path, "ring.shm")
    size = 64 + cap_slots * slot_size
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)

    struct.pack_into("<Q", mm, 0, cap_slots)
    struct.pack_into("<Q", mm, 8, 0)               # ProdIdx
    struct.pack_into("<Q", mm, 16, 0)              # ConsIdx
    struct.pack_into("<I", mm, 24, slot_size)
    struct.pack_into("<Q", mm, 32, 0xFFFFFFFFFFFFFFFF)  # no eventfd
    struct.pack_into("<Q", mm, 40, 0xFFFFFFFFFFFFFFFF)
    mm.close()

    ring = Ring.from_fd(fd)
    os.close(fd)
    os.unlink(path)
    return ring


def make_payload(size):
    """Deterministic payload with CRC32 trailer (matches Go's GeneratePayload)."""
    buf = bytearray(size)
    for i in range(size - 4):
        buf[i] = (i * 7 + 13) & 0xFF
    crc = zlib.crc32(buf[:size - 4]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, size - 4, crc)
    return bytes(buf)


def verify_payload(data):
    """Verify CRC32 trailer."""
    if len(data) < 8:
        return False
    expected = struct.unpack_from("<I", data, len(data) - 4)[0]
    actual = zlib.crc32(data[:len(data) - 4]) & 0xFFFFFFFF
    return expected == actual


def make_header(payload_size, msg_id=1, schema=SCHEMA_RAW, kind=KIND_PROCESS):
    return Header(
        length=payload_size,
        kind=kind,
        ts_ns=int(time.time() * 1e9),
        concept_id=0x8A7311CCDD55002A,
        concept_bits=24,
        schema_id=schema,
        src_id=1001,
        msg_id=msg_id,
        ver=1,
    )


# ================================================================
# Header Tests
# ================================================================

class TestHeader:
    def test_pack_unpack_roundtrip(self):
        hdr = Header(
            length=1234, kind=KIND_PROCESS, flags=FLAG_LOA_PTR,
            ts_ns=0xDEADBEEFCAFE0001, concept_id=0x8A7311CCDD55002A,
            concept_bits=28, schema_id=42, src_id=1001, msg_id=7,
            hop=3, ver=1, trace_id=0x1122334455667788, checksum32=0xAABBCCDD,
        )
        packed = hdr.pack()
        assert len(packed) == HDR_SIZE

        got = Header.unpack(packed)
        assert got.length == 1234
        assert got.kind == KIND_PROCESS
        assert got.flags == FLAG_LOA_PTR
        assert got.ts_ns == 0xDEADBEEFCAFE0001
        assert got.concept_id == 0x8A7311CCDD55002A
        assert got.concept_bits == 28
        assert got.schema_id == 42
        assert got.src_id == 1001
        assert got.msg_id == 7
        assert got.hop == 3
        assert got.ver == 1
        assert got.trace_id == 0x1122334455667788
        assert got.checksum32 == 0xAABBCCDD

    def test_is_loa(self):
        hdr = Header(flags=FLAG_LOA_PTR)
        assert hdr.is_loa
        hdr2 = Header(flags=0)
        assert not hdr2.is_loa


# ================================================================
# LOARef Tests
# ================================================================

class TestLOARef:
    def test_encode_decode(self):
        ref = LOARef(pool_id=5, slot_id=1234, offset=0, length=65536)
        data = ref.encode()
        assert len(data) == LOA_REF_SIZE

        got = LOARef.decode(data)
        assert got.pool_id == 5
        assert got.slot_id == 1234
        assert got.offset == 0
        assert got.length == 65536


# ================================================================
# Ring Tests
# ================================================================

class TestRing:
    def test_write_read_roundtrip(self, tmp_path):
        ring = make_ring(str(tmp_path))
        payload = make_payload(128)
        hdr = make_header(128, msg_id=42)

        assert ring.try_write(hdr, payload)
        msg = ring.try_read()
        assert msg is not None
        assert msg.header.msg_id == 42
        assert msg.header.length == 128
        assert verify_payload(msg.payload)
        ring.close()

    def test_empty_read(self, tmp_path):
        ring = make_ring(str(tmp_path))
        assert ring.try_read() is None
        ring.close()

    def test_full_ring(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=4, slot_size=256)
        payload = make_payload(64)
        hdr = make_header(64)

        for i in range(4):
            assert ring.try_write(hdr, payload), f"write {i} should succeed"
        assert not ring.try_write(hdr, payload), "ring should be full"

        # Drain one
        msg = ring.try_read()
        assert msg is not None
        assert ring.try_write(hdr, payload), "write after drain should succeed"
        ring.close()

    def test_ordering(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=128)
        payload = make_payload(32)

        for i in range(50):
            hdr = make_header(32, msg_id=i)
            assert ring.try_write(hdr, payload)

        for i in range(50):
            msg = ring.try_read()
            assert msg is not None
            assert msg.header.msg_id == i

        ring.close()

    def test_read_within_timeout(self, tmp_path):
        ring = make_ring(str(tmp_path))
        start = time.monotonic()
        msg = ring.read_within(0.05)
        elapsed = time.monotonic() - start

        assert msg is None
        assert elapsed >= 0.04
        ring.close()

    def test_various_payload_sizes(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=32, slot_size=1024)
        for size in [8, 64, 256, 512, 900]:
            payload = make_payload(size)
            hdr = make_header(size, msg_id=size)
            assert ring.try_write(hdr, payload)
            msg = ring.try_read()
            assert msg is not None
            assert msg.header.msg_id == size
            assert verify_payload(msg.payload)
        ring.close()


# ================================================================
# LOA Pool Tests
# ================================================================

class TestLOAPool:
    def test_create_open(self, tmp_path):
        path = os.path.join(str(tmp_path), "test.loa")
        pool = LOAPool.create(path, pool_id=7, num_slots=8, slot_size=1024)
        assert pool.pool_id == 7
        assert pool.num_slots == 8
        assert pool.slot_size == 1024
        pool.close()

        pool2 = LOAPool.open(path)
        assert pool2.pool_id == 7
        assert pool2.num_slots == 8
        pool2.close()

    def test_alloc_commit_deref_release(self, tmp_path):
        path = os.path.join(str(tmp_path), "test.loa")
        pool = LOAPool.create(path, pool_id=1, num_slots=8, slot_size=4096)

        data = make_payload(256)
        buf, ref = pool.alloc(256, owner_frag_id=1001)
        buf[:] = data
        pool.commit(ref)

        assert ref.pool_id == 1
        assert ref.length == 256

        got = pool.deref(ref)
        assert bytes(got) == data
        assert verify_payload(bytes(got))

        pool.release(ref)
        pool.close()

    def test_pool_full(self, tmp_path):
        path = os.path.join(str(tmp_path), "test.loa")
        pool = LOAPool.create(path, pool_id=1, num_slots=2, slot_size=256)

        _, ref0 = pool.alloc(10, 1)
        pool.commit(ref0)
        _, ref1 = pool.alloc(10, 1)
        pool.commit(ref1)

        with pytest.raises(RuntimeError, match="full"):
            pool.alloc(10, 1)

        pool.release(ref0)
        _, ref2 = pool.alloc(10, 1)
        pool.commit(ref2)
        pool.release(ref1)
        pool.release(ref2)
        pool.close()

    def test_oversized(self, tmp_path):
        path = os.path.join(str(tmp_path), "test.loa")
        pool = LOAPool.create(path, pool_id=1, num_slots=4, slot_size=256)
        with pytest.raises(ValueError, match="exceeds"):
            pool.alloc(1024, 1)
        pool.close()

    def test_multiple_readers(self, tmp_path):
        path = os.path.join(str(tmp_path), "test.loa")
        pool = LOAPool.create(path, pool_id=1, num_slots=4, slot_size=1024)

        data = b"shared tensor data" + b"\x00" * (64 - 18)
        buf, ref = pool.alloc(64, 1)
        buf[:] = data
        pool.commit(ref)

        for _ in range(5):
            got = pool.deref(ref)
            assert bytes(got) == data

        for _ in range(5):
            pool.release(ref)

        pool.close()


# ================================================================
# LOA + Ring Integration
# ================================================================

class TestLOARing:
    def test_write_loa_read_loa(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=32, slot_size=256)
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=16, slot_size=4096)

        data = make_payload(2048)
        hdr = make_header(2048, msg_id=99, schema=SCHEMA_WEIGHT_SHARD)

        ref = write_loa(ring, pool, hdr, data, owner_frag_id=1001)
        assert ref is not None

        result = read_loa(ring, pool)
        assert result is not None
        msg, got_data, got_ref = result
        assert msg.is_loa
        assert msg.header.msg_id == 99
        assert msg.header.schema_id == SCHEMA_WEIGHT_SHARD
        assert verify_payload(bytes(got_data))
        assert got_ref is not None
        pool.release(got_ref)

        ring.close()
        pool.close()

    def test_inline_via_read_loa(self, tmp_path):
        ring = make_ring(str(tmp_path), cap_slots=16, slot_size=256)
        pool_path = os.path.join(str(tmp_path), "loa.shm")
        pool = LOAPool.create(pool_path, pool_id=1, num_slots=4, slot_size=1024)

        payload = b"hello inline"
        hdr = Header(length=len(payload), kind=KIND_PING, ver=1)
        assert ring.try_write(hdr, payload)

        result = read_loa(ring, pool)
        assert result is not None
        msg, data, ref = result
        assert not msg.is_loa
        assert ref is None
        assert bytes(data) == b"hello inline"

        ring.close()
        pool.close()


# ================================================================
# Cross-Language Interop (Python writes, Go reads — layout compat)
# ================================================================

class TestInterop:
    def test_header_binary_compat(self):
        """Verify Python header packing matches Go's binary layout exactly."""
        hdr = Header(
            length=512, kind=KIND_PROCESS, flags=0,
            ts_ns=1234567890, concept_id=0xAABBCCDDEEFF0011,
            concept_bits=24, schema_id=1, src_id=1001,
            msg_id=7, hop=0, ver=1, trace_id=0, checksum32=0,
        )
        packed = hdr.pack()

        # Manually verify field offsets (must match Go header.go)
        assert struct.unpack_from("<I", packed, 0)[0] == 512          # Len at 0
        assert struct.unpack_from("<H", packed, 4)[0] == KIND_PROCESS # Kind at 4
        assert struct.unpack_from("<H", packed, 6)[0] == 0            # Flags at 6
        assert struct.unpack_from("<Q", packed, 8)[0] == 1234567890   # TSns at 8
        assert struct.unpack_from("<Q", packed, 16)[0] == 0xAABBCCDDEEFF0011  # ConceptID at 16
        assert struct.unpack_from("<H", packed, 24)[0] == 24          # ConceptBits at 24
        assert struct.unpack_from("<H", packed, 26)[0] == 1           # SchemaID at 26
        assert struct.unpack_from("<I", packed, 28)[0] == 1001        # SrcID at 28
        assert struct.unpack_from("<I", packed, 32)[0] == 7           # MsgID at 32

    def test_loa_ref_binary_compat(self):
        """Verify LOARef encoding matches Go layout."""
        ref = LOARef(pool_id=1, slot_id=42, offset=0, length=65536)
        data = ref.encode()
        assert struct.unpack_from("<H", data, 0)[0] == 1     # PoolID
        assert struct.unpack_from("<H", data, 2)[0] == 42    # SlotID
        assert struct.unpack_from("<I", data, 4)[0] == 0     # Offset
        assert struct.unpack_from("<I", data, 8)[0] == 65536 # Length

    def test_loa_pool_magic_compat(self, tmp_path):
        """Verify Python-created LOA pool has correct magic for Go to open."""
        path = os.path.join(str(tmp_path), "interop.loa")
        pool = LOAPool.create(path, pool_id=1, num_slots=4, slot_size=1024)
        pool.close()

        with open(path, "rb") as f:
            magic = struct.unpack("<Q", f.read(8))[0]
        assert magic == LOA_MAGIC

    def test_payload_crc_compat(self):
        """Verify CRC32 matches Go's GeneratePayload/VerifyPayload."""
        # This is the exact same algorithm as Go's proof/harness.go
        payload = make_payload(64)
        assert verify_payload(payload)
        # Corrupt one byte
        corrupted = bytearray(payload)
        corrupted[0] ^= 0xFF
        assert not verify_payload(bytes(corrupted))


# ================================================================
# Benchmark-style performance test
# ================================================================

class TestPerformance:
    def test_ring_throughput(self, tmp_path):
        """Measure ring write+read throughput from Python."""
        ring = make_ring(str(tmp_path), cap_slots=1024, slot_size=256)
        payload = make_payload(128)
        hdr = make_header(128)
        n = 10_000

        start = time.monotonic()
        for i in range(n):
            hdr.msg_id = i
            assert ring.try_write(hdr, payload)
            msg = ring.try_read()
            assert msg is not None
        elapsed = time.monotonic() - start

        throughput_mb = (n * 128) / elapsed / 1e6
        ops_per_sec = n / elapsed
        print(f"\nPython ring: {ops_per_sec:,.0f} ops/s, {throughput_mb:.1f} MB/s ({elapsed*1000:.1f}ms)")
        ring.close()

    def test_loa_throughput(self, tmp_path):
        """Measure LOA alloc+commit+deref+release throughput from Python."""
        path = os.path.join(str(tmp_path), "perf.loa")
        pool = LOAPool.create(path, pool_id=1, num_slots=256, slot_size=4096)
        n = 10_000
        payload = make_payload(1024)

        start = time.monotonic()
        for i in range(n):
            buf, ref = pool.alloc(1024, 1)
            buf[:] = payload
            pool.commit(ref)
            data = pool.deref(ref)
            _ = bytes(data[0:1])  # touch it
            pool.release(ref)
        elapsed = time.monotonic() - start

        throughput_mb = (n * 1024) / elapsed / 1e6
        ops_per_sec = n / elapsed
        print(f"\nPython LOA: {ops_per_sec:,.0f} ops/s, {throughput_mb:.1f} MB/s ({elapsed*1000:.1f}ms)")
        pool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
