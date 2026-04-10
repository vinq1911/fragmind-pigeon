"""
fragpigeon — Python bindings for fragmind-pigeon shared-memory IPC.

Pure Python (ctypes + mmap), no compiled extensions needed.
Matches the Go binary layout byte-for-byte so Python processes can
read/write the same shm rings and LOA pools the Go pigeon creates.

Usage:
    from fragpigeon import Ring, LOAPool, Header, LOARef

    # Open an existing ring (fd passed from pigeon or coordinator)
    ring = Ring.from_fd(fd)
    ring.try_write(header, payload)
    msg = ring.try_read()

    # Open an existing LOA pool
    pool = LOAPool.open("/dev/shm/fragmind.loa.1")
    buf, ref = pool.alloc(4096, owner_frag_id=1)
    buf[:] = tensor_data
    pool.commit(ref)
    data = pool.deref(ref)
    pool.release(ref)
"""

import ctypes
import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

# ================================================================
# Constants
# ================================================================

HDR_SIZE = 64
LOA_HEADER_SIZE = 64
LOA_SLOT_META_SIZE = 32
LOA_MAGIC = 0x4C4F41504F4F4C31  # "LOAPOOL1"
LOA_REF_SIZE = 12

# Message kinds
KIND_PROCESS = 0x0001
KIND_LEARN = 0x0002
KIND_SHARE = 0x0003
KIND_PING = 0x0004

# Flags
FLAG_END_OF_STREAM = 0x0001
FLAG_URGENT = 0x0002
FLAG_REPLY = 0x0004
FLAG_DROP_OK = 0x0008
FLAG_LOA_PTR = 0x0010

# Schema IDs
SCHEMA_RAW = 0
SCHEMA_WEIGHT_SHARD = 1
SCHEMA_KV_CACHE = 2
SCHEMA_ACTIVATION = 3
SCHEMA_TOKEN_BATCH = 4
SCHEMA_GRADIENT = 5
SCHEMA_CONTROL = 0xFFFF

# Data types
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2
DTYPE_FP8E4 = 3
DTYPE_FP8E5 = 4
DTYPE_I8 = 5
DTYPE_I32 = 6
DTYPE_U32 = 7

DTYPE_SIZES = {
    DTYPE_F32: 4, DTYPE_F16: 2, DTYPE_BF16: 2,
    DTYPE_FP8E4: 1, DTYPE_FP8E5: 1, DTYPE_I8: 1,
    DTYPE_I32: 4, DTYPE_U32: 4,
}

# Slot states
SLOT_FREE = 0
SLOT_ALLOCATING = 1
SLOT_READY = 2


# ================================================================
# Header
# ================================================================

# Header layout: all little-endian
#  0: u32  len
#  4: u16  kind
#  6: u16  flags
#  8: u64  ts_ns
# 16: u64  concept_id
# 24: u16  concept_bits
# 26: u16  schema_id
# 28: u32  src_id
# 32: u32  msg_id
# 36: u16  hop
# 38: u16  ver
# 40: u64  trace_id
# 48: u32  checksum32
# 52: u32  _reserved

_HDR_FMT = "<IHHQQHHIIHHQI"  # 52 bytes of fields
_HDR_STRUCT = struct.Struct(_HDR_FMT)
# Go header is 64 bytes: 52 of fields + 4 reserved + 8 padding
assert _HDR_STRUCT.size == 52


@dataclass
class Header:
    length: int = 0
    kind: int = KIND_PROCESS
    flags: int = 0
    ts_ns: int = 0
    concept_id: int = 0
    concept_bits: int = 0
    schema_id: int = 0
    src_id: int = 0
    msg_id: int = 0
    hop: int = 0
    ver: int = 1
    trace_id: int = 0
    checksum32: int = 0

    def pack(self) -> bytes:
        buf = _HDR_STRUCT.pack(
            self.length, self.kind, self.flags, self.ts_ns,
            self.concept_id, self.concept_bits, self.schema_id,
            self.src_id, self.msg_id, self.hop, self.ver,
            self.trace_id, self.checksum32,
        )
        return buf + b"\x00" * (HDR_SIZE - len(buf))  # pad to 64

    @classmethod
    def unpack(cls, data: bytes) -> "Header":
        vals = _HDR_STRUCT.unpack(data[:_HDR_STRUCT.size])
        return cls(
            length=vals[0], kind=vals[1], flags=vals[2], ts_ns=vals[3],
            concept_id=vals[4], concept_bits=vals[5], schema_id=vals[6],
            src_id=vals[7], msg_id=vals[8], hop=vals[9], ver=vals[10],
            trace_id=vals[11], checksum32=vals[12],
        )

    @property
    def is_loa(self) -> bool:
        return bool(self.flags & FLAG_LOA_PTR)


# ================================================================
# LOARef
# ================================================================

_LOA_REF_FMT = "<HHII"
_LOA_REF_STRUCT = struct.Struct(_LOA_REF_FMT)
assert _LOA_REF_STRUCT.size == LOA_REF_SIZE


@dataclass
class LOARef:
    pool_id: int = 0
    slot_id: int = 0
    offset: int = 0
    length: int = 0

    def encode(self) -> bytes:
        return _LOA_REF_STRUCT.pack(self.pool_id, self.slot_id, self.offset, self.length)

    @classmethod
    def decode(cls, data: bytes) -> "LOARef":
        vals = _LOA_REF_STRUCT.unpack(data[:LOA_REF_SIZE])
        return cls(pool_id=vals[0], slot_id=vals[1], offset=vals[2], length=vals[3])


# ================================================================
# Message
# ================================================================

@dataclass
class Msg:
    header: Header
    payload: bytes

    @property
    def is_loa(self) -> bool:
        return self.header.is_loa

    def loa_ref(self) -> LOARef:
        return LOARef.decode(self.payload)


# ================================================================
# Ring Buffer
# ================================================================

class Ring:
    """SPSC shared-memory ring buffer, matching Go's Ring layout."""

    def __init__(self, mm: mmap.mmap, size: int):
        self._mm = mm
        self._size = size
        # Read control header
        ctrl = mm[:64]
        self._cap_slots = struct.unpack_from("<Q", ctrl, 0)[0]
        self._slot_size = struct.unpack_from("<I", ctrl, 24)[0]
        self._slots_base = 64

    @classmethod
    def from_fd(cls, fd: int) -> "Ring":
        """Open a ring from a file descriptor (e.g., passed via SCM_RIGHTS)."""
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        return cls(mm, size)

    @classmethod
    def from_path(cls, path: str) -> "Ring":
        """Open a ring from a file path."""
        fd = os.open(path, os.O_RDWR)
        try:
            return cls.from_fd(fd)
        finally:
            os.close(fd)

    def close(self):
        self._mm.close()

    def _slot_offset(self, idx: int) -> int:
        return self._slots_base + (idx & (self._cap_slots - 1)) * self._slot_size

    def _read_u64(self, offset: int) -> int:
        return struct.unpack_from("<Q", self._mm, offset)[0]

    def _write_u64(self, offset: int, val: int):
        struct.pack_into("<Q", self._mm, offset, val)

    @property
    def prod_idx(self) -> int:
        return self._read_u64(8)

    @property
    def cons_idx(self) -> int:
        return self._read_u64(16)

    def try_write(self, hdr: Header, payload: bytes) -> bool:
        """Write a message to the ring. Returns False if full."""
        prod = self.prod_idx
        cons = self.cons_idx
        if prod - cons >= self._cap_slots:
            return False
        if hdr.length + HDR_SIZE > self._slot_size:
            return False

        off = self._slot_offset(prod)
        packed_hdr = hdr.pack()
        self._mm[off:off + HDR_SIZE] = packed_hdr
        self._mm[off + HDR_SIZE:off + HDR_SIZE + hdr.length] = payload
        self._write_u64(8, prod + 1)
        return True

    def try_read(self) -> Optional[Msg]:
        """Read a message from the ring. Returns None if empty."""
        prod = self.prod_idx
        cons = self.cons_idx
        if prod == cons:
            return None

        off = self._slot_offset(cons)
        hdr = Header.unpack(bytes(self._mm[off:off + HDR_SIZE]))
        payload = bytes(self._mm[off + HDR_SIZE:off + HDR_SIZE + hdr.length])
        self._write_u64(16, cons + 1)
        return Msg(header=hdr, payload=payload)

    def read_within(self, timeout_s: float) -> Optional[Msg]:
        """Poll for a message with timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            msg = self.try_read()
            if msg is not None:
                return msg
            if time.monotonic() >= deadline:
                return None
            time.sleep(50e-6)  # 50µs spin


# ================================================================
# LOA Pool
# ================================================================

class LOAPool:
    """Shared-memory LOA (Large Object Attach) arena."""

    def __init__(self, mm: mmap.mmap, size: int):
        self._mm = mm
        self._size = size
        self._views: list[memoryview] = []  # track outstanding views for cleanup

        # Parse header
        magic = struct.unpack_from("<Q", mm, 0)[0]
        if magic != LOA_MAGIC:
            raise ValueError(f"bad LOA magic: {magic:#x}")

        self.version = struct.unpack_from("<I", mm, 8)[0]
        self.num_slots = struct.unpack_from("<I", mm, 12)[0]
        self.slot_size = struct.unpack_from("<I", mm, 16)[0]
        self.pool_id = struct.unpack_from("<H", mm, 20)[0]
        self.data_base = struct.unpack_from("<I", mm, 24)[0]

        # Build free list from slot states
        self._free_list: list[int] = []
        for i in range(self.num_slots):
            state = self._slot_state(i)
            if state == SLOT_FREE:
                self._free_list.append(i)

    @classmethod
    def open(cls, path: str) -> "LOAPool":
        """Open an existing LOA pool."""
        fd = os.open(path, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        return cls(mm, size)

    @classmethod
    def create(cls, path: str, pool_id: int = 0, num_slots: int = 4096,
               slot_size: int = 65536) -> "LOAPool":
        """Create a new LOA pool."""
        meta_region = LOA_SLOT_META_SIZE * num_slots
        data_offset = _align_up(LOA_HEADER_SIZE + meta_region, 4096)
        total_size = data_offset + num_slots * slot_size

        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, total_size)
            mm = mmap.mmap(fd, total_size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)

        # Write header
        struct.pack_into("<Q", mm, 0, LOA_MAGIC)
        struct.pack_into("<I", mm, 8, 1)          # version
        struct.pack_into("<I", mm, 12, num_slots)
        struct.pack_into("<I", mm, 16, slot_size)
        struct.pack_into("<H", mm, 20, pool_id)
        struct.pack_into("<I", mm, 24, data_offset)

        return cls(mm, total_size)

    def close(self):
        # Release tracked memoryviews
        for v in self._views:
            try:
                v.release()
            except ValueError:
                pass
        self._views.clear()
        try:
            self._mm.close()
        except BufferError:
            pass  # numpy/torch may hold derived buffer refs; gc cleans up

    def _meta_offset(self, slot_id: int) -> int:
        return LOA_HEADER_SIZE + slot_id * LOA_SLOT_META_SIZE

    def _data_offset(self, slot_id: int) -> int:
        return self.data_base + slot_id * self.slot_size

    def _slot_state(self, slot_id: int) -> int:
        return struct.unpack_from("<I", self._mm, self._meta_offset(slot_id))[0]

    def _set_slot_state(self, slot_id: int, state: int):
        struct.pack_into("<I", self._mm, self._meta_offset(slot_id), state)

    def _slot_refcnt(self, slot_id: int) -> int:
        return struct.unpack_from("<i", self._mm, self._meta_offset(slot_id) + 4)[0]

    def _add_refcnt(self, slot_id: int, delta: int) -> int:
        off = self._meta_offset(slot_id) + 4
        cur = struct.unpack_from("<i", self._mm, off)[0]
        new = cur + delta
        struct.pack_into("<i", self._mm, off, new)
        return new

    def _set_slot_owner(self, slot_id: int, owner: int):
        struct.pack_into("<I", self._mm, self._meta_offset(slot_id) + 8, owner)

    def _set_slot_size(self, slot_id: int, size: int):
        struct.pack_into("<I", self._mm, self._meta_offset(slot_id) + 12, size)

    def alloc(self, size: int, owner_frag_id: int = 0) -> tuple[memoryview, LOARef]:
        """Allocate a slot. Returns (writable memoryview, LOARef)."""
        if size > self.slot_size:
            raise ValueError(f"payload {size} exceeds slot size {self.slot_size}")
        if not self._free_list:
            raise RuntimeError("LOA pool full")

        slot_id = self._free_list.pop()
        self._set_slot_state(slot_id, SLOT_ALLOCATING)
        struct.pack_into("<i", self._mm, self._meta_offset(slot_id) + 4, 0)  # refcnt=0
        self._set_slot_owner(slot_id, owner_frag_id)
        self._set_slot_size(slot_id, size)

        off = self._data_offset(slot_id)
        buf = memoryview(self._mm)[off:off + size]
        self._views.append(buf)

        ref = LOARef(pool_id=self.pool_id, slot_id=slot_id, offset=0, length=size)
        return buf, ref

    def commit(self, ref: LOARef):
        """Mark a slot as ready for readers."""
        self._set_slot_state(ref.slot_id, SLOT_READY)

    def deref(self, ref: LOARef) -> bytes:
        """Get a copy of slot data. Call release() when done."""
        if ref.slot_id >= self.num_slots:
            raise ValueError(f"slot {ref.slot_id} out of range")
        state = self._slot_state(ref.slot_id)
        if state != SLOT_READY:
            raise RuntimeError(f"slot {ref.slot_id} not ready (state={state})")
        self._add_refcnt(ref.slot_id, 1)
        off = self._data_offset(ref.slot_id) + ref.offset
        return bytes(self._mm[off:off + ref.length])

    def deref_view(self, ref: LOARef) -> memoryview:
        """Get a zero-copy memoryview of slot data. Caller must del the view before pool.close()."""
        if ref.slot_id >= self.num_slots:
            raise ValueError(f"slot {ref.slot_id} out of range")
        state = self._slot_state(ref.slot_id)
        if state != SLOT_READY:
            raise RuntimeError(f"slot {ref.slot_id} not ready (state={state})")
        self._add_refcnt(ref.slot_id, 1)
        off = self._data_offset(ref.slot_id) + ref.offset
        return memoryview(self._mm)[off:off + ref.length]

    def release(self, ref: LOARef):
        """Release a slot. Freed when refcount reaches 0."""
        new_rc = self._add_refcnt(ref.slot_id, -1)
        if new_rc <= 0:
            self._set_slot_state(ref.slot_id, SLOT_FREE)
            self._free_list.append(ref.slot_id)


# ================================================================
# COI Table (read-only)
# ================================================================

@dataclass
class COIEntry:
    concept_id: int
    bits: int
    schema_id: int
    flags: int


class COITable:
    """Read-only shared-memory COI directory (seqlock reader)."""

    def __init__(self, mm: mmap.mmap):
        self._mm = mm

    @classmethod
    def open(cls, path: str) -> "COITable":
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        return cls(mm)

    def close(self):
        self._mm.close()

    def snapshot(self) -> tuple[int, int, list[COIEntry]]:
        """Read COI table with seqlock consistency. Returns (version, updated_ns, entries)."""
        while True:
            s1 = struct.unpack_from("<Q", self._mm, 0)[0]
            if s1 & 1:
                continue  # writer active
            count = struct.unpack_from("<I", self._mm, 12)[0]
            version = struct.unpack_from("<I", self._mm, 8)[0]
            updated_ns = struct.unpack_from("<Q", self._mm, 16)[0]

            entries = []
            off = 64
            for _ in range(count):
                if off + 16 > len(self._mm):
                    break
                cid = struct.unpack_from("<Q", self._mm, off)[0]
                bits = struct.unpack_from("<H", self._mm, off + 8)[0]
                schema = struct.unpack_from("<H", self._mm, off + 10)[0]
                flags = struct.unpack_from("<H", self._mm, off + 12)[0]
                entries.append(COIEntry(cid, bits, schema, flags))
                off += 16

            s2 = struct.unpack_from("<Q", self._mm, 0)[0]
            if s1 == s2:
                return version, updated_ns, entries


# ================================================================
# High-level LOA + Ring helpers
# ================================================================

def write_loa(ring: Ring, pool: LOAPool, hdr: Header, data: bytes,
              owner_frag_id: int = 0) -> Optional[LOARef]:
    """Write data via LOA and send pointer over ring."""
    buf, ref = pool.alloc(len(data), owner_frag_id)
    buf[:] = data
    pool.commit(ref)

    hdr.length = LOA_REF_SIZE
    hdr.flags |= FLAG_LOA_PTR
    ref_bytes = ref.encode()
    if not ring.try_write(hdr, ref_bytes):
        pool.release(ref)
        return None
    return ref


def read_loa(ring: Ring, pool: LOAPool, timeout_s: float = 0
             ) -> Optional[tuple[Msg, bytes, Optional[LOARef]]]:
    """Read a ring message; if LOA, deref the pool."""
    if timeout_s > 0:
        msg = ring.read_within(timeout_s)
    else:
        msg = ring.try_read()
    if msg is None:
        return None

    if msg.is_loa:
        ref = msg.loa_ref()
        data = pool.deref(ref)
        return msg, data, ref
    else:
        return msg, memoryview(msg.payload), None


# ================================================================
# Utilities
# ================================================================

def _align_up(v: int, align: int) -> int:
    return (v + align - 1) & ~(align - 1)
