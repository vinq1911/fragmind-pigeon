# Python Bindings

Pure Python bindings for fragmind-pigeon shared-memory IPC. No compiled extensions — just `mmap` + `struct`.

## Install

```bash
# Copy the module into your project (single file, no dependencies beyond stdlib)
cp bindings/python/fragpigeon.py your_project/

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/fragmind-pigeon/bindings/python:$PYTHONPATH
```

Requirements: Python 3.10+ (dataclasses, typing). No pip packages needed.

## Quick Start

### Ring (small messages)

```python
from fragpigeon import Ring, Header, KIND_PROCESS, SCHEMA_TOKEN_BATCH
import time

# Open a ring from a file descriptor (passed by pigeon or coordinator)
ring = Ring.from_fd(fd)
# Or from a path (for testing)
ring = Ring.from_path("/dev/shm/my-ring.shm")

# Write a message
hdr = Header(
    length=len(payload),
    kind=KIND_PROCESS,
    ts_ns=int(time.time() * 1e9),
    concept_id=0x0001000000000000,
    concept_bits=16,
    schema_id=SCHEMA_TOKEN_BATCH,
    src_id=1001,
    msg_id=1,
)
if ring.try_write(hdr, payload):
    print("sent")

# Read a message (non-blocking)
msg = ring.try_read()
if msg:
    print(f"got {msg.header.schema_id}: {len(msg.payload)} bytes")

# Read with timeout
msg = ring.read_within(timeout_s=1.0)
```

### LOA Pool (large tensors)

```python
from fragpigeon import LOAPool, write_loa, read_loa

# Open the pigeon's LOA pool
pool = LOAPool.open("/dev/shm/fragmind.loa.1")

# --- Writer side ---
# Option 1: copy-in (simple)
buf, ref = pool.alloc(tensor.nbytes, owner_frag_id=1)
buf[:] = tensor.tobytes()
pool.commit(ref)

# Option 2: write via ring (LOA pointer)
ref = write_loa(ring, pool, header, tensor.tobytes(), owner_frag_id=1)

# --- Reader side ---
# Option 1: direct deref (returns bytes copy)
data = pool.deref(ref)
tensor = np.frombuffer(data, dtype=np.float16)
pool.release(ref)

# Option 2: read via ring (auto-derefs LOA)
result = read_loa(ring, pool, timeout_s=1.0)
if result:
    msg, data, ref = result
    tensor = np.frombuffer(data, dtype=np.float16)
    if ref:
        pool.release(ref)

# Option 3: zero-copy memoryview (advanced, must release before pool.close())
view = pool.deref_view(ref)  # memoryview into shm, no copy
np_tensor = np.frombuffer(view, dtype=np.float16)
# ... use tensor ...
del np_tensor, view  # must release before pool.close()
pool.release(ref)
```

### COI Table (read-only)

```python
from fragpigeon import COITable

table = COITable.open("/dev/shm/fragmind.coi.local")
version, updated_ns, entries = table.snapshot()
for e in entries:
    print(f"concept={e.concept_id:#x} bits={e.bits} schema={e.schema_id}")
table.close()
```

### Creating Pools (for testing)

```python
# Python can create LOA pools (same binary format as Go)
pool = LOAPool.create("/tmp/test.loa", pool_id=1, num_slots=256, slot_size=65536)
# Go pigeon or other Python processes can open this pool
```

## API Reference

### Header

| Field | Type | Description |
|-------|------|-------------|
| `length` | int | Payload size in bytes |
| `kind` | int | `KIND_PROCESS`, `KIND_LEARN`, `KIND_SHARE`, `KIND_PING` |
| `flags` | int | Bitmask: `FLAG_LOA_PTR`, `FLAG_URGENT`, `FLAG_DROP_OK`, etc. |
| `ts_ns` | int | Timestamp (nanoseconds since epoch) |
| `concept_id` | int | 64-bit concept identifier for routing |
| `concept_bits` | int | Prefix bits for COI matching |
| `schema_id` | int | `SCHEMA_RAW`, `SCHEMA_WEIGHT_SHARD`, `SCHEMA_KV_CACHE`, etc. |
| `src_id` | int | Source fragment ID |
| `msg_id` | int | Message sequence number |
| `hop` | int | Hop count |
| `ver` | int | Protocol version |
| `trace_id` | int | Distributed trace ID |
| `checksum32` | int | CRC32 of payload |

Methods: `pack() -> bytes`, `unpack(data) -> Header`, `is_loa -> bool`

### Ring

| Method | Description |
|--------|-------------|
| `Ring.from_fd(fd)` | Open from file descriptor |
| `Ring.from_path(path)` | Open from file path |
| `try_write(hdr, payload) -> bool` | Non-blocking write. Returns False if full. |
| `try_read() -> Msg or None` | Non-blocking read. Returns None if empty. |
| `read_within(timeout_s) -> Msg or None` | Polling read with timeout. |
| `close()` | Unmap shared memory. |

### LOAPool

| Method | Description |
|--------|-------------|
| `LOAPool.open(path)` | Open existing pool (can alloc and deref). |
| `LOAPool.create(path, ...)` | Create new pool. |
| `alloc(size, owner) -> (memoryview, LOARef)` | Reserve a slot. Write into the memoryview. |
| `commit(ref)` | Mark slot as ready for readers. |
| `deref(ref) -> bytes` | Get a copy of slot data. Safe to use after pool.close(). |
| `deref_view(ref) -> memoryview` | Zero-copy view. Must del before pool.close(). |
| `release(ref)` | Decrement refcount. Frees slot when zero. |
| `close()` | Unmap shared memory. |

### LOARef

| Field | Type | Description |
|-------|------|-------------|
| `pool_id` | int | LOA pool identifier |
| `slot_id` | int | Slot index within pool |
| `offset` | int | Byte offset within slot (usually 0) |
| `length` | int | Payload length in bytes |

Methods: `encode() -> bytes`, `decode(data) -> LOARef`

### High-level Helpers

| Function | Description |
|----------|-------------|
| `write_loa(ring, pool, hdr, data, owner)` | Alloc LOA slot, copy data, send pointer over ring. |
| `read_loa(ring, pool, timeout)` | Read ring msg; if LOA, deref pool. Returns `(Msg, data, ref)`. |

## Cross-Language Interop

The Python bindings produce the exact same binary layout as the Go implementation. You can:

- **Python writes, Go reads**: Python `ring.try_write()` -> Go `ring.Read()`
- **Go writes, Python reads**: Go `ring.TryWrite()` -> Python `ring.try_read()`
- **Share LOA pools**: both languages mmap the same file, same struct layout
- **Share COI tables**: Python `COITable` reads the same seqlock shm Go writes

This is verified by the cross-language tests in `bindings/python/test_cross_language.py`.

## Performance

Measured on Apple M4 (ARM64), Python 3.13:

| Operation | Python | Go |
|-----------|--------|-----|
| Ring write+read | 505K ops/s | 2.7M ops/s (1KB) |
| LOA alloc+deref+release | 596K ops/s | 130M ops/s |

Python is ~5x slower than Go for ring operations due to struct pack/unpack overhead. For real LLM workloads, the hot path is the tensor data (written by GPU/C++ engine directly into LOA shm); Python only handles the control path.

## Testing

```bash
cd bindings/python

# Unit tests (22 tests)
python -m pytest test_fragpigeon.py -v

# Cross-language tests (Python writes, Go reads)
python -m pytest test_cross_language.py -v -s
```
