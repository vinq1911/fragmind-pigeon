# fragmind-pigeon

COI-aware, ultra-fast IPC + cluster routing for fragmind fragments.
Local delivery via shared-memory SPSC rings; cross-host via QUIC pigeons.

---

## What is Fragmind?

Fragmind ("fragmented mind") is a system for distributing LLM inference and training across multiple hosts with near-DMA latency. When an LLM is too large for a single machine, its layers, weights, KV-cache, and activations must be shared between hosts with minimal overhead. Fragmind provides the data plane for this.

**fragmind-pigeon** is the routing daemon and IPC library that connects **fragments** (individual processing units that own a slice of the model) on the same host and across a cluster.

## Core Concepts

### Fragment

A **fragment** is an independent process that owns part of an LLM workload: a set of layers, a KV-cache partition, an attention head group, or a token batch processor. Fragments communicate through shared-memory rings (local) or via pigeons (cross-host).

### Pigeon

The **pigeon** is a per-host daemon that:
- Manages COI registrations from local fragments
- Publishes a read-only shared-memory COI directory
- Gossips COI state with peer pigeons over QUIC
- Routes messages between fragments based on concept-of-interest matching

### COI (Concept of Interest)

A **COI** declares what data a fragment cares about. It consists of:

| Field | Type | Description |
|-------|------|-------------|
| `ConceptID` | `uint64` | Identifies the concept (e.g., model+layer encoded as a 64-bit key) |
| `Bits` | `uint16` | Prefix length for hierarchical matching (28 = match top 28 bits) |
| `SchemaID` | `uint16` | What kind of data (weight shard, KV-cache, activation, etc.) |
| `Flags` | `uint16` | Subscription flags |

COIs use prefix-based matching: a fragment registering `ConceptID=0x8A731100, Bits=24` receives any message whose top 24 bits match `0x8A7311xx`. This enables hierarchical routing: subscribe to a whole model (few bits), a layer range (more bits), or a specific head (all bits).

Fragments register COIs with the local pigeon over a Unix domain socket (UDS). The pigeon tracks leases (5s TTL + 15s blackout grace) and gossips changes to peer pigeons.

### Ring (SPSC Shared-Memory Buffer)

The **ring** is a lock-free, single-producer/single-consumer circular buffer in shared memory. It is the fastest local IPC path:

- Memory-mapped file descriptor shared between producer and consumer
- 64-byte control header: capacity, producer index, consumer index, slot size, eventfd descriptors
- Each slot holds a 64-byte message header + payload
- Uses atomic load/store for index synchronization (ARM64-safe)
- Signaling via eventfd for blocking reads

Typical ring parameters: 1024 slots x 1024 bytes/slot = 1 MB ring.

### UDS (Unix Domain Socket)

Fragments communicate with the local pigeon over a **UDS** at `/tmp/pigeon.sock` (configurable via `FM_PIGEON_SOCK`). The protocol is:

```
[op:1][count:1][pad:2][entries... 16B each]
```

Operations:
- `REGISTER (0)` — register COIs, starts lease
- `RENEW (1)` — extend lease
- `UNREGISTER (2)` — remove COIs
- `LIST (3)` — dump current local COI table

### LOA (Large Object Attach)

For payloads larger than a ring slot (tensors, weight shards, KV-cache slices), **LOA** provides a zero-copy shared-memory arena:

1. **Writer** calls `pool.Alloc(size)` to reserve a slot in the LOA pool
2. Writer fills the slot directly (zero-copy into shared memory)
3. Writer calls `pool.Commit(ref)` to mark the slot as ready
4. Writer sends a small ring message with `FlagLOAPtr` set and an `LOARef` (12 bytes) as payload
5. **Reader** receives the ring message, calls `pool.Deref(ref)` to get a read-only view
6. Reader calls `pool.Release(ref)` when done — slot is freed when refcount hits 0

LOA pool layout:
```
[LOAHeader: 64B] [SlotMeta[0]: 32B] [SlotMeta[1]: 32B] ... [page-aligned data region]
```

Default configuration: 4096 slots x 64 KB = 256 MB arena. Customizable per deployment.

### IPC (Inter-Process Communication)

Fragmind uses a layered IPC approach:

| Layer | Mechanism | Latency | Use Case |
|-------|-----------|---------|----------|
| Local small | Ring (SPSC shm) | ~ns | Control messages, token batches, small activations |
| Local large | LOA + Ring pointer | ~ns | Weight shards, KV-cache, large activations |
| Cross-host control | QUIC gossip | ~ms | COI sync, topology, heartbeat |
| Cross-host data | QUIC streams (future: RDMA) | ~ms | Tensor forwarding between hosts |

### Router

The **router** maps `(ConceptBits, ConceptID)` to destinations:
- `LocalFragIDs` — fragments on this host
- `RemoteSites` — peer pigeon siteIDs

The router is populated from local COI registrations and remote gossip. When a message arrives, the pigeon looks up destinations and either delivers locally (via ring) or forwards to remote sites (via QUIC).

### Gossip Protocol

Pigeons exchange COI state over QUIC using a simple gossip protocol:

| Op | Code | Description |
|----|------|-------------|
| `ADD` | 10 | New COI registered |
| `DEL` | 11 | COI removed |
| `RENEW` | 12 | COI lease extended |
| `SNAPRQ` | 13 | Request full snapshot |
| `SNAPRS` | 14 | Full snapshot response |
| `HELLO` | 15 | Peer identification (siteID + epoch) |

Wire format: `[op:1][count:1][siteID:2][epoch:4][entries... 16B each]`

Gossip is debounced (75ms) to batch rapid COI changes before broadcasting.

### Schemas

SchemaID identifies the payload format. Well-known schemas for LLM workloads:

| SchemaID | Name | Payload |
|----------|------|---------|
| 0 | `Raw` | Pass-through, no metadata |
| 1 | `WeightShard` | 32B meta + fp16/bf16/fp8 tensor data |
| 2 | `KVCache` | 32B meta + interleaved K,V data |
| 3 | `Activation` | 24B meta + tensor data |
| 4 | `TokenBatch` | 16B meta + token IDs + attention mask |
| 5 | `Gradient` | 32B meta + gradient data |
| 0xFFFF | `Control` | Internal pigeon control |

Each schema has a fixed-size metadata prefix followed by raw tensor/token data. Metadata includes model ID, layer range, data type (fp32/fp16/bf16/fp8), shape, and a CRC32 checksum.

### Message Header

Every ring message has a 64-byte header:

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | Len | u32 | Payload length |
| 4 | Kind | u16 | Message type (Process, Learn, Share, Ping) |
| 6 | Flags | u16 | FlagEndOfStream, FlagUrgent, FlagReply, FlagDropOK, FlagLOAPtr |
| 8 | TSns | u64 | Timestamp (nanoseconds) |
| 16 | ConceptID | u64 | Concept identifier |
| 24 | ConceptBits | u16 | Prefix bits for routing |
| 26 | SchemaID | u16 | Payload schema |
| 28 | SrcID | u32 | Source fragment ID |
| 32 | MsgID | u32 | Message sequence number |
| 36 | Hop | u16 | Hop count |
| 38 | Ver | u16 | Protocol version |
| 40 | TraceID | u64 | Distributed trace ID |
| 48 | Checksum32 | u32 | Header checksum |

---

## Install

```bash
go get github.com/vinq1911/fragmind-pigeon@latest
```

## Run the Pigeon Daemon

```bash
FM_SITE_ID=1 FM_PIGEON_SOCK=/tmp/pigeon.sock ./pigeon
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FM_SITE_ID` | `1` | Unique site identifier for this host |
| `FM_PIGEON_SOCK` | `/tmp/pigeon.sock` | UDS path for fragment control |
| `FM_MODE` | `none` | Peer mode: `quic` or `none` |
| `FM_BIND` | `:4433` | QUIC listen address |
| `FM_PEERS` | (empty) | Comma-separated peer addresses |
| `FM_COI_SHM_PATH` | `/dev/shm/fragmind.coi.local` (Linux) | COI shared-memory path |
| `FM_LOA_SHM_PATH` | `/dev/shm/fragmind.loa.<siteID>` | LOA pool path |
| `FM_LOA_SLOTS` | `4096` | Number of LOA slots |
| `FM_LOA_SLOT_SIZE` | `65536` | Bytes per LOA slot |

### Multi-Pigeon Cluster

```bash
# Host A
FM_SITE_ID=1 FM_MODE=quic FM_BIND=:4433 FM_PEERS=hostB:4433 ./pigeon

# Host B
FM_SITE_ID=2 FM_MODE=quic FM_BIND=:4433 FM_PEERS=hostA:4433 ./pigeon
```

Build with QUIC support:
```bash
go build -tags=quic -o pigeon ./cmd/pigeon
```

## Fragment Usage

### Basic Ring Communication

```go
import fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"

// Open rings (FDs passed from coordinator)
in, _ := fp.OpenRingFromFD(inFD)
out, _ := fp.OpenRingFromFD(outFD)
defer in.Close()
defer out.Close()

// Read a message
msg, _ := in.Read(true) // blocking

// Write a message
payload := []byte("processed result")
hdr := fp.Header{
    Len: uint32(len(payload)), Kind: fp.KindProcess,
    ConceptID: 0x8A7311CCDD55002A, ConceptBits: 24, SchemaID: 1,
}
for !out.TryWrite(hdr, payload) {
    // ring full, spin
}
```

### COI Registration (Auto-Renew)

```go
cois := []fp.COI{{ConceptID: 0x8A7311CCDD55002A, Bits: 28, SchemaID: 1}}
h, _ := fp.StartCOI(fp.COIOptions{SocketPath: "/tmp/pigeon.sock"}, cois)
defer h.Close()
// Lease auto-renews in background every 1s
```

### LOA (Large Object Transfer)

```go
// Create or open an LOA pool
pool, _ := fp.CreateLOAPool(fp.LOAPoolOptions{
    Path:     "/dev/shm/fragmind.loa.0",
    NumSlots: 4096,
    SlotSize: 64 * 1024, // 64 KB per slot
})
defer pool.Close()

// Writer: allocate, fill, commit, send pointer over ring
tensorData := make([]byte, 32*1024) // 32 KB tensor
ref, ok := fp.WriteLOA(outRing, pool, fp.Header{
    Kind: fp.KindProcess, SchemaID: fp.SchemaWeightShard,
    ConceptID: 0x8A731100, ConceptBits: 24,
}, tensorData, myFragID)

// Reader: receive ring message, deref LOA, process, release
msg, data, ref, _ := fp.ReadLOA(inRing, pool, true)
// data is a zero-copy view into shared memory
processWeightShard(data)
pool.Release(ref) // free the slot
```

### Zero-Copy Producer Path

```go
// For producers that generate data in-place (e.g., GPU -> shm)
buf, ref, _ := fp.WriteLOAZeroCopy(pool, 65536, myFragID)
// Write directly into buf (shared memory) — no copy
generateTensorInto(buf)
fp.CommitLOA(outRing, pool, ref, header)
```

### Reading the COI Directory

```go
table, _ := fp.OpenLocalCOITable("")
defer table.Close()

ver, updated, entries := table.Snapshot()
for _, e := range entries {
    fmt.Printf("COI: concept=%x bits=%d schema=%d\n", e.ConceptID, e.Bits, e.SchemaID)
}
```

### Fragment Attachment (Pigeon-Managed Routing)

```go
// Fragment attaches to pigeon — receives ring pair for routed messaging
att, _ := fp.Attach("/tmp/pigeon.sock")
defer att.Close()

// Register COIs so other fragments' messages get routed to you
cois := []fp.COI{{ConceptID: 0x0001000000000000, Bits: 16, SchemaID: fp.SchemaWeightShard}}
h, _ := fp.StartCOI(fp.COIOptions{
    SocketPath: "/tmp/pigeon.sock",
    FragID:     uint16(att.FragID),
}, cois)
defer h.Close()

// Write to your outbound ring — pigeon routes to subscribers
hdr := fp.Header{Kind: fp.KindProcess, ConceptID: 0x0001000000000000, ConceptBits: 16}
att.OutRing.TryWrite(hdr, payload)

// Read from your inbound ring — messages routed to you by pigeon
msg, _ := att.InRing.ReadWithin(time.Second)
```

### Backpressure

```go
// Write with exponential backoff (blocks up to timeout)
err := fp.RingWriteWithBackoff(ring, hdr, payload, 5*time.Second)

// LOA alloc with retry on pool-full
buf, ref, err := fp.LOAAllocWithBackoff(pool, size, fragID, 5*time.Second)

// Full LOA write path with backpressure on both pool and ring
ref, err := fp.WriteLOAWithBackoff(ring, pool, hdr, data, fragID, 5*time.Second)

// FlagDropOK: if ring is still full at deadline, message is dropped (not an error)
hdr.Flags |= fp.FlagDropOK
err = fp.RingWriteWithBackoff(ring, hdr, payload, 100*time.Millisecond)
// err == fp.ErrDropped means safe drop, not a failure
```

### Python Bindings

```python
from fragpigeon import Ring, LOAPool, Header, write_loa, read_loa

# Open ring and LOA pool (same shm files the Go pigeon creates)
ring = Ring.from_path("/dev/shm/my-ring.shm")
pool = LOAPool.open("/dev/shm/fragmind.loa.1")

# Write a weight shard via LOA
ref = write_loa(ring, pool, header, tensor_bytes, owner_frag_id=1)

# Read and deref
msg, data, ref = read_loa(ring, pool, timeout_s=1.0)
tensor = np.frombuffer(data, dtype=np.float16)
pool.release(ref)
```

See [docs/python-bindings.md](docs/python-bindings.md) for full API reference.

## Performance

Benchmarked on Apple M4 (ARM64):

| Operation | Latency | Allocations |
|-----------|---------|-------------|
| Header pack + unpack | ~6 ns | 0 |
| Router lookup | ~13 ns | 0 |
| LOA alloc + commit + release | ~9 ns | 0 |
| LOA full cycle (alloc + commit + deref + release) | ~9 ns | 0 |
| LOARef encode + decode | <1 ns | 0 |
| Schema meta encode + decode | ~2 ns | 0 |

All hot paths are zero-allocation.

### Comparative (end-to-end, Apple M4)

**Small messages (P50 latency, lower = better):**

| Transport | 64B | 1KB |
|-----------|-----|-----|
| **fragmind-ring** | **167 ns** | **208 ns** |
| UDS | 2,959 ns (18x) | 3,250 ns (16x) |
| Pipe | 2,792 ns (17x) | 3,042 ns (15x) |
| TCP | 10,333 ns (62x) | 10,208 ns (49x) |

**Large objects (throughput MB/s, higher = better):**

| Transport | 64KB | 1MB | 16MB |
|-----------|------|-----|------|
| **fragmind-loa** | **9,369** | **8,950** | **8,455** |
| TCP | 1,483 | 4,952 | 4,920 |
| Pipe | 3,708 | 3,771 | 3,275 |
| UDS | 1,888 | 2,696 | 2,459 |

Run `bash proof/run.sh` to reproduce. Results saved to `proof/results/` as JSON for tracking over time.

## Build

```bash
# Standard build (no QUIC)
go build -o pigeon ./cmd/pigeon

# With QUIC peer support
go build -tags=quic -o pigeon ./cmd/pigeon

# ARM64 Linux (for deployment)
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o pigeon ./cmd/pigeon

# Docker
docker buildx build --platform linux/arm64 -t fragmind-pigeon:latest -f Dockerfile .

# Tests
go test ./...

# Benchmarks
go test -bench=. -benchmem ./pkg/fragpigeon/
```

## Architecture

```
+------------------+       +------------------+
|   Fragment A     |       |   Fragment B     |
|  (layers 0-15)  |       |  (layers 16-31)  |
+--------+---------+       +--------+---------+
         |  Ring (shm)              |  Ring (shm)
         v                          v
+------------------------------------------+
|              Pigeon (site=1)             |
|  COI Table | Router | LOA Pool | Gossip |
+------------------------------------------+
         |                          |
         |  QUIC (cross-host)       |
         v                          v
+------------------------------------------+
|              Pigeon (site=2)             |
|  COI Table | Router | LOA Pool | Gossip |
+------------------------------------------+
         |  Ring (shm)              |  Ring (shm)
         v                          v
+--------+---------+       +--------+---------+
|   Fragment C     |       |   Fragment D     |
|  (KV-cache L)   |       |  (KV-cache R)    |
+------------------+       +------------------+
```

## Project Structure

```
cmd/pigeon/              Pigeon daemon entry point
pkg/fragpigeon/
  pigeon.go              Daemon core (COI shm, LOA pool, UDS server, gossip)
  ring.go                SPSC shared-memory ring buffer (atomic, ARM64-safe)
  header.go              64-byte message header (pack/unpack)
  coi.go                 COI client, table reader, auto-renew, LOA discovery
  router.go              Concept-based routing table
  loa.go                 LOA shared-memory arena (alloc/commit/deref/release)
  loa_multi.go           Multi-slot LOA for large payloads
  loa_ring.go            LOA + Ring integration helpers
  attach.go              Fragment attachment, ring pair creation, COI routing
  backpressure.go        Exponential backoff for ring + LOA writes
  schema.go              LLM schemas (WeightShard, KVCache, Activation, etc.)
  peers.go               PeerManager interface
  peers_quic.go          QUIC peer manager (build tag: quic)
  gossip_quic.go         Gossip protocol (build tag: quic)
  forward_quic.go        Cross-host forwarding with LOA resolution (build tag: quic)
  *_stub.go              No-op stubs for non-QUIC builds
  defaults_*.go          Platform-specific paths (Linux /dev/shm, macOS /tmp)
  e2e_test.go            19 end-to-end integration tests
  *_test.go              Unit tests and benchmarks
  internal/
    memfd_linux.go       memfd_create for anonymous shm
    memfd_other_unix.go  Temp-file fallback for macOS/BSD
bindings/
  c/fragpigeon.h         C header — all struct definitions + inline helpers
  python/
    fragpigeon.py        Pure Python bindings (mmap + struct, no extensions)
    test_fragpigeon.py   22 unit tests
    test_cross_language.py  Python-writes-Go-reads interop tests
proof/
  proof_test.go          Comparative benchmark suite (fragmind vs baselines)
  harness.go             Payload gen, latency histogram, JSON reporting
  fragmind.go            Ring + LOA benchmark wrappers
  baselines.go           UDS, TCP, Pipe transports (end-to-end)
  run.sh                 One-command benchmark runner
  results/               JSON results (gitignored, machine-specific)
examples/
  loa_weight_demo/       Zero-copy weight shard transfer demo
  local_ring_demo/       Coordinator + sender + receiver over shared rings
  scripts/
    two_pigeons_local.sh Launch a 3-pigeon QUIC cluster locally
docs/
  architecture.md        System design, message flow, memory layouts
  wire-protocol.md       Binary protocol reference (UDS, gossip, shm layouts)
  python-bindings.md     Python API reference and usage guide
  testing.md             Test structure, coverage, and patterns
```

## Documentation

- [Architecture](docs/architecture.md) — system design, message flow, concurrency model
- [Wire Protocol](docs/wire-protocol.md) — binary format reference for all protocols
- [Python Bindings](docs/python-bindings.md) — API reference and usage guide
- [Testing Guide](docs/testing.md) — test structure, coverage, how to write new tests
