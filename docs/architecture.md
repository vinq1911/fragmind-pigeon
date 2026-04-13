# Architecture

## System Overview

Fragmind-pigeon is a three-layer IPC system for distributed LLM inference:

```
                        ┌─────────────────────────────────────┐
                        │           Control Plane              │
                        │  COI registration, lease renewal,    │
                        │  gossip, topology discovery          │
                        │  (UDS + QUIC)                        │
                        └────────────────┬────────────────────┘
                                         │
┌────────────────────────────────────────┼────────────────────────────────────────┐
│                        Data Plane (local host)                                  │
│                                        │                                        │
│  ┌──────────┐    Ring (shm)    ┌───────┴───────┐    Ring (shm)    ┌──────────┐ │
│  │Fragment A ├────────────────►│    Pigeon      ├────────────────►│Fragment B │ │
│  │(producer) │◄────────────────┤   (router)     │◄────────────────┤(consumer) │ │
│  └─────┬─────┘   Ring (shm)   └───────┬───────┘   Ring (shm)    └─────┬─────┘ │
│        │                               │                               │        │
│        │         LOA Pool (shm)        │                               │        │
│        └──────────►┌───────────────────┴──┐◄───────────────────────────┘        │
│                    │  Shared-memory arena  │  zero-copy: writer allocs,          │
│                    │  (4096 slots x 64KB)  │  reader derefs same pages           │
│                    └──────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                    QUIC streams
                                         │
┌─────────────────────────────────────────┼───────────────────────────────────────┐
│                        Data Plane (remote host)                                 │
│                                        │                                        │
│                                ┌───────┴───────┐                                │
│                                │    Pigeon      │                                │
│                                │   (site=2)     │                                │
│                                └───────┬───────┘                                │
│                                        │                                        │
│                              Fragment C, D ...                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Message Flow

### Local delivery (same host)

1. Fragment A writes a message to its **outbound ring** (shm SPSC buffer)
2. Pigeon's routing goroutine reads from A's outbound ring
3. Pigeon looks up `(ConceptBits, ConceptID)` in the **Router**
4. Router returns `Destinations{LocalFragIDs: [B], RemoteSites: []}`
5. Pigeon writes the message to Fragment B's **inbound ring**
6. Fragment B reads from its inbound ring

For LOA messages (large tensors), only a 12-byte `LOARef` pointer travels through the ring. The actual data lives in the shared LOA pool — both fragments access the same physical pages.

### Cross-host delivery

1. Same as steps 1-3 above
2. Router returns `Destinations{RemoteSites: [2]}`
3. Pigeon calls `ForwardRemote()`:
   - If the message has `FlagLOAPtr`: deref the LOA pool to get actual data, clear the flag, send inline over QUIC (remote can't access local shm)
   - Otherwise: send header + payload over a QUIC stream
4. Remote pigeon receives, looks up local subscribers, delivers via rings

### COI lifecycle

```
Fragment                    Pigeon                     Peer Pigeons
   │                          │                            │
   ├──REGISTER(COIs)─────────►│                            │
   │                          ├──update local table        │
   │                          ├──update router             │
   │                          ├──publish COI shm           │
   │                          ├──enqueue gossip ADD────────►
   │                          │  (75ms debounce)           │
   ├──RENEW(COIs)────────────►│                            │
   │  (every 1s)              ├──extend lease              │
   │                          │                            │
   │  (close/crash)           │                            │
   ├──UNREGISTER(COIs)───────►│                            │
   │                          ├──remove from table         │
   │                          ├──enqueue gossip DEL────────►
   │                          │                            │
   │                          │  (if no RENEW for 5s+15s)  │
   │                          ├──expire entry              │
```

## Memory Layout

### Ring Buffer (64B control + N x SlotSize)

```
Offset  Size  Field
─────────────────────────
 0      8     CapSlots (uint64, power of 2)
 8      8     ProdIdx  (uint64, atomic)
16      8     ConsIdx  (uint64, atomic)
24      4     SlotSize (uint32)
28      4     (padding)
32      8     ProdEvtFD (uint64, 0xFF...FF = none)
40      8     ConsEvtFD (uint64, 0xFF...FF = none)
48     16     (reserved)
─────────────────────────
64      -     Slot[0]
64+S    -     Slot[1]
...           ...

Each slot: [Header: 64B][Payload: hdr.len bytes][unused padding to SlotSize]
```

### LOA Pool

```
Offset      Size            Field
────────────────────────────────────
 0          64              LOAHeader (magic, version, num_slots, slot_size, pool_id, data_base)
64          32 * num_slots  SlotMeta[] (state, refcnt, owner, size per slot)
(page-aligned)              Data region: num_slots * slot_size bytes contiguous
```

### Message Header (64 bytes)

```
Offset  Size  Field         Description
───────────────────────────────────────
 0      4     Len           Payload length (bytes)
 4      2     Kind          KindProcess=1, KindLearn=2, KindShare=3, KindPing=4
 6      2     Flags         FlagEOS=1, FlagUrgent=2, FlagReply=4, FlagDropOK=8, FlagLOAPtr=16
 8      8     TSns          Timestamp (nanoseconds since epoch)
16      8     ConceptID     Routing key
24      2     ConceptBits   Prefix bits for COI matching
26      2     SchemaID      Payload format identifier
28      4     SrcID         Source fragment ID
32      4     MsgID         Sequence number
36      2     Hop           Hop count (for loop detection)
38      2     Ver           Protocol version
40      8     TraceID       Distributed trace correlation
48      4     Checksum32    CRC32 of payload
52     12     (reserved)
```

## Concurrency Model

- **Ring**: lock-free SPSC. One producer, one consumer. Atomic load/store on ProdIdx/ConsIdx. No mutexes.
- **LOA Pool**: free list protected by `sync.Mutex`. Slot state and refcount use `sync/atomic`. Multiple concurrent readers safe.
- **Router**: `sync.RWMutex`. Concurrent reads (message routing) with exclusive writes (COI register/unregister).
- **COI shm**: seqlock for writers (pigeon), spin-wait for readers (fragments).
- **Gossip**: 75ms debounce timer batches rapid COI changes. Pending queues protected by pigeon's main `sync.RWMutex`.

## Build Tags

| Tag | Effect |
|-----|--------|
| (none) | Local-only mode. QUIC/RDMA stubs compiled. Peer manager is no-op. |
| `quic` | QUIC peer networking. Self-signed TLS 1.3. Gossip + forwarding over QUIC streams. |
| `rdma` | RDMA over Thunderbolt 5 (macOS 26.2+). libibverbs via CGo. LOA pool registered as RDMA memory region. Remote sites RDMA-read tensor data directly from LOA slots. |

## Distributed Training Data Flow

The gorch integration demo shows the full training pipeline:

```
                        Forward Pass
                        ════════════
 ┌──────────────────┐                      ┌──────────────────┐
 │    Worker A       │   activations (LOA)  │    Worker B       │
 │  Linear(784→128)  ├─────────────────────►│  Linear(128→10)  │
 │  + ReLU           │   32KB via ring ptr  │  + CrossEntropy   │
 └──────────────────┘                      └──────────────────┘
         ▲                                          │
         │              Backward Pass               │
         │              ═════════════               ▼
         │                                  loss.Backward()
         │   gradients (LOA)                        │
         └──────────────────────────────────────────┘
              32KB via ring ptr

 Both workers: Adam optimizer step on their own parameters
```

Per batch: 2 LOA transfers (32KB activations + 32KB gradients), 2 ring messages.
Measured overhead: 6.2% vs single-process (dominated by CPU compute, not IPC).

## RDMA over Thunderbolt

For cross-host deployment on Apple Silicon clusters:

```
 ┌─────────────┐  Thunderbolt 5 cable  ┌─────────────┐
 │  Mac A       │◄─────────────────────►│  Mac B       │
 │  Pigeon(1)   │   RDMA: 80 Gb/s      │  Pigeon(2)   │
 │  LOA Pool    │   Latency: 3-9 µs    │  LOA Pool    │
 └──────┬──────┘                       └──────┬──────┘
        │                                      │
   Fragment A                             Fragment B
```

How it works:
1. Each pigeon registers its LOA pool as an RDMA memory region (`ibv_reg_mr`)
2. During site handshake, pigeons exchange memory region keys (rkey)
3. When forwarding LOA messages cross-host, instead of serializing over QUIC, the remote pigeon issues an `IBV_WR_RDMA_READ` directly from the source LOA pool
4. Tensor data flows DMA from one Mac's memory to another — no CPU memcpy, no kernel networking stack

Requirements: macOS 26.2+, `rdma_ctl enable` (Recovery Mode), direct Thunderbolt 4/5 cable.
Uses standard libibverbs API (same as Linux InfiniBand).

## Platform Support

| Platform | Ring | LOA | COI shm | Pigeon | QUIC | RDMA |
|----------|------|-----|---------|--------|------|------|
| Linux (x86_64, ARM64) | memfd_create | /dev/shm | /dev/shm | Full | Full | InfiniBand/RoCE |
| macOS (ARM64) | temp file + unlink | /tmp | /tmp | Full | Full | Thunderbolt 5 (26.2+) |
| FreeBSD/other Unix | temp file + unlink | /tmp | /tmp | Full | Full | — |
