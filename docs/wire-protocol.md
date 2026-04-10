# Wire Protocol Reference

All multi-byte values are little-endian.

## UDS Control Protocol (Fragment <-> Pigeon)

Socket: Unix domain stream socket at `FM_PIGEON_SOCK` (default `/tmp/pigeon.sock`).

### Request frame

```
Offset  Size  Field
───────────────────
 0      1     Op        operation code
 1      1     Count     number of COI entries (0-255)
 2      2     FragID    fragment ID (from Attach; 0 if not attached)
 4      16*N  Entries   COI entries (Count * 16 bytes each)
```

### COI Entry (16 bytes)

```
Offset  Size  Field
───────────────────
 0      8     ConceptID
 8      2     Bits
10      2     SchemaID
12      2     Flags
14      2     (padding)
```

### Operations

| Op | Code | Direction | Payload | Response |
|----|------|-----------|---------|----------|
| REGISTER | 0 | frag -> pigeon | COI entries | (none) |
| RENEW | 1 | frag -> pigeon | COI entries | (none) |
| UNREGISTER | 2 | frag -> pigeon | COI entries | (none) |
| LIST | 3 | frag -> pigeon | (none) | `[u16 count][entries...]` |
| LOAINFO | 4 | frag -> pigeon | (none) | `[u16 pathlen][path bytes]` |
| ATTACH | 5 | frag -> pigeon | (none) | SCM_RIGHTS: `[inFD, outFD]` + `[u32 fragID]` |

### LIST response

```
Offset  Size  Field
───────────────────
 0      2     Count     number of entries
 2      16*N  Entries   COI entries
```

### LOAINFO response

```
Offset  Size  Field
───────────────────
 0      2     PathLen   length of LOA pool path string
 2      N     Path      UTF-8 path bytes
```

### ATTACH response

Sent via `sendmsg` with `SCM_RIGHTS` ancillary data containing two file descriptors:

| FD | Purpose |
|----|---------|
| fds[0] | Inbound ring (pigeon writes, fragment reads) |
| fds[1] | Outbound ring (fragment writes, pigeon reads for routing) |

Data payload: `[u32 fragID]`

## Gossip Protocol (Pigeon <-> Pigeon, over QUIC)

Each QUIC stream carries one or more gossip frames.

### Gossip frame

```
Offset  Size  Field
───────────────────
 0      1     Op        operation code
 1      1     Count     number of entries
 2      2     SiteID    originating pigeon site ID
 4      4     Epoch     monotonic counter for ordering
 8      16*N  Entries   COI entries (same format as UDS)
```

### Operations

| Op | Code | Description |
|----|------|-------------|
| ADD | 10 | New COI(s) registered at SiteID |
| DEL | 11 | COI(s) removed at SiteID |
| RENEW | 12 | COI lease(s) extended at SiteID |
| SNAPRQ | 13 | Request full COI snapshot from peer |
| SNAPRS | 14 | Full snapshot response (all COIs at SiteID) |
| HELLO | 15 | Peer identification (SiteID + Epoch, no entries) |

### Connection lifecycle

1. Dialer opens QUIC connection to peer
2. Opens a stream, sends `HELLO` frame
3. Peer receives `HELLO`, registers connection under `SiteID`
4. Peer replies with `SNAPRS` (full COI snapshot)
5. Both sides enter steady-state: `ADD`/`DEL`/`RENEW` as COIs change

### Forward frame (data plane, over QUIC)

For cross-host message forwarding:

```
Offset  Size  Field
───────────────────
 0      2     FrameLen  total bytes following (header + payload)
 2      64    Header    message header
66      N     Payload   message payload (LOA data resolved to inline)
```

LOA messages are resolved before forwarding: the pigeon dereferences the local LOA pool and sends the actual tensor data inline, since the remote host cannot access local shared memory.

## Shared Memory Layouts

### Ring Control Header (64 bytes, offset 0 of ring file)

```
Offset  Size  Type     Field
─────────────────────────────
 0      8     uint64   CapSlots     (power of 2)
 8      8     uint64   ProdIdx      (atomic, monotonic)
16      8     uint64   ConsIdx      (atomic, monotonic)
24      4     uint32   SlotSize     (bytes per slot)
28      4     -        (padding)
32      8     uint64   ProdEvtFD    (0xFFFFFFFFFFFFFFFF = poll mode)
40      8     uint64   ConsEvtFD    (0xFFFFFFFFFFFFFFFF = poll mode)
48      32    -        (reserved)
```

Slots start at offset 64. Each slot is `SlotSize` bytes. Slot index = `ProdIdx & (CapSlots - 1)`.

### LOA Pool Header (64 bytes, offset 0 of LOA file)

```
Offset  Size  Type     Field
─────────────────────────────
 0      8     uint64   Magic        (0x4C4F41504F4F4C31 = "LOAPOOL1")
 8      4     uint32   Version      (1)
12      4     uint32   NumSlots
16      4     uint32   SlotSize     (bytes per slot)
20      2     uint16   PoolID
22      2     -        (padding)
24      4     uint32   DataBase     (byte offset of data region, page-aligned)
28      36    -        (reserved)
```

### LOA Slot Metadata (32 bytes per slot, starting at offset 64)

```
Offset  Size  Type     Field
─────────────────────────────
 0      4     uint32   State        (0=free, 1=allocating, 2=ready; atomic)
 4      4     int32    RefCnt       (atomic; freed when <= 0)
 8      4     uint32   Owner        (fragment ID of allocator)
12      4     uint32   Size         (actual payload bytes)
16      16    -        (reserved)
```

### COI Shared Memory Table (offset 0 of COI file)

```
Offset  Size  Type     Field
─────────────────────────────
 0      8     uint64   Seq          (seqlock: odd = write in progress; atomic)
 8      4     uint32   Version      (bumped on each update; atomic)
12      4     uint32   Count        (number of entries; atomic)
16      8     uint64   UpdatedNs    (last update timestamp; atomic)
24      40    -        (reserved, pad to 64)
64      16*N  -        COI entries  (same 16-byte format as UDS/gossip)
```

Reader protocol: spin while `Seq` is odd, read `Count`+`Version`+`UpdatedNs`, copy entries, re-read `Seq`. If `Seq` changed, retry.

## LOA Reference (12 bytes)

Sent as ring payload when `FlagLOAPtr` (0x0010) is set:

```
Offset  Size  Type     Field
─────────────────────────────
 0      2     uint16   PoolID
 2      2     uint16   SlotID
 4      4     uint32   Offset       (byte offset within slot, usually 0)
 8      4     uint32   Length       (payload bytes)
```

## Multi-slot LOA Reference (16 bytes)

For payloads spanning multiple contiguous slots:

```
Offset  Size  Type     Field
─────────────────────────────
 0      2     uint16   PoolID
 2      2     uint16   StartSlot
 4      2     uint16   NumSlots
 6      2     -        (padding)
 8      4     uint32   Offset
12      4     uint32   Length
```
