# RDMA over Thunderbolt Setup Guide

## Prerequisites

- Two Macs with Thunderbolt 4 or 5 ports
- macOS 26.2 or later on both machines
- Thunderbolt cable connecting the two Macs
- Xcode Command Line Tools installed

## Step 1: Enable RDMA (both Macs)

RDMA is disabled by default for security. You must enable it from Recovery Mode:

```bash
# On each Mac:
# 1. Shut down
# 2. Press and hold power button until "Loading startup options" appears
# 3. Select "Options" → Recovery Mode
# 4. Open Terminal (Utilities → Terminal)
# 5. Run:
rdma_ctl enable
# 6. Reboot
```

## Step 2: Verify RDMA

After rebooting, verify RDMA is working:

```bash
# Should show a device (not "disabled")
ibv_devices

# Expected output:
#     device              node GUID
#     ------              ----------------
#     mlx5_0              xxxxxxxxxxxx

# More details:
ibv_devinfo
```

If it still shows "disabled", ensure:
- The Thunderbolt cable is connected
- Both Macs have been rebooted after `rdma_ctl enable`
- macOS 26.2+ is installed (`sw_vers`)

## Step 3: Build Fragmind with RDMA

```bash
cd fragmind-pigeon

# Build the RDMA demo
CGO_ENABLED=1 go build -tags=rdma -o rdma_demo ./examples/rdma_demo

# Build the pigeon daemon with RDMA transport
CGO_ENABLED=1 go build -tags=rdma -o pigeon ./cmd/pigeon
```

## Step 4: Run the Demo

### Mac A (server — the one with the tensor data):

```bash
./rdma_demo -mode server -bind :9999 -size 65536
```

This will:
1. Open the RDMA device
2. Create an LOA pool and register it for RDMA access
3. Write a 64KB tensor into the LOA pool
4. Print the RDMA address and rkey
5. Wait for client connection on TCP port 9999

### Mac B (client — reads the tensor):

```bash
./rdma_demo -mode client -peer <mac-a-ip>:9999 -size 65536
```

This will:
1. Open the RDMA device
2. Connect to Mac A's TCP control channel
3. Receive the RDMA memory region info (address, rkey, length)
4. Report that the RDMA path is ready

### Expected Output (Server):

```
=== Fragmind RDMA over Thunderbolt Demo ===
Mode: server | Tensor: 65536 bytes

[OK] librdma.dylib loaded
[OK] RDMA device opened (1 device(s) found)
[OK] Protection domain allocated
[OK] GID: fe80000000000000xxxxxxxxxxxx
[OK] LOA pool registered for RDMA: 1114176 bytes, lkey=12345, rkey=67890
Tensor: slot=15, 65536 bytes, crc=abcdef01
RDMA: addr=7f8000000000 rkey=67890
Waiting for client on :9999...
Sent RDMA ref to client
Client done. RDMA demo complete.
```

## Step 5: Run the Pigeon Daemon with RDMA

For full distributed operation (not just the demo):

### Mac A:
```bash
FM_SITE_ID=1 FM_MODE=rdma FM_BIND=:9999 FM_PEERS=<mac-b-ip>:9999 \
    ./pigeon
```

### Mac B:
```bash
FM_SITE_ID=2 FM_MODE=rdma FM_BIND=:9999 FM_PEERS=<mac-a-ip>:9999 \
    ./pigeon
```

Fragments on Mac A attach to their local pigeon. When a fragment sends data targeting a COI registered on Mac B, the pigeon forwards via RDMA instead of QUIC.

## What's Working vs What's Next

### Working Now
- RDMA device detection and initialization via dlopen (no header files needed)
- Protection domain allocation
- LOA pool registration as RDMA memory region (`ibv_reg_mr`)
- GID query for peer identification
- TCP control channel for handshake (exchanging rkey, addr, GID)
- Graceful fallback when RDMA not enabled

### Coming Next (needs hardware testing)
- QP (Queue Pair) connection: `ibv_create_qp`, QP state transitions (INIT→RTR→RTS)
- `ibv_post_send(IBV_WR_RDMA_READ)` for actual zero-copy tensor pull
- Completion polling (`ibv_poll_cq`)
- Full integration with pigeon routing (auto-forward LOA refs via RDMA)

The QP setup is ~100 LOC of struct initialization that needs to be tested with real hardware. The memory registration and handshake protocol are complete.

## Performance Expectations

| Metric | Thunderbolt 5 | Thunderbolt 4 |
|--------|--------------|--------------|
| Bandwidth | 80 Gb/s (10 GB/s) | 40 Gb/s (5 GB/s) |
| Latency | 3-9 us | 5-15 us |
| 64KB tensor | ~6 us | ~13 us |
| 1MB tensor | ~100 us | ~200 us |
| 16MB weight shard | ~1.6 ms | ~3.2 ms |

For comparison: QUIC over localhost is ~1-3 ms for 16MB. RDMA is **100-1000x lower latency** for the same data.

## Troubleshooting

**"No RDMA devices"**: Run `rdma_ctl enable` in Recovery Mode and reboot.

**"ibv_alloc_pd failed"**: The Thunderbolt cable may not be connected, or the peer Mac may not have RDMA enabled.

**"ibv_reg_mr failed"**: The IOMMU may be blocking the memory registration. Ensure both Macs are running macOS 26.2+.

**GID shows all zeros**: The RDMA port may not be active. Check that the Thunderbolt cable is connected and both sides are ready.

## References

- [Apple TN3205: RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
- [MLX JACCL PR (reference implementation)](https://github.com/ml-explore/mlx/pull/2808)
- [exo distributed inference with RDMA](https://github.com/exo-explore/exo)
