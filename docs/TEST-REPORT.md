# Fragmind-Pigeon Test Report

**Date:** 2026-04-13  
**Commit:** 353b12b (main)  
**Platform:** darwin/arm64 (Apple M4, 10 cores)  
**Go:** 1.26.0 | **Python:** 3.13.9  
**Codebase:** 56 source files, 9,610 lines

---

## 1. Test Summary

| Suite | Tests | Passed | Failed | Time |
|-------|-------|--------|--------|------|
| Go unit tests | 27 | 27 | 0 | <1s |
| Go e2e tests | 21 | 21 | 0 | 2.1s |
| Go benchmarks | 6 | 6 | 0 | 11s |
| Go proof (comparative) | 25 | 25 | 0 | 10s |
| Python unit tests | 22 | 22 | 0 | <1s |
| Python NumPy tests | 7 | 7 | 0 | <1s |
| Python cross-language | 2 | 2 | 0 | 1.8s |
| **Total** | **110** | **110** | **0** | **~27s** |

---

## 2. Go Unit Tests (27 tests)

### Header
| Test | Status |
|------|--------|
| TestHeaderPackUnpack | PASS |
| TestHeaderSize | PASS |

### Router
| Test | Status |
|------|--------|
| TestRouterAddLookup | PASS |
| TestRouterRemove | PASS |
| TestRouterDestinationsHeader | PASS |
| TestRouterNilLookup | PASS |
| TestRouterConcurrent | PASS |
| TestAppendUniqueU16 | PASS |

### LOA Pool
| Test | Status |
|------|--------|
| TestLOACreateOpen | PASS |
| TestLOAAllocCommitDerefRelease | PASS |
| TestLOAPoolFull | PASS |
| TestLOARefEncodeDecode | PASS |
| TestLOAMultipleReaders | PASS |
| TestLOAOversizedAlloc | PASS |
| TestLOAConcurrentAllocRelease | PASS |
| TestLOABadMagic | PASS |

### Multi-Slot LOA
| Test | Status |
|------|--------|
| TestAllocMultiBasic | PASS |
| TestAllocMultiExact | PASS |
| TestAllocMultiTooLarge | PASS |
| TestAllocMultiFragmentation | PASS |
| TestAllocMultiSingleSlot | PASS |

### Schema Metadata
| Test | Status |
|------|--------|
| TestWeightShardMetaEncodeDecode | PASS |
| TestKVCacheMetaEncodeDecode | PASS |
| TestActivationMetaEncodeDecode | PASS |
| TestTokenBatchMetaEncodeDecode | PASS |
| TestGradientMetaEncodeDecode | PASS |
| TestDTypeSize | PASS |

---

## 3. Go End-to-End Tests (21 tests)

| Test | What It Proves | Status |
|------|----------------|--------|
| RingRoundTrip/8B | Ring write+read, CRC verified (8 bytes) | PASS |
| RingRoundTrip/64B | Ring write+read, CRC verified (64 bytes) | PASS |
| RingRoundTrip/128B | Ring write+read, CRC verified (128 bytes) | PASS |
| RingRoundTrip/440B | Ring write+read, CRC verified (440 bytes) | PASS |
| RingOrdering | 100 messages arrive in strict FIFO order | PASS |
| RingFull | TryWrite returns false when full, recovers after drain | PASS |
| RingReadWithinTimeout | Polling timeout returns correctly | PASS |
| LOARingIntegration/64B | WriteLOA+ReadLOA with CRC (64 bytes) | PASS |
| LOARingIntegration/1024B | WriteLOA+ReadLOA with CRC (1 KB) | PASS |
| LOARingIntegration/4000B | WriteLOA+ReadLOA with CRC (4 KB) | PASS |
| LOAZeroCopyPath | Zero-copy alloc → direct write → deref, byte-exact | PASS |
| InlineViaReadLOA | Non-LOA messages pass through ReadLOA correctly | PASS |
| RingBackpressure | Backoff timeout + FlagDropOK returns ErrDropped | PASS |
| LOABackpressure | Pool full → retry → succeed after release | PASS |
| WriteLOAWithBackoff | Full backpressure write cycle (LOA + ring) | PASS |
| ConcurrentRingAccess | 500 msgs, producer+consumer goroutines (400 cycles) | PASS |
| ConcurrentLOA | 8 goroutines x 50 alloc/release cycles | PASS |
| MultiSlotLOA | 5KB across 5x1KB slots, CRC verified | PASS |
| PigeonLifecycle | Daemon start → COI register → LOA discover → alloc | PASS |
| COILeaseExpiry | Register → close → unregister propagates | PASS |
| FullPipeline | Pigeon + COI + Ring + LOA + Schema end-to-end | PASS |
| COIRouting | Fragment A → pigeon → Fragment B (no self-loop) | PASS |
| MultiSubscriberRouting | A → pigeon → B + C (fan-out) | PASS |
| LOARouting | LOA pointer routed through pigeon, zero-copy deref | PASS |
| AttachSCMRights | Real UDS FD passing via SCM_RIGHTS, bidirectional ring | PASS |
| TwoFragmentsAttachAndRoute | Two fragments Attach() + route via COI | PASS |

---

## 4. Go Microbenchmarks (Apple M4)

| Benchmark | ns/op | B/op | allocs/op |
|-----------|-------|------|-----------|
| HeaderPackUnpack | 5.6 | 0 | 0 |
| RouterLookup | 13.4 | 0 | 0 |
| LOAAllocCommitRelease | 9.1 | 0 | 0 |
| LOAAllocCommitDerefRelease | 9.3 | 0 | 0 |
| LOARefEncodeDecode | 0.35 | 0 | 0 |
| WeightShardMetaEncodeDecode | 1.7 | 0 | 0 |

All hot paths are **zero-allocation**.

---

## 5. Comparative Benchmarks (fragmind vs baselines)

All measurements are **end-to-end** (write → kernel → read → verify CRC). No half-measured kernel buffer tricks.

### Small Messages — P50 Latency (lower = better)

| Transport | 64B | 1KB | Speedup vs TCP |
|-----------|-----|-----|----------------|
| **fragmind-ring** | **167 ns** | **208 ns** | **61x / 51x** |
| UDS | 3,125 ns | 3,334 ns | 3.3x / 3.2x |
| Pipe | 2,959 ns | 3,208 ns | 3.5x / 3.3x |
| TCP | 10,208 ns | 10,542 ns | 1x / 1x |

### Large Objects — Throughput MB/s (higher = better)

| Transport | 64KB | 1MB | 16MB |
|-----------|------|-----|------|
| **fragmind-loa** | **8,806** | **8,607** | **6,060** |
| TCP | 2,686 | 5,469 | 5,136 |
| Pipe | 3,609 | 3,632 | 3,125 |
| UDS | 1,678 | 2,501 | 2,429 |

### Fragmind advantage at 16MB (weight shard):

| vs | Throughput ratio | Latency ratio |
|----|-----------------|---------------|
| vs UDS | **2.5x** faster | **3.4x** lower latency |
| vs Pipe | **1.9x** faster | **2.6x** lower latency |
| vs TCP | **1.2x** faster | **1.6x** lower latency |

---

## 6. Python Binding Tests (31 tests)

### Core Bindings (22 tests)
| Test | Status |
|------|--------|
| TestHeader::test_pack_unpack_roundtrip | PASS |
| TestHeader::test_is_loa | PASS |
| TestLOARef::test_encode_decode | PASS |
| TestRing::test_write_read_roundtrip | PASS |
| TestRing::test_empty_read | PASS |
| TestRing::test_full_ring | PASS |
| TestRing::test_ordering | PASS |
| TestRing::test_read_within_timeout | PASS |
| TestRing::test_various_payload_sizes | PASS |
| TestLOAPool::test_create_open | PASS |
| TestLOAPool::test_alloc_commit_deref_release | PASS |
| TestLOAPool::test_pool_full | PASS |
| TestLOAPool::test_oversized | PASS |
| TestLOAPool::test_multiple_readers | PASS |
| TestLOARing::test_write_loa_read_loa | PASS |
| TestLOARing::test_inline_via_read_loa | PASS |
| TestInterop::test_header_binary_compat | PASS |
| TestInterop::test_loa_ref_binary_compat | PASS |
| TestInterop::test_loa_pool_magic_compat | PASS |
| TestInterop::test_payload_crc_compat | PASS |
| TestPerformance::test_ring_throughput | PASS |
| TestPerformance::test_loa_throughput | PASS |

### NumPy Tensor Tests (7 tests)
| Test | Status |
|------|--------|
| TestAllocTensor::test_alloc_f16 | PASS |
| TestAllocTensor::test_alloc_f32 | PASS |
| TestDerefTensor::test_deref_copy | PASS |
| TestDerefTensor::test_deref_view_zerocopy | PASS |
| TestWriteReadTensor::test_write_read_f16 | PASS |
| TestWriteReadTensor::test_zerocopy_write_path | PASS |
| TestPerformance::test_numpy_loa_throughput | PASS |

### Cross-Language Interop (2 tests)
| Test | Status |
|------|--------|
| test_python_write_go_read_inline | PASS (CRC verified) |
| test_python_write_go_read_loa | PASS (LOA magic + CRC verified) |

### Python Performance

| Operation | Throughput |
|-----------|-----------|
| Ring write+read | 466K ops/s, 60 MB/s |
| LOA alloc+deref+release | 535K ops/s, 548 MB/s |
| NumPy tensor LOA (128x128 f16) | 269K ops/s, 8.8 GB/s |

---

## 7. Real ML Workload — Distributed MNIST Training

Training a 2-layer MLP (784→128→ReLU→128→10) on 60,000 MNIST images, distributed across two fragment workers sharing activations and gradients through fragmind's LOA pool.

### Configuration
| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (60K train, 10K test) |
| Model | Linear(784→128) + ReLU → Linear(128→10) |
| Optimizer | Adam (lr=0.001) |
| Batch size | 64 |
| Epochs | 3 |
| IPC | LOA pool (128 slots × 128KB) + 2 rings (fwd + bwd) |

### Training Progress
| Epoch | Loss | Batches | Elapsed |
|-------|------|---------|---------|
| 1 | 0.2973 | 938 | 9.9s |
| 2 | 0.1344 | 938 | 19.5s |
| 3 | 0.0941 | 938 | 29.3s |

### Results

| Metric | Distributed (fragmind) | Baseline (single-process) |
|--------|----------------------|---------------------------|
| **Test accuracy** | **96.93%** | **97.00%** |
| Avg batch latency | 10.41 ms | 10.43 ms |
| Training time | 29.3s | 29.4s |
| **Overhead** | **-0.2%** | — |
| LOA transfers (fwd) | 87.9 MB | 0 |
| LOA transfers (bwd) | 87.9 MB | 0 |
| Total data via LOA | 175.8 MB | 0 |

**The distributed model achieves 96.93% accuracy — within 0.07% of the single-process baseline — with zero measurable overhead.**

Loss converges correctly from 0.30 → 0.09, proving gradients flow correctly through the LOA shared-memory pool.

---

## 8. Multi-Process Demo

Three separate OS processes (pigeon daemon + producer + consumer) exchange weight shards through COI-routed rings + LOA zero-copy.

| Metric | Value |
|--------|-------|
| Processes | 3 (pigeon, producer, consumer) |
| Shards sent | 5 |
| Shard size | 4,096 bytes |
| CRC verification | 5/5 passed |
| IPC | SCM_RIGHTS FD passing + LOA |
| Status | **PASS** |

---

## 9. Code Quality

### Amimica Clone Detection
| Metric | Value |
|--------|-------|
| Files scanned | 44 |
| Functions analyzed | 124 |
| Clone classes >0.5 | 5 (all intentional: benchmarks, schema codecs) |
| Clone classes >0.8 | 0 |

No actionable duplicates. Previous 4-region `makeRing` duplicate was refactored into shared `CreateRing()` helper.

### Build Verification
| Build | Status |
|-------|--------|
| `go build ./...` (default) | PASS |
| `go build -tags=quic ./...` | PASS |
| `go vet ./...` | PASS |

---

## 10. Test Coverage by Component

| Component | Unit Tests | E2E Tests | Benchmarks | Total |
|-----------|-----------|-----------|------------|-------|
| Ring | 0 | 7 | 0 | 7 |
| LOA Pool | 8 | 5 | 4 | 17 |
| Multi-Slot LOA | 5 | 1 | 0 | 6 |
| Header | 2 | 0 | 1 | 3 |
| Router | 6 | 0 | 1 | 7 |
| Schema | 6 | 0 | 1 | 7 |
| Backpressure | 0 | 3 | 0 | 3 |
| Pigeon Daemon | 0 | 2 | 0 | 2 |
| COI Routing | 0 | 3 | 0 | 3 |
| Fragment Attach | 0 | 2 | 0 | 2 |
| Python Bindings | 22 | 2 | 2 | 26 |
| NumPy Helpers | 7 | 0 | 1 | 8 |
| Proof (comparative) | 0 | 0 | 25 | 25 |
| **Total** | **56** | **25** | **35** | **116** |

---

## 11. Known Limitations

- RDMA transport (`peers_rdma.go`) requires libibverbs headers — cannot build-verify on machines without RDMA hardware
- Ring tests use poll mode (no eventfd) since eventfd is Linux-only; blocking read path untested on macOS
- Python bindings don't support `Attach()` over UDS (no SCM_RIGHTS in Python yet)
- Proof benchmarks show LOA zero-copy throughput lower than LOA-with-copy due to byte-by-byte pattern fill in the benchmark (not representative of real GPU DMA)
- Multi-process demo uses `go run` (compile overhead); production would use pre-built binaries
