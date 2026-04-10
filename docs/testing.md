# Testing Guide

## Test Structure

```
pkg/fragpigeon/
  header_test.go       Header pack/unpack, size assertions
  router_test.go       Router CRUD, concurrent access
  loa_test.go          LOA pool lifecycle, pool full, bad magic, concurrent
  loa_multi_test.go    Multi-slot allocation, fragmentation
  schema_test.go       All schema metadata encode/decode, DType sizes
  bench_test.go        Microbenchmarks (header, router, LOA, schema)
  e2e_test.go          End-to-end integration tests (19 tests)

proof/
  proof_test.go        Comparative benchmarks (fragmind vs UDS/TCP/Pipe)
  harness.go           Payload generation, latency histogram, JSON reporting
  fragmind.go          Ring + LOA benchmark wrappers
  baselines.go         UDS, TCP, Pipe transport implementations
  run.sh               One-command benchmark runner

bindings/python/
  test_fragpigeon.py         Python binding unit tests (22 tests)
  test_cross_language.py     Python-writes-Go-reads interop (2 tests)
```

## Running Tests

```bash
# All Go tests (unit + e2e)
go test -v ./pkg/fragpigeon/

# Just e2e tests
go test -v -run TestE2E ./pkg/fragpigeon/

# Microbenchmarks
go test -bench=. -benchmem ./pkg/fragpigeon/

# Comparative proof benchmarks (with JSON report)
go test -v -run TestProofRun ./proof/

# Full proof run (report + Go benchmarks)
bash proof/run.sh

# Python tests
cd bindings/python
python -m pytest test_fragpigeon.py -v
python -m pytest test_cross_language.py -v -s

# Everything
go test ./... && cd bindings/python && python -m pytest -v
```

## E2E Test Coverage

| Test | What It Proves |
|------|----------------|
| `RingRoundTrip` | Ring write/read with CRC at 4 payload sizes (8B-440B) |
| `RingOrdering` | 100 messages arrive in strict FIFO order |
| `RingFull` | TryWrite returns false when full, recovers after drain |
| `RingReadWithinTimeout` | Polling timeout returns after deadline |
| `LOARingIntegration` | WriteLOA/ReadLOA with CRC at 3 sizes (64B-4KB) |
| `LOAZeroCopyPath` | Zero-copy alloc -> direct write -> deref, byte-exact |
| `InlineViaReadLOA` | Non-LOA messages pass through ReadLOA correctly |
| `RingBackpressure` | Backoff timeout, FlagDropOK returns ErrDropped |
| `LOABackpressure` | Pool full -> retry -> succeed after release |
| `WriteLOAWithBackoff` | Full backpressure cycle (LOA alloc + ring write) |
| `ConcurrentRingAccess` | 500 msgs, producer/consumer goroutines, no race |
| `ConcurrentLOA` | 8 goroutines x 50 cycles = 400 alloc/release cycles |
| `MultiSlotLOA` | 5KB across 5 x 1KB slots, CRC verified |
| `PigeonLifecycle` | Daemon start -> COI register -> LOA discover -> alloc |
| `COILeaseExpiry` | Register -> close -> unregister propagates |
| `FullPipeline` | Pigeon + COI + Ring + LOA + Schema end-to-end |
| `COIRouting` | Fragment A -> pigeon -> Fragment B, no self-loop |
| `MultiSubscriberRouting` | A -> pigeon -> B + C (fan-out to two subscribers) |
| `LOARouting` | LOA pointer routed through pigeon, zero-copy deref |

## Proof Benchmarks

The proof suite compares fragmind against standard IPC at realistic LLM payload sizes:

| Payload | Represents |
|---------|------------|
| 64 B | Heartbeat/ping |
| 1 KB | Token batch |
| 64 KB | Activation slice |
| 1 MB | KV-cache chunk |
| 16 MB | Weight shard |

Transports tested: `fragmind-ring`, `fragmind-loa`, `fragmind-loa-zerocopy`, `uds`, `tcp`, `pipe`.

All benchmarks measure **end-to-end**: writer sends, blocks until reader confirms receipt and verifies CRC. No half-measured kernel buffer tricks.

Results are saved to `proof/results/proof-YYYYMMDD-HHMMSS.json` with metadata (Go version, OS, arch, CPU count) for tracking regressions over time.

## Writing New Tests

Follow these patterns:

**Ring tests**: use `testRing(t, capSlots, slotSize)` helper to create a temp ring.

**LOA tests**: use `tempLOAPool(t, numSlots, slotSize)` helper.

**Pigeon tests**: set env vars with `t.Setenv()`, call `p.initCOIShm()` + `p.initLOAPool()` + `p.serveUDS()` + `go p.housekeep()` manually (don't call `p.Run()` which blocks).

**Routing tests**: create `AttachedFragment` with `done` and `stopped` channels, `defer frag.Stop()` before test returns to prevent goroutine leaks. Use short UDS paths (`/tmp/fp-test-*.sock`) on macOS to avoid the 108-char limit.

**Payload verification**: use `testPayload(size)` + `verifyPayload(buf)` which generate deterministic CRC32-protected data (same algorithm as the proof suite and Python bindings).
