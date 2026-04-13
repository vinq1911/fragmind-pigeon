package proof

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

// mkRing delegates to the shared CreateRing helper.
func mkRing(dir string, name string, capSlots, slotSize int) (*fp.Ring, func(), error) {
	return fp.CreateRing(dir, name, capSlots, slotSize)
}

// BenchFragmindRing benchmarks the Ring transport for a given payload size.
// It writes N messages on the producer side and reads them on the consumer side,
// measuring per-message latency.
func BenchFragmindRing(payloadSize, count int) (BenchResult, error) {
	dir := os.TempDir()

	// SlotSize must fit header + payload
	slotSize := fp.HdrSize + payloadSize
	if slotSize < 256 {
		slotSize = 256
	}
	// Round up to power of 2 for slot count
	capSlots := 1024
	if payloadSize > 32*1024 {
		capSlots = 256 // fewer slots for large payloads to limit memory
	}

	ring, cleanup, err := mkRing(dir, fmt.Sprintf("proof-ring-%d", payloadSize), capSlots, slotSize)
	if err != nil {
		return BenchResult{}, fmt.Errorf("mkRing: %w", err)
	}
	defer cleanup()

	payload := GeneratePayload(payloadSize)
	lr := NewLatencyRecorder(count)
	verified := true

	hdr := fp.Header{
		Len:         uint32(payloadSize),
		Kind:        fp.KindProcess,
		ConceptID:   0x8A7311CCDD55002A,
		ConceptBits: 24,
		SchemaID:    fp.SchemaWeightShard,
		SrcID:       1001,
		Ver:         1,
	}

	// Single-threaded producer-consumer (same goroutine, alternating)
	// This measures the raw ring write+read latency without goroutine scheduling noise.
	start := time.Now()
	for i := 0; i < count; i++ {
		hdr.MsgID = uint32(i)
		hdr.TSns = uint64(time.Now().UnixNano())

		t0 := time.Now()
		for !ring.TryWrite(hdr, payload) {
			// ring full — shouldn't happen in single-threaded mode
		}
		msg, err := ring.Read(false)
		if err != nil {
			return BenchResult{}, fmt.Errorf("ring read: %w", err)
		}
		lr.Record(time.Since(t0))

		if !VerifyPayload(msg.Payload) {
			verified = false
		}
	}
	wall := time.Since(start)

	stats := lr.Stats()
	totalBytes := int64(count) * int64(payloadSize)

	return BenchResult{
		Transport:     "fragmind-ring",
		PayloadName:   payloadName(payloadSize),
		PayloadSize:   payloadSize,
		Ops:           count,
		TotalBytes:    totalBytes,
		WallTime:      wall,
		ThroughputMBs: float64(totalBytes) / wall.Seconds() / 1e6,
		MsgsPerSec:    float64(count) / wall.Seconds(),
		LatencyP50:    stats.P50.Nanoseconds(),
		LatencyP95:    stats.P95.Nanoseconds(),
		LatencyP99:    stats.P99.Nanoseconds(),
		LatencyMin:    stats.Min.Nanoseconds(),
		LatencyMax:    stats.Max.Nanoseconds(),
		Verified:      verified,
	}, nil
}

// BenchFragmindLOA benchmarks the LOA zero-copy path.
// Alloc -> fill -> commit -> deref -> verify -> release per iteration.
func BenchFragmindLOA(payloadSize, count int) (BenchResult, error) {
	dir := os.TempDir()
	slotSize := uint32(payloadSize)
	if slotSize < 64*1024 {
		slotSize = 64 * 1024
	}

	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path:     filepath.Join(dir, fmt.Sprintf("proof-loa-%d.shm", payloadSize)),
		PoolID:   1,
		NumSlots: 256,
		SlotSize: slotSize,
	})
	if err != nil {
		return BenchResult{}, fmt.Errorf("create LOA pool: %w", err)
	}
	defer pool.Close()

	payload := GeneratePayload(payloadSize)
	lr := NewLatencyRecorder(count)
	verified := true

	start := time.Now()
	for i := 0; i < count; i++ {
		t0 := time.Now()

		// Alloc + fill (simulates zero-copy write path)
		buf, ref, err := pool.Alloc(uint32(payloadSize), 1001)
		if err != nil {
			return BenchResult{}, fmt.Errorf("loa alloc: %w", err)
		}
		copy(buf, payload)
		pool.Commit(ref)

		// Deref (reader side)
		data, err := pool.Deref(ref)
		if err != nil {
			return BenchResult{}, fmt.Errorf("loa deref: %w", err)
		}
		if !VerifyPayload(data) {
			verified = false
		}
		pool.Release(ref)

		lr.Record(time.Since(t0))
	}
	wall := time.Since(start)

	stats := lr.Stats()
	totalBytes := int64(count) * int64(payloadSize)

	return BenchResult{
		Transport:     "fragmind-loa",
		PayloadName:   payloadName(payloadSize),
		PayloadSize:   payloadSize,
		Ops:           count,
		TotalBytes:    totalBytes,
		WallTime:      wall,
		ThroughputMBs: float64(totalBytes) / wall.Seconds() / 1e6,
		MsgsPerSec:    float64(count) / wall.Seconds(),
		LatencyP50:    stats.P50.Nanoseconds(),
		LatencyP95:    stats.P95.Nanoseconds(),
		LatencyP99:    stats.P99.Nanoseconds(),
		LatencyMin:    stats.Min.Nanoseconds(),
		LatencyMax:    stats.Max.Nanoseconds(),
		Verified:      verified,
	}, nil
}

// BenchFragmindLOAZeroCopy benchmarks the true zero-copy LOA path.
// Producer writes directly into the shm slot (no memcpy from Go heap).
// This simulates the real LLM workload where a GPU/engine writes directly
// into the LOA pool, and the consumer reads via pointer deref.
func BenchFragmindLOAZeroCopy(payloadSize, count int) (BenchResult, error) {
	dir := os.TempDir()
	slotSize := uint32(payloadSize)
	if slotSize < 64*1024 {
		slotSize = 64 * 1024
	}

	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path:     filepath.Join(dir, fmt.Sprintf("proof-loa-zc-%d.shm", payloadSize)),
		PoolID:   1,
		NumSlots: 256,
		SlotSize: slotSize,
	})
	if err != nil {
		return BenchResult{}, fmt.Errorf("create LOA pool: %w", err)
	}
	defer pool.Close()

	lr := NewLatencyRecorder(count)
	verified := true

	start := time.Now()
	for i := 0; i < count; i++ {
		t0 := time.Now()

		// Alloc returns a writable slice directly into shm.
		// In real use, the GPU/inference engine writes here — no memcpy.
		buf, ref, err := pool.Alloc(uint32(payloadSize), 1001)
		if err != nil {
			return BenchResult{}, fmt.Errorf("loa alloc: %w", err)
		}
		// Simulate in-place write: fill pattern + CRC directly in shm.
		// This is the zero-copy path — data never exists in Go heap.
		fillPayloadInPlace(buf)
		pool.Commit(ref)

		// Deref (reader side) — pointer into same shm, no copy
		data, err := pool.Deref(ref)
		if err != nil {
			return BenchResult{}, fmt.Errorf("loa deref: %w", err)
		}
		if !VerifyPayload(data) {
			verified = false
		}
		pool.Release(ref)

		lr.Record(time.Since(t0))
	}
	wall := time.Since(start)

	stats := lr.Stats()
	totalBytes := int64(count) * int64(payloadSize)

	return BenchResult{
		Transport:     "fragmind-loa-zerocopy",
		PayloadName:   payloadName(payloadSize),
		PayloadSize:   payloadSize,
		Ops:           count,
		TotalBytes:    totalBytes,
		WallTime:      wall,
		ThroughputMBs: float64(totalBytes) / wall.Seconds() / 1e6,
		MsgsPerSec:    float64(count) / wall.Seconds(),
		LatencyP50:    stats.P50.Nanoseconds(),
		LatencyP95:    stats.P95.Nanoseconds(),
		LatencyP99:    stats.P99.Nanoseconds(),
		LatencyMin:    stats.Min.Nanoseconds(),
		LatencyMax:    stats.Max.Nanoseconds(),
		Verified:      verified,
	}, nil
}

// fillPayloadInPlace writes the deterministic pattern + CRC32 directly
// into a buffer (simulates a producer writing directly into shm).
func fillPayloadInPlace(buf []byte) {
	size := len(buf)
	for i := 0; i < size-4; i++ {
		buf[i] = byte(i*7 + 13)
	}
	csum := crc32IEEE(buf[:size-4])
	binary.LittleEndian.PutUint32(buf[size-4:], csum)
}

func crc32IEEE(b []byte) uint32 { return crc32.ChecksumIEEE(b) }

func payloadName(size int) string {
	for _, p := range Payloads {
		if p.Size == size {
			return p.Name
		}
	}
	return fmt.Sprintf("%dB", size)
}
