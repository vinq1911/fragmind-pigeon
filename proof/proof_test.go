package proof

import (
	"fmt"
	"testing"
)

// opsForSize picks iteration count — more ops for small payloads,
// fewer for large ones to keep total runtime reasonable.
func opsForSize(size int) int {
	switch {
	case size <= 1024:
		return 100_000
	case size <= 64*1024:
		return 10_000
	case size <= 1024*1024:
		return 1_000
	default: // 16 MB
		return 100
	}
}

// --- Fragmind Ring (inline payloads, up to ~64KB) ---

func BenchmarkFragmindRing(b *testing.B) {
	// Ring handles inline payloads up to slot size.
	// For large payloads (>64KB), LOA is the right path.
	ringPayloads := []PayloadSpec{
		{"Tiny-64B", 64},
		{"TokenBatch-1KB", 1024},
		{"Activation-64KB", 64 * 1024},
	}
	for _, p := range ringPayloads {
		b.Run(p.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := BenchFragmindRing(p.Size, opsForSize(p.Size))
				if err != nil {
					b.Fatal(err)
				}
				if !result.Verified {
					b.Fatal("data verification failed")
				}
			}
		})
	}
}

// --- Fragmind LOA (zero-copy large objects) ---

func BenchmarkFragmindLOA(b *testing.B) {
	loaPayloads := []PayloadSpec{
		{"Activation-64KB", 64 * 1024},
		{"KVSlice-1MB", 1 * 1024 * 1024},
		{"WeightShard-16MB", 16 * 1024 * 1024},
	}
	for _, p := range loaPayloads {
		b.Run(p.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := BenchFragmindLOA(p.Size, opsForSize(p.Size))
				if err != nil {
					b.Fatal(err)
				}
				if !result.Verified {
					b.Fatal("data verification failed")
				}
			}
		})
	}
}

// --- Fragmind LOA Zero-Copy (no memcpy, direct-to-shm) ---

func BenchmarkFragmindLOAZeroCopy(b *testing.B) {
	loaPayloads := []PayloadSpec{
		{"Activation-64KB", 64 * 1024},
		{"KVSlice-1MB", 1 * 1024 * 1024},
		{"WeightShard-16MB", 16 * 1024 * 1024},
	}
	for _, p := range loaPayloads {
		b.Run(p.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := BenchFragmindLOAZeroCopy(p.Size, opsForSize(p.Size))
				if err != nil {
					b.Fatal(err)
				}
				if !result.Verified {
					b.Fatal("data verification failed")
				}
			}
		})
	}
}

// --- UDS Baseline ---

func BenchmarkUDS(b *testing.B) {
	for _, p := range Payloads {
		b.Run(p.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := BenchUDS(p.Size, opsForSize(p.Size))
				if err != nil {
					b.Fatal(err)
				}
				if !result.Verified {
					b.Fatal("data verification failed")
				}
			}
		})
	}
}

// --- TCP Baseline ---

func BenchmarkTCP(b *testing.B) {
	for _, p := range Payloads {
		b.Run(p.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := BenchTCP(p.Size, opsForSize(p.Size))
				if err != nil {
					b.Fatal(err)
				}
				if !result.Verified {
					b.Fatal("data verification failed")
				}
			}
		})
	}
}

// --- Pipe Baseline ---

func BenchmarkPipe(b *testing.B) {
	for _, p := range Payloads {
		b.Run(p.Name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := BenchPipe(p.Size, opsForSize(p.Size))
				if err != nil {
					b.Fatal(err)
				}
				if !result.Verified {
					b.Fatal("data verification failed")
				}
			}
		})
	}
}

// --- Full Proof Run (non-benchmark, produces JSON report) ---

func TestProofRun(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping proof run in short mode")
	}

	report := NewRunReport()

	type benchFunc func(size, count int) (BenchResult, error)
	transports := []struct {
		name string
		fn   benchFunc
		// max payload size this transport handles
		maxSize int
	}{
		{"fragmind-ring", BenchFragmindRing, 64 * 1024},                  // up to 64KB inline
		{"fragmind-loa", BenchFragmindLOA, 16 * 1024 * 1024},            // 64KB+ (with memcpy)
		{"fragmind-loa-zerocopy", BenchFragmindLOAZeroCopy, 16 * 1024 * 1024}, // 64KB+ (direct to shm)
		{"uds", BenchUDS, 16 * 1024 * 1024},
		{"tcp", BenchTCP, 16 * 1024 * 1024},
		{"pipe", BenchPipe, 16 * 1024 * 1024},
	}

	for _, tr := range transports {
		for _, p := range Payloads {
			if p.Size > tr.maxSize {
				continue
			}
			// Skip ring for sizes > 64KB (LOA handles those)
			if tr.name == "fragmind-ring" && p.Size > 64*1024 {
				continue
			}
			// LOA starts at 64KB
			if (tr.name == "fragmind-loa" || tr.name == "fragmind-loa-zerocopy") && p.Size < 64*1024 {
				continue
			}

			count := opsForSize(p.Size)
			t.Run(fmt.Sprintf("%s/%s", tr.name, p.Name), func(t *testing.T) {
				result, err := tr.fn(p.Size, count)
				if err != nil {
					t.Fatalf("%s/%s: %v", tr.name, p.Name, err)
				}
				if !result.Verified {
					t.Fatalf("%s/%s: data verification failed", tr.name, p.Name)
				}
				report.Add(result)
				t.Logf("%s/%s: %.1f MB/s, p50=%dns, p99=%dns",
					result.Transport, result.PayloadName,
					result.ThroughputMBs,
					result.LatencyP50, result.LatencyP99)
			})
		}
	}

	report.PrintSummary()

	path, err := report.SaveJSON("results")
	if err != nil {
		t.Logf("warning: could not save JSON: %v", err)
	} else {
		t.Logf("results saved to %s", path)
	}
}
