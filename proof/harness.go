package proof

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"
)

// Payload sizes representing realistic LLM workloads.
type PayloadSpec struct {
	Name string
	Size int
}

var Payloads = []PayloadSpec{
	{"Tiny-64B", 64},
	{"TokenBatch-1KB", 1024},
	{"Activation-64KB", 64 * 1024},
	{"KVSlice-1MB", 1 * 1024 * 1024},
	{"WeightShard-16MB", 16 * 1024 * 1024},
}

// GeneratePayload creates a deterministic payload with a CRC32 trailer.
// Layout: [size-4 bytes of pattern data] [4 bytes CRC32]
func GeneratePayload(size int) []byte {
	if size < 8 {
		size = 8
	}
	buf := make([]byte, size)
	// Fill with deterministic pattern (position-dependent)
	for i := 0; i < size-4; i++ {
		buf[i] = byte(i*7 + 13)
	}
	// Last 4 bytes = CRC32 of the data portion
	csum := crc32.ChecksumIEEE(buf[:size-4])
	binary.LittleEndian.PutUint32(buf[size-4:], csum)
	return buf
}

// VerifyPayload checks the CRC32 trailer. Returns true if data is intact.
func VerifyPayload(buf []byte) bool {
	if len(buf) < 8 {
		return false
	}
	size := len(buf)
	expected := binary.LittleEndian.Uint32(buf[size-4:])
	actual := crc32.ChecksumIEEE(buf[:size-4])
	return expected == actual
}

// LatencyRecorder collects per-operation timings for percentile reporting.
type LatencyRecorder struct {
	samples []time.Duration
}

func NewLatencyRecorder(capacity int) *LatencyRecorder {
	return &LatencyRecorder{samples: make([]time.Duration, 0, capacity)}
}

func (lr *LatencyRecorder) Record(d time.Duration) {
	lr.samples = append(lr.samples, d)
}

func (lr *LatencyRecorder) Count() int { return len(lr.samples) }

type LatencyStats struct {
	P50  time.Duration
	P95  time.Duration
	P99  time.Duration
	Min  time.Duration
	Max  time.Duration
	Mean time.Duration
}

func (lr *LatencyRecorder) Stats() LatencyStats {
	n := len(lr.samples)
	if n == 0 {
		return LatencyStats{}
	}
	sorted := make([]time.Duration, n)
	copy(sorted, lr.samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var total time.Duration
	for _, s := range sorted {
		total += s
	}

	return LatencyStats{
		P50:  sorted[n*50/100],
		P95:  sorted[n*95/100],
		P99:  sorted[n*99/100],
		Min:  sorted[0],
		Max:  sorted[n-1],
		Mean: total / time.Duration(n),
	}
}

// BenchResult captures one benchmark run for JSON output.
type BenchResult struct {
	Transport   string        `json:"transport"`
	PayloadName string        `json:"payload_name"`
	PayloadSize int           `json:"payload_size"`
	Ops         int           `json:"ops"`
	TotalBytes  int64         `json:"total_bytes"`
	WallTime    time.Duration `json:"wall_time_ns"`
	ThroughputMBs float64    `json:"throughput_mb_s"`
	MsgsPerSec  float64       `json:"msgs_per_sec"`
	LatencyP50  int64         `json:"latency_p50_ns"`
	LatencyP95  int64         `json:"latency_p95_ns"`
	LatencyP99  int64         `json:"latency_p99_ns"`
	LatencyMin  int64         `json:"latency_min_ns"`
	LatencyMax  int64         `json:"latency_max_ns"`
	Verified    bool          `json:"verified"`
}

// RunReport is the top-level output for a complete benchmark run.
type RunReport struct {
	Timestamp string        `json:"timestamp"`
	GoVersion string        `json:"go_version"`
	OS        string        `json:"os"`
	Arch      string        `json:"arch"`
	NumCPU    int           `json:"num_cpu"`
	Results   []BenchResult `json:"results"`
}

func NewRunReport() RunReport {
	return RunReport{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		GoVersion: runtime.Version(),
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
		NumCPU:    runtime.NumCPU(),
	}
}

func (r *RunReport) Add(br BenchResult) {
	r.Results = append(r.Results, br)
}

func (r *RunReport) SaveJSON(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("proof-%s.json", time.Now().Format("20060102-150405"))
	path := filepath.Join(dir, name)
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", err
	}
	return path, os.WriteFile(path, data, 0o644)
}

func (r *RunReport) PrintSummary() {
	fmt.Println()
	fmt.Println("=== Fragmind Proof Benchmark Results ===")
	fmt.Printf("Date: %s | Go: %s | %s/%s | CPUs: %d\n\n",
		r.Timestamp, r.GoVersion, r.OS, r.Arch, r.NumCPU)

	fmt.Printf("%-20s %-18s %12s %12s %12s %12s %8s\n",
		"Transport", "Payload", "Throughput", "Msgs/s", "P50", "P99", "OK")
	fmt.Println("────────────────────────────────────────────────────────────────────────────────────────────────────")

	for _, br := range r.Results {
		ok := "yes"
		if !br.Verified {
			ok = "FAIL"
		}
		fmt.Printf("%-20s %-18s %10.1f MB/s %10.0f %10s %10s %8s\n",
			br.Transport, br.PayloadName,
			br.ThroughputMBs, br.MsgsPerSec,
			time.Duration(br.LatencyP50), time.Duration(br.LatencyP99), ok)
	}
	fmt.Println()
}
