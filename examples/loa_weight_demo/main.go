// loa_weight_demo: end-to-end demo of LOA zero-copy weight shard transfer.
//
// Runs a pigeon daemon, creates two fragments connected by a ring,
// and transfers a simulated bf16 weight shard (32KB–1MB) through the
// LOA pool with zero-copy on the reader side.
//
// Usage:
//
//	go run ./examples/loa_weight_demo
//	go run ./examples/loa_weight_demo -size 1048576  # 1 MB shard
//	go run ./examples/loa_weight_demo -count 100      # 100 iterations
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"hash/crc32"
	"log"
	"os"
	"path/filepath"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
	"golang.org/x/sys/unix"
)

func main() {
	shardSize := flag.Int("size", 64*1024, "weight shard size in bytes")
	count := flag.Int("count", 10, "number of shards to transfer")
	flag.Parse()

	dir, err := os.MkdirTemp("", "fragmind-loa-demo-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// --- 1. Create LOA pool (simulates what pigeon daemon does) ---
	loaPath := filepath.Join(dir, "fragmind.loa.1")
	slotSize := uint32(*shardSize)
	if slotSize < 64*1024 {
		slotSize = 64 * 1024
	}
	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path:     loaPath,
		PoolID:   1,
		NumSlots: 64,
		SlotSize: slotSize,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	// --- 2. Create shared-memory ring (producer → consumer) ---
	ringSlotSize := fp.HdrSize + fp.LOARefSize + 16 // header + LOARef payload + padding
	ring, ringCleanup, err := makeRing(dir, "ring-prod-cons", 256, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer ringCleanup()

	// --- 3. Generate test weight shard data ---
	shard := makeWeightShard(*shardSize)
	csum := crc32.ChecksumIEEE(shard)

	meta := fp.WeightShardMeta{
		ModelID:     1,
		LayerStart:  0,
		LayerEnd:    32,
		DType:       fp.DTypeBF16,
		NumElements: uint32(*shardSize / fp.DTypeSize(fp.DTypeBF16)),
		Shape:       [4]uint16{uint16(*shardSize / 2 / 1024), 1024, 0, 0},
		Checksum:    csum,
	}

	fmt.Printf("=== Fragmind LOA Weight Shard Demo ===\n")
	fmt.Printf("Shard size:  %d bytes (%s)\n", *shardSize, humanBytes(*shardSize))
	fmt.Printf("Iterations:  %d\n", *count)
	fmt.Printf("LOA pool:    %s (slots=%d slotSize=%s)\n", loaPath, 64, humanBytes(int(slotSize)))
	fmt.Printf("Schema:      WeightShard (bf16, layers 0-32, %d elements)\n", meta.NumElements)
	fmt.Println()

	// --- 4. Run producer → consumer loop ---
	var totalLatency time.Duration
	var minLat, maxLat time.Duration
	verified := 0

	for i := 0; i < *count; i++ {
		t0 := time.Now()

		// PRODUCER: alloc LOA slot, write shard zero-copy, send ring pointer
		buf, ref, err := fp.WriteLOAZeroCopy(pool, uint32(*shardSize), 1001)
		if err != nil {
			log.Fatalf("alloc: %v", err)
		}
		copy(buf, shard) // in real use: GPU writes directly here

		hdr := fp.Header{
			Kind:        fp.KindProcess,
			Flags:       fp.FlagLOAPtr,
			TSns:        uint64(time.Now().UnixNano()),
			ConceptID:   0x0001000000000000, // model 1, all layers
			ConceptBits: 16,
			SchemaID:    fp.SchemaWeightShard,
			SrcID:       1001,
			MsgID:       uint32(i),
			Ver:         1,
		}
		if !fp.CommitLOA(ring, pool, ref, hdr) {
			log.Fatal("ring full")
		}

		// CONSUMER: read ring message, deref LOA, verify, release
		msg, data, ref2, err := fp.ReadLOA(ring, pool, false)
		if err != nil {
			log.Fatalf("read: %v", err)
		}

		lat := time.Since(t0)
		totalLatency += lat
		if minLat == 0 || lat < minLat {
			minLat = lat
		}
		if lat > maxLat {
			maxLat = lat
		}

		// Verify data integrity
		if msg.Header.SchemaID != fp.SchemaWeightShard {
			log.Fatalf("wrong schema: %d", msg.Header.SchemaID)
		}
		if msg.Header.Flags&fp.FlagLOAPtr == 0 {
			log.Fatal("missing FlagLOAPtr")
		}
		gotCsum := crc32.ChecksumIEEE(data)
		if gotCsum == csum {
			verified++
		} else {
			log.Printf("[%d] CRC mismatch: got %08x want %08x", i, gotCsum, csum)
		}

		pool.Release(ref2)

		if i == 0 || (i+1)%10 == 0 || i == *count-1 {
			fmt.Printf("  [%d/%d] shard transferred in %s (verified=%v)\n",
				i+1, *count, lat, gotCsum == csum)
		}
	}

	// --- 5. Summary ---
	avgLat := totalLatency / time.Duration(*count)
	throughput := float64(*count) * float64(*shardSize) / totalLatency.Seconds() / 1e6

	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("Verified:    %d/%d\n", verified, *count)
	fmt.Printf("Avg latency: %s\n", avgLat)
	fmt.Printf("Min latency: %s\n", minLat)
	fmt.Printf("Max latency: %s\n", maxLat)
	fmt.Printf("Throughput:  %.1f MB/s\n", throughput)
	fmt.Printf("Copies:      1 (producer→shm) + 0 (consumer reads shm directly)\n")
	if verified == *count {
		fmt.Println("Status:      PASS")
	} else {
		fmt.Println("Status:      FAIL")
		os.Exit(1)
	}
}

func makeWeightShard(size int) []byte {
	buf := make([]byte, size)
	// Fill with pseudo-random bf16 pattern
	for i := range buf {
		buf[i] = byte((i * 31 + 17) & 0xFF)
	}
	return buf
}

func makeRing(dir, name string, capSlots, slotSize int) (*fp.Ring, func(), error) {
	size := 64 + capSlots*slotSize
	path := filepath.Join(dir, name+".shm")

	fd, err := unix.Open(path, unix.O_CREAT|unix.O_EXCL|unix.O_RDWR, 0600)
	if err != nil {
		return nil, nil, err
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		unix.Close(fd)
		return nil, nil, err
	}
	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		unix.Close(fd)
		return nil, nil, err
	}
	binary.LittleEndian.PutUint64(mem[0:], uint64(capSlots))
	binary.LittleEndian.PutUint64(mem[8:], 0)
	binary.LittleEndian.PutUint64(mem[16:], 0)
	binary.LittleEndian.PutUint32(mem[24:], uint32(slotSize))
	binary.LittleEndian.PutUint64(mem[32:], ^uint64(0))
	binary.LittleEndian.PutUint64(mem[40:], ^uint64(0))
	_ = unix.Munmap(mem)
	_ = os.Remove(path)

	ring, err := fp.OpenRingFromFD(fd)
	if err != nil {
		unix.Close(fd)
		return nil, nil, err
	}
	return ring, func() { ring.Close(); unix.Close(fd) }, nil
}

func humanBytes(b int) string {
	switch {
	case b >= 1024*1024:
		return fmt.Sprintf("%.1f MB", float64(b)/1024/1024)
	case b >= 1024:
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
