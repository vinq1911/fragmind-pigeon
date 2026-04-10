package fragpigeon

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// --- Ring helpers for tests ---

func testRing(t *testing.T, capSlots, slotSize int) (*Ring, func()) {
	t.Helper()
	dir := t.TempDir()
	size := 64 + capSlots*slotSize
	path := filepath.Join(dir, "ring.shm")

	fd, err := unix.Open(path, unix.O_CREAT|unix.O_EXCL|unix.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		t.Fatal(err)
	}
	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint64(mem[0:], uint64(capSlots))
	binary.LittleEndian.PutUint64(mem[8:], 0)
	binary.LittleEndian.PutUint64(mem[16:], 0)
	binary.LittleEndian.PutUint32(mem[24:], uint32(slotSize))
	binary.LittleEndian.PutUint64(mem[32:], ^uint64(0)) // no eventfd
	binary.LittleEndian.PutUint64(mem[40:], ^uint64(0))
	_ = unix.Munmap(mem)
	_ = os.Remove(path)

	ring, err := OpenRingFromFD(fd)
	if err != nil {
		t.Fatal(err)
	}
	return ring, func() { ring.Close(); unix.Close(fd) }
}

func testPayload(size int) []byte {
	buf := make([]byte, size)
	for i := 0; i < size-4; i++ {
		buf[i] = byte(i*7 + 13)
	}
	csum := crc32.ChecksumIEEE(buf[:size-4])
	binary.LittleEndian.PutUint32(buf[size-4:], csum)
	return buf
}

func verifyPayload(buf []byte) bool {
	if len(buf) < 8 {
		return false
	}
	expected := binary.LittleEndian.Uint32(buf[len(buf)-4:])
	return crc32.ChecksumIEEE(buf[:len(buf)-4]) == expected
}

func testHeader(payloadSize int, msgID uint32) Header {
	return Header{
		Len:         uint32(payloadSize),
		Kind:        KindProcess,
		TSns:        uint64(time.Now().UnixNano()),
		ConceptID:   0x8A7311CCDD55002A,
		ConceptBits: 24,
		SchemaID:    SchemaWeightShard,
		SrcID:       1001,
		MsgID:       msgID,
		Ver:         1,
	}
}

// ================================================================
// E2E Test 1: Ring write/read round-trip with CRC verification
// ================================================================

func TestE2E_RingRoundTrip(t *testing.T) {
	ring, cleanup := testRing(t, 64, 512)
	defer cleanup()

	for _, size := range []int{8, 64, 128, 440} {
		t.Run(fmt.Sprintf("%dB", size), func(t *testing.T) {
			payload := testPayload(size)
			hdr := testHeader(size, 1)

			if !ring.TryWrite(hdr, payload) {
				t.Fatal("TryWrite failed")
			}
			msg, err := ring.Read(false)
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			if msg.Header.MsgID != 1 {
				t.Fatalf("MsgID: got %d, want 1", msg.Header.MsgID)
			}
			if !verifyPayload(msg.Payload) {
				t.Fatal("payload CRC mismatch")
			}
		})
	}
}

// ================================================================
// E2E Test 2: Ring ordering — messages arrive in FIFO order
// ================================================================

func TestE2E_RingOrdering(t *testing.T) {
	ring, cleanup := testRing(t, 256, 256)
	defer cleanup()

	n := 100
	payload := testPayload(64)

	for i := 0; i < n; i++ {
		hdr := testHeader(64, uint32(i))
		if !ring.TryWrite(hdr, payload) {
			t.Fatalf("write %d failed", i)
		}
	}
	for i := 0; i < n; i++ {
		msg, err := ring.Read(false)
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		if msg.Header.MsgID != uint32(i) {
			t.Fatalf("ordering: got MsgID=%d at position %d", msg.Header.MsgID, i)
		}
	}
}

// ================================================================
// E2E Test 3: Ring full — TryWrite returns false
// ================================================================

func TestE2E_RingFull(t *testing.T) {
	ring, cleanup := testRing(t, 4, 256)
	defer cleanup()

	payload := testPayload(64)
	hdr := testHeader(64, 0)

	// Fill the ring
	for i := 0; i < 4; i++ {
		if !ring.TryWrite(hdr, payload) {
			t.Fatalf("write %d should succeed", i)
		}
	}
	// 5th write should fail
	if ring.TryWrite(hdr, payload) {
		t.Fatal("expected TryWrite to return false on full ring")
	}

	// Drain one, write should succeed again
	_, _ = ring.Read(false)
	if !ring.TryWrite(hdr, payload) {
		t.Fatal("write after drain should succeed")
	}
}

// ================================================================
// E2E Test 4: Ring ReadWithin timeout
// ================================================================

func TestE2E_RingReadWithinTimeout(t *testing.T) {
	ring, cleanup := testRing(t, 4, 256)
	defer cleanup()

	start := time.Now()
	_, err := ring.ReadWithin(50 * time.Millisecond)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error")
	}
	if elapsed < 40*time.Millisecond {
		t.Fatalf("returned too fast: %v", elapsed)
	}
}

// ================================================================
// E2E Test 5: LOA + Ring integration — WriteLOA / ReadLOA
// ================================================================

func TestE2E_LOARingIntegration(t *testing.T) {
	ring, ringCleanup := testRing(t, 64, 256)
	defer ringCleanup()

	pool := tempLOAPool(t, 16, 4096)
	defer pool.Close()

	for _, size := range []int{64, 1024, 4000} {
		t.Run(fmt.Sprintf("%dB", size), func(t *testing.T) {
			data := testPayload(size)
			hdr := testHeader(size, uint32(size))

			ref, ok := WriteLOA(ring, pool, hdr, data, 1001)
			if !ok {
				t.Fatal("WriteLOA failed")
			}

			msg, gotData, gotRef, err := ReadLOA(ring, pool, false)
			if err != nil {
				t.Fatalf("ReadLOA: %v", err)
			}
			if !msg.IsLOA() {
				t.Fatal("expected LOA flag")
			}
			if gotRef.SlotID != ref.SlotID {
				t.Fatalf("ref mismatch: got slot %d, want %d", gotRef.SlotID, ref.SlotID)
			}
			if !verifyPayload(gotData) {
				t.Fatal("LOA payload CRC mismatch")
			}

			pool.Release(gotRef)
		})
	}
}

// ================================================================
// E2E Test 6: LOA zero-copy write path
// ================================================================

func TestE2E_LOAZeroCopyPath(t *testing.T) {
	ring, ringCleanup := testRing(t, 64, 256)
	defer ringCleanup()

	pool := tempLOAPool(t, 8, 8192)
	defer pool.Close()

	size := uint32(4096)
	buf, ref, err := WriteLOAZeroCopy(pool, size, 1001)
	if err != nil {
		t.Fatal(err)
	}
	// Write directly into shm (zero-copy producer path)
	for i := range buf {
		buf[i] = byte(i * 31)
	}

	hdr := NewLOAHeader(KindProcess, 0x1234, 24, SchemaActivation, 1001)
	if !CommitLOA(ring, pool, ref, hdr) {
		t.Fatal("CommitLOA failed")
	}

	msg, data, gotRef, err := ReadLOA(ring, pool, false)
	if err != nil {
		t.Fatal(err)
	}
	if !msg.IsLOA() {
		t.Fatal("expected LOA")
	}
	if msg.Header.SchemaID != SchemaActivation {
		t.Fatalf("schema: got %d, want %d", msg.Header.SchemaID, SchemaActivation)
	}
	// Verify the data matches what was written
	for i := 0; i < int(size); i++ {
		if data[i] != byte(i*31) {
			t.Fatalf("data[%d]: got %d, want %d", i, data[i], byte(i*31))
		}
	}
	pool.Release(gotRef)
}

// ================================================================
// E2E Test 7: Inline (non-LOA) message through ReadLOA
// ================================================================

func TestE2E_InlineViaReadLOA(t *testing.T) {
	ring, cleanup := testRing(t, 16, 512)
	defer cleanup()

	pool := tempLOAPool(t, 4, 1024)
	defer pool.Close()

	payload := []byte("hello inline")
	hdr := Header{
		Len:  uint32(len(payload)),
		Kind: KindPing,
		Ver:  1,
	}
	if !ring.TryWrite(hdr, payload) {
		t.Fatal("write failed")
	}

	msg, data, ref, err := ReadLOA(ring, pool, false)
	if err != nil {
		t.Fatal(err)
	}
	if msg.IsLOA() {
		t.Fatal("should not be LOA")
	}
	if ref != (LOARef{}) {
		t.Fatal("ref should be zero for inline")
	}
	if string(data) != "hello inline" {
		t.Fatalf("data: got %q", data)
	}
}

// ================================================================
// E2E Test 8: Backpressure — RingWriteWithBackoff
// ================================================================

func TestE2E_RingBackpressure(t *testing.T) {
	ring, cleanup := testRing(t, 2, 256)
	defer cleanup()

	payload := testPayload(64)

	// Fill the ring
	for i := 0; i < 2; i++ {
		hdr := testHeader(64, uint32(i))
		if err := RingWriteWithBackoff(ring, hdr, payload, time.Second); err != nil {
			t.Fatal(err)
		}
	}

	// Ring is full — write with short timeout should fail
	hdr := testHeader(64, 99)
	err := RingWriteWithBackoff(ring, hdr, payload, 50*time.Millisecond)
	if err != ErrRingFull {
		t.Fatalf("expected ErrRingFull, got %v", err)
	}

	// With FlagDropOK, should get ErrDropped instead
	hdr.Flags = FlagDropOK
	err = RingWriteWithBackoff(ring, hdr, payload, 50*time.Millisecond)
	if err != ErrDropped {
		t.Fatalf("expected ErrDropped, got %v", err)
	}

	// Drain and retry — should succeed
	_, _ = ring.Read(false)
	hdr.Flags = 0
	if err := RingWriteWithBackoff(ring, hdr, payload, time.Second); err != nil {
		t.Fatalf("write after drain: %v", err)
	}
}

// ================================================================
// E2E Test 9: Backpressure — LOAAllocWithBackoff
// ================================================================

func TestE2E_LOABackpressure(t *testing.T) {
	pool := tempLOAPool(t, 2, 1024)
	defer pool.Close()

	// Fill pool
	_, ref0, _ := pool.Alloc(100, 1)
	pool.Commit(ref0)
	_, ref1, _ := pool.Alloc(100, 1)
	pool.Commit(ref1)

	// Should timeout
	_, _, err := LOAAllocWithBackoff(pool, 100, 1, 50*time.Millisecond)
	if err != ErrLOAFull {
		t.Fatalf("expected ErrLOAFull, got %v", err)
	}

	// Release one, retry should succeed
	pool.Release(ref0)
	buf, ref2, err := LOAAllocWithBackoff(pool, 100, 1, time.Second)
	if err != nil {
		t.Fatalf("alloc after release: %v", err)
	}
	buf[0] = 42
	pool.Commit(ref2)
	pool.Release(ref1)
	pool.Release(ref2)
}

// ================================================================
// E2E Test 10: WriteLOAWithBackoff full cycle
// ================================================================

func TestE2E_WriteLOAWithBackoff(t *testing.T) {
	ring, ringCleanup := testRing(t, 16, 256)
	defer ringCleanup()

	pool := tempLOAPool(t, 8, 4096)
	defer pool.Close()

	data := testPayload(2048)
	hdr := testHeader(2048, 1)

	ref, err := WriteLOAWithBackoff(ring, pool, hdr, data, 1001, time.Second)
	if err != nil {
		t.Fatal(err)
	}

	msg, gotData, gotRef, err := ReadLOA(ring, pool, false)
	if err != nil {
		t.Fatal(err)
	}
	if !msg.IsLOA() {
		t.Fatal("expected LOA")
	}
	if !verifyPayload(gotData) {
		t.Fatal("CRC mismatch")
	}
	_ = ref
	pool.Release(gotRef)
}

// ================================================================
// E2E Test 11: Concurrent ring producers and consumers
// ================================================================

func TestE2E_ConcurrentRingAccess(t *testing.T) {
	ring, cleanup := testRing(t, 1024, 256)
	defer cleanup()

	n := 500
	payload := testPayload(64)

	// Producer goroutine
	go func() {
		for i := 0; i < n; i++ {
			hdr := testHeader(64, uint32(i))
			for !ring.TryWrite(hdr, payload) {
				time.Sleep(10 * time.Microsecond)
			}
		}
	}()

	// Consumer — collect all messages
	received := 0
	for received < n {
		msg, err := ring.ReadWithin(5 * time.Second)
		if err != nil {
			t.Fatalf("read %d: %v", received, err)
		}
		if !verifyPayload(msg.Payload) {
			t.Fatalf("CRC fail at msg %d", received)
		}
		received++
	}
}

// ================================================================
// E2E Test 12: Concurrent LOA alloc/release under contention
// ================================================================

func TestE2E_ConcurrentLOA(t *testing.T) {
	pool := tempLOAPool(t, 16, 1024)
	defer pool.Close()

	var wg sync.WaitGroup
	var success atomic.Int64

	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				buf, ref, err := pool.Alloc(128, uint32(id))
				if err != nil {
					continue // pool full, expected
				}
				buf[0] = byte(id)
				pool.Commit(ref)

				data, err := pool.Deref(ref)
				if err != nil {
					t.Errorf("deref failed: %v", err)
					continue
				}
				if data[0] != byte(id) {
					t.Errorf("data mismatch: got %d, want %d", data[0], id)
				}
				pool.Release(ref)
				success.Add(1)
			}
		}(g)
	}
	wg.Wait()

	if success.Load() == 0 {
		t.Fatal("no successful alloc/release cycles")
	}
	t.Logf("concurrent LOA: %d successful cycles", success.Load())
}

// ================================================================
// E2E Test 13: Multi-slot LOA end-to-end
// ================================================================

func TestE2E_MultiSlotLOA(t *testing.T) {
	dir := t.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "multi.loa"),
		PoolID:   1,
		NumSlots: 32,
		SlotSize: 1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Allocate 5KB (needs 5 × 1KB slots)
	size := uint32(5000)
	buf, ref, err := pool.AllocMulti(size, 1001)
	if err != nil {
		t.Fatal(err)
	}
	if ref.NumSlots != 5 {
		t.Fatalf("expected 5 slots, got %d", ref.NumSlots)
	}

	// Write CRC-protected payload
	for i := 0; i < int(size)-4; i++ {
		buf[i] = byte(i*7 + 13)
	}
	csum := crc32.ChecksumIEEE(buf[:size-4])
	binary.LittleEndian.PutUint32(buf[size-4:], csum)

	pool.CommitMulti(ref)

	// Deref and verify
	data, err := pool.DerefMulti(ref)
	if err != nil {
		t.Fatal(err)
	}
	gotCsum := crc32.ChecksumIEEE(data[:size-4])
	if gotCsum != csum {
		t.Fatalf("CRC mismatch: got %08x, want %08x", gotCsum, csum)
	}

	pool.ReleaseMulti(ref)
}

// ================================================================
// E2E Test 14: Pigeon daemon lifecycle — start, COI register, LOA query
// ================================================================

func TestE2E_PigeonLifecycle(t *testing.T) {
	dir := t.TempDir()
	sock := filepath.Join(dir, "pigeon.sock")
	loaPath := filepath.Join(dir, "fragmind.loa.1")
	coiPath := filepath.Join(dir, "fragmind.coi.local")

	// Set env vars for the pigeon
	t.Setenv(COIShmEnv, coiPath)
	t.Setenv(LOAShmEnv, loaPath)

	p := NewPigeon(1, sock)

	// Start pigeon in background (Run blocks, so we need to init manually)
	if err := p.initCOIShm(); err != nil {
		t.Fatal(err)
	}
	if err := p.initLOAPool(); err != nil {
		t.Fatal(err)
	}
	if err := p.serveUDS(); err != nil {
		t.Fatal(err)
	}
	go p.housekeep()

	// Give UDS server a moment
	time.Sleep(50 * time.Millisecond)

	// Register COIs
	cois := []COI{
		{ConceptID: 0x8A7311CCDD55002A, Bits: 24, SchemaID: SchemaWeightShard},
		{ConceptID: 0x1234560000000000, Bits: 16, SchemaID: SchemaKVCache},
	}
	handle, err := StartCOI(COIOptions{SocketPath: sock, RenewEvery: 500 * time.Millisecond}, cois)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Close()

	// Wait for registration to propagate
	time.Sleep(100 * time.Millisecond)

	// Verify COIs are in the shm table
	table, err := OpenLocalCOITable(coiPath)
	if err != nil {
		t.Fatal(err)
	}
	defer table.Close()

	_, _, entries := table.Snapshot()
	if len(entries) != 2 {
		t.Fatalf("expected 2 COI entries, got %d", len(entries))
	}

	// Query LOA pool path
	loaPool, err := DiscoverLOAPool(sock)
	if err != nil {
		t.Fatal(err)
	}
	defer loaPool.Close()

	// Verify we can alloc from the discovered pool
	buf, ref, err := loaPool.Alloc(256, 2001)
	if err != nil {
		t.Fatalf("alloc from discovered pool: %v", err)
	}
	buf[0] = 0xAB
	loaPool.Commit(ref)
	loaPool.Release(ref)

	t.Log("pigeon lifecycle: PASS (COI register + LOA discover + alloc)")
}

// ================================================================
// E2E Test 15: COI lease expiration
// ================================================================

func TestE2E_COILeaseExpiry(t *testing.T) {
	dir := t.TempDir()
	sock := filepath.Join(dir, "pigeon.sock")
	coiPath := filepath.Join(dir, "fragmind.coi.local")
	loaPath := filepath.Join(dir, "fragmind.loa.1")

	t.Setenv(COIShmEnv, coiPath)
	t.Setenv(LOAShmEnv, loaPath)

	p := NewPigeon(1, sock)
	if err := p.initCOIShm(); err != nil {
		t.Fatal(err)
	}
	if err := p.initLOAPool(); err != nil {
		t.Fatal(err)
	}
	if err := p.serveUDS(); err != nil {
		t.Fatal(err)
	}
	go p.housekeep()
	time.Sleep(50 * time.Millisecond)

	// Register a COI, then close immediately (stops renewal)
	cois := []COI{{ConceptID: 0xDEAD, Bits: 16, SchemaID: 1}}
	handle, err := StartCOI(COIOptions{SocketPath: sock, RenewEvery: 100 * time.Millisecond}, cois)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)

	// Should see 1 entry
	p.mu.RLock()
	count1 := len(p.local)
	p.mu.RUnlock()
	if count1 != 1 {
		t.Fatalf("expected 1 COI, got %d", count1)
	}

	// Close handle (stops renewal, sends UNREGISTER)
	handle.Close()
	time.Sleep(200 * time.Millisecond)

	// After grace period, should be gone
	// LeaseTTL=5s + BlackoutGrace=15s is too long for a test,
	// but UNREGISTER should remove it immediately
	p.mu.RLock()
	count2 := len(p.local)
	p.mu.RUnlock()
	if count2 != 0 {
		t.Fatalf("expected 0 COIs after close, got %d", count2)
	}
}

// ================================================================
// E2E Test 16: Full pipeline — pigeon + ring + LOA + schema
// ================================================================

func TestE2E_FullPipeline(t *testing.T) {
	dir := t.TempDir()
	sock := filepath.Join(dir, "pigeon.sock")
	coiPath := filepath.Join(dir, "fragmind.coi.local")
	loaPath := filepath.Join(dir, "fragmind.loa.1")

	t.Setenv(COIShmEnv, coiPath)
	t.Setenv(LOAShmEnv, loaPath)

	// Start pigeon
	p := NewPigeon(1, sock)
	if err := p.initCOIShm(); err != nil {
		t.Fatal(err)
	}
	if err := p.initLOAPool(); err != nil {
		t.Fatal(err)
	}
	if err := p.serveUDS(); err != nil {
		t.Fatal(err)
	}
	go p.housekeep()
	time.Sleep(50 * time.Millisecond)

	// Create ring
	ring, ringCleanup := testRing(t, 64, 256)
	defer ringCleanup()

	// Discover LOA pool from pigeon
	pool, err := DiscoverLOAPool(sock)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Register COI
	cois := []COI{{ConceptID: 0x0001000000000000, Bits: 16, SchemaID: SchemaWeightShard}}
	handle, err := StartCOI(COIOptions{SocketPath: sock}, cois)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Close()
	time.Sleep(50 * time.Millisecond)

	// Producer: create weight shard, write via LOA
	shardSize := uint32(8192)
	shard := make([]byte, shardSize)
	for i := range shard {
		shard[i] = byte(i * 31)
	}
	shardCRC := crc32.ChecksumIEEE(shard)

	meta := WeightShardMeta{
		ModelID: 1, LayerStart: 0, LayerEnd: 32,
		DType: DTypeBF16, NumElements: shardSize / 2,
		Shape: [4]uint16{64, 64, 0, 0}, Checksum: shardCRC,
	}

	hdr := Header{
		Kind:        KindProcess,
		TSns:        uint64(time.Now().UnixNano()),
		ConceptID:   0x0001000000000000,
		ConceptBits: 16,
		SchemaID:    SchemaWeightShard,
		SrcID:       1001,
		MsgID:       1,
		Ver:         1,
	}

	ref, err := WriteLOAWithBackoff(ring, pool, hdr, shard, 1001, time.Second)
	if err != nil {
		t.Fatal(err)
	}

	// Consumer: read, verify schema, verify data
	msg, data, gotRef, err := ReadLOA(ring, pool, false)
	if err != nil {
		t.Fatal(err)
	}
	if !msg.IsLOA() {
		t.Fatal("expected LOA message")
	}
	if msg.Header.SchemaID != SchemaWeightShard {
		t.Fatalf("schema: got %d, want %d", msg.Header.SchemaID, SchemaWeightShard)
	}

	// Verify metadata would decode correctly
	var metaBuf [WeightShardMetaSize]byte
	meta.Encode(metaBuf[:])
	decoded := DecodeWeightShardMeta(metaBuf[:])
	if decoded.Checksum != shardCRC {
		t.Fatalf("meta checksum: got %08x, want %08x", decoded.Checksum, shardCRC)
	}

	// Verify raw data integrity
	gotCRC := crc32.ChecksumIEEE(data)
	if gotCRC != shardCRC {
		t.Fatalf("data CRC: got %08x, want %08x", gotCRC, shardCRC)
	}

	pool.Release(gotRef)
	_ = ref

	t.Log("full pipeline: pigeon + COI + ring + LOA + schema = PASS")
}

// ================================================================
// E2E Test 17: COI-based routing — Fragment A sends, Fragment B receives
// via pigeon routing (no direct knowledge of each other)
// ================================================================

func TestE2E_COIRouting(t *testing.T) {
	dir := t.TempDir()
	sock := filepath.Join(dir, "pigeon.sock")
	coiPath := filepath.Join(dir, "fragmind.coi.local")
	loaPath := filepath.Join(dir, "fragmind.loa.1")

	t.Setenv(COIShmEnv, coiPath)
	t.Setenv(LOAShmEnv, loaPath)

	// Start pigeon
	p := NewPigeon(1, sock)
	if err := p.initCOIShm(); err != nil {
		t.Fatal(err)
	}
	if err := p.initLOAPool(); err != nil {
		t.Fatal(err)
	}
	if err := p.serveUDS(); err != nil {
		t.Fatal(err)
	}
	go p.housekeep()
	time.Sleep(50 * time.Millisecond)

	// Create ring pairs for two fragments (in-process, bypassing SCM_RIGHTS)
	fragA_ID := uint32(1)
	fragB_ID := uint32(2)

	slotSize := HdrSize + 256
	// Fragment A: outRing (A writes, pigeon reads), inRing (pigeon writes, A reads)
	fragA_outRing, fragA_outCleanup := testRing(t, 64, slotSize)
	defer fragA_outCleanup()
	fragA_inRing, fragA_inCleanup := testRing(t, 64, slotSize)
	defer fragA_inCleanup()

	// Fragment B: outRing (B writes, pigeon reads), inRing (pigeon writes, B reads)
	fragB_outRing, fragB_outCleanup := testRing(t, 64, slotSize)
	defer fragB_outCleanup()
	fragB_inRing, fragB_inCleanup := testRing(t, 64, slotSize)
	defer fragB_inCleanup()

	// Register fragments with pigeon
	fragA := &AttachedFragment{FragID: fragA_ID, InRing: fragA_inRing, OutRing: fragA_outRing, done: make(chan struct{}), stopped: make(chan struct{})}
	fragB := &AttachedFragment{FragID: fragB_ID, InRing: fragB_inRing, OutRing: fragB_outRing, done: make(chan struct{}), stopped: make(chan struct{})}
	p.frags.add(fragA)
	p.frags.add(fragB)
	defer fragA.Stop()
	defer fragB.Stop()

	// Start routing goroutines (pigeon reads fragment outbound rings)
	go p.routeFragment(fragA)
	go p.routeFragment(fragB)

	// Fragment B subscribes to a COI (model-1, layers 0-31)
	conceptID := uint64(0x0001000000000000)
	p.router.Add(16, conceptID, []uint16{uint16(fragB_ID)}, nil)

	time.Sleep(50 * time.Millisecond)

	// Fragment A sends a message targeting that COI
	payload := testPayload(128)
	hdr := Header{
		Len:         uint32(len(payload)),
		Kind:        KindProcess,
		TSns:        uint64(time.Now().UnixNano()),
		ConceptID:   conceptID,
		ConceptBits: 16,
		SchemaID:    SchemaWeightShard,
		SrcID:       fragA_ID,
		MsgID:       1,
		Ver:         1,
	}

	// Write to Fragment A's outbound ring
	if !fragA_outRing.TryWrite(hdr, payload) {
		t.Fatal("fragment A write failed")
	}

	// Fragment B should receive the message on its inbound ring
	msg, err := fragB_inRing.ReadWithin(2 * time.Second)
	if err != nil {
		t.Fatalf("fragment B read: %v", err)
	}

	if msg.Header.MsgID != 1 {
		t.Fatalf("expected MsgID=1, got %d", msg.Header.MsgID)
	}
	if msg.Header.SrcID != fragA_ID {
		t.Fatalf("expected SrcID=%d, got %d", fragA_ID, msg.Header.SrcID)
	}
	if msg.Header.SchemaID != SchemaWeightShard {
		t.Fatalf("expected schema=%d, got %d", SchemaWeightShard, msg.Header.SchemaID)
	}
	if !verifyPayload(msg.Payload) {
		t.Fatal("payload CRC mismatch")
	}

	// Fragment A should NOT receive its own message
	_, err = fragA_inRing.ReadWithin(100 * time.Millisecond)
	if err == nil {
		t.Fatal("fragment A should not receive its own message")
	}

	t.Log("COI routing: Fragment A → pigeon → Fragment B = PASS")
}

// ================================================================
// E2E Test 18: Multi-subscriber routing — one sender, two receivers
// ================================================================

func TestE2E_MultiSubscriberRouting(t *testing.T) {
	dir := t.TempDir()
	// Use /tmp for socket to avoid macOS 108-char UDS path limit
	sock := fmt.Sprintf("/tmp/fp-test-multi-%d.sock", time.Now().UnixNano()%100000)
	defer os.Remove(sock)
	coiPath := filepath.Join(dir, "fragmind.coi.local")
	loaPath := filepath.Join(dir, "fragmind.loa.1")

	t.Setenv(COIShmEnv, coiPath)
	t.Setenv(LOAShmEnv, loaPath)

	p := NewPigeon(1, sock)
	if err := p.initCOIShm(); err != nil {
		t.Fatal(err)
	}
	if err := p.initLOAPool(); err != nil {
		t.Fatal(err)
	}
	if err := p.serveUDS(); err != nil {
		t.Fatal(err)
	}
	go p.housekeep()
	time.Sleep(50 * time.Millisecond)

	slotSize := HdrSize + 256

	// Three fragments: A (sender), B and C (receivers)
	fragA_out, fragA_outCleanup := testRing(t, 64, slotSize)
	defer fragA_outCleanup()
	fragA_in, fragA_inCleanup := testRing(t, 64, slotSize)
	defer fragA_inCleanup()
	fragB_in, fragB_inCleanup := testRing(t, 64, slotSize)
	defer fragB_inCleanup()
	fragB_out, fragB_outCleanup := testRing(t, 64, slotSize)
	defer fragB_outCleanup()
	fragC_in, fragC_inCleanup := testRing(t, 64, slotSize)
	defer fragC_inCleanup()
	fragC_out, fragC_outCleanup := testRing(t, 64, slotSize)
	defer fragC_outCleanup()

	fA := &AttachedFragment{FragID: 1, InRing: fragA_in, OutRing: fragA_out, done: make(chan struct{}), stopped: make(chan struct{})}
	fB := &AttachedFragment{FragID: 2, InRing: fragB_in, OutRing: fragB_out, done: make(chan struct{}), stopped: make(chan struct{})}
	fC := &AttachedFragment{FragID: 3, InRing: fragC_in, OutRing: fragC_out, done: make(chan struct{}), stopped: make(chan struct{})}
	p.frags.add(fA)
	p.frags.add(fB)
	p.frags.add(fC)
	defer fA.Stop()
	defer fB.Stop()
	defer fC.Stop()

	go p.routeFragment(fA)
	go p.routeFragment(fB)
	go p.routeFragment(fC)

	// Both B and C subscribe to the same COI
	conceptID := uint64(0x00AA000000000000)
	p.router.Add(16, conceptID, []uint16{2, 3}, nil)
	time.Sleep(50 * time.Millisecond)

	// A sends one message
	payload := testPayload(64)
	hdr := Header{
		Len: uint32(len(payload)), Kind: KindProcess,
		ConceptID: conceptID, ConceptBits: 16,
		SchemaID: SchemaActivation, SrcID: 1, MsgID: 42, Ver: 1,
	}
	if !fragA_out.TryWrite(hdr, payload) {
		t.Fatal("A write failed")
	}

	// Both B and C should receive it
	msgB, err := fragB_in.ReadWithin(2 * time.Second)
	if err != nil {
		t.Fatalf("B read: %v", err)
	}
	msgC, err := fragC_in.ReadWithin(2 * time.Second)
	if err != nil {
		t.Fatalf("C read: %v", err)
	}

	if msgB.Header.MsgID != 42 || msgC.Header.MsgID != 42 {
		t.Fatalf("MsgID mismatch: B=%d C=%d", msgB.Header.MsgID, msgC.Header.MsgID)
	}
	if !verifyPayload(msgB.Payload) || !verifyPayload(msgC.Payload) {
		t.Fatal("payload verification failed")
	}

	t.Log("multi-subscriber routing: A → pigeon → B + C = PASS")
}

// ================================================================
// E2E Test 19: LOA routing — sender uses LOA, receiver gets LOA pointer
// ================================================================

func TestE2E_LOARouting(t *testing.T) {
	dir := t.TempDir()
	sock := filepath.Join(dir, "pigeon.sock")
	coiPath := filepath.Join(dir, "fragmind.coi.local")
	loaPath := filepath.Join(dir, "fragmind.loa.1")

	t.Setenv(COIShmEnv, coiPath)
	t.Setenv(LOAShmEnv, loaPath)

	p := NewPigeon(1, sock)
	if err := p.initCOIShm(); err != nil {
		t.Fatal(err)
	}
	if err := p.initLOAPool(); err != nil {
		t.Fatal(err)
	}
	if err := p.serveUDS(); err != nil {
		t.Fatal(err)
	}
	go p.housekeep()
	time.Sleep(50 * time.Millisecond)

	pool := p.LOAPool()

	// Slot size for rings: just enough for LOA pointer messages
	ringSlotSize := HdrSize + LOARefSize + 16

	fragA_out, fragA_outCleanup := testRing(t, 64, ringSlotSize)
	defer fragA_outCleanup()
	fragA_in, fragA_inCleanup := testRing(t, 64, ringSlotSize)
	defer fragA_inCleanup()
	fragB_out, fragB_outCleanup := testRing(t, 64, ringSlotSize)
	defer fragB_outCleanup()
	fragB_in, fragB_inCleanup := testRing(t, 64, ringSlotSize)
	defer fragB_inCleanup()

	fA := &AttachedFragment{FragID: 1, InRing: fragA_in, OutRing: fragA_out, done: make(chan struct{}), stopped: make(chan struct{})}
	fB := &AttachedFragment{FragID: 2, InRing: fragB_in, OutRing: fragB_out, done: make(chan struct{}), stopped: make(chan struct{})}
	p.frags.add(fA)
	p.frags.add(fB)
	defer fA.Stop()
	defer fB.Stop()

	go p.routeFragment(fA)
	go p.routeFragment(fB)

	conceptID := uint64(0x00BB000000000000)
	p.router.Add(16, conceptID, []uint16{2}, nil)
	time.Sleep(50 * time.Millisecond)

	// Fragment A writes a 4KB weight shard via LOA
	shardSize := 4096
	shard := make([]byte, shardSize)
	for i := range shard {
		shard[i] = byte(i * 17)
	}
	shardCRC := crc32.ChecksumIEEE(shard)

	hdr := Header{
		Kind:        KindProcess,
		ConceptID:   conceptID,
		ConceptBits: 16,
		SchemaID:    SchemaWeightShard,
		SrcID:       1,
		MsgID:       99,
		Ver:         1,
	}

	ref, ok := WriteLOA(fragA_out, pool, hdr, shard, 1)
	if !ok {
		t.Fatal("WriteLOA failed")
	}

	// Fragment B receives the LOA pointer via pigeon routing
	msg, err := fragB_in.ReadWithin(2 * time.Second)
	if err != nil {
		t.Fatalf("B read: %v", err)
	}
	if !msg.IsLOA() {
		t.Fatal("expected LOA message")
	}
	if msg.Header.MsgID != 99 {
		t.Fatalf("MsgID: got %d, want 99", msg.Header.MsgID)
	}

	// Deref the LOA pointer — same shm pool, zero-copy
	gotRef := msg.LOARef()
	data, err := pool.Deref(gotRef)
	if err != nil {
		t.Fatalf("LOA deref: %v", err)
	}

	gotCRC := crc32.ChecksumIEEE(data)
	if gotCRC != shardCRC {
		t.Fatalf("CRC mismatch: got %08x, want %08x", gotCRC, shardCRC)
	}

	pool.Release(gotRef)
	_ = ref

	t.Log("LOA routing: A writes LOA → pigeon → B reads zero-copy = PASS")
}
