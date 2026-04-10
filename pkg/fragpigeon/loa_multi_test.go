package fragpigeon

import (
	"path/filepath"
	"testing"
)

func TestAllocMultiBasic(t *testing.T) {
	dir := t.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "multi.loa"),
		PoolID:   1,
		NumSlots: 16,
		SlotSize: 1024, // 1KB slots
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Allocate 3KB (needs 3 slots of 1KB each)
	payloadSize := uint32(3000)
	buf, ref, err := pool.AllocMulti(payloadSize, 1001)
	if err != nil {
		t.Fatal(err)
	}
	if ref.NumSlots != 3 {
		t.Fatalf("expected 3 slots, got %d", ref.NumSlots)
	}
	if uint32(len(buf)) != payloadSize {
		t.Fatalf("expected buf len %d, got %d", payloadSize, len(buf))
	}

	// Write pattern
	for i := range buf {
		buf[i] = byte(i & 0xFF)
	}
	pool.CommitMulti(ref)

	// Deref
	data, err := pool.DerefMulti(ref)
	if err != nil {
		t.Fatal(err)
	}
	if uint32(len(data)) != payloadSize {
		t.Fatalf("deref len: got %d, want %d", len(data), payloadSize)
	}
	for i := range data {
		if data[i] != byte(i&0xFF) {
			t.Fatalf("data mismatch at %d: got %d, want %d", i, data[i], byte(i&0xFF))
		}
	}

	pool.ReleaseMulti(ref)
}

func TestAllocMultiExact(t *testing.T) {
	dir := t.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "multi.loa"),
		PoolID:   1,
		NumSlots: 8,
		SlotSize: 4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Allocate exactly 2 slots worth
	buf, ref, err := pool.AllocMulti(8192, 1)
	if err != nil {
		t.Fatal(err)
	}
	if ref.NumSlots != 2 {
		t.Fatalf("expected 2 slots, got %d", ref.NumSlots)
	}
	if len(buf) != 8192 {
		t.Fatalf("expected 8192 bytes, got %d", len(buf))
	}
	pool.CommitMulti(ref)
	pool.ReleaseMulti(ref)
}

func TestAllocMultiTooLarge(t *testing.T) {
	dir := t.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "multi.loa"),
		PoolID:   1,
		NumSlots: 4,
		SlotSize: 1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Request more than total pool
	_, _, err = pool.AllocMulti(5000, 1)
	if err == nil {
		t.Fatal("expected error for oversized multi-alloc")
	}
}

func TestAllocMultiFragmentation(t *testing.T) {
	dir := t.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "multi.loa"),
		PoolID:   1,
		NumSlots: 8,
		SlotSize: 1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Allocate slots 0,1 (single) and 2,3 (single) to fragment the pool
	_, ref0, _ := pool.Alloc(512, 1)
	pool.Commit(ref0)
	_, ref1, _ := pool.Alloc(512, 1)
	pool.Commit(ref1)

	// Now free slot 0 but keep slot 1 — this fragments the contiguous space
	pool.Release(ref0)

	// Allocating 3 contiguous slots should still work from slots 2-7 range
	// (slots 2-4 are free, slot 1 is occupied)
	// Actually: we allocated ref0=slot7 (pop from end), ref1=slot6.
	// Free ref0 returns slot7. Free list now: [0,1,2,3,4,5,7]
	// 3 contiguous from slots 0-5 should work.
	buf, ref, err := pool.AllocMulti(3000, 1)
	if err != nil {
		t.Fatalf("multi-alloc after fragmentation: %v", err)
	}
	if ref.NumSlots != 3 {
		t.Fatalf("expected 3 slots, got %d", ref.NumSlots)
	}
	for i := range buf {
		buf[i] = 0xAB
	}
	pool.CommitMulti(ref)
	pool.ReleaseMulti(ref)
	pool.Release(ref1) // cleanup
}

func TestAllocMultiSingleSlot(t *testing.T) {
	dir := t.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "multi.loa"),
		PoolID:   1,
		NumSlots: 4,
		SlotSize: 4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// Single slot via AllocMulti
	buf, ref, err := pool.AllocMulti(100, 1)
	if err != nil {
		t.Fatal(err)
	}
	if ref.NumSlots != 1 {
		t.Fatalf("expected 1 slot, got %d", ref.NumSlots)
	}
	buf[0] = 42
	pool.CommitMulti(ref)

	data, err := pool.DerefMulti(ref)
	if err != nil {
		t.Fatal(err)
	}
	if data[0] != 42 {
		t.Fatalf("data mismatch: got %d, want 42", data[0])
	}
	pool.ReleaseMulti(ref)
}
