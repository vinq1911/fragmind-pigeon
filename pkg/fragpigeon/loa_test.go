package fragpigeon

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func tempLOAPool(t *testing.T, slots uint32, slotSize uint32) *LOAPool {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.loa")
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     path,
		PoolID:   1,
		NumSlots: slots,
		SlotSize: slotSize,
	})
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func TestLOACreateOpen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.loa")

	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     path,
		PoolID:   42,
		NumSlots: 16,
		SlotSize: 1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	if pool.poolID != 42 {
		t.Fatalf("poolID: got %d, want 42", pool.poolID)
	}
	if pool.numSlots != 16 {
		t.Fatalf("numSlots: got %d, want 16", pool.numSlots)
	}

	// Open the same pool from another handle
	pool2, err := OpenLOAPool(path)
	if err != nil {
		t.Fatal(err)
	}
	defer pool2.Close()

	if pool2.poolID != 42 {
		t.Fatalf("pool2 poolID: got %d, want 42", pool2.poolID)
	}
}

func TestLOAAllocCommitDerefRelease(t *testing.T) {
	pool := tempLOAPool(t, 8, 4096)
	defer pool.Close()

	data := []byte("hello fragmind tensor data here")

	buf, ref, err := pool.Alloc(uint32(len(data)), 1001)
	if err != nil {
		t.Fatal(err)
	}
	copy(buf, data)
	pool.Commit(ref)

	if ref.PoolID != 1 {
		t.Fatalf("ref.PoolID: got %d, want 1", ref.PoolID)
	}
	if ref.Length != uint32(len(data)) {
		t.Fatalf("ref.Length: got %d, want %d", ref.Length, len(data))
	}

	// Deref from a reader
	got, err := pool.Deref(ref)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(data) {
		t.Fatalf("deref data mismatch: got %q, want %q", got, data)
	}

	// Release
	pool.Release(ref)

	// Slot should now be free — we can alloc it again
	_, ref2, err := pool.Alloc(100, 1002)
	if err != nil {
		t.Fatal(err)
	}
	pool.Commit(ref2)
	pool.Release(ref2)
}

func TestLOAPoolFull(t *testing.T) {
	pool := tempLOAPool(t, 2, 256)
	defer pool.Close()

	_, ref1, err := pool.Alloc(10, 1)
	if err != nil {
		t.Fatal(err)
	}
	pool.Commit(ref1)

	_, ref2, err := pool.Alloc(10, 1)
	if err != nil {
		t.Fatal(err)
	}
	pool.Commit(ref2)

	// Third alloc should fail
	_, _, err = pool.Alloc(10, 1)
	if err != ErrLOAFull {
		t.Fatalf("expected ErrLOAFull, got %v", err)
	}

	// Release one and try again
	pool.Release(ref1)
	_, ref3, err := pool.Alloc(10, 1)
	if err != nil {
		t.Fatalf("expected alloc after release, got %v", err)
	}
	pool.Commit(ref3)
	pool.Release(ref2)
	pool.Release(ref3)
}

func TestLOARefEncodeDecode(t *testing.T) {
	ref := LOARef{
		PoolID: 5,
		SlotID: 1234,
		Offset: 0,
		Length: 65536,
	}
	var buf [LOARefSize]byte
	ref.Encode(buf[:])

	got := DecodeLOARef(buf[:])
	if got != ref {
		t.Fatalf("encode/decode mismatch: got %+v, want %+v", got, ref)
	}
}

func TestLOAMultipleReaders(t *testing.T) {
	pool := tempLOAPool(t, 4, 1024)
	defer pool.Close()

	data := []byte("shared tensor data")
	buf, ref, err := pool.Alloc(uint32(len(data)), 1)
	if err != nil {
		t.Fatal(err)
	}
	copy(buf, data)
	pool.Commit(ref)

	// Multiple readers deref the same slot
	for i := 0; i < 5; i++ {
		got, err := pool.Deref(ref)
		if err != nil {
			t.Fatalf("deref %d: %v", i, err)
		}
		if string(got) != string(data) {
			t.Fatalf("deref %d: data mismatch", i)
		}
	}

	// Release all readers — refcnt should go to 0 after 5 releases
	for i := 0; i < 5; i++ {
		pool.Release(ref)
	}
}

func TestLOAOversizedAlloc(t *testing.T) {
	pool := tempLOAPool(t, 4, 256)
	defer pool.Close()

	_, _, err := pool.Alloc(1024, 1) // larger than slot
	if err == nil {
		t.Fatal("expected error for oversized alloc")
	}
}

func TestLOAConcurrentAllocRelease(t *testing.T) {
	pool := tempLOAPool(t, 64, 512)
	defer pool.Close()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf, ref, err := pool.Alloc(32, 1)
			if err != nil {
				return // pool full, ok
			}
			copy(buf, []byte("test"))
			pool.Commit(ref)
			pool.Release(ref)
		}()
	}
	wg.Wait()
}

func TestLOABadMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.loa")
	if err := os.WriteFile(path, make([]byte, 256), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := OpenLOAPool(path)
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}
