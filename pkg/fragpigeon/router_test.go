package fragpigeon

import (
	"sync"
	"testing"
)

func TestRouterAddLookup(t *testing.T) {
	r := NewRouter()

	r.Add(24, 0xABCD00, []uint16{1, 2}, []uint16{10})
	r.Add(24, 0xABCD00, []uint16{3}, nil) // append local
	r.Add(16, 0xFF00, nil, []uint16{20, 21})

	d := r.DestinationsBy(24, 0xABCD00)
	if d == nil {
		t.Fatal("expected destinations for 24/0xABCD00")
	}
	if len(d.LocalFragIDs) != 3 {
		t.Fatalf("expected 3 locals, got %d", len(d.LocalFragIDs))
	}
	if len(d.RemoteSites) != 1 || d.RemoteSites[0] != 10 {
		t.Fatalf("expected remote [10], got %v", d.RemoteSites)
	}

	d2 := r.DestinationsBy(16, 0xFF00)
	if d2 == nil || len(d2.RemoteSites) != 2 {
		t.Fatalf("expected 2 remotes for 16/0xFF00, got %v", d2)
	}
}

func TestRouterRemove(t *testing.T) {
	r := NewRouter()
	r.Add(24, 0xABCD00, []uint16{1}, []uint16{10})
	r.Remove(24, 0xABCD00)

	if d := r.DestinationsBy(24, 0xABCD00); d != nil {
		t.Fatalf("expected nil after remove, got %v", d)
	}
}

func TestRouterDestinationsHeader(t *testing.T) {
	r := NewRouter()
	r.Add(28, 0x12345670, nil, []uint16{5})

	h := Header{ConceptID: 0x12345670, ConceptBits: 28}
	d := r.Destinations(h)
	if d == nil || len(d.RemoteSites) != 1 {
		t.Fatalf("expected destination from header lookup, got %v", d)
	}
}

func TestRouterNilLookup(t *testing.T) {
	r := NewRouter()
	if d := r.DestinationsBy(24, 0xDEAD); d != nil {
		t.Fatal("expected nil for unknown concept")
	}
	h := Header{ConceptID: 0xDEAD, ConceptBits: 24}
	if d := r.Destinations(h); d != nil {
		t.Fatal("expected nil for unknown header")
	}
}

func TestRouterConcurrent(t *testing.T) {
	r := NewRouter()
	var wg sync.WaitGroup

	// Concurrent writers
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			cid := uint64(i) << 8
			r.Add(24, cid, []uint16{uint16(i)}, nil)
		}(i)
	}

	// Concurrent readers
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			cid := uint64(i) << 8
			_ = r.DestinationsBy(24, cid)
		}(i)
	}

	wg.Wait()
}

func TestAppendUniqueU16(t *testing.T) {
	result := appendUniqueU16([]uint16{1, 2, 3}, 2, 4, 3, 5)
	if len(result) != 5 {
		t.Fatalf("expected 5 elements, got %d: %v", len(result), result)
	}
}
