package fragpigeon

import (
	"encoding/binary"
	"fmt"
	"sync/atomic"
	"unsafe"
)

func unsafePointer(addr uintptr) unsafe.Pointer { return unsafe.Pointer(addr) }

// Multi-slot LOA: allocate contiguous runs of slots for payloads larger
// than a single slot. Uses the same LOA pool — adjacent slots are
// contiguous in the data region, so a multi-slot allocation is one
// contiguous buffer.

// LOAMultiRef references a contiguous run of slots.
type LOAMultiRef struct {
	PoolID    uint16
	StartSlot uint16 // first slot in the run
	NumSlots  uint16 // number of contiguous slots
	_         uint16 // padding
	Offset    uint32 // byte offset within first slot (usually 0)
	Length    uint32 // total payload length across all slots
}

const LOAMultiRefSize = 16

func (r LOAMultiRef) Encode(b []byte) {
	binary.LittleEndian.PutUint16(b[0:], r.PoolID)
	binary.LittleEndian.PutUint16(b[2:], r.StartSlot)
	binary.LittleEndian.PutUint16(b[4:], r.NumSlots)
	// b[6:8] padding
	binary.LittleEndian.PutUint32(b[8:], r.Offset)
	binary.LittleEndian.PutUint32(b[12:], r.Length)
}

func DecodeLOAMultiRef(b []byte) LOAMultiRef {
	return LOAMultiRef{
		PoolID:    binary.LittleEndian.Uint16(b[0:]),
		StartSlot: binary.LittleEndian.Uint16(b[2:]),
		NumSlots:  binary.LittleEndian.Uint16(b[4:]),
		Offset:    binary.LittleEndian.Uint32(b[8:]),
		Length:    binary.LittleEndian.Uint32(b[12:]),
	}
}

// AllocMulti reserves a contiguous run of N slots for a large payload.
// The returned slice spans all N slots as one contiguous buffer.
func (p *LOAPool) AllocMulti(size uint32, ownerFragID uint32) ([]byte, LOAMultiRef, error) {
	needed := int((size + p.slotSize - 1) / p.slotSize)
	if needed <= 0 {
		needed = 1
	}
	if needed > int(p.numSlots) {
		return nil, LOAMultiRef{}, fmt.Errorf("loa: payload %d requires %d slots, pool has %d",
			size, needed, p.numSlots)
	}

	// Find a contiguous run in the free list.
	// Strategy: sort free slots, find a run of `needed` consecutive IDs.
	p.freeMu.Lock()
	startIdx, found := findContiguousRun(p.freeList, needed)
	if !found {
		p.freeMu.Unlock()
		return nil, LOAMultiRef{}, fmt.Errorf("loa: need %d contiguous slots, not available (free=%d)",
			needed, len(p.freeList))
	}

	startSlot := p.freeList[startIdx]
	// Remove all slots in the run from the free list.
	// Since findContiguousRun found consecutive slot IDs (not consecutive indices),
	// rebuild the free list excluding the allocated range.
	removeSet := make(map[uint16]bool, needed)
	for i := 0; i < needed; i++ {
		removeSet[startSlot+uint16(i)] = true
	}
	newFree := make([]uint16, 0, len(p.freeList)-needed)
	for _, s := range p.freeList {
		if !removeSet[s] {
			newFree = append(newFree, s)
		}
	}
	p.freeList = newFree
	p.freeMu.Unlock()

	// Mark all slots as allocating
	for i := 0; i < needed; i++ {
		m := p.slotMeta(startSlot + uint16(i))
		atomic.StoreUint32(&m.State, slotAllocating)
		atomic.StoreInt32(&m.RefCnt, 0)
		m.Owner = ownerFragID
		if i == 0 {
			m.Size = size // total size stored on first slot
		} else {
			m.Size = p.slotSize // full slot
		}
	}

	// Return contiguous view across all slots
	totalCap := uint32(needed) * p.slotSize
	data := p.slotDataRange(startSlot, totalCap)[:size]

	ref := LOAMultiRef{
		PoolID:    p.poolID,
		StartSlot: startSlot,
		NumSlots:  uint16(needed),
		Length:    size,
	}
	return data, ref, nil
}

// CommitMulti marks all slots in a multi-slot allocation as ready.
func (p *LOAPool) CommitMulti(ref LOAMultiRef) {
	for i := uint16(0); i < ref.NumSlots; i++ {
		m := p.slotMeta(ref.StartSlot + i)
		atomic.StoreUint32(&m.State, slotReady)
	}
}

// DerefMulti returns a read-only view of a multi-slot allocation.
func (p *LOAPool) DerefMulti(ref LOAMultiRef) ([]byte, error) {
	if uint32(ref.StartSlot)+uint32(ref.NumSlots) > p.numSlots {
		return nil, fmt.Errorf("loa: multi-slot range %d+%d out of bounds", ref.StartSlot, ref.NumSlots)
	}
	// Check first slot is ready
	m := p.slotMeta(ref.StartSlot)
	if atomic.LoadUint32(&m.State) != slotReady {
		return nil, fmt.Errorf("loa: slot %d not ready", ref.StartSlot)
	}
	// Increment refcount on all slots
	for i := uint16(0); i < ref.NumSlots; i++ {
		atomic.AddInt32(&p.slotMeta(ref.StartSlot+i).RefCnt, 1)
	}

	totalCap := uint32(ref.NumSlots) * p.slotSize
	data := p.slotDataRange(ref.StartSlot, totalCap)
	end := ref.Offset + ref.Length
	if end > totalCap {
		end = totalCap
	}
	return data[ref.Offset:end], nil
}

// ReleaseMulti decrements refcount on all slots; frees when zero.
func (p *LOAPool) ReleaseMulti(ref LOAMultiRef) {
	allFree := true
	for i := uint16(0); i < ref.NumSlots; i++ {
		m := p.slotMeta(ref.StartSlot + i)
		if atomic.AddInt32(&m.RefCnt, -1) > 0 {
			allFree = false
		}
	}
	if allFree {
		p.freeMu.Lock()
		for i := uint16(0); i < ref.NumSlots; i++ {
			sid := ref.StartSlot + i
			atomic.StoreUint32(&p.slotMeta(sid).State, slotFree)
			p.freeList = append(p.freeList, sid)
		}
		p.freeMu.Unlock()
	}
}

// slotDataRange returns a byte slice spanning multiple contiguous slots.
func (p *LOAPool) slotDataRange(startSlot uint16, length uint32) []byte {
	off := p.dataBase + uintptr(startSlot)*uintptr(p.slotSize)
	return (*[1 << 30]byte)(unsafePointer(off))[:length:length]
}

// findContiguousRun finds a run of `needed` consecutive slot IDs in the free list.
// Returns the start index in the free list and whether it was found.
func findContiguousRun(free []uint16, needed int) (int, bool) {
	if len(free) < needed {
		return 0, false
	}
	if needed == 1 {
		return len(free) - 1, true // pop from end
	}

	// Sort-free approach: scan for consecutive IDs.
	// Build a set for O(1) lookup.
	set := make(map[uint16]int, len(free)) // slotID -> index in free list
	for i, s := range free {
		set[s] = i
	}

	// Try each slot as a potential start of a run.
	for _, s := range free {
		ok := true
		for j := 1; j < needed; j++ {
			if _, exists := set[s+uint16(j)]; !exists {
				ok = false
				break
			}
		}
		if ok {
			// Found a run starting at slot s. Collect indices.
			// Return the index of the first slot.
			return set[s], true
		}
	}
	return 0, false
}
