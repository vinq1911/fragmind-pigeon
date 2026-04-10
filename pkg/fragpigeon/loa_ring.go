package fragpigeon

import "time"

// LOA + Ring integration helpers.
//
// When FlagLOAPtr is set on a ring message, the payload contains an
// LOARef (12 bytes) pointing to the actual data in the LOA pool.
// This enables zero-copy transfer of large objects through the ring.

// WriteLOA allocates an LOA slot, copies data into it, commits it,
// and writes a ring message with FlagLOAPtr referencing the slot.
// Returns false if the ring or LOA pool is full.
func WriteLOA(ring *Ring, pool *LOAPool, h Header, data []byte, ownerFragID uint32) (LOARef, bool) {
	buf, ref, err := pool.Alloc(uint32(len(data)), ownerFragID)
	if err != nil {
		return LOARef{}, false
	}
	copy(buf, data)
	pool.Commit(ref)

	// Build the ring payload: just the LOARef
	var refBuf [LOARefSize]byte
	ref.Encode(refBuf[:])

	h.Len = LOARefSize
	h.Flags |= FlagLOAPtr
	if !ring.TryWrite(h, refBuf[:]) {
		// Ring full — release the LOA slot
		pool.Release(ref)
		return LOARef{}, false
	}
	return ref, true
}

// WriteLOAZeroCopy allocates an LOA slot and returns the writable slice
// so the caller can fill it directly (true zero-copy for producers that
// generate data in-place). Call CommitLOA after writing data.
func WriteLOAZeroCopy(pool *LOAPool, size uint32, ownerFragID uint32) ([]byte, LOARef, error) {
	return pool.Alloc(size, ownerFragID)
}

// CommitLOA commits the LOA slot and writes the ring message.
func CommitLOA(ring *Ring, pool *LOAPool, ref LOARef, h Header) bool {
	pool.Commit(ref)

	var refBuf [LOARefSize]byte
	ref.Encode(refBuf[:])

	h.Len = LOARefSize
	h.Flags |= FlagLOAPtr
	if !ring.TryWrite(h, refBuf[:]) {
		pool.Release(ref)
		return false
	}
	return true
}

// IsLOA returns true if the message uses LOA (large object attach).
func (m *Msg) IsLOA() bool {
	return m.Header.Flags&FlagLOAPtr != 0
}

// LOARef decodes the LOA reference from the message payload.
// Only valid when IsLOA() returns true.
func (m *Msg) LOARef() LOARef {
	if len(m.Payload) < LOARefSize {
		return LOARef{}
	}
	return DecodeLOARef(m.Payload)
}

// ReadLOA is a convenience that reads a ring message, and if it's an LOA
// message, dereferences the LOA pool to return the actual data.
// The caller must call pool.Release(ref) when done with the data.
// For non-LOA messages, ref is zero-value and data is the inline payload.
func ReadLOA(ring *Ring, pool *LOAPool, block bool) (Msg, []byte, LOARef, error) {
	msg, err := ring.Read(block)
	if err != nil {
		return msg, nil, LOARef{}, err
	}
	if msg.Header.Flags&FlagLOAPtr == 0 {
		// Inline payload, no LOA
		return msg, msg.Payload, LOARef{}, nil
	}
	ref := DecodeLOARef(msg.Payload)
	data, err := pool.Deref(ref)
	if err != nil {
		return msg, nil, ref, err
	}
	return msg, data, ref, nil
}

// NewLOAHeader builds a Header suitable for LOA messages.
func NewLOAHeader(kind Kind, conceptID uint64, conceptBits uint16, schemaID uint16, srcID uint32) Header {
	return Header{
		Kind:        kind,
		Flags:       FlagLOAPtr,
		TSns:        uint64(time.Now().UnixNano()),
		ConceptID:   conceptID,
		ConceptBits: conceptBits,
		SchemaID:    schemaID,
		SrcID:       srcID,
	}
}
