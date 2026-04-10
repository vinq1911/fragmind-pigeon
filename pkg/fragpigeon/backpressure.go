package fragpigeon

import (
	"errors"
	"time"
)

// Backpressure provides blocking/timeout write paths for ring and LOA.
// When the downstream consumer is slow, these functions stall the producer
// rather than dropping data or busy-spinning.

var (
	ErrRingFull = errors.New("ring full: backpressure timeout")
	ErrDropped  = errors.New("message dropped (FlagDropOK)")
)

// RingWriteWithBackoff attempts to write to the ring, backing off with
// exponential sleep up to the deadline. Returns nil on success.
// If FlagDropOK is set on the header and the ring is still full at deadline,
// the message is dropped (returns ErrDropped, not an error to abort on).
func RingWriteWithBackoff(r *Ring, h Header, payload []byte, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	backoff := 50 * time.Microsecond
	maxBackoff := 1 * time.Millisecond

	for {
		if r.TryWrite(h, payload) {
			return nil
		}
		if time.Now().After(deadline) {
			if h.Flags&FlagDropOK != 0 {
				return ErrDropped
			}
			return ErrRingFull
		}
		time.Sleep(backoff)
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// LOAAllocWithBackoff attempts to allocate from the LOA pool, retrying
// with backoff until the deadline. Useful when the pool is temporarily
// full because consumers haven't released slots yet.
func LOAAllocWithBackoff(pool *LOAPool, size uint32, ownerFragID uint32, timeout time.Duration) ([]byte, LOARef, error) {
	deadline := time.Now().Add(timeout)
	backoff := 50 * time.Microsecond
	maxBackoff := 1 * time.Millisecond

	for {
		buf, ref, err := pool.Alloc(size, ownerFragID)
		if err == nil {
			return buf, ref, nil
		}
		if !errors.Is(err, ErrLOAFull) {
			return nil, LOARef{}, err // real error (oversized, etc.)
		}
		if time.Now().After(deadline) {
			return nil, LOARef{}, ErrLOAFull
		}
		time.Sleep(backoff)
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// LOAAllocMultiWithBackoff is the multi-slot version of LOAAllocWithBackoff.
func LOAAllocMultiWithBackoff(pool *LOAPool, size uint32, ownerFragID uint32, timeout time.Duration) ([]byte, LOAMultiRef, error) {
	deadline := time.Now().Add(timeout)
	backoff := 50 * time.Microsecond
	maxBackoff := 1 * time.Millisecond

	for {
		buf, ref, err := pool.AllocMulti(size, ownerFragID)
		if err == nil {
			return buf, ref, nil
		}
		if time.Now().After(deadline) {
			return nil, LOAMultiRef{}, err
		}
		time.Sleep(backoff)
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

// WriteLOAWithBackoff is the full write path with backpressure on both
// the LOA pool (alloc) and the ring (write). Blocks up to timeout.
func WriteLOAWithBackoff(ring *Ring, pool *LOAPool, h Header, data []byte, ownerFragID uint32, timeout time.Duration) (LOARef, error) {
	buf, ref, err := LOAAllocWithBackoff(pool, uint32(len(data)), ownerFragID, timeout)
	if err != nil {
		return LOARef{}, err
	}
	copy(buf, data)
	pool.Commit(ref)

	var refBuf [LOARefSize]byte
	ref.Encode(refBuf[:])

	h.Len = LOARefSize
	h.Flags |= FlagLOAPtr

	if err := RingWriteWithBackoff(ring, h, refBuf[:], timeout); err != nil {
		pool.Release(ref)
		return LOARef{}, err
	}
	return ref, nil
}
