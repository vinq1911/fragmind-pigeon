//go:build quic

package fragpigeon

import (
	"context"
	"encoding/binary"
	"log"
	"unsafe"
)

// ForwardRemote forwards a message to remote pigeon sites.
// For LOA messages (FlagLOAPtr), it dereferences the local LOA pool to get
// the actual data, then sends header + full payload over QUIC — since remote
// hosts can't access local shared memory.
func (p *Pigeon) ForwardRemote(h Header, payload []byte) {
	ds := p.router.Destinations(h)
	if ds == nil || len(ds.RemoteSites) == 0 {
		return
	}

	// If this is an LOA pointer, resolve it to actual data for the wire.
	wirePayload := payload
	var loaRef LOARef
	if h.Flags&FlagLOAPtr != 0 && p.loaPool != nil && len(payload) >= LOARefSize {
		loaRef = DecodeLOARef(payload)
		data, err := p.loaPool.Deref(loaRef)
		if err != nil {
			log.Printf("forward: LOA deref failed: %v", err)
			return
		}
		// Send actual data, clear LOAPtr flag (remote receives inline)
		wirePayload = data
		h.Flags &^= FlagLOAPtr
		h.Len = uint32(len(wirePayload))
		// Release after we've sent to all peers (deferred below)
		defer p.loaPool.Release(loaRef)
	}

	frame := encodeFrame(h, wirePayload)
	qp, _ := p.pm.(*quicPeers)

	qp.mu.RLock()
	defer qp.mu.RUnlock()
	for _, site := range ds.RemoteSites {
		raw := qp.conns[site]
		c, ok := raw.(connLike)
		if !ok || c == nil {
			continue
		}
		str, err := c.OpenStreamSync(context.Background())
		if err != nil {
			continue
		}
		if _, err := str.Write(frame); err != nil {
			log.Printf("forward write err: %v", err)
		}
		_ = str.Close()
	}
}

// forwardToRemotes wraps ForwardRemote for the attach routing path.
func (p *Pigeon) forwardToRemotes(h Header, payload []byte) {
	p.ForwardRemote(h, payload)
}

func encodeFrame(h Header, payload []byte) []byte {
	n := 2 + HdrSize + len(payload)
	b := make([]byte, n)
	binary.LittleEndian.PutUint16(b[:2], uint16(HdrSize+len(payload)))
	h.Pack(unsafe.Pointer(&b[2]))
	copy(b[2+HdrSize:], payload)
	return b
}
