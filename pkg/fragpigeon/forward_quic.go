//go:build quic

package fragpigeon

import (
	"context"
	"encoding/binary"
	"log"
	"unsafe"
)

func (p *Pigeon) ForwardRemote(h Header, payload []byte) {

	ds := p.router.Destinations(h)

	if ds == nil || len(ds.RemoteSites) == 0 {
		return
	}

	frame := encodeFrame(h, payload)
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

func encodeFrame(h Header, payload []byte) []byte {
	n := 2 + HdrSize + len(payload)
	b := make([]byte, n)
	binary.LittleEndian.PutUint16(b[:2], uint16(HdrSize+len(payload)))
	h.Pack(unsafe.Pointer(&b[2]))
	copy(b[2+HdrSize:], payload)
	return b
}
