//go:build quic

package fragpigeon

import (
	"context"
	"crypto/tls"
	"encoding/binary"
	"time"

	"github.com/quic-go/quic-go"
)

const (
	ctlADD    = 10
	ctlDEL    = 11
	ctlRENEW  = 12
	ctlSNAPRQ = 13
	ctlSNAPRS = 14
	ctlHELLO  = 15
)

type GossipEntry struct {
	ConceptID uint64
	Bits      uint16
	SchemaID  uint16
	Flags     uint16
}

type Gossip struct {
	SiteID uint16
	Epoch  uint32
}

func (g *Gossip) Encode(op byte, ents []GossipEntry, into []byte) []byte {
	n := 8 + len(ents)*16
	if cap(into) < n+2 {
		into = make([]byte, n+2)
	}
	b := into[:n+2]
	b[0] = op
	b[1] = byte(len(ents))
	binary.LittleEndian.PutUint16(b[2:], g.SiteID)
	binary.LittleEndian.PutUint32(b[4:], g.Epoch)
	off := 8
	for _, e := range ents {
		binary.LittleEndian.PutUint64(b[off+0:], e.ConceptID)
		binary.LittleEndian.PutUint16(b[off+8:], e.Bits)
		binary.LittleEndian.PutUint16(b[off+10:], e.SchemaID)
		binary.LittleEndian.PutUint16(b[off+12:], e.Flags)
		off += 16
	}
	return b
}

func DialQUIC(ctx context.Context, addr string, tlsConf *tls.Config) (any, error) {
	qc := &quic.Config{
		KeepAlivePeriod: 5 * time.Second,
		MaxIdleTimeout:  30 * time.Second,
	}
	conn, err := quic.DialAddr(ctx, addr, tlsConf, qc)
	if err != nil {
		return nil, err
	}
	return conn, nil
}
