//go:build !quic

package fragpigeon

import (
	"context"
	"errors"
)

// Keep the same public surface so other files compile without QUIC.

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
	// Minimal encoder to keep callers happy even without QUIC.
	n := 8 + len(ents)*16
	if cap(into) < n {
		into = make([]byte, n)
	}
	b := into[:n]
	b[0] = op
	b[1] = byte(len(ents))
	// We don’t need to actually serialize fully in stub builds.
	return b
}

// DialQUIC is unavailable in non-QUIC builds.
func DialQUIC(ctx context.Context, addr string, _ any) (any, error) {
	return nil, errors.New("quic support is disabled (build with -tags=quic)")
}
