//go:build rdma

package fragpigeon

// GossipEntry and Gossip types for rdma builds.
// Same as gossip_stub.go but compiled when rdma tag is active.

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
	if cap(into) < n {
		into = make([]byte, n)
	}
	b := into[:n]
	b[0] = op
	b[1] = byte(len(ents))
	return b
}
