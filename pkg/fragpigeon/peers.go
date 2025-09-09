package fragpigeon

type PeerManager interface {
	Start(p *Pigeon) error
	Broadcast(op byte, ents []GossipEntry) // non-blocking best-effort
	Close() error
}
