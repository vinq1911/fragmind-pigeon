package fragpigeon

type noopPeers struct{}

func (n *noopPeers) Start(*Pigeon) error           { return nil }
func (n *noopPeers) Broadcast(byte, []GossipEntry) {}
func (n *noopPeers) Close() error                  { return nil }
