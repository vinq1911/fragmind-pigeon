//go:build !quic && !rdma

package fragpigeon

func newPeerManager(mode string) PeerManager {
	// QUIC not compiled in; always return noop
	return &noopPeers{}
}
