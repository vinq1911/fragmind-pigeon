//go:build !quic

package fragpigeon

func newPeerManager(mode string) PeerManager {
	// QUIC not compiled in; always return noop
	return &noopPeers{}
}
