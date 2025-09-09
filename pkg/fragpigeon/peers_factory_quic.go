//go:build quic

package fragpigeon

func newPeerManager(mode string) PeerManager {
	if mode == "quic" {
		return &quicPeers{} // only defined in quic builds
	}
	return &noopPeers{}
}
