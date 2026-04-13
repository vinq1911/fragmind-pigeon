//go:build !quic && !rdma

package fragpigeon

// forwardToRemotes is a no-op when QUIC is not compiled in.
func (p *Pigeon) forwardToRemotes(h Header, payload []byte) {}
