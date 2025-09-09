//go:build quic

package fragpigeon

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/binary"
	"io"
	"log"
	"math/big"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/quic-go/quic-go"
)

const alpn = "fragmind-pigeon"

func clientTLS() *tls.Config {
	return &tls.Config{
		InsecureSkipVerify: true, // we accept self-signed from peers
		NextProtos:         []string{alpn},
		MinVersion:         tls.VersionTLS13,
	}
}

func serverTLS() *tls.Config {
	cert := mustSelfSignedCert()
	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{alpn},
		MinVersion:   tls.VersionTLS13,
	}
}

func mustSelfSignedCert() tls.Certificate {
	// 1) key
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic(err)
	}

	// 2) template
	serial, _ := rand.Int(rand.Reader, big.NewInt(1<<62))
	tmpl := x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: "fragmind-pigeon"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{"localhost"},
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
	}

	// 3) self-sign
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &key.PublicKey, key)
	if err != nil {
		panic(err)
	}

	cert := tls.Certificate{
		Certificate: [][]byte{der},
		PrivateKey:  key,
	}
	return cert
}

/*** minimal local interfaces ***/
type streamLike interface {
	Read([]byte) (int, error)
	Write([]byte) (int, error)
	Close() error
}
type connLike interface {
	OpenStreamSync(context.Context) (streamLike, error)
	AcceptStream(context.Context) (streamLike, error)
	CloseWithAppError(code uint64, msg string) error
}

/*** adapters that hold POINTERS to quic-go types ***/
type qStream struct{ inner *quic.Stream }

func (s qStream) Read(p []byte) (int, error)  { return s.inner.Read(p) }
func (s qStream) Write(p []byte) (int, error) { return s.inner.Write(p) }
func (s qStream) Close() error                { return s.inner.Close() }

type qConn struct{ inner *quic.Conn }

func (c qConn) OpenStreamSync(ctx context.Context) (streamLike, error) {
	st, err := c.inner.OpenStreamSync(ctx) // st is a pointer
	if err != nil {
		return nil, err
	}
	return qStream{inner: st}, nil // take POINTER so pointer methods compile
}
func (c qConn) AcceptStream(ctx context.Context) (streamLike, error) {
	st, err := c.inner.AcceptStream(ctx) // pointer
	if err != nil {
		return nil, err
	}
	return qStream{inner: st}, nil // wrap pointer
}
func (c qConn) CloseWithAppError(code uint64, msg string) error {
	return c.inner.CloseWithError(quic.ApplicationErrorCode(code), msg)
}

/*** peer manager using connLike/streamLike ***/
type quicPeers struct {
	p     *Pigeon
	mu    sync.RWMutex
	conns map[uint16]connLike // siteID -> conn
}

func (qp *quicPeers) Start(p *Pigeon) error {

	qp.p = p
	qp.conns = make(map[uint16]connLike)

	bind := p.bind
	if bind == "" {
		bind = ":4433"
	}
	ln, err := quic.ListenAddr(bind, serverTLS(), &quic.Config{
		KeepAlivePeriod: 5 * time.Second,
		MaxIdleTimeout:  30 * time.Second,
	})
	//log.Printf("pigeon (%d): quic listening on %s", p.siteID, bind)
	if err != nil {
		return err
	}
	go qp.acceptLoop(ln)

	for _, tgt := range p.peers {
		t := strings.TrimSpace(tgt)
		if t == "" {
			continue
		}
		go qp.dialLoop(t)
	}
	return nil
}

func (qp *quicPeers) acceptLoop(ln *quic.Listener) {
	for {
		c, err := ln.Accept(context.Background())
		if err != nil {
			log.Printf("quic accept: %v", err)
			continue
		}
		qp.attachConn(qConn{inner: c})
	}
}

func (qp *quicPeers) dialLoop(addr string) {
	for {

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		c, err := quic.DialAddr(ctx, addr, clientTLS(), &quic.Config{
			KeepAlivePeriod: 5 * time.Second,
			MaxIdleTimeout:  30 * time.Second,
		})
		cancel()
		if err != nil {
			log.Printf("quic dial %s: %v", addr, err)
			time.Sleep(2 * time.Second)
			continue
		}
		qp.attachConn(qConn{inner: c})
		return
	}
}

func (qp *quicPeers) attachConn(c connLike) {
	go qp.acceptReadLoop(c) // IMPORTANT: accept peer's streams

	if str, err := c.OpenStreamSync(context.Background()); err == nil {
		//n (%d): quic connected to peer", qp.p.siteID)
		qp.sendHello(str)
		qp.sendSnapshot(str)
		_ = str.Close()
	}

	qp.mu.Lock()
	qp.conns[0] = c // temporary until we learn siteID from HELLO
	qp.mu.Unlock()
}

func (qp *quicPeers) acceptReadLoop(c connLike) {
	for {
		str, err := c.AcceptStream(context.Background())
		//log.Printf("pigeon (%d): quic accepted stream", qp.p.siteID)
		if err != nil {
			log.Printf("control accept: %v", err)
			_ = c.CloseWithAppError(0, "")
			return
		}
		go qp.readControlStream(c, str)
	}
}

func (qp *quicPeers) readControlStream(c connLike, str streamLike) {
	buf := make([]byte, 64*1024)
	for {
		//log.Printf("pigeon (%d): quic reading control stream", qp.p.siteID)
		n, err := str.Read(buf)
		//log.Printf("control read: %d bytes", n)
		if err != nil {
			if err == io.EOF || err == net.ErrClosed {
				qp.handleControlFrames(c, buf[:n])
			} else {
				log.Printf("control error: %v", err)
			}
			_ = str.Close()
			return
		}
	}
}

func (qp *quicPeers) handleControlFrames(c connLike, b []byte) {
	log.Printf("control frame: %v %d bytes", b, len(b))
	i := 0
	for i+8 <= len(b) {
		op := b[i+0]
		cnt := int(b[i+1])
		site := binary.LittleEndian.Uint16(b[i+2:])
		epoch := binary.LittleEndian.Uint32(b[i+4:])
		i += 8

		ents := make([]GossipEntry, 0, cnt)
		for j := 0; j < cnt && i+16 <= len(b); j++ {
			ents = append(ents, GossipEntry{
				ConceptID: binary.LittleEndian.Uint64(b[i+0:]),
				Bits:      binary.LittleEndian.Uint16(b[i+8:]),
				SchemaID:  binary.LittleEndian.Uint16(b[i+10:]),
				Flags:     binary.LittleEndian.Uint16(b[i+12:]),
			})
			i += 16
		}

		switch op {
		case ctlHELLO:
			qp.mu.Lock()
			delete(qp.conns, 0)
			qp.conns[site] = c
			qp.mu.Unlock()
			if str, err := c.OpenStreamSync(context.Background()); err == nil {
				qp.sendSnapshot(str)
				_ = str.Close()
			}
		case ctlSNAPRS, ctlADD, ctlRENEW, ctlDEL:
			qp.p.ApplyRemote(op, site, ents)
		}
		_ = epoch
	}
}

func (qp *quicPeers) Broadcast(op byte, ents []GossipEntry) {
	//log.Printf("pigeon (%d): broadcast op=%d ents=%v conns=%v", qp.p.siteID, op, ents, qp.conns)
	frame := opFrame(op, qp.p.siteID, qp.bumpEpoch(), ents)
	qp.mu.RLock()
	defer qp.mu.RUnlock()
	for _, c := range qp.conns {
		if c == nil {
			continue
		}
		str, err := c.OpenStreamSync(context.Background())
		if err != nil {
			continue
		}
		_, err = str.Write(frame)
		//log.Printf("broadcast write: %v", err)
		_ = str.Close()
	}
}

func (qp *quicPeers) Close() error {
	qp.mu.Lock()
	defer qp.mu.Unlock()
	for _, c := range qp.conns {
		_ = c.CloseWithAppError(0, "")
	}
	return nil
}

func (qp *quicPeers) sendHello(str streamLike) {
	qp.p.epoch++
	var b [8]byte
	b[0], b[1] = ctlHELLO, 0
	binary.LittleEndian.PutUint16(b[2:], qp.p.siteID)
	binary.LittleEndian.PutUint32(b[4:], qp.p.epoch)
	_, _ = str.Write(b[:])
}

func (qp *quicPeers) sendSnapshot(str streamLike) {
	ents := qp.p.SnapshotCOIs()
	f := opFrame(ctlSNAPRS, qp.p.siteID, qp.bumpEpoch(), ents)
	_, _ = str.Write(f)
}

func (qp *quicPeers) bumpEpoch() uint32 { qp.p.epoch++; return qp.p.epoch }

func opFrame(op byte, site uint16, epoch uint32, ents []GossipEntry) []byte {
	n := 8 + len(ents)*16
	b := make([]byte, n)
	b[0] = op
	b[1] = byte(len(ents))
	binary.LittleEndian.PutUint16(b[2:], site)
	binary.LittleEndian.PutUint32(b[4:], epoch)
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
