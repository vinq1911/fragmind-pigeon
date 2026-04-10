package fragpigeon

import (
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	DefaultPigeonSock = "/tmp/pigeon.sock"
)

// COI subscription item
type COI struct {
	ConceptID uint64
	Bits      uint16
	SchemaID  uint16
	Flags     uint16
}

/******** Fragment <-> Pigeon UDS control ********/

type coiClient struct {
	mu     sync.Mutex
	c      net.Conn
	buf    []byte
	fragID uint16 // set after Attach; included in REGISTER/RENEW headers
}

func newCOIClient(sockPath string) (*coiClient, error) {
	if sockPath == "" {
		sockPath = DefaultPigeonSock
	}
	c, err := net.Dial("unix", sockPath)
	if err != nil {
		return nil, err
	}
	return &coiClient{c: c, buf: make([]byte, 0, 1024)}, nil
}

const (
	opREGISTER = 0
	opRENEW    = 1
	opUNREG    = 2
	opLIST     = 3
	opLOAINFO  = 4 // query LOA pool path
)

func (cc *coiClient) send(op byte, set []COI) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	n := 4 + len(set)*16
	if cap(cc.buf) < n {
		cc.buf = make([]byte, n)
	}
	b := cc.buf[:n]
	b[0] = op
	b[1] = byte(len(set))
	binary.LittleEndian.PutUint16(b[2:4], cc.fragID)
	off := 4
	for _, s := range set {
		binary.LittleEndian.PutUint64(b[off+0:], s.ConceptID)
		binary.LittleEndian.PutUint16(b[off+8:], s.Bits)
		binary.LittleEndian.PutUint16(b[off+10:], s.SchemaID)
		binary.LittleEndian.PutUint16(b[off+12:], s.Flags)
		off += 16
	}
	_, err := cc.c.Write(b)
	return err
}

func (cc *coiClient) Register(set []COI) error   { return cc.send(opREGISTER, set) }
func (cc *coiClient) Renew(set []COI) error      { return cc.send(opRENEW, set) }
func (cc *coiClient) Unregister(set []COI) error { return cc.send(opUNREG, set) }

/******** Local COI directory (read-only shm) ********/

type COITable struct {
	mem   []byte
	hdr   *coiHdr
	items []byte
}

type coiHdr struct {
	Seq       uint64
	Version   uint32
	Count     uint32
	UpdatedNs uint64
	_         [48]byte // pad to 64
}

const (
	coiEntSize   = 16
	coiOffCID    = 0
	coiOffBits   = 8
	coiOffSchema = 10
	coiOffFlags  = 12
)

func OpenLocalCOITable(path string) (*COITable, error) {
	if path == "" {
		if env := os.Getenv(COIShmEnv); env != "" {
			path = env
		}
	}
	if path == "" {
		path = DefaultCOIShmPath
	}

	fd, err := unix.Open(path, unix.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open COI shm %q: %w", path, err)
	}

	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("fstat %q: %w", path, err)
	}
	size := int(st.Size)
	if size < 64 {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("coi shm too small: %d", size)
	}
	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ, unix.MAP_SHARED)
	_ = unix.Close(fd)
	if err != nil {
		return nil, fmt.Errorf("mmap %q: %w", path, err)
	}

	h := (*coiHdr)(unsafe.Pointer(&mem[0]))
	return &COITable{mem: mem, hdr: h, items: mem[64:]}, nil
}

func (t *COITable) Close() error { return unix.Munmap(t.mem) }

func (t *COITable) Snapshot() (version uint32, updated time.Time, entries []COI) {
	for {
		s1 := atomic.LoadUint64(&t.hdr.Seq)
		if s1&1 != 0 {
			continue
		}
		cnt := atomic.LoadUint32(&t.hdr.Count)
		ver := atomic.LoadUint32(&t.hdr.Version)
		up := atomic.LoadUint64(&t.hdr.UpdatedNs)

		need := int(cnt) * coiEntSize
		if need < 0 {
			need = 0
		}
		if need > len(t.items) {
			need = len(t.items)
		}
		buf := make([]byte, need)
		copy(buf, t.items[:need])

		s2 := atomic.LoadUint64(&t.hdr.Seq)
		if s1 == s2 {
			es := make([]COI, 0, cnt)
			for i := 0; i+coiEntSize <= len(buf); i += coiEntSize {
				cid := binary.LittleEndian.Uint64(buf[i+coiOffCID:])
				bits := binary.LittleEndian.Uint16(buf[i+coiOffBits:])
				schema := binary.LittleEndian.Uint16(buf[i+coiOffSchema:])
				flags := binary.LittleEndian.Uint16(buf[i+coiOffFlags:])
				es = append(es, COI{ConceptID: cid, Bits: bits, SchemaID: schema, Flags: flags})
			}
			return ver, time.Unix(0, int64(up)), es
		}
	}
}

/******** Public fragment helper (auto-renew) ********/

type COIHandle struct {
	cc     *coiClient
	setMu  sync.RWMutex
	set    []COI
	stop   chan struct{}
	closed atomic.Bool
}

type COIOptions struct {
	SocketPath string
	RenewEvery time.Duration // default 1s
	FragID     uint16        // fragment ID (from Attach); 0 if not attached
}

func StartCOI(opts COIOptions, initial []COI) (*COIHandle, error) {
	if opts.RenewEvery <= 0 {
		opts.RenewEvery = time.Second
	}
	cc, err := newCOIClient(opts.SocketPath)
	if err != nil {
		return nil, err
	}
	cc.fragID = opts.FragID
	h := &COIHandle{cc: cc, stop: make(chan struct{}, 1)}
	if len(initial) > 0 {
		if err := cc.Register(initial); err != nil {
			return nil, err
		}
		h.set = append([]COI(nil), initial...)
	}
	go func() {
		t := time.NewTicker(opts.RenewEvery)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				h.setMu.RLock()
				s := append([]COI(nil), h.set...)
				h.setMu.RUnlock()
				if len(s) != 0 {
					_ = h.cc.Renew(s)
				}
			case <-h.stop:
				return
			}
		}
	}()
	return h, nil
}

func (h *COIHandle) Update(set []COI) error {
	h.setMu.Lock()
	h.set = append([]COI(nil), set...)
	h.setMu.Unlock()
	return h.cc.Renew(set)
}

// List asks the local pigeon for its current local COIs (best-effort).
func (cc *coiClient) List() ([]COI, error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	// send LIST (no payload)
	var hdr [4]byte
	hdr[0] = opLIST
	if _, err := cc.c.Write(hdr[:]); err != nil {
		return nil, err
	}
	// read reply: [u16 count][entries...each 16B]
	// keep this simple (small buffers); OK to block briefly
	buf := make([]byte, 64*1024)
	n, err := cc.c.Read(buf)
	if err != nil || n < 2 {
		return nil, errors.New("short read on LIST")
	}
	cnt := int(binary.LittleEndian.Uint16(buf[:2]))
	out := make([]COI, 0, cnt)
	off := 2
	for i := 0; i < cnt && off+16 <= n; i++ {
		out = append(out, COI{
			ConceptID: binary.LittleEndian.Uint64(buf[off+0:]),
			Bits:      binary.LittleEndian.Uint16(buf[off+8:]),
			SchemaID:  binary.LittleEndian.Uint16(buf[off+10:]),
			Flags:     binary.LittleEndian.Uint16(buf[off+12:]),
		})
		off += 16
	}
	return out, nil
}

// LOAInfo queries the local pigeon for its LOA pool shm path.
func (cc *coiClient) LOAInfo() (string, error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	var hdr [4]byte
	hdr[0] = opLOAINFO
	if _, err := cc.c.Write(hdr[:]); err != nil {
		return "", err
	}
	// Reply: [u16 pathLen][path bytes]
	buf := make([]byte, 1024)
	n, err := cc.c.Read(buf)
	if err != nil || n < 2 {
		return "", errors.New("short read on LOAINFO")
	}
	pathLen := int(binary.LittleEndian.Uint16(buf[:2]))
	if 2+pathLen > n {
		return "", errors.New("truncated LOAINFO reply")
	}
	return string(buf[2 : 2+pathLen]), nil
}

// DiscoverLOAPool connects to the local pigeon, queries the LOA pool path,
// and opens it. Returns the pool handle for alloc/deref/release.
func DiscoverLOAPool(socketPath string) (*LOAPool, error) {
	cc, err := newCOIClient(socketPath)
	if err != nil {
		return nil, fmt.Errorf("connect to pigeon: %w", err)
	}
	defer cc.c.Close()
	path, err := cc.LOAInfo()
	if err != nil {
		return nil, fmt.Errorf("query LOA info: %w", err)
	}
	if path == "" {
		return nil, errors.New("pigeon has no LOA pool")
	}
	return OpenLOAPool(path)
}

func (h *COIHandle) Close() error {
	if h.closed.Swap(true) {
		return nil
	}
	close(h.stop)
	h.setMu.RLock()
	s := append([]COI(nil), h.set...)
	h.setMu.RUnlock()
	_ = h.cc.Unregister(s)
	return h.cc.c.Close()
}
