package fragpigeon

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	LeaseTTL       = 5 * time.Second
	BlackoutGrace  = 15 * time.Second
	GossipDebounce = 75 * time.Millisecond
)

type LocalCOI struct {
	COI
	OwnerFragID uint32
	LeaseUntil  time.Time
	GraceUntil  time.Time
}

type Pigeon struct {
	siteID uint16
	router *Router
	local  map[uint64]LocalCOI
	mu     sync.RWMutex

	// COI shm
	shmFD   int
	shmMem  []byte
	udsPath string

	// LOA pool
	loaPool *LOAPool
	loaPath string

	// Fragment registry (attached fragments with ring pairs)
	frags *fragmentRegistry

	// --- peers ---
	mode       string      // "quic" or "none"
	bind       string      // e.g., ":4433"
	peers      []string    // dial targets
	pm         PeerManager // interface (quic or stub)
	epoch      atomic.Uint32
	debouncing bool
	debounceMu sync.Mutex
	// gossip queues (protected by mu)
	pendingAdds []GossipEntry
	pendingDels []GossipEntry
}

func NewPigeon(siteID uint16, udsPath string) *Pigeon {
	if udsPath == "" {
		udsPath = DefaultPigeonSock
	}
	return &Pigeon{
		siteID:  siteID,
		router:  NewRouter(),
		local:   make(map[uint64]LocalCOI),
		udsPath: udsPath,
		frags:   newFragmentRegistry(),
	}
}

func NewPigeonWithNet(siteID uint16, udsPath, mode, bind string, peers []string) *Pigeon {
	if udsPath == "" {
		udsPath = DefaultPigeonSock
	}
	p := &Pigeon{
		siteID:  siteID,
		router:  NewRouter(),
		local:   make(map[uint64]LocalCOI),
		udsPath: udsPath,
		frags:   newFragmentRegistry(),
		mode:    mode,
		bind:    bind,
		peers:   peers,
	}
	p.pm = newPeerManager(mode)
	return p
}

func key(bits uint16, cid uint64) uint64 {
	return (uint64(bits) << 56) | (cid & ^(uint64(0) >> (bits)))
}

func (p *Pigeon) Run() error {
	// 0) Start peer manager (if any)
	if p.pm != nil {
		log.Printf("pigeon (%d): peer manager %T (mode=%s bind=%s peers=%v)\n", p.siteID, p.pm, p.mode, p.bind, p.peers)
		if err := p.pm.Start(p); err != nil {
			return err
		}
	} else {
		log.Printf("pigeon (%d): peer manager <nil> (mode=%s) — NO inter-pigeon links\n", p.siteID, p.mode)
	}

	// 1) COI shm table
	if err := p.initCOIShm(); err != nil {
		return err
	}
	p.publishCOIShm() // initialize version/updated once so watchers see non-zero header

	// 2) LOA pool
	if err := p.initLOAPool(); err != nil {
		return err
	}

	// 3) UDS server (REGISTER/RENEW/UNREGISTER)
	if err := p.serveUDS(); err != nil {
		return err
	}

	// 4) leases & gossip debounce
	go p.housekeep()

	select {}
}

func (p *Pigeon) housekeep() {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for now := range t.C {
		p.mu.Lock()
		changed := false
		for k, v := range p.local {
			if now.After(v.GraceUntil) {
				delete(p.local, k)
				// enqueue gossip DEL...
				changed = true
				continue
			}
		}
		p.mu.Unlock()
		if changed {
			p.publishCOIShm()
		}
	}
}

func (p *Pigeon) initCOIShm() error {
	path := os.Getenv(COIShmEnv)
	if path == "" {
		path = DefaultCOIShmPath
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mk dir for coi shm: %w", err)
	}

	const size = 64 + 64*1024
	fd, err := unix.Open(path, unix.O_CREAT|unix.O_RDWR|unix.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("open coi shm: %w", err)
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		_ = unix.Close(fd)
		return fmt.Errorf("ftruncate coi shm: %w", err)
	}
	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = unix.Close(fd)
		return fmt.Errorf("mmap coi shm: %w", err)
	}
	// Do NOT unlink; keep path visible so other processes can open it.
	p.shmFD, p.shmMem = fd, mem

	log.Printf("pigeon (%d): COI shm at %s", p.siteID, path)
	p.publishCOIShm()
	return nil
}

// called by PeerManager when a full snapshot is requested
func (p *Pigeon) SnapshotCOIs() []GossipEntry {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]GossipEntry, 0, len(p.local))
	for _, v := range p.local {
		out = append(out, GossipEntry{
			ConceptID: v.ConceptID, Bits: v.Bits, SchemaID: v.SchemaID, Flags: v.Flags,
		})
	}
	return out
}

// called when we learn remote COIs
func (p *Pigeon) ApplyRemote(op byte, siteID uint16, ents []GossipEntry) {
	// Update router buckets for remote site; no COI shm (local only)
	// log.Printf("pigeon (%d): router learned site=%d %d ent(s) via op=%d", p.siteID, siteID, len(ents), op)
	p.mu.Lock()
	for _, e := range ents {
		switch op {
		case ctlADD, ctlRENEW:
			p.router.Add(e.Bits, e.ConceptID, nil, []uint16{siteID})
		case ctlDEL:
			p.router.Remove(e.Bits, e.ConceptID)
		}
	}
	p.mu.Unlock()
}

type coiHdrW struct {
	Seq       uint64
	Version   uint32
	Count     uint32
	UpdatedNs uint64
	_         [48]byte
}

func (p *Pigeon) publishCOIShm() {
	h := (*coiHdrW)(unsafe.Pointer(&p.shmMem[0]))
	// enter write
	h.Seq++
	entries := p.encodeLocalEntries(p.shmMem[64:])
	// write header
	h.Count = uint32(entries)
	h.Version++
	h.UpdatedNs = uint64(time.Now().UnixNano())
	// leave write
	h.Seq++
}

func (p *Pigeon) encodeLocalEntries(buf []byte) int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	i := 0
	for _, v := range p.local {
		if i+16 > len(buf) {
			break
		}
		putU64(buf[i+0:], v.ConceptID)
		putU16(buf[i+8:], v.Bits)
		putU16(buf[i+10:], v.SchemaID)
		putU16(buf[i+12:], v.Flags)
		i += 16
	}
	return i / 16
}

// enqueue helpers (call these after local REGISTER/UNREGISTER)
func (p *Pigeon) enqueueGossipAdds(ents []GossipEntry) {
	p.mu.Lock()
	p.pendingAdds = append(p.pendingAdds, ents...)
	p.mu.Unlock()
	p.armDebounce()
}
func (p *Pigeon) enqueueGossipDels(ents []GossipEntry) {
	p.mu.Lock()
	p.pendingDels = append(p.pendingDels, ents...)
	p.mu.Unlock()
	p.armDebounce()
}

func (p *Pigeon) armDebounce() {
	p.debounceMu.Lock()
	if p.debouncing {
		p.debounceMu.Unlock()
		return
	}
	p.debouncing = true
	p.debounceMu.Unlock()

	go func() {
		time.Sleep(GossipDebounce)
		p.flushGossip()
	}()
}

func (p *Pigeon) flushGossip() {
	// snapshot and clear pending queues
	//log.Printf("pigeon (%d): flushing gossip", p.siteID)
	p.mu.Lock()
	adds := append([]GossipEntry(nil), p.pendingAdds...)
	dels := append([]GossipEntry(nil), p.pendingDels...)
	p.pendingAdds = p.pendingAdds[:0]
	p.pendingDels = p.pendingDels[:0]
	p.mu.Unlock()

	// broadcast (best-effort)
	if p.pm != nil {
		if len(adds) > 0 {
			p.pm.Broadcast(ctlADD, adds)
		}
		if len(dels) > 0 {
			p.pm.Broadcast(ctlDEL, dels)
		}
	}

	p.debounceMu.Lock()
	p.debouncing = false
	p.debounceMu.Unlock()
}

func putU64(b []byte, v uint64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}
func putU16(b []byte, v uint16) { b[0] = byte(v); b[1] = byte(v >> 8) }

func (p *Pigeon) serveUDS() error {
	_ = unix.Unlink(p.udsPath)
	l, err := net.Listen("unix", p.udsPath)
	if err != nil {
		return err
	}
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				continue
			}
			go p.handleFragControl(c)
		}
	}()
	return nil
}

func (p *Pigeon) handleFragControl(c net.Conn) {
	defer c.Close()
	hdr := make([]byte, 4)
	for {
		// Read the fixed 4-byte header reliably
		if _, err := io.ReadFull(c, hdr); err != nil {
			return
		}
		op := hdr[0]
		cnt := int(hdr[1])

		// Read the entry payload (cnt * 16 bytes) if any
		payloadLen := cnt * 16
		var buf []byte
		if payloadLen > 0 {
			buf = make([]byte, payloadLen)
			if _, err := io.ReadFull(c, buf); err != nil {
				return
			}
		}

		switch op {
		case opREGISTER, opRENEW:
			now := time.Now()
			ents := make([]GossipEntry, 0, cnt)
			// Extract fragID from header byte 2-3 (if fragment provided it)
			fragID := leU16(hdr[2:])
			p.mu.Lock()
			for i := 0; i < cnt; i++ {
				off := i * 16
				cid := leU64(buf[off:])
				bits := leU16(buf[off+8:])
				schema := leU16(buf[off+10:])
				flags := leU16(buf[off+12:])
				k := key(bits, cid)
				p.local[k] = LocalCOI{
					COI:        COI{ConceptID: cid, Bits: bits, SchemaID: schema, Flags: flags},
					OwnerFragID: uint32(fragID),
					LeaseUntil:  now.Add(LeaseTTL),
					GraceUntil:  now.Add(LeaseTTL + BlackoutGrace),
				}
				ents = append(ents, GossipEntry{ConceptID: cid, Bits: bits, SchemaID: schema, Flags: flags})
				// Register in router for local delivery
				if fragID > 0 {
					p.router.Add(bits, cid, []uint16{fragID}, nil)
				}
			}
			p.mu.Unlock()
			p.enqueueGossipAdds(ents)
			p.publishCOIShm()

		case opLIST:
			p.mu.RLock()
			tmp := make([]LocalCOI, 0, len(p.local))
			for _, v := range p.local {
				tmp = append(tmp, v)
			}
			p.mu.RUnlock()

			resp := make([]byte, 2+len(tmp)*16)
			binary.LittleEndian.PutUint16(resp[:2], uint16(len(tmp)))
			off := 2
			for _, v := range tmp {
				putU64(resp[off+0:], v.ConceptID)
				putU16(resp[off+8:], v.Bits)
				putU16(resp[off+10:], v.SchemaID)
				putU16(resp[off+12:], v.Flags)
				off += 16
			}
			_, _ = c.Write(resp)

		case opUNREG:
			ents := make([]GossipEntry, 0, cnt)
			p.mu.Lock()
			for i := 0; i < cnt; i++ {
				off := i * 16
				cid := leU64(buf[off:])
				bits := leU16(buf[off+8:])
				schema := leU16(buf[off+10:])
				flags := leU16(buf[off+12:])
				k := key(bits, cid)
				delete(p.local, k)
				ents = append(ents, GossipEntry{ConceptID: cid, Bits: bits, SchemaID: schema, Flags: flags})
			}
			p.mu.Unlock()
			p.enqueueGossipDels(ents)
			p.publishCOIShm()

		case opLOAINFO:
			// Reply with LOA pool path: [u16 len][path bytes]
			pathBytes := []byte(p.loaPath)
			resp := make([]byte, 2+len(pathBytes))
			binary.LittleEndian.PutUint16(resp[:2], uint16(len(pathBytes)))
			copy(resp[2:], pathBytes)
			_, _ = c.Write(resp)

		case opATTACH:
			p.handleAttach(c)
			return // attachment takes over the connection lifecycle
		}
	}
}

const LOAShmEnv = "FM_LOA_SHM_PATH"

func (p *Pigeon) initLOAPool() error {
	path := os.Getenv(LOAShmEnv)
	if path == "" {
		path = filepath.Join(defaultLOADir(), fmt.Sprintf("fragmind.loa.%d", p.siteID))
	}

	numSlots := uint32(DefaultLOASlots)
	slotSize := uint32(DefaultLOASlotSize)
	if s := os.Getenv("FM_LOA_SLOTS"); s != "" {
		if v, err := fmt.Sscanf(s, "%d", &numSlots); err != nil || v != 1 {
			numSlots = DefaultLOASlots
		}
	}
	if s := os.Getenv("FM_LOA_SLOT_SIZE"); s != "" {
		if v, err := fmt.Sscanf(s, "%d", &slotSize); err != nil || v != 1 {
			slotSize = DefaultLOASlotSize
		}
	}

	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     path,
		PoolID:   p.siteID,
		NumSlots: numSlots,
		SlotSize: slotSize,
	})
	if err != nil {
		return fmt.Errorf("init LOA pool: %w", err)
	}
	p.loaPool = pool
	p.loaPath = path
	log.Printf("pigeon (%d): LOA pool at %s (slots=%d slotSize=%d)", p.siteID, path, numSlots, slotSize)
	return nil
}

// LOAPool returns the pigeon's LOA pool (for in-process use / tests).
func (p *Pigeon) LOAPool() *LOAPool { return p.loaPool }

// LOAPath returns the shm path of the LOA pool.
func (p *Pigeon) LOAPath() string { return p.loaPath }

func leU64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}
func leU16(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }
