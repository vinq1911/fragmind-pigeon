package fragpigeon

import (
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sys/unix"
)

// Fragment attachment: pigeon creates ring pairs, passes FDs to fragments,
// and routes messages between them based on COI subscriptions.

const (
	opATTACH = 5 // fragment requests ring pair from pigeon
	// Reply: SCM_RIGHTS with [inFD, outFD] + [4B fragID]
)

// AttachedFragment represents a connected fragment with its ring pair.
type AttachedFragment struct {
	FragID  uint32
	InRing  *Ring // pigeon writes TO fragment (fragment reads)
	OutRing *Ring // fragment writes TO pigeon (pigeon reads for routing)
	InFD    int   // underlying fd for InRing
	OutFD   int   // underlying fd for OutRing
	COIs    []COI // fragment's subscribed COIs
	done    chan struct{}
	stopped chan struct{}
}

// fragmentRegistry tracks all attached fragments.
type fragmentRegistry struct {
	mu        sync.RWMutex
	fragments map[uint32]*AttachedFragment
	nextID    atomic.Uint32
}

func newFragmentRegistry() *fragmentRegistry {
	r := &fragmentRegistry{
		fragments: make(map[uint32]*AttachedFragment),
	}
	r.nextID.Store(1)
	return r
}

func (r *fragmentRegistry) add(f *AttachedFragment) {
	r.mu.Lock()
	r.fragments[f.FragID] = f
	r.mu.Unlock()
}

func (r *fragmentRegistry) remove(fragID uint32) {
	r.mu.Lock()
	delete(r.fragments, fragID)
	r.mu.Unlock()
}

func (r *fragmentRegistry) get(fragID uint32) *AttachedFragment {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.fragments[fragID]
}

func (r *fragmentRegistry) all() []*AttachedFragment {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*AttachedFragment, 0, len(r.fragments))
	for _, f := range r.fragments {
		out = append(out, f)
	}
	return out
}

// createRingPair creates two shared-memory rings:
//   - inRing:  pigeon → fragment (fragment reads from this)
//   - outRing: fragment → pigeon (pigeon reads from this for routing)
func createRingPair(dir string, fragID uint32, capSlots, slotSize int) (inRing *Ring, inFD int, outRing *Ring, outFD int, err error) {
	inFD, err = createRingFD(dir, fmt.Sprintf("frag-%d-in", fragID), capSlots, slotSize)
	if err != nil {
		return nil, -1, nil, -1, fmt.Errorf("create in-ring: %w", err)
	}
	inRing, err = OpenRingFromFD(inFD)
	if err != nil {
		unix.Close(inFD)
		return nil, -1, nil, -1, fmt.Errorf("open in-ring: %w", err)
	}

	outFD, err = createRingFD(dir, fmt.Sprintf("frag-%d-out", fragID), capSlots, slotSize)
	if err != nil {
		inRing.Close()
		unix.Close(inFD)
		return nil, -1, nil, -1, fmt.Errorf("create out-ring: %w", err)
	}
	outRing, err = OpenRingFromFD(outFD)
	if err != nil {
		inRing.Close()
		unix.Close(inFD)
		unix.Close(outFD)
		return nil, -1, nil, -1, fmt.Errorf("open out-ring: %w", err)
	}

	return inRing, inFD, outRing, outFD, nil
}

func createRingFD(dir, name string, capSlots, slotSize int) (int, error) {
	size := 64 + capSlots*slotSize
	path := filepath.Join(dir, name+".shm")

	fd, err := unix.Open(path, unix.O_CREAT|unix.O_EXCL|unix.O_RDWR, 0600)
	if err != nil {
		fd, err = unix.Open(path, unix.O_RDWR, 0600)
		if err != nil {
			return -1, err
		}
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		unix.Close(fd)
		return -1, err
	}

	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		unix.Close(fd)
		return -1, err
	}
	binary.LittleEndian.PutUint64(mem[0:], uint64(capSlots))
	binary.LittleEndian.PutUint64(mem[8:], 0)                // ProdIdx
	binary.LittleEndian.PutUint64(mem[16:], 0)               // ConsIdx
	binary.LittleEndian.PutUint32(mem[24:], uint32(slotSize))
	binary.LittleEndian.PutUint64(mem[32:], ^uint64(0))      // ProdEvtFD=-1
	binary.LittleEndian.PutUint64(mem[40:], ^uint64(0))      // ConsEvtFD=-1
	_ = unix.Munmap(mem)
	_ = os.Remove(path) // unlink, keep fd open

	return fd, nil
}

// sendFDs sends file descriptors over a Unix domain socket using SCM_RIGHTS.
// Uses Go's native UnixConn.WriteMsgUnix to avoid File() dup issues.
func sendFDs(conn net.Conn, fds []int, data []byte) error {
	uc, ok := conn.(*net.UnixConn)
	if !ok {
		return fmt.Errorf("not a unix conn")
	}
	rights := unix.UnixRights(fds...)
	_, _, err := uc.WriteMsgUnix(data, rights, nil)
	return err
}

// recvFDs receives file descriptors from a Unix domain socket.
// Uses Go's native UnixConn.ReadMsgUnix to avoid File() dup issues.
func recvFDs(conn net.Conn, numFDs int) ([]int, []byte, error) {
	uc, ok := conn.(*net.UnixConn)
	if !ok {
		return nil, nil, fmt.Errorf("not a unix conn")
	}

	buf := make([]byte, 64)
	oob := make([]byte, unix.CmsgLen(numFDs*4))
	n, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
	if err != nil {
		return nil, nil, err
	}

	scms, err := unix.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		return nil, nil, err
	}
	var fds []int
	for _, scm := range scms {
		gotFDs, err := unix.ParseUnixRights(&scm)
		if err != nil {
			continue
		}
		fds = append(fds, gotFDs...)
	}
	return fds, buf[:n], nil
}

// --- Pigeon-side attachment ---

// handleAttach creates a ring pair for a new fragment and sends FDs back.
func (p *Pigeon) handleAttach(c net.Conn) {
	fragID := p.frags.nextID.Add(1) - 1

	dir := os.TempDir()
	capSlots := 1024
	slotSize := 1024 // 64B header + up to 960B payload (or LOA pointers)

	inRing, inFD, outRing, outFD, err := createRingPair(dir, fragID, capSlots, slotSize)
	if err != nil {
		log.Printf("pigeon (%d): attach failed: %v", p.siteID, err)
		return
	}

	frag := &AttachedFragment{
		FragID:  fragID,
		InRing:  inRing,
		OutRing: outRing,
		InFD:    inFD,
		OutFD:   outFD,
	}
	p.frags.add(frag)

	// Send FDs to fragment: [inFD, outFD] + [4B fragID]
	var resp [4]byte
	binary.LittleEndian.PutUint32(resp[:], fragID)
	if err := sendFDs(c, []int{inFD, outFD}, resp[:]); err != nil {
		log.Printf("pigeon (%d): send FDs failed: %v", p.siteID, err)
		p.frags.remove(fragID)
		return
	}

	log.Printf("pigeon (%d): fragment %d attached (in=%d out=%d)", p.siteID, fragID, inFD, outFD)

	// Start routing goroutine: read fragment's outbound ring, route messages
	go p.routeFragment(frag)
}

// routeFragment reads messages from a fragment's outbound ring and
// delivers them to other fragments that subscribe to matching COIs.
// StopFragment signals the routing goroutine to exit and waits for it.
func (f *AttachedFragment) Stop() {
	if f.done != nil {
		select {
		case <-f.done:
		default:
			close(f.done)
		}
	}
	if f.stopped != nil {
		<-f.stopped
	}
}

func (p *Pigeon) routeFragment(frag *AttachedFragment) {
	if frag.done == nil {
		frag.done = make(chan struct{})
	}
	if frag.stopped == nil {
		frag.stopped = make(chan struct{})
	}
	defer close(frag.stopped)
	for {
		select {
		case <-frag.done:
			return
		default:
		}
		msg, err := frag.OutRing.ReadWithin(100 * time.Millisecond)
		if err != nil {
			continue
		}

		// Look up destinations for this message's concept
		dests := p.router.Destinations(msg.Header)
		if dests == nil {
			continue // no subscribers
		}

		// Deliver to local fragments
		for _, fragID := range dests.LocalFragIDs {
			target := p.frags.get(uint32(fragID))
			if target == nil || target.FragID == frag.FragID {
				continue // don't loop back to sender
			}

			// If LOA message, just forward the pointer (same shm pool)
			if !target.InRing.TryWrite(msg.Header, msg.Payload) {
				// Target's inbound ring is full — drop if allowed
				if msg.Header.Flags&FlagDropOK == 0 {
					log.Printf("pigeon: routing drop fragID=%d (ring full)", target.FragID)
				}
			}
		}

		// Forward to remote sites (if QUIC) — handled by forwardToRemotes
		if len(dests.RemoteSites) > 0 {
			p.forwardToRemotes(msg.Header, msg.Payload)
		}
	}
}

// --- Fragment-side attachment ---

// AttachResult is returned to a fragment after attaching to the pigeon.
type AttachResult struct {
	FragID  uint32
	InRing  *Ring // read from this (messages routed to you)
	OutRing *Ring // write to this (pigeon routes your messages)
}

// Attach connects to the local pigeon and receives a ring pair for messaging.
// The pigeon creates the rings and sends the FDs via SCM_RIGHTS.
func Attach(socketPath string) (*AttachResult, error) {
	if socketPath == "" {
		socketPath = DefaultPigeonSock
	}

	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("connect to pigeon: %w", err)
	}

	// Send ATTACH request
	var hdr [4]byte
	hdr[0] = opATTACH
	if _, err := conn.Write(hdr[:]); err != nil {
		conn.Close()
		return nil, fmt.Errorf("send attach: %w", err)
	}

	// Receive FDs + fragID
	fds, data, err := recvFDs(conn, 2)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("recv FDs: %w", err)
	}
	conn.Close()

	if len(fds) < 2 {
		return nil, fmt.Errorf("expected 2 FDs, got %d", len(fds))
	}
	if len(data) < 4 {
		return nil, fmt.Errorf("expected 4B fragID, got %d", len(data))
	}

	fragID := binary.LittleEndian.Uint32(data[:4])

	// Fragment's perspective: inFD is what pigeon writes to (fragment reads),
	// outFD is what fragment writes to (pigeon reads)
	inRing, err := OpenRingFromFD(fds[0])
	if err != nil {
		return nil, fmt.Errorf("open in-ring: %w", err)
	}
	outRing, err := OpenRingFromFD(fds[1])
	if err != nil {
		inRing.Close()
		return nil, fmt.Errorf("open out-ring: %w", err)
	}

	return &AttachResult{
		FragID:  fragID,
		InRing:  inRing,
		OutRing: outRing,
	}, nil
}

func (a *AttachResult) Close() error {
	a.InRing.Close()
	a.OutRing.Close()
	return nil
}
