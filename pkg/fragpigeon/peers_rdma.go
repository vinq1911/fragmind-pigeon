//go:build rdma

package fragpigeon

/*
#cgo LDFLAGS: -libverbs

#include <infiniband/verbs.h>
#include <stdlib.h>
#include <string.h>

// Helper to get first RDMA device
static struct ibv_context *open_first_device() {
	struct ibv_device **dev_list = ibv_get_device_list(NULL);
	if (!dev_list || !dev_list[0]) return NULL;
	struct ibv_context *ctx = ibv_open_device(dev_list[0]);
	ibv_free_device_list(dev_list);
	return ctx;
}
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"log"
	"sync"
	"unsafe"
)

// rdmaPeers implements PeerManager using RDMA over Thunderbolt (libibverbs).
type rdmaPeers struct {
	p   *Pigeon
	ctx *C.struct_ibv_context
	pd  *C.struct_ibv_pd
	cq  *C.struct_ibv_cq

	// LOA memory region registered for RDMA access
	loaMR *C.struct_ibv_mr

	mu    sync.RWMutex
	conns map[uint16]*rdmaConn // siteID -> connection
}

// rdmaConn represents an RDMA connection to a peer pigeon.
type rdmaConn struct {
	siteID uint16
	qp     *C.struct_ibv_qp
	// Remote LOA memory region info (received during handshake)
	remoteLOA RDMAMemoryRegion
}

func newPeerManager(mode string) PeerManager {
	if mode == "rdma" {
		return &rdmaPeers{conns: make(map[uint16]*rdmaConn)}
	}
	if mode == "quic" {
		return &quicPeers{}
	}
	return &noopPeers{}
}

func (rp *rdmaPeers) Start(p *Pigeon) error {
	rp.p = p

	// Open RDMA device (Thunderbolt RDMA on macOS)
	ctx := C.open_first_device()
	if ctx == nil {
		return fmt.Errorf("rdma: no RDMA devices found (is rdma_ctl enabled?)")
	}
	rp.ctx = ctx

	// Allocate protection domain
	pd := C.ibv_alloc_pd(ctx)
	if pd == nil {
		return fmt.Errorf("rdma: ibv_alloc_pd failed")
	}
	rp.pd = pd

	// Create completion queue
	cq := C.ibv_create_cq(ctx, 128, nil, nil, 0)
	if cq == nil {
		return fmt.Errorf("rdma: ibv_create_cq failed")
	}
	rp.cq = cq

	// Register LOA pool memory for RDMA access
	if p.loaPool != nil && len(p.loaPool.mem) > 0 {
		mr := C.ibv_reg_mr(pd,
			unsafe.Pointer(&p.loaPool.mem[0]),
			C.size_t(len(p.loaPool.mem)),
			C.IBV_ACCESS_LOCAL_WRITE|C.IBV_ACCESS_REMOTE_READ|C.IBV_ACCESS_REMOTE_WRITE)
		if mr == nil {
			return fmt.Errorf("rdma: ibv_reg_mr failed for LOA pool")
		}
		rp.loaMR = mr
		log.Printf("pigeon (%d): RDMA registered LOA pool (%d bytes, lkey=%d, rkey=%d)",
			p.siteID, len(p.loaPool.mem), mr.lkey, mr.rkey)
	}

	log.Printf("pigeon (%d): RDMA transport started", p.siteID)
	return nil
}

// LOAMemoryRegion returns the RDMA memory region info for the LOA pool.
// This is exchanged with peers during handshake so they can RDMA read our LOA data.
func (rp *rdmaPeers) LOAMemoryRegion() RDMAMemoryRegion {
	if rp.loaMR == nil {
		return RDMAMemoryRegion{}
	}
	return RDMAMemoryRegion{
		Addr:   uint64(uintptr(rp.loaMR.addr)),
		Length: uint32(rp.loaMR.length),
		RKey:   uint32(rp.loaMR.rkey),
		LKey:   uint32(rp.loaMR.lkey),
	}
}

// RDMARead performs an RDMA read from a remote site's LOA pool into local memory.
// This is the zero-copy cross-host path: no serialization, no kernel networking stack.
func (rp *rdmaPeers) RDMARead(siteID uint16, remoteAddr uint64, rkey uint32,
	localBuf unsafe.Pointer, length uint32) error {

	rp.mu.RLock()
	conn := rp.conns[siteID]
	rp.mu.RUnlock()
	if conn == nil {
		return fmt.Errorf("rdma: no connection to site %d", siteID)
	}
	if rp.loaMR == nil {
		return fmt.Errorf("rdma: no local memory region registered")
	}

	// Post RDMA read work request
	var sge C.struct_ibv_sge
	sge.addr = C.uint64_t(uintptr(localBuf))
	sge.length = C.uint32_t(length)
	sge.lkey = rp.loaMR.lkey

	var wr C.struct_ibv_send_wr
	wr.wr_id = 0
	wr.sg_list = &sge
	wr.num_sge = 1
	wr.opcode = C.IBV_WR_RDMA_READ
	wr.send_flags = C.IBV_SEND_SIGNALED
	// Set remote address and key using the rdma union field
	*(*C.uint64_t)(unsafe.Pointer(&wr.wr[0])) = C.uint64_t(remoteAddr) // remote_addr
	*(*C.uint32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(&wr.wr[0])) + 8)) = C.uint32_t(rkey) // rkey

	var badWr *C.struct_ibv_send_wr
	ret := C.ibv_post_send(conn.qp, &wr, &badWr)
	if ret != 0 {
		return fmt.Errorf("rdma: ibv_post_send failed: %d", ret)
	}

	// Poll for completion
	var wc C.struct_ibv_wc
	for {
		n := C.ibv_poll_cq(rp.cq, 1, &wc)
		if n > 0 {
			if wc.status != C.IBV_WC_SUCCESS {
				return fmt.Errorf("rdma: WC error: %d", wc.status)
			}
			return nil
		}
	}
}

// Broadcast sends gossip to all peers (falls back to QUIC-style control messages
// over a side channel, since RDMA is for data plane only).
func (rp *rdmaPeers) Broadcast(op byte, ents []GossipEntry) {
	// RDMA is data-plane only. Gossip uses a separate TCP/QUIC control channel.
	// For now, this is a no-op — gossip would need a separate control transport.
	// In production, you'd pair RDMA data plane with a QUIC control plane.
}

func (rp *rdmaPeers) Close() error {
	if rp.loaMR != nil {
		C.ibv_dereg_mr(rp.loaMR)
	}
	if rp.cq != nil {
		C.ibv_destroy_cq(rp.cq)
	}
	if rp.pd != nil {
		C.ibv_dealloc_pd(rp.pd)
	}
	if rp.ctx != nil {
		C.ibv_close_device(rp.ctx)
	}
	return nil
}

// forwardToRemotes for RDMA mode: instead of serializing data,
// send an RDMALOARef so the remote can RDMA read directly.
func (p *Pigeon) forwardToRemotes(h Header, payload []byte) {
	rp, ok := p.pm.(*rdmaPeers)
	if !ok {
		return
	}

	ds := p.router.Destinations(h)
	if ds == nil || len(ds.RemoteSites) == 0 {
		return
	}

	// If LOA message, convert LOARef to RDMALOARef
	if h.Flags&FlagLOAPtr != 0 && len(payload) >= LOARefSize {
		ref := DecodeLOARef(payload)
		mr := rp.LOAMemoryRegion()

		// Calculate the remote address for this LOA slot
		slotDataOffset := uint64(p.loaPool.hdr.DataBase) + uint64(ref.SlotID)*uint64(p.loaPool.slotSize)
		remoteAddr := mr.Addr + slotDataOffset + uint64(ref.Offset)

		// Build RDMA LOA ref
		rdmaRef := RDMALOARef{
			LOARef:     ref,
			RemoteAddr: remoteAddr,
			RKey:       mr.RKey,
		}

		// Encode and send as control message
		var buf [RDMALOARefSize]byte
		ref.Encode(buf[:LOARefSize])
		binary.LittleEndian.PutUint64(buf[LOARefSize:], rdmaRef.RemoteAddr)
		binary.LittleEndian.PutUint32(buf[LOARefSize+8:], rdmaRef.RKey)

		h.Len = RDMALOARefSize
		// Send over the control channel to each remote site
		// (In production, this would go via TCP/QUIC control plane)
		_ = buf
		log.Printf("pigeon (%d): RDMA forward ref slot=%d raddr=%x rkey=%d to %v",
			p.siteID, ref.SlotID, remoteAddr, mr.RKey, ds.RemoteSites)
	}
}
