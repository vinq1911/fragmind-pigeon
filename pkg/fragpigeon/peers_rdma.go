//go:build rdma && darwin

package fragpigeon

/*
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Apple ships libibverbs inside the dyld shared cache (librdma.dylib).
// No headers are provided, so we use dlopen/dlsym to access the verbs API.
// These wrappers match the standard libibverbs ABI.

// Opaque types
typedef void ibv_context;
typedef void ibv_pd;
typedef void ibv_cq;
typedef void ibv_qp;
typedef void ibv_mr;

// ibv_device_attr (partial — we only need a few fields)
struct rdma_device_attr {
    char     fw_ver[64];
    uint64_t node_guid;
    uint64_t sys_image_guid;
    uint64_t max_mr_size;
    uint64_t page_size_cap;
    uint32_t vendor_id;
    uint32_t vendor_part_id;
    uint32_t hw_ver;
    int      max_qp;
    int      max_qp_wr;
    // ... more fields we don't need
};

// ibv_port_attr (partial)
struct rdma_port_attr {
    uint32_t state;      // enum ibv_port_state
    uint32_t max_mtu;
    uint32_t active_mtu;
    int      gid_tbl_len;
    uint32_t port_cap_flags;
    uint32_t max_msg_sz;
    uint32_t bad_pkey_cntr;
    uint32_t qkey_viol_cntr;
    uint16_t pkey_tbl_len;
    uint16_t lid;
    uint16_t sm_lid;
    uint8_t  lmc;
    uint8_t  max_vl_num;
};

// GID (16 bytes)
typedef union {
    uint8_t  raw[16];
    struct { uint64_t subnet_prefix; uint64_t interface_id; } global;
} ibv_gid;

// Function pointers loaded at runtime
static void *librdma = NULL;

// Device list
typedef void** (*fn_get_device_list)(int *num);
typedef void   (*fn_free_device_list)(void **list);
typedef void*  (*fn_open_device)(void *dev);
typedef int    (*fn_close_device)(void *ctx);
typedef const char* (*fn_get_device_name)(void *dev);

// PD, CQ, MR
typedef void*  (*fn_alloc_pd)(void *ctx);
typedef int    (*fn_dealloc_pd)(void *pd);
typedef void*  (*fn_create_cq)(void *ctx, int cqe, void *cq_context, void *channel, int comp_vector);
typedef int    (*fn_destroy_cq)(void *cq);
typedef void*  (*fn_reg_mr)(void *pd, void *addr, size_t length, int access);
typedef int    (*fn_dereg_mr)(void *mr);

// MR fields (offset-based access since we don't have the struct definition)
// Standard libibverbs ibv_mr layout: { void *context; void *pd; void *addr; size_t length; uint32_t handle; uint32_t lkey; uint32_t rkey; }
static uint32_t mr_lkey(void *mr) {
    // lkey is at offset: context(8) + pd(8) + addr(8) + length(8) + handle(4) = 36
    return *(uint32_t*)((char*)mr + 36);
}
static uint32_t mr_rkey(void *mr) {
    return *(uint32_t*)((char*)mr + 40);
}
static void* mr_addr(void *mr) {
    return *(void**)((char*)mr + 16);
}
static size_t mr_length(void *mr) {
    return *(size_t*)((char*)mr + 24);
}

// Port and GID
typedef int (*fn_query_port)(void *ctx, uint8_t port, void *attr);
typedef int (*fn_query_gid)(void *ctx, uint8_t port, int index, ibv_gid *gid);

// QP
typedef void* (*fn_create_qp)(void *pd, void *qp_init_attr);
typedef int   (*fn_modify_qp)(void *qp, void *attr, int attr_mask);
typedef int   (*fn_destroy_qp)(void *qp);
typedef int   (*fn_post_send)(void *qp, void *wr, void **bad_wr);
typedef int   (*fn_post_recv)(void *qp, void *wr, void **bad_wr);
typedef int   (*fn_poll_cq)(void *cq, int num_entries, void *wc);

// Load the library
static int rdma_init() {
    if (librdma) return 0;
    librdma = dlopen("librdma.dylib", RTLD_LAZY);
    if (!librdma) return -1;
    return 0;
}

static void *rdma_sym(const char *name) {
    if (!librdma) return NULL;
    return dlsym(librdma, name);
}

// Convenience wrappers
static void** rdma_get_device_list(int *num) {
    fn_get_device_list fn = (fn_get_device_list)rdma_sym("ibv_get_device_list");
    return fn ? fn(num) : NULL;
}

static void rdma_free_device_list(void **list) {
    fn_free_device_list fn = (fn_free_device_list)rdma_sym("ibv_free_device_list");
    if (fn) fn(list);
}

static void* rdma_open_device(void *dev) {
    fn_open_device fn = (fn_open_device)rdma_sym("ibv_open_device");
    return fn ? fn(dev) : NULL;
}

static int rdma_close_device(void *ctx) {
    fn_close_device fn = (fn_close_device)rdma_sym("ibv_close_device");
    return fn ? fn(ctx) : -1;
}

static void* rdma_alloc_pd(void *ctx) {
    fn_alloc_pd fn = (fn_alloc_pd)rdma_sym("ibv_alloc_pd");
    return fn ? fn(ctx) : NULL;
}

static int rdma_dealloc_pd(void *pd) {
    fn_dealloc_pd fn = (fn_dealloc_pd)rdma_sym("ibv_dealloc_pd");
    return fn ? fn(pd) : -1;
}

static void* rdma_create_cq(void *ctx, int cqe) {
    fn_create_cq fn = (fn_create_cq)rdma_sym("ibv_create_cq");
    return fn ? fn(ctx, cqe, NULL, NULL, 0) : NULL;
}

static int rdma_destroy_cq(void *cq) {
    fn_destroy_cq fn = (fn_destroy_cq)rdma_sym("ibv_destroy_cq");
    return fn ? fn(cq) : -1;
}

// IBV_ACCESS flags
#define MY_IBV_ACCESS_LOCAL_WRITE  (1)
#define MY_IBV_ACCESS_REMOTE_WRITE (2)
#define MY_IBV_ACCESS_REMOTE_READ  (4)

static void* rdma_reg_mr(void *pd, void *addr, size_t length, int access) {
    fn_reg_mr fn = (fn_reg_mr)rdma_sym("ibv_reg_mr");
    return fn ? fn(pd, addr, length, access) : NULL;
}

static int rdma_dereg_mr(void *mr) {
    fn_dereg_mr fn = (fn_dereg_mr)rdma_sym("ibv_dereg_mr");
    return fn ? fn(mr) : -1;
}

static int rdma_query_port(void *ctx, uint8_t port, struct rdma_port_attr *attr) {
    fn_query_port fn = (fn_query_port)rdma_sym("ibv_query_port");
    return fn ? fn(ctx, port, attr) : -1;
}

static int rdma_query_gid(void *ctx, uint8_t port, int index, ibv_gid *gid) {
    fn_query_gid fn = (fn_query_gid)rdma_sym("ibv_query_gid");
    return fn ? fn(ctx, port, index, gid) : -1;
}
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"unsafe"
)

// rdmaPeers implements PeerManager using RDMA over Thunderbolt (libibverbs via dlopen).
type rdmaPeers struct {
	p   *Pigeon
	ctx unsafe.Pointer // ibv_context*
	pd  unsafe.Pointer // ibv_pd*
	cq  unsafe.Pointer // ibv_cq*

	loaMR unsafe.Pointer // ibv_mr* for LOA pool

	mu    sync.RWMutex
	conns map[uint16]*rdmaConn
}

type rdmaConn struct {
	siteID    uint16
	qp        unsafe.Pointer // ibv_qp*
	remoteLOA RDMAMemoryRegion
}

func newPeerManager(mode string) PeerManager {
	if mode == "rdma" {
		return &rdmaPeers{conns: make(map[uint16]*rdmaConn)}
	}
	return &noopPeers{}
}

func (rp *rdmaPeers) Start(p *Pigeon) error {
	rp.p = p

	// Initialize dlopen
	if C.rdma_init() != 0 {
		return fmt.Errorf("rdma: failed to load librdma.dylib")
	}

	// Open first RDMA device
	var numDevices C.int
	devList := C.rdma_get_device_list(&numDevices)
	if devList == nil || numDevices == 0 {
		return fmt.Errorf("rdma: no devices found (run 'rdma_ctl enable' in Recovery Mode)")
	}
	// devList[0] is the first device
	firstDev := *(*unsafe.Pointer)(unsafe.Pointer(devList))
	ctx := C.rdma_open_device(firstDev)
	C.rdma_free_device_list(devList)
	if ctx == nil {
		return fmt.Errorf("rdma: failed to open device")
	}
	rp.ctx = ctx

	// Allocate protection domain
	pd := C.rdma_alloc_pd(ctx)
	if pd == nil {
		return fmt.Errorf("rdma: ibv_alloc_pd failed")
	}
	rp.pd = pd

	// Create completion queue
	cq := C.rdma_create_cq(ctx, 256)
	if cq == nil {
		return fmt.Errorf("rdma: ibv_create_cq failed")
	}
	rp.cq = cq

	// Register LOA pool memory for RDMA
	if p.loaPool != nil && len(p.loaPool.mem) > 0 {
		access := C.MY_IBV_ACCESS_LOCAL_WRITE | C.MY_IBV_ACCESS_REMOTE_READ | C.MY_IBV_ACCESS_REMOTE_WRITE
		mr := C.rdma_reg_mr(pd, unsafe.Pointer(&p.loaPool.mem[0]), C.size_t(len(p.loaPool.mem)), C.int(access))
		if mr == nil {
			return fmt.Errorf("rdma: ibv_reg_mr failed for LOA pool (%d bytes)", len(p.loaPool.mem))
		}
		rp.loaMR = mr
		log.Printf("pigeon (%d): RDMA registered LOA pool (%d bytes, lkey=%d, rkey=%d)",
			p.siteID, len(p.loaPool.mem), C.mr_lkey(mr), C.mr_rkey(mr))
	}

	// Start listening for peer connections on the control channel (TCP)
	if p.bind != "" {
		go rp.listenControl(p.bind)
	}

	// Dial configured peers
	for _, peer := range p.peers {
		if peer != "" {
			go rp.dialPeer(peer)
		}
	}

	log.Printf("pigeon (%d): RDMA transport started (bind=%s peers=%v)", p.siteID, p.bind, p.peers)
	return nil
}

// LOA memory region info for handshake
func (rp *rdmaPeers) loaMemRegion() RDMAMemoryRegion {
	if rp.loaMR == nil {
		return RDMAMemoryRegion{}
	}
	return RDMAMemoryRegion{
		Addr:   uint64(uintptr(C.mr_addr(rp.loaMR))),
		Length: uint32(C.mr_length(rp.loaMR)),
		RKey:   uint32(C.mr_rkey(rp.loaMR)),
		LKey:   uint32(C.mr_lkey(rp.loaMR)),
	}
}

// --- Control channel (TCP) for handshake ---
// RDMA requires out-of-band exchange of QPN, GID, and memory region keys.

// Handshake message: [2B siteID][16B GID][4B QPN][4B rkey][8B addr][4B length]
const handshakeSize = 38

func (rp *rdmaPeers) listenControl(bind string) {
	ln, err := net.Listen("tcp", bind)
	if err != nil {
		log.Printf("rdma control listen: %v", err)
		return
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			continue
		}
		go rp.handlePeerHandshake(conn)
	}
}

func (rp *rdmaPeers) dialPeer(addr string) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		log.Printf("rdma dial %s: %v", addr, err)
		return
	}
	rp.handlePeerHandshake(conn)
}

func (rp *rdmaPeers) handlePeerHandshake(conn net.Conn) {
	defer conn.Close()

	// Get our GID
	var gid C.ibv_gid
	C.rdma_query_gid(rp.ctx, 1, 0, &gid)

	mr := rp.loaMemRegion()

	// Send our info
	var msg [handshakeSize]byte
	binary.LittleEndian.PutUint16(msg[0:], rp.p.siteID)
	copy(msg[2:18], C.GoBytes(unsafe.Pointer(&gid), 16))
	// QPN would go at msg[18:22] — 0 for now until QP creation
	binary.LittleEndian.PutUint32(msg[22:], mr.RKey)
	binary.LittleEndian.PutUint64(msg[26:], mr.Addr)
	binary.LittleEndian.PutUint32(msg[34:], mr.Length)
	conn.Write(msg[:])

	// Read peer info
	var peer [handshakeSize]byte
	if _, err := io.ReadFull(conn, peer[:]); err != nil {
		log.Printf("rdma handshake read: %v", err)
		return
	}

	peerSiteID := binary.LittleEndian.Uint16(peer[0:])
	peerRKey := binary.LittleEndian.Uint32(peer[22:])
	peerAddr := binary.LittleEndian.Uint64(peer[26:])
	peerLength := binary.LittleEndian.Uint32(peer[34:])

	log.Printf("pigeon (%d): RDMA handshake with site=%d rkey=%d addr=%x len=%d",
		rp.p.siteID, peerSiteID, peerRKey, peerAddr, peerLength)

	// Store peer connection info (QP creation would go here)
	rp.mu.Lock()
	rp.conns[peerSiteID] = &rdmaConn{
		siteID: peerSiteID,
		remoteLOA: RDMAMemoryRegion{
			Addr:   peerAddr,
			Length: peerLength,
			RKey:   peerRKey,
		},
	}
	rp.mu.Unlock()

	// TODO: Create QP, exchange QPN, transition QP to RTS
	// This requires: ibv_create_qp, ibv_modify_qp (INIT→RTR→RTS)
	// with the peer's GID and QPN. The QP setup is ~100 LOC of
	// struct initialization. Left as next step for hardware testing.
}

func (rp *rdmaPeers) Broadcast(op byte, ents []GossipEntry) {
	// Gossip uses TCP control channel, not RDMA data plane
	// TODO: implement gossip over the TCP connections
}

func (rp *rdmaPeers) Close() error {
	if rp.loaMR != nil {
		C.rdma_dereg_mr(rp.loaMR)
	}
	if rp.cq != nil {
		C.rdma_destroy_cq(rp.cq)
	}
	if rp.pd != nil {
		C.rdma_dealloc_pd(rp.pd)
	}
	if rp.ctx != nil {
		C.rdma_close_device(rp.ctx)
	}
	return nil
}

// forwardToRemotes for RDMA: send LOARef + rkey so remote can RDMA read.
func (p *Pigeon) forwardToRemotes(h Header, payload []byte) {
	rp, ok := p.pm.(*rdmaPeers)
	if !ok {
		return
	}
	ds := p.router.Destinations(h)
	if ds == nil || len(ds.RemoteSites) == 0 {
		return
	}

	if h.Flags&FlagLOAPtr != 0 && len(payload) >= LOARefSize {
		ref := DecodeLOARef(payload)
		mr := rp.loaMemRegion()
		slotOff := uint64(p.loaPool.hdr.DataBase) + uint64(ref.SlotID)*uint64(p.loaPool.slotSize)
		remoteAddr := mr.Addr + slotOff + uint64(ref.Offset)

		log.Printf("pigeon (%d): RDMA forward slot=%d raddr=%x rkey=%d to %v",
			p.siteID, ref.SlotID, remoteAddr, mr.RKey, ds.RemoteSites)
		// TODO: send RDMALOARef via TCP control channel to each remote site
		// Remote side would then ibv_post_send(IBV_WR_RDMA_READ) to pull the data
	}
}
