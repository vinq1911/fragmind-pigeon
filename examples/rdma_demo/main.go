//go:build rdma && darwin

// rdma_demo: RDMA over Thunderbolt LOA tensor transfer between two Macs.
//
// Prerequisites:
//   1. macOS 26.2+ on both Macs
//   2. Enable RDMA: boot to Recovery Mode, open Terminal, run: rdma_ctl enable
//   3. Reboot both Macs
//   4. Connect via Thunderbolt 4/5 cable
//   5. Verify: ibv_devices should show a device (not "disabled")
//
// Build:
//   CGO_ENABLED=1 go build -tags=rdma -o rdma_demo ./examples/rdma_demo
//
// Run on Mac A (server):
//   ./rdma_demo -mode server -bind :9999
//
// Run on Mac B (client):
//   ./rdma_demo -mode client -peer <macA-ip>:9999
//
// What happens:
//   1. Both sides open the RDMA device and register LOA pool memory
//   2. Exchange GID + memory region keys over TCP
//   3. Server writes a tensor into its LOA pool
//   4. Client sees the RDMA info and could issue an RDMA read
//      (full QP setup needed for actual RDMA read — see below)
package main

/*
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void ibv_context;
typedef union { uint8_t raw[16]; } ibv_gid;

static void *librdma = NULL;

typedef void** (*fn_get_device_list)(int *num);
typedef void   (*fn_free_device_list)(void **list);
typedef void*  (*fn_open_device)(void *dev);
typedef int    (*fn_close_device)(void *ctx);
typedef void*  (*fn_alloc_pd)(void *ctx);
typedef int    (*fn_dealloc_pd)(void *pd);
typedef void*  (*fn_reg_mr)(void *pd, void *addr, size_t length, int access);
typedef int    (*fn_dereg_mr)(void *mr);
typedef int    (*fn_query_gid)(void *ctx, uint8_t port, int index, ibv_gid *gid);

static int rdma_init() {
    if (librdma) return 0;
    librdma = dlopen("librdma.dylib", RTLD_LAZY);
    return librdma ? 0 : -1;
}

static void *rdma_sym(const char *name) {
    return librdma ? dlsym(librdma, name) : NULL;
}

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
static int rdma_query_gid(void *ctx, uint8_t port, int index, ibv_gid *gid) {
    fn_query_gid fn = (fn_query_gid)rdma_sym("ibv_query_gid");
    return fn ? fn(ctx, port, index, gid) : -1;
}

static uint32_t mr_lkey(void *mr) { return *(uint32_t*)((char*)mr + 36); }
static uint32_t mr_rkey(void *mr) { return *(uint32_t*)((char*)mr + 40); }
*/
import "C"

import (
	"encoding/binary"
	"flag"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"unsafe"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	mode := flag.String("mode", "server", "server or client")
	bind := flag.String("bind", ":9999", "TCP bind for control channel")
	peer := flag.String("peer", "", "peer address (client mode)")
	size := flag.Int("size", 64*1024, "tensor size in bytes")
	flag.Parse()

	fmt.Println("=== Fragmind RDMA over Thunderbolt Demo ===")
	fmt.Printf("Mode: %s | Tensor: %d bytes\n\n", *mode, *size)

	// Init RDMA
	if C.rdma_init() != 0 {
		log.Fatal("Failed to load librdma.dylib. Is macOS 26.2+ installed?")
	}
	fmt.Println("[OK] librdma.dylib loaded")

	// Open device
	var numDev C.int
	devList := C.rdma_get_device_list(&numDev)
	if devList == nil || numDev == 0 {
		log.Fatal("No RDMA devices. Enable: boot Recovery Mode → Terminal → rdma_ctl enable → reboot")
	}
	firstDev := *(*unsafe.Pointer)(unsafe.Pointer(devList))
	ctx := C.rdma_open_device(firstDev)
	C.rdma_free_device_list(devList)
	if ctx == nil {
		log.Fatal("Failed to open RDMA device")
	}
	defer C.rdma_close_device(ctx)
	fmt.Printf("[OK] RDMA device opened (%d device(s) found)\n", numDev)

	// PD
	pd := C.rdma_alloc_pd(ctx)
	if pd == nil {
		log.Fatal("ibv_alloc_pd failed")
	}
	defer C.rdma_dealloc_pd(pd)
	fmt.Println("[OK] Protection domain allocated")

	// GID
	var gid C.ibv_gid
	if C.rdma_query_gid(ctx, 1, 0, &gid) != 0 {
		fmt.Println("[WARN] Could not query GID (port may not be active)")
	} else {
		gidBytes := C.GoBytes(unsafe.Pointer(&gid), 16)
		fmt.Printf("[OK] GID: %x\n", gidBytes)
	}

	// Create LOA pool and register for RDMA
	dir, _ := os.MkdirTemp("", "rdma-demo-*")
	defer os.RemoveAll(dir)

	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path: filepath.Join(dir, "loa"), PoolID: 1,
		NumSlots: 16, SlotSize: uint32(*size),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	access := C.MY_IBV_ACCESS_LOCAL_WRITE | C.MY_IBV_ACCESS_REMOTE_READ | C.MY_IBV_ACCESS_REMOTE_WRITE
	mr := C.rdma_reg_mr(pd, unsafe.Pointer(&pool.Mem()[0]), C.size_t(len(pool.Mem())), C.int(access))
	if mr == nil {
		log.Fatal("ibv_reg_mr failed — RDMA may not be enabled")
	}
	defer C.rdma_dereg_mr(mr)
	fmt.Printf("[OK] LOA pool registered for RDMA: %d bytes, lkey=%d, rkey=%d\n",
		len(pool.Mem()), C.mr_lkey(mr), C.mr_rkey(mr))

	switch *mode {
	case "server":
		runServer(pool, mr, ctx, *bind, *size)
	case "client":
		if *peer == "" {
			log.Fatal("-peer required in client mode")
		}
		runClient(pool, mr, ctx, *peer, *size)
	}
}

func runServer(pool *fp.LOAPool, mr unsafe.Pointer, ctx unsafe.Pointer, bind string, size int) {
	// Write tensor to LOA
	buf, ref, _ := pool.Alloc(uint32(size), 1)
	for i := range buf[:size] {
		buf[i] = byte((i*31 + 17) & 0xFF)
	}
	pool.Commit(ref)
	crc := crc32.ChecksumIEEE(buf[:size])
	fmt.Printf("Tensor: slot=%d, %d bytes, crc=%08x\n", ref.SlotID, size, crc)

	// Compute RDMA address
	loaAddr := uint64(uintptr(unsafe.Pointer(&pool.Mem()[0])))
	dataOff := uint64(pool.DataBase()) + uint64(ref.SlotID)*uint64(pool.SlotSize())
	rdmaAddr := loaAddr + dataOff
	rkey := uint32(C.mr_rkey(mr))

	fmt.Printf("RDMA: addr=%x rkey=%d\n", rdmaAddr, rkey)

	// Listen for client
	ln, err := net.Listen("tcp", bind)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Waiting for client on %s...\n", bind)
	conn, err := ln.Accept()
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	// Send: [8B addr][4B rkey][4B length][4B crc]
	var msg [20]byte
	binary.LittleEndian.PutUint64(msg[0:], rdmaAddr)
	binary.LittleEndian.PutUint32(msg[8:], rkey)
	binary.LittleEndian.PutUint32(msg[12:], uint32(size))
	binary.LittleEndian.PutUint32(msg[16:], crc)
	conn.Write(msg[:])
	fmt.Println("Sent RDMA ref to client")

	// Wait for done signal
	var done [1]byte
	conn.Read(done[:])
	fmt.Println("Client done. RDMA demo complete.")
	pool.Release(ref)
}

func runClient(pool *fp.LOAPool, mr unsafe.Pointer, ctx unsafe.Pointer, peer string, size int) {
	conn, err := net.Dial("tcp", peer)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer conn.Close()

	var msg [20]byte
	if _, err := io.ReadFull(conn, msg[:]); err != nil {
		log.Fatal(err)
	}
	rdmaAddr := binary.LittleEndian.Uint64(msg[0:])
	rkey := binary.LittleEndian.Uint32(msg[8:])
	length := binary.LittleEndian.Uint32(msg[12:])
	expectedCRC := binary.LittleEndian.Uint32(msg[16:])

	fmt.Printf("Received: addr=%x rkey=%d length=%d crc=%08x\n", rdmaAddr, rkey, length, expectedCRC)
	fmt.Println()
	fmt.Println("RDMA device and memory region are set up.")
	fmt.Println("To complete the data transfer, QP connection setup is needed:")
	fmt.Println("  1. ibv_create_qp (RC type)")
	fmt.Println("  2. Exchange QPN between peers")
	fmt.Println("  3. ibv_modify_qp: RESET → INIT → RTR → RTS")
	fmt.Println("  4. ibv_post_send(IBV_WR_RDMA_READ) to pull tensor data")
	fmt.Println()
	fmt.Printf("The server's LOA slot at RDMA addr %x with rkey=%d is ready.\n", rdmaAddr, rkey)
	fmt.Printf("An RDMA READ of %d bytes would transfer the tensor at 80 Gb/s.\n", length)

	conn.Write([]byte{1})
}
