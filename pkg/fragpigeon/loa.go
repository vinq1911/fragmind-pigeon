package fragpigeon

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/unix"
)

// LOA (Large Object Attach) provides a shared-memory arena for zero-copy
// transfer of large payloads (tensors, weight shards, KV-cache slices)
// between fragments on the same host.
//
// Layout:
//   [LOAHeader: 64 bytes]
//   [SlotMeta[0]: 32 bytes] [SlotMeta[1]: 32 bytes] ...
//   [... padding to align dataBase ...]
//   [Slot data region ...]
//
// A writer calls Alloc(size) -> LOARef, writes into the returned slice,
// then sends a ring message with FlagLOAPtr and the LOARef encoded in
// the payload. The reader calls Deref(ref) to get a read-only view,
// and Release(ref) when done.

const (
	loaHeaderSize   = 64
	loaSlotMetaSize = 32
	loaAlignment    = 4096 // page-align the data region

	// Default pool: 256 MB arena, 4096 slots of 64 KB each.
	DefaultLOASlots    = 4096
	DefaultLOASlotSize = 64 * 1024 // 64 KB per slot
)

// LOARef is a handle to an allocated LOA slot, sent over the ring.
type LOARef struct {
	PoolID uint16 // identifies which LOA pool (for multi-pool setups)
	SlotID uint16 // slot index within the pool
	Offset uint32 // byte offset within the slot (usually 0)
	Length uint32 // payload length in bytes
}

// LOARefSize is the wire size of an LOARef (12 bytes).
const LOARefSize = 12

func (r LOARef) Encode(b []byte) {
	binary.LittleEndian.PutUint16(b[0:], r.PoolID)
	binary.LittleEndian.PutUint16(b[2:], r.SlotID)
	binary.LittleEndian.PutUint32(b[4:], r.Offset)
	binary.LittleEndian.PutUint32(b[8:], r.Length)
}

func DecodeLOARef(b []byte) LOARef {
	return LOARef{
		PoolID: binary.LittleEndian.Uint16(b[0:]),
		SlotID: binary.LittleEndian.Uint16(b[2:]),
		Offset: binary.LittleEndian.Uint32(b[4:]),
		Length: binary.LittleEndian.Uint32(b[8:]),
	}
}

// LOAHeader lives at offset 0 of the shm region.
type LOAHeader struct {
	Magic     uint64 // 0x4C4F41504F4F4C31 ("LOAPOOL1")
	Version   uint32
	NumSlots  uint32
	SlotSize  uint32
	PoolID    uint16
	_         [2]byte
	DataBase  uint32 // byte offset where slot data begins
	_         [32]byte
}

const loaMagic = 0x4C4F41504F4F4C31

// slotMeta tracks per-slot state in the shm header region.
type slotMeta struct {
	State  uint32 // 0=free, 1=allocating, 2=ready, 3=released
	RefCnt int32  // reference count (readers holding this slot)
	Owner  uint32 // fragment ID of the allocator
	Size   uint32 // actual payload size (<= slotSize)
	_      [16]byte
}

const (
	slotFree       = 0
	slotAllocating = 1
	slotReady      = 2
)

// LOAPool is a writer/reader handle to an LOA shared-memory arena.
type LOAPool struct {
	poolID   uint16
	mem      []byte
	hdr      *LOAHeader
	metaBase uintptr
	dataBase uintptr
	numSlots uint32
	slotSize uint32
	fd       int

	// Writer-side free list (only the allocator process uses this)
	freeMu   sync.Mutex
	freeList []uint16
}

type LOAPoolOptions struct {
	Path     string // shm path (default: /dev/shm/fragmind.loa.<poolID> or /tmp/ on macOS)
	PoolID   uint16
	NumSlots uint32
	SlotSize uint32 // bytes per slot
}

// CreateLOAPool creates a new LOA shared-memory pool. Typically called by the pigeon daemon.
func CreateLOAPool(opts LOAPoolOptions) (*LOAPool, error) {
	if opts.NumSlots == 0 {
		opts.NumSlots = DefaultLOASlots
	}
	if opts.SlotSize == 0 {
		opts.SlotSize = DefaultLOASlotSize
	}
	path := opts.Path
	if path == "" {
		path = defaultLOAPath(opts.PoolID)
	}

	metaRegion := uint32(loaSlotMetaSize) * opts.NumSlots
	dataOffset := alignUp(uint32(loaHeaderSize)+metaRegion, loaAlignment)
	totalSize := int(dataOffset) + int(opts.NumSlots)*int(opts.SlotSize)

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("mkdir for loa: %w", err)
	}

	fd, err := unix.Open(path, unix.O_CREAT|unix.O_RDWR|unix.O_TRUNC, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open loa shm: %w", err)
	}
	if err := unix.Ftruncate(fd, int64(totalSize)); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("ftruncate loa: %w", err)
	}
	mem, err := unix.Mmap(fd, 0, totalSize, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("mmap loa: %w", err)
	}

	// Initialize header
	hdr := (*LOAHeader)(unsafe.Pointer(&mem[0]))
	hdr.Magic = loaMagic
	hdr.Version = 1
	hdr.NumSlots = opts.NumSlots
	hdr.SlotSize = opts.SlotSize
	hdr.PoolID = opts.PoolID
	hdr.DataBase = dataOffset

	// Build free list
	free := make([]uint16, opts.NumSlots)
	for i := range free {
		free[i] = uint16(i)
	}

	pool := &LOAPool{
		poolID:   opts.PoolID,
		mem:      mem,
		hdr:      hdr,
		metaBase: uintptr(unsafe.Pointer(&mem[loaHeaderSize])),
		dataBase: uintptr(unsafe.Pointer(&mem[dataOffset])),
		numSlots: opts.NumSlots,
		slotSize: opts.SlotSize,
		fd:       fd,
		freeList: free,
	}
	return pool, nil
}

// OpenLOAPool opens an existing LOA pool for reading (and optional writing).
func OpenLOAPool(path string) (*LOAPool, error) {
	if path == "" {
		return nil, errors.New("loa pool path required")
	}
	fd, err := unix.Open(path, unix.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("open loa: %w", err)
	}
	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("fstat loa: %w", err)
	}
	size := int(st.Size)
	if size < loaHeaderSize {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("loa shm too small: %d", size)
	}
	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("mmap loa: %w", err)
	}

	// Validate magic via byte read before unsafe pointer cast
	magic := binary.LittleEndian.Uint64(mem[0:8])
	if magic != loaMagic {
		_ = unix.Munmap(mem)
		_ = unix.Close(fd)
		return nil, fmt.Errorf("bad loa magic: %x", magic)
	}
	hdr := (*LOAHeader)(unsafe.Pointer(&mem[0]))
	if int(hdr.DataBase) >= size {
		_ = unix.Munmap(mem)
		_ = unix.Close(fd)
		return nil, fmt.Errorf("loa: dataBase %d out of range (size %d)", hdr.DataBase, size)
	}

	pool := &LOAPool{
		poolID:   hdr.PoolID,
		mem:      mem,
		hdr:      hdr,
		metaBase: uintptr(unsafe.Pointer(&mem[loaHeaderSize])),
		dataBase: uintptr(unsafe.Pointer(&mem[hdr.DataBase])),
		numSlots: hdr.NumSlots,
		slotSize: hdr.SlotSize,
		fd:       fd,
	}
	// Rebuild free list from slot metadata (enables alloc from opened pools)
	free := make([]uint16, 0, hdr.NumSlots)
	for i := uint32(0); i < hdr.NumSlots; i++ {
		m := pool.slotMeta(uint16(i))
		if atomic.LoadUint32(&m.State) == slotFree {
			free = append(free, uint16(i))
		}
	}
	pool.freeList = free
	return pool, nil
}

// Mem returns the raw mmap'd memory (for RDMA memory registration).
func (p *LOAPool) Mem() []byte { return p.mem }

// DataBase returns the byte offset where slot data begins.
func (p *LOAPool) DataBase() uint32 { return p.hdr.DataBase }

// SlotSize returns bytes per slot.
func (p *LOAPool) SlotSize() uint32 { return p.slotSize }

func (p *LOAPool) Close() error {
	if err := unix.Munmap(p.mem); err != nil {
		return err
	}
	return unix.Close(p.fd)
}

func (p *LOAPool) slotMeta(id uint16) *slotMeta {
	off := p.metaBase + uintptr(id)*loaSlotMetaSize
	return (*slotMeta)(unsafe.Pointer(off))
}

func (p *LOAPool) slotData(id uint16) []byte {
	off := p.dataBase + uintptr(id)*uintptr(p.slotSize)
	return unsafe.Slice((*byte)(unsafe.Pointer(off)), p.slotSize)
}

// Alloc reserves a slot and returns a writable slice + LOARef.
// The caller writes payload into the slice, then calls Commit(ref).
// Returns ErrLOAFull if no slots available.
func (p *LOAPool) Alloc(size uint32, ownerFragID uint32) ([]byte, LOARef, error) {
	if size > p.slotSize {
		return nil, LOARef{}, fmt.Errorf("loa: payload %d exceeds slot size %d", size, p.slotSize)
	}

	p.freeMu.Lock()
	if len(p.freeList) == 0 {
		p.freeMu.Unlock()
		return nil, LOARef{}, ErrLOAFull
	}
	slotID := p.freeList[len(p.freeList)-1]
	p.freeList = p.freeList[:len(p.freeList)-1]
	p.freeMu.Unlock()

	m := p.slotMeta(slotID)
	atomic.StoreUint32(&m.State, slotAllocating)
	atomic.StoreInt32(&m.RefCnt, 0)
	m.Owner = ownerFragID
	m.Size = size

	data := p.slotData(slotID)[:size]

	ref := LOARef{
		PoolID: p.poolID,
		SlotID: slotID,
		Offset: 0,
		Length: size,
	}
	return data, ref, nil
}

// Commit marks a slot as ready for readers after the writer has finished
// writing data into the slice returned by Alloc.
func (p *LOAPool) Commit(ref LOARef) {
	m := p.slotMeta(ref.SlotID)
	atomic.StoreUint32(&m.State, slotReady)
}

// Deref returns a read-only view of the LOA slot's data.
// Caller must call Release when done reading.
func (p *LOAPool) Deref(ref LOARef) ([]byte, error) {
	if uint32(ref.SlotID) >= p.numSlots {
		return nil, fmt.Errorf("loa: slot %d out of range", ref.SlotID)
	}
	m := p.slotMeta(ref.SlotID)
	state := atomic.LoadUint32(&m.State)
	if state != slotReady {
		return nil, fmt.Errorf("loa: slot %d not ready (state=%d)", ref.SlotID, state)
	}
	atomic.AddInt32(&m.RefCnt, 1)

	data := p.slotData(ref.SlotID)
	end := ref.Offset + ref.Length
	if end > p.slotSize {
		end = p.slotSize
	}
	return data[ref.Offset:end], nil
}

// Release decrements the reference count. When it hits 0, the slot is freed.
func (p *LOAPool) Release(ref LOARef) {
	m := p.slotMeta(ref.SlotID)
	newRC := atomic.AddInt32(&m.RefCnt, -1)
	if newRC <= 0 {
		atomic.StoreUint32(&m.State, slotFree)
		p.freeMu.Lock()
		p.freeList = append(p.freeList, ref.SlotID)
		p.freeMu.Unlock()
	}
}

var ErrLOAFull = errors.New("loa: pool full, no free slots")

func defaultLOAPath(poolID uint16) string {
	return filepath.Join(defaultLOADir(), fmt.Sprintf("fragmind.loa.%d", poolID))
}

func alignUp(v, align uint32) uint32 {
	return (v + align - 1) &^ (align - 1)
}
