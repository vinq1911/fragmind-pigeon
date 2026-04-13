package fragpigeon

import (
	"encoding/binary"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// CreateRingFD creates a shared-memory ring buffer file and returns the fd.
// The file is unlinked after creation (anonymous shm). The ring is initialized
// with poll mode (no eventfd). Use OpenRingFromFD to get a Ring handle.
func CreateRingFD(dir, name string, capSlots, slotSize int) (int, error) {
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
	binary.LittleEndian.PutUint64(mem[32:], ^uint64(0))      // ProdEvtFD=-1 (poll mode)
	binary.LittleEndian.PutUint64(mem[40:], ^uint64(0))      // ConsEvtFD=-1
	_ = unix.Munmap(mem)
	_ = os.Remove(path) // unlink, keep fd open

	return fd, nil
}

// CreateRing creates a ring and returns it ready to use. Convenience wrapper.
func CreateRing(dir, name string, capSlots, slotSize int) (*Ring, func(), error) {
	fd, err := CreateRingFD(dir, name, capSlots, slotSize)
	if err != nil {
		return nil, nil, err
	}
	ring, err := OpenRingFromFD(fd)
	if err != nil {
		unix.Close(fd)
		return nil, nil, err
	}
	cleanup := func() {
		ring.Close()
		unix.Close(fd)
	}
	return ring, cleanup, nil
}
