package fragpigeon

import (
	"errors"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

type Ring struct {
	mem       []byte
	ctrl      *ringCtrl
	slotsBase uintptr
}

type ringCtrl struct {
	CapSlots  uint64
	ProdIdx   uint64
	ConsIdx   uint64
	SlotSize  uint32
	_         uint32
	ProdEvtFD uint64
	ConsEvtFD uint64
	_         [4]uint64
}

func OpenRingFromFD(fd int) (*Ring, error) {
	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		return nil, err
	}
	size := int(st.Size)
	mem, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	r := &Ring{mem: mem}
	r.ctrl = (*ringCtrl)(unsafe.Pointer(&mem[0]))
	r.slotsBase = uintptr(unsafe.Pointer(&mem[64]))
	return r, nil
}

func (r *Ring) Close() error { return unix.Munmap(r.mem) }

func (r *Ring) slotPtr(idx uint64) uintptr {
	cap := r.ctrl.CapSlots
	slot := (idx & (cap - 1)) * uint64(r.ctrl.SlotSize)
	return r.slotsBase + uintptr(slot)
}

type Msg struct {
	Header  Header
	Payload []byte // read-only view into shm
}

func (r *Ring) TryWrite(h Header, payload []byte) bool {
	prod := atomic.LoadUint64(&r.ctrl.ProdIdx)
	cons := atomic.LoadUint64(&r.ctrl.ConsIdx)
	if prod-cons >= r.ctrl.CapSlots {
		return false
	} // full
	if int(h.Len)+HdrSize > int(r.ctrl.SlotSize) {
		return false
	}
	slot := r.slotPtr(prod)
	h.Pack(unsafe.Pointer(slot))
	ps := unsafe.Slice((*byte)(unsafe.Pointer(slot+HdrSize)), int(h.Len))
	copy(ps, payload)
	atomic.StoreUint64(&r.ctrl.ProdIdx, prod+1)
	_, _ = unix.Write(int(r.ctrl.ConsEvtFD), []byte{1, 0, 0, 0, 0, 0, 0, 0})
	return true
}

func (r *Ring) Read(block bool) (Msg, error) {
	for {
		prod := atomic.LoadUint64(&r.ctrl.ProdIdx)
		cons := atomic.LoadUint64(&r.ctrl.ConsIdx)
		if prod == cons {
			if !block {
				return Msg{}, unix.EAGAIN
			}
			var buf [8]byte
			_, _ = unix.Read(int(r.ctrl.ConsEvtFD), buf[:]) // drain
			_, err := unix.Poll([]unix.PollFd{{Fd: int32(r.ctrl.ConsEvtFD), Events: unix.POLLIN}}, -1)
			if err != nil {
				return Msg{}, err
			}
			continue
		}
		slot := r.slotPtr(cons)
		h := UnpackHeader(unsafe.Pointer(slot))
		p := unsafe.Slice((*byte)(unsafe.Pointer(slot+HdrSize)), int(h.Len))
		atomic.StoreUint64(&r.ctrl.ConsIdx, cons+1)
		return Msg{Header: h, Payload: p}, nil
	}
}

func (r *Ring) ReadWithin(d time.Duration) (Msg, error) {
	dead := time.Now().Add(d)
	for {
		m, err := r.Read(false)
		if err == nil {
			return m, nil
		}
		if !errors.Is(err, unix.EAGAIN) {
			return Msg{}, err
		}
		if time.Now().After(dead) {
			return Msg{}, unix.EAGAIN
		}
		time.Sleep(50 * time.Microsecond)
	}
}

