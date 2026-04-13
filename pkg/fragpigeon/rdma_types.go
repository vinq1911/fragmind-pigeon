package fragpigeon

// RDMA types shared between rdma and non-rdma builds.

// RDMAMemoryRegion describes a remote-accessible memory region.
// Exchanged during site handshake so remote pigeons can issue RDMA reads.
type RDMAMemoryRegion struct {
	Addr   uint64 // virtual address of the region
	Length uint32 // size in bytes
	RKey   uint32 // remote access key
	LKey   uint32 // local access key
}

// RDMALOARef extends LOARef with RDMA remote access info.
// Sent over the control channel when forwarding LOA data cross-host via RDMA.
type RDMALOARef struct {
	LOARef
	RemoteAddr uint64 // address of the LOA slot data in the remote site's memory
	RKey       uint32 // remote key for RDMA read
}

const RDMALOARefSize = LOARefSize + 12 // 12 + 12 = 24 bytes
