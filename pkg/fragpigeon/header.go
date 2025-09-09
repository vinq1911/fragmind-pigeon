package fragpigeon

import (
	"encoding/binary"
	"unsafe"
)

const (
	HdrSize        = 64
	offLen         = 0
	offKind        = 4
	offFlags       = 6
	offTS          = 8
	offConceptID   = 16
	offConceptBits = 24
	offSchemaID    = 26
	offSrcID       = 28
	offMsgID       = 32
	offHop         = 36
	offVer         = 38
	offTraceID     = 40
	offChecksum32  = 48
)

type Kind uint16
type Flags uint16

const (
	KindProcess Kind = 0x0001
	KindLearn   Kind = 0x0002
	KindShare   Kind = 0x0003
	KindPing    Kind = 0x0004
)

const (
	FlagEndOfStream Flags = 0x0001
	FlagUrgent      Flags = 0x0002
	FlagReply       Flags = 0x0004
	FlagDropOK      Flags = 0x0008
	FlagLOAPtr      Flags = 0x0010
)

type Header struct {
	Len         uint32
	Kind        Kind
	Flags       Flags
	TSns        uint64
	ConceptID   uint64
	ConceptBits uint16
	SchemaID    uint16
	SrcID       uint32
	MsgID       uint32
	Hop         uint16
	Ver         uint16
	TraceID     uint64
	Checksum32  uint32
	_           uint32
}

func (h *Header) Pack(slot unsafe.Pointer) {
	b := unsafe.Slice((*byte)(slot), HdrSize)
	binary.LittleEndian.PutUint32(b[offLen:], h.Len)
	binary.LittleEndian.PutUint16(b[offKind:], uint16(h.Kind))
	binary.LittleEndian.PutUint16(b[offFlags:], uint16(h.Flags))
	binary.LittleEndian.PutUint64(b[offTS:], h.TSns)
	binary.LittleEndian.PutUint64(b[offConceptID:], h.ConceptID)
	binary.LittleEndian.PutUint16(b[offConceptBits:], h.ConceptBits)
	binary.LittleEndian.PutUint16(b[offSchemaID:], h.SchemaID)
	binary.LittleEndian.PutUint32(b[offSrcID:], h.SrcID)
	binary.LittleEndian.PutUint32(b[offMsgID:], h.MsgID)
	binary.LittleEndian.PutUint16(b[offHop:], h.Hop)
	binary.LittleEndian.PutUint16(b[offVer:], h.Ver)
	binary.LittleEndian.PutUint64(b[offTraceID:], h.TraceID)
	binary.LittleEndian.PutUint32(b[offChecksum32:], h.Checksum32)
}

func UnpackHeader(slot unsafe.Pointer) Header {
	b := unsafe.Slice((*byte)(slot), HdrSize)
	return Header{
		Len:         binary.LittleEndian.Uint32(b[offLen:]),
		Kind:        Kind(binary.LittleEndian.Uint16(b[offKind:])),
		Flags:       Flags(binary.LittleEndian.Uint16(b[offFlags:])),
		TSns:        binary.LittleEndian.Uint64(b[offTS:]),
		ConceptID:   binary.LittleEndian.Uint64(b[offConceptID:]),
		ConceptBits: binary.LittleEndian.Uint16(b[offConceptBits:]),
		SchemaID:    binary.LittleEndian.Uint16(b[offSchemaID:]),
		SrcID:       binary.LittleEndian.Uint32(b[offSrcID:]),
		MsgID:       binary.LittleEndian.Uint32(b[offMsgID:]),
		Hop:         binary.LittleEndian.Uint16(b[offHop:]),
		Ver:         binary.LittleEndian.Uint16(b[offVer:]),
		TraceID:     binary.LittleEndian.Uint64(b[offTraceID:]),
		Checksum32:  binary.LittleEndian.Uint32(b[offChecksum32:]),
	}
}
