package fragpigeon

import (
	"testing"
	"unsafe"
)

func TestHeaderPackUnpack(t *testing.T) {
	orig := Header{
		Len:         1234,
		Kind:        KindProcess,
		Flags:       FlagUrgent | FlagLOAPtr,
		TSns:        0xDEADBEEFCAFE0001,
		ConceptID:   0x8A7311CCDD55002A,
		ConceptBits: 28,
		SchemaID:    42,
		SrcID:       1001,
		MsgID:       7,
		Hop:         3,
		Ver:         1,
		TraceID:     0x1122334455667788,
		Checksum32:  0xAABBCCDD,
	}

	var buf [HdrSize]byte
	orig.Pack(unsafe.Pointer(&buf[0]))
	got := UnpackHeader(unsafe.Pointer(&buf[0]))

	if got.Len != orig.Len {
		t.Errorf("Len: got %d, want %d", got.Len, orig.Len)
	}
	if got.Kind != orig.Kind {
		t.Errorf("Kind: got %d, want %d", got.Kind, orig.Kind)
	}
	if got.Flags != orig.Flags {
		t.Errorf("Flags: got %d, want %d", got.Flags, orig.Flags)
	}
	if got.TSns != orig.TSns {
		t.Errorf("TSns: got %x, want %x", got.TSns, orig.TSns)
	}
	if got.ConceptID != orig.ConceptID {
		t.Errorf("ConceptID: got %x, want %x", got.ConceptID, orig.ConceptID)
	}
	if got.ConceptBits != orig.ConceptBits {
		t.Errorf("ConceptBits: got %d, want %d", got.ConceptBits, orig.ConceptBits)
	}
	if got.SchemaID != orig.SchemaID {
		t.Errorf("SchemaID: got %d, want %d", got.SchemaID, orig.SchemaID)
	}
	if got.SrcID != orig.SrcID {
		t.Errorf("SrcID: got %d, want %d", got.SrcID, orig.SrcID)
	}
	if got.MsgID != orig.MsgID {
		t.Errorf("MsgID: got %d, want %d", got.MsgID, orig.MsgID)
	}
	if got.Hop != orig.Hop {
		t.Errorf("Hop: got %d, want %d", got.Hop, orig.Hop)
	}
	if got.Ver != orig.Ver {
		t.Errorf("Ver: got %d, want %d", got.Ver, orig.Ver)
	}
	if got.TraceID != orig.TraceID {
		t.Errorf("TraceID: got %x, want %x", got.TraceID, orig.TraceID)
	}
	if got.Checksum32 != orig.Checksum32 {
		t.Errorf("Checksum32: got %x, want %x", got.Checksum32, orig.Checksum32)
	}
}

func TestHeaderSize(t *testing.T) {
	if HdrSize != 64 {
		t.Fatalf("HdrSize should be 64, got %d", HdrSize)
	}
}
