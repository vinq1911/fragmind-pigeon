package fragpigeon

import (
	"path/filepath"
	"testing"
	"unsafe"
)

func BenchmarkHeaderPackUnpack(b *testing.B) {
	h := Header{
		Len: 512, Kind: KindProcess, Flags: FlagUrgent,
		TSns: 0xDEADBEEF, ConceptID: 0x8A7311CCDD55002A,
		ConceptBits: 28, SchemaID: 42, SrcID: 1001, MsgID: 7,
	}
	var buf [HdrSize]byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Pack(unsafe.Pointer(&buf[0]))
		_ = UnpackHeader(unsafe.Pointer(&buf[0]))
	}
}

func BenchmarkRouterLookup(b *testing.B) {
	r := NewRouter()
	// Populate with 1000 concepts across 10 bit-widths
	for bits := uint16(16); bits <= 32; bits += 2 {
		for i := 0; i < 100; i++ {
			cid := uint64(i) << bits
			r.Add(bits, cid, []uint16{uint16(i)}, []uint16{uint16(i + 100)})
		}
	}
	h := Header{ConceptID: 50 << 24, ConceptBits: 24}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = r.Destinations(h)
	}
}

func BenchmarkLOAAllocCommitRelease(b *testing.B) {
	dir := b.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "bench.loa"),
		PoolID:   1,
		NumSlots: 4096,
		SlotSize: 64 * 1024,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, ref, err := pool.Alloc(1024, 1)
		if err != nil {
			b.Fatal(err)
		}
		// Simulate writing a small tensor
		buf[0] = byte(i)
		pool.Commit(ref)
		pool.Release(ref)
	}
}

func BenchmarkLOAAllocCommitDerefRelease(b *testing.B) {
	dir := b.TempDir()
	pool, err := CreateLOAPool(LOAPoolOptions{
		Path:     filepath.Join(dir, "bench.loa"),
		PoolID:   1,
		NumSlots: 4096,
		SlotSize: 64 * 1024,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf, ref, err := pool.Alloc(4096, 1)
		if err != nil {
			b.Fatal(err)
		}
		buf[0] = byte(i)
		pool.Commit(ref)

		data, err := pool.Deref(ref)
		if err != nil {
			b.Fatal(err)
		}
		_ = data[0] // simulate read
		pool.Release(ref)
	}
}

func BenchmarkLOARefEncodeDecode(b *testing.B) {
	ref := LOARef{PoolID: 1, SlotID: 42, Offset: 0, Length: 65536}
	var buf [LOARefSize]byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref.Encode(buf[:])
		_ = DecodeLOARef(buf[:])
	}
}

func BenchmarkWeightShardMetaEncodeDecode(b *testing.B) {
	m := WeightShardMeta{
		ModelID: 1, LayerStart: 0, LayerEnd: 32,
		DType: DTypeBF16, NumElements: 1024 * 1024,
		Shape: [4]uint16{1024, 1024, 0, 0}, Checksum: 0xDEADBEEF,
	}
	var buf [WeightShardMetaSize]byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Encode(buf[:])
		_ = DecodeWeightShardMeta(buf[:])
	}
}
