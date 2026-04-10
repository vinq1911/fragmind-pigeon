package fragpigeon

import "testing"

func TestWeightShardMetaEncodeDecode(t *testing.T) {
	orig := WeightShardMeta{
		ModelID:     1,
		LayerStart:  0,
		LayerEnd:    32,
		DType:       DTypeBF16,
		NumElements: 1024 * 1024,
		Shape:       [4]uint16{1024, 1024, 0, 0},
		Checksum:    0xDEADBEEF,
	}
	var buf [WeightShardMetaSize]byte
	orig.Encode(buf[:])
	got := DecodeWeightShardMeta(buf[:])

	if got.ModelID != orig.ModelID || got.LayerStart != orig.LayerStart ||
		got.LayerEnd != orig.LayerEnd || got.DType != orig.DType ||
		got.NumElements != orig.NumElements || got.Shape != orig.Shape ||
		got.Checksum != orig.Checksum {
		t.Fatalf("mismatch:\n  got  %+v\n  want %+v", got, orig)
	}
}

func TestKVCacheMetaEncodeDecode(t *testing.T) {
	orig := KVCacheMeta{
		ModelID:     2,
		Layer:       15,
		HeadStart:   0,
		HeadEnd:     32,
		SeqStart:    100,
		SeqLen:      512,
		HeadDim:     128,
		DType:       DTypeF16,
		NumElements: 32 * 512 * 128,
		Checksum:    0xCAFEBABE,
	}
	var buf [KVCacheMetaSize]byte
	orig.Encode(buf[:])
	got := DecodeKVCacheMeta(buf[:])

	if got.ModelID != orig.ModelID || got.Layer != orig.Layer ||
		got.HeadStart != orig.HeadStart || got.HeadEnd != orig.HeadEnd ||
		got.SeqStart != orig.SeqStart || got.SeqLen != orig.SeqLen ||
		got.HeadDim != orig.HeadDim || got.DType != orig.DType ||
		got.NumElements != orig.NumElements || got.Checksum != orig.Checksum {
		t.Fatalf("mismatch:\n  got  %+v\n  want %+v", got, orig)
	}
}

func TestActivationMetaEncodeDecode(t *testing.T) {
	orig := ActivationMeta{
		ModelID:   3,
		Layer:     10,
		DType:     DTypeF32,
		BatchIdx:  7,
		SeqLen:    2048,
		HiddenDim: 4096,
		Checksum:  0x12345678,
	}
	var buf [ActivationMetaSize]byte
	orig.Encode(buf[:])
	got := DecodeActivationMeta(buf[:])

	if got.ModelID != orig.ModelID || got.Layer != orig.Layer ||
		got.DType != orig.DType || got.BatchIdx != orig.BatchIdx ||
		got.SeqLen != orig.SeqLen || got.HiddenDim != orig.HiddenDim ||
		got.Checksum != orig.Checksum {
		t.Fatalf("mismatch:\n  got  %+v\n  want %+v", got, orig)
	}
}

func TestTokenBatchMetaEncodeDecode(t *testing.T) {
	orig := TokenBatchMeta{
		ModelID:   1,
		BatchIdx:  0,
		NumTokens: 256,
		MaxSeqLen: 4096,
	}
	var buf [TokenBatchMetaSize]byte
	orig.Encode(buf[:])
	got := DecodeTokenBatchMeta(buf[:])

	if got.ModelID != orig.ModelID || got.BatchIdx != orig.BatchIdx ||
		got.NumTokens != orig.NumTokens || got.MaxSeqLen != orig.MaxSeqLen {
		t.Fatalf("mismatch:\n  got  %+v\n  want %+v", got, orig)
	}
}

func TestGradientMetaEncodeDecode(t *testing.T) {
	orig := GradientMeta{
		ModelID:     1,
		LayerStart:  0,
		LayerEnd:    16,
		DType:       DTypeF32,
		NumElements: 2048 * 2048,
		Step:        100000,
		Checksum:    0xABCDEF01,
	}
	var buf [GradientMetaSize]byte
	orig.Encode(buf[:])
	got := DecodeGradientMeta(buf[:])

	if got.ModelID != orig.ModelID || got.LayerStart != orig.LayerStart ||
		got.LayerEnd != orig.LayerEnd || got.DType != orig.DType ||
		got.NumElements != orig.NumElements || got.Step != orig.Step ||
		got.Checksum != orig.Checksum {
		t.Fatalf("mismatch:\n  got  %+v\n  want %+v", got, orig)
	}
}

func TestDTypeSize(t *testing.T) {
	cases := []struct {
		dt   DType
		want int
	}{
		{DTypeF32, 4}, {DTypeF16, 2}, {DTypeBF16, 2},
		{DTypeFP8E4, 1}, {DTypeFP8E5, 1}, {DTypeI8, 1},
		{DTypeI32, 4}, {DTypeU32, 4},
	}
	for _, c := range cases {
		if got := DTypeSize(c.dt); got != c.want {
			t.Errorf("DTypeSize(%d) = %d, want %d", c.dt, got, c.want)
		}
	}
}
