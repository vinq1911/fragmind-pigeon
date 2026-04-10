package fragpigeon

import "encoding/binary"

// Schema IDs for LLM workloads.
// SchemaID is a uint16 in the Header; these constants define the well-known
// schemas for fragmind's primary use case: distributed LLM inference/training.

const (
	// SchemaRaw is a pass-through schema with no metadata envelope.
	SchemaRaw uint16 = 0

	// SchemaWeightShard carries a slice of model weights (fp16/bf16/fp8).
	// Payload: WeightShardMeta (32B) + raw tensor data.
	SchemaWeightShard uint16 = 1

	// SchemaKVCache carries key/value cache slices for attention layers.
	// Payload: KVCacheMeta (32B) + interleaved K,V data.
	SchemaKVCache uint16 = 2

	// SchemaActivation carries intermediate activations for pipeline parallelism.
	// Payload: ActivationMeta (24B) + tensor data.
	SchemaActivation uint16 = 3

	// SchemaTokenBatch carries token IDs + attention masks for prompt routing.
	// Payload: TokenBatchMeta (16B) + token IDs (uint32[]) + mask (uint8[]).
	SchemaTokenBatch uint16 = 4

	// SchemaGradient carries gradient accumulation buffers for distributed training.
	// Payload: GradientMeta (32B) + gradient data.
	SchemaGradient uint16 = 5

	// SchemaControl carries internal pigeon control messages (heartbeat, topology, etc.)
	SchemaControl uint16 = 0xFFFF
)

// Tensor data types (matches common ML frameworks).
type DType uint8

const (
	DTypeF32    DType = 0
	DTypeF16    DType = 1
	DTypeBF16   DType = 2
	DTypeFP8E4  DType = 3 // FP8 E4M3
	DTypeFP8E5  DType = 4 // FP8 E5M2
	DTypeI8     DType = 5
	DTypeI32    DType = 6
	DTypeU32    DType = 7
)

// DTypeSize returns the byte size of one element.
func DTypeSize(dt DType) int {
	switch dt {
	case DTypeF32, DTypeI32, DTypeU32:
		return 4
	case DTypeF16, DTypeBF16:
		return 2
	case DTypeFP8E4, DTypeFP8E5, DTypeI8:
		return 1
	default:
		return 0
	}
}

// WeightShardMeta is the metadata prefix for SchemaWeightShard payloads.
// Total: 32 bytes, followed by raw tensor data.
type WeightShardMeta struct {
	ModelID    uint32   // model identifier
	LayerStart uint16   // first layer index in this shard
	LayerEnd   uint16   // last layer index (exclusive)
	DType      DType    // tensor element type
	_          [3]byte  // padding
	NumElements uint32  // total number of elements
	Shape      [4]uint16 // up to 4 dimensions (0 = unused)
	Checksum   uint32   // CRC32 of the tensor data
}

const WeightShardMetaSize = 32

func (m *WeightShardMeta) Encode(b []byte) {
	binary.LittleEndian.PutUint32(b[0:], m.ModelID)
	binary.LittleEndian.PutUint16(b[4:], m.LayerStart)
	binary.LittleEndian.PutUint16(b[6:], m.LayerEnd)
	b[8] = byte(m.DType)
	// b[9:12] padding
	binary.LittleEndian.PutUint32(b[12:], m.NumElements)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint16(b[16+i*2:], m.Shape[i])
	}
	binary.LittleEndian.PutUint32(b[24:], m.Checksum)
}

func DecodeWeightShardMeta(b []byte) WeightShardMeta {
	return WeightShardMeta{
		ModelID:     binary.LittleEndian.Uint32(b[0:]),
		LayerStart:  binary.LittleEndian.Uint16(b[4:]),
		LayerEnd:    binary.LittleEndian.Uint16(b[6:]),
		DType:       DType(b[8]),
		NumElements: binary.LittleEndian.Uint32(b[12:]),
		Shape: [4]uint16{
			binary.LittleEndian.Uint16(b[16:]),
			binary.LittleEndian.Uint16(b[18:]),
			binary.LittleEndian.Uint16(b[20:]),
			binary.LittleEndian.Uint16(b[22:]),
		},
		Checksum: binary.LittleEndian.Uint32(b[24:]),
	}
}

// KVCacheMeta is the metadata prefix for SchemaKVCache payloads.
// Total: 32 bytes, followed by interleaved K,V tensor data.
type KVCacheMeta struct {
	ModelID     uint32
	Layer       uint16
	HeadStart   uint16 // first attention head
	HeadEnd     uint16 // last head (exclusive)
	SeqStart    uint32 // sequence position start
	SeqLen      uint32 // number of sequence positions
	HeadDim     uint16 // dimension per head
	DType       DType
	_           [1]byte
	NumElements uint32
	Checksum    uint32
}

const KVCacheMetaSize = 32

func (m *KVCacheMeta) Encode(b []byte) {
	binary.LittleEndian.PutUint32(b[0:], m.ModelID)
	binary.LittleEndian.PutUint16(b[4:], m.Layer)
	binary.LittleEndian.PutUint16(b[6:], m.HeadStart)
	binary.LittleEndian.PutUint16(b[8:], m.HeadEnd)
	binary.LittleEndian.PutUint32(b[10:], m.SeqStart)
	binary.LittleEndian.PutUint32(b[14:], m.SeqLen)
	binary.LittleEndian.PutUint16(b[18:], m.HeadDim)
	b[20] = byte(m.DType)
	// b[21] padding
	binary.LittleEndian.PutUint32(b[22:], m.NumElements)
	binary.LittleEndian.PutUint32(b[26:], m.Checksum)
}

func DecodeKVCacheMeta(b []byte) KVCacheMeta {
	return KVCacheMeta{
		ModelID:     binary.LittleEndian.Uint32(b[0:]),
		Layer:       binary.LittleEndian.Uint16(b[4:]),
		HeadStart:   binary.LittleEndian.Uint16(b[6:]),
		HeadEnd:     binary.LittleEndian.Uint16(b[8:]),
		SeqStart:    binary.LittleEndian.Uint32(b[10:]),
		SeqLen:      binary.LittleEndian.Uint32(b[14:]),
		HeadDim:     binary.LittleEndian.Uint16(b[18:]),
		DType:       DType(b[20]),
		NumElements: binary.LittleEndian.Uint32(b[22:]),
		Checksum:    binary.LittleEndian.Uint32(b[26:]),
	}
}

// ActivationMeta is the metadata prefix for SchemaActivation payloads.
// Total: 24 bytes, followed by tensor data.
type ActivationMeta struct {
	ModelID  uint32
	Layer    uint16
	DType    DType
	_        byte
	BatchIdx uint32 // micro-batch index for pipeline parallelism
	SeqLen   uint32 // sequence length
	HiddenDim uint32 // hidden dimension size
	Checksum uint32
}

const ActivationMetaSize = 24

func (m *ActivationMeta) Encode(b []byte) {
	binary.LittleEndian.PutUint32(b[0:], m.ModelID)
	binary.LittleEndian.PutUint16(b[4:], m.Layer)
	b[6] = byte(m.DType)
	// b[7] padding
	binary.LittleEndian.PutUint32(b[8:], m.BatchIdx)
	binary.LittleEndian.PutUint32(b[12:], m.SeqLen)
	binary.LittleEndian.PutUint32(b[16:], m.HiddenDim)
	binary.LittleEndian.PutUint32(b[20:], m.Checksum)
}

func DecodeActivationMeta(b []byte) ActivationMeta {
	return ActivationMeta{
		ModelID:   binary.LittleEndian.Uint32(b[0:]),
		Layer:     binary.LittleEndian.Uint16(b[4:]),
		DType:     DType(b[6]),
		BatchIdx:  binary.LittleEndian.Uint32(b[8:]),
		SeqLen:    binary.LittleEndian.Uint32(b[12:]),
		HiddenDim: binary.LittleEndian.Uint32(b[16:]),
		Checksum:  binary.LittleEndian.Uint32(b[20:]),
	}
}

// TokenBatchMeta is the metadata prefix for SchemaTokenBatch payloads.
// Total: 16 bytes, followed by token IDs (uint32[NumTokens]) + mask (uint8[NumTokens]).
type TokenBatchMeta struct {
	ModelID   uint32
	BatchIdx  uint32
	NumTokens uint32
	MaxSeqLen uint16
	_         uint16
}

const TokenBatchMetaSize = 16

func (m *TokenBatchMeta) Encode(b []byte) {
	binary.LittleEndian.PutUint32(b[0:], m.ModelID)
	binary.LittleEndian.PutUint32(b[4:], m.BatchIdx)
	binary.LittleEndian.PutUint32(b[8:], m.NumTokens)
	binary.LittleEndian.PutUint16(b[12:], m.MaxSeqLen)
}

func DecodeTokenBatchMeta(b []byte) TokenBatchMeta {
	return TokenBatchMeta{
		ModelID:   binary.LittleEndian.Uint32(b[0:]),
		BatchIdx:  binary.LittleEndian.Uint32(b[4:]),
		NumTokens: binary.LittleEndian.Uint32(b[8:]),
		MaxSeqLen: binary.LittleEndian.Uint16(b[12:]),
	}
}

// GradientMeta is the metadata prefix for SchemaGradient payloads.
// Total: 32 bytes, followed by gradient tensor data.
type GradientMeta struct {
	ModelID     uint32
	LayerStart  uint16
	LayerEnd    uint16
	DType       DType
	_           [3]byte
	NumElements uint32
	Step        uint64 // training step number
	Checksum    uint32
	_           [4]byte
}

const GradientMetaSize = 32

func (m *GradientMeta) Encode(b []byte) {
	binary.LittleEndian.PutUint32(b[0:], m.ModelID)
	binary.LittleEndian.PutUint16(b[4:], m.LayerStart)
	binary.LittleEndian.PutUint16(b[6:], m.LayerEnd)
	b[8] = byte(m.DType)
	// b[9:12] padding
	binary.LittleEndian.PutUint32(b[12:], m.NumElements)
	binary.LittleEndian.PutUint64(b[16:], m.Step)
	binary.LittleEndian.PutUint32(b[24:], m.Checksum)
}

func DecodeGradientMeta(b []byte) GradientMeta {
	return GradientMeta{
		ModelID:     binary.LittleEndian.Uint32(b[0:]),
		LayerStart:  binary.LittleEndian.Uint16(b[4:]),
		LayerEnd:    binary.LittleEndian.Uint16(b[6:]),
		DType:       DType(b[8]),
		NumElements: binary.LittleEndian.Uint32(b[12:]),
		Step:        binary.LittleEndian.Uint64(b[16:]),
		Checksum:    binary.LittleEndian.Uint32(b[24:]),
	}
}
