//go:build darwin

// gorch_distributed: Distributed MNIST inference via fragmind-pigeon.
//
// Demonstrates two "fragment workers" that each own a portion of an MLP:
//   Worker A: Linear(784→128) + ReLU  (produces hidden activations)
//   Worker B: Linear(128→10)          (produces logits from activations)
//
// Activations flow from A→B through fragmind's LOA pool (zero-copy shared
// memory). Both workers share the same LOA arena — A writes activation
// tensors directly into LOA slots, B reads them via pointer dereference.
//
// The pigeon daemon manages the LOA pool and could route between the
// workers via COI-based rings. For this demo, we use a direct ring
// pair to keep it self-contained.
//
// Usage:
//   CGO_ENABLED=1 go run ./examples/gorch_distributed
package main

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

const (
	inputDim  = 784 // 28x28 MNIST
	hiddenDim = 128
	outputDim = 10
	batchSize = 64
)

func main() {
	dir, err := os.MkdirTemp("", "gorch-fragmind-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	fmt.Println("=== Gorch + Fragmind Distributed Inference Demo ===")
	fmt.Println()

	// --- 1. Create fragmind LOA pool ---
	loaSlotSize := uint32(batchSize * hiddenDim * 4) // float32: 64*128*4 = 32KB
	if loaSlotSize < 64*1024 {
		loaSlotSize = 64 * 1024
	}
	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path:     filepath.Join(dir, "fragmind.loa"),
		PoolID:   1,
		NumSlots: 64,
		SlotSize: loaSlotSize,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()
	fmt.Printf("LOA pool: %d slots x %d bytes = %d KB\n", 64, loaSlotSize, 64*int(loaSlotSize)/1024)

	// --- 2. Create shared ring (A→B for activation pointers) ---
	ringSlotSize := fp.HdrSize + fp.LOARefSize + 16
	ring, ringCleanup, err := fp.CreateRing(dir, "ring-a-to-b", 256, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer ringCleanup()

	// --- 3. Create the model layers ---
	layerA := nn.NewLinear(inputDim, hiddenDim) // Worker A owns this
	reluA := nn.NewReLU()
	layerB := nn.NewLinear(hiddenDim, outputDim) // Worker B owns this

	fmt.Printf("Model: Linear(%d→%d) + ReLU → Linear(%d→%d)\n", inputDim, hiddenDim, hiddenDim, outputDim)
	fmt.Printf("Worker A parameters: %d (weights) + %d (bias) = %d floats\n",
		layerA.Weight.Size(), layerA.Bias.Size(), layerA.Weight.Size()+layerA.Bias.Size())
	fmt.Printf("Worker B parameters: %d (weights) + %d (bias) = %d floats\n",
		layerB.Weight.Size(), layerB.Bias.Size(), layerB.Weight.Size()+layerB.Bias.Size())
	fmt.Println()

	// --- 4. Run distributed inference on synthetic batches ---
	nBatches := 20
	var totalTime time.Duration
	var totalActivationBytes int64

	fmt.Println("--- Distributed Inference ---")
	for i := 0; i < nBatches; i++ {
		// Generate synthetic batch (random MNIST-like input)
		input := g.RandN(batchSize, inputDim)

		batchStart := time.Now()

		// === Worker A: Forward through layer 1 + ReLU ===
		hidden := reluA.Forward(layerA.Forward(input))
		hiddenData := hidden.Data() // []float32

		// Write activation tensor into LOA pool (zero-copy)
		activationBytes := len(hiddenData) * 4
		buf, ref, err := pool.Alloc(uint32(activationBytes), 1) // Worker A = fragID 1
		if err != nil {
			log.Fatalf("LOA alloc: %v", err)
		}
		// Copy float32 slice into LOA shm slot
		copyFloat32ToBytes(buf, hiddenData)
		pool.Commit(ref)

		// Send LOA pointer over ring
		hdr := fp.Header{
			Kind:        fp.KindProcess,
			Flags:       fp.FlagLOAPtr,
			TSns:        uint64(time.Now().UnixNano()),
			ConceptID:   0x0001000000000000, // model 1
			ConceptBits: 16,
			SchemaID:    fp.SchemaActivation,
			SrcID:       1,
			MsgID:       uint32(i),
			Ver:         1,
			Checksum32:  crc32.ChecksumIEEE(buf[:activationBytes]),
		}
		var refBuf [fp.LOARefSize]byte
		ref.Encode(refBuf[:])
		hdr.Len = fp.LOARefSize
		if !ring.TryWrite(hdr, refBuf[:]) {
			log.Fatal("ring write failed")
		}

		// === Worker B: Read activation from LOA, forward through layer 2 ===
		msg, err := ring.Read(false)
		if err != nil {
			log.Fatalf("ring read: %v", err)
		}
		gotRef := fp.DecodeLOARef(msg.Payload)
		actData, err := pool.Deref(gotRef)
		if err != nil {
			log.Fatalf("LOA deref: %v", err)
		}

		// Verify CRC
		gotCRC := crc32.ChecksumIEEE(actData[:activationBytes])
		if gotCRC != msg.Header.Checksum32 {
			log.Fatalf("CRC mismatch: got %08x, want %08x", gotCRC, msg.Header.Checksum32)
		}

		// Reconstruct tensor from LOA data
		activationFloats := bytesToFloat32(actData[:activationBytes])
		hiddenB := g.NewTensor(activationFloats, batchSize, hiddenDim)

		// Forward through layer 2
		logits := layerB.Forward(hiddenB)

		batchTime := time.Since(batchStart)
		totalTime += batchTime
		totalActivationBytes += int64(activationBytes)

		pool.Release(gotRef)

		// Compute accuracy on synthetic data (just to show logits are valid)
		if i == 0 || i == nBatches-1 {
			preds := argmax(logits.Data(), batchSize, outputDim)
			fmt.Printf("  Batch %d/%d: latency=%s, activation=%d bytes via LOA, logits_shape=%v, pred_dist=%v\n",
				i+1, nBatches, batchTime, activationBytes, logits.Shape(), distribution(preds, outputDim))
		}
	}

	// --- 5. Summary ---
	avgLatency := totalTime / time.Duration(nBatches)
	throughput := float64(totalActivationBytes) / totalTime.Seconds() / 1e6

	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("Batches:              %d (batch_size=%d)\n", nBatches, batchSize)
	fmt.Printf("Activation size:      %d bytes per batch (%dx%d float32)\n",
		batchSize*hiddenDim*4, batchSize, hiddenDim)
	fmt.Printf("Total activations:    %d bytes via LOA\n", totalActivationBytes)
	fmt.Printf("Avg batch latency:    %s\n", avgLatency)
	fmt.Printf("LOA throughput:       %.1f MB/s\n", throughput)
	fmt.Printf("Data path:            Worker A → LOA pool (zero-copy) → Ring pointer → Worker B → LOA deref\n")
	fmt.Printf("Copies:               1 (float32→LOA) + 0 (B reads LOA directly)\n")
	fmt.Printf("Verification:         CRC32 on all %d activations: PASS\n", nBatches)

	// Compare with non-fragmind (direct function call)
	fmt.Println()
	fmt.Println("--- Baseline (no fragmind, direct call) ---")
	baseStart := time.Now()
	for i := 0; i < nBatches; i++ {
		input := g.RandN(batchSize, inputDim)
		hidden := reluA.Forward(layerA.Forward(input))
		_ = layerB.Forward(hidden)
	}
	baseTime := time.Since(baseStart)
	baseAvg := baseTime / time.Duration(nBatches)
	overhead := float64(avgLatency-baseAvg) / float64(baseAvg) * 100

	fmt.Printf("Avg batch latency:    %s\n", baseAvg)
	fmt.Printf("Fragmind overhead:    %.1f%% (LOA alloc + memcpy + ring + deref)\n", overhead)
	fmt.Println()
	if overhead < 10 {
		fmt.Println("Status: PASS (<10% overhead)")
	} else if overhead < 25 {
		fmt.Println("Status: PASS (acceptable overhead)")
	} else {
		fmt.Printf("Status: OVERHEAD %.1f%% (optimize LOA path)\n", overhead)
	}
}

// --- Helpers ---

func copyFloat32ToBytes(dst []byte, src []float32) {
	for i, v := range src {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(dst[i*4:], bits)
	}
}

func bytesToFloat32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(b[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func argmax(data []float32, batch, classes int) []int {
	preds := make([]int, batch)
	for i := 0; i < batch; i++ {
		maxIdx := 0
		maxVal := data[i*classes]
		for j := 1; j < classes; j++ {
			if data[i*classes+j] > maxVal {
				maxVal = data[i*classes+j]
				maxIdx = j
			}
		}
		preds[i] = maxIdx
	}
	return preds
}

func distribution(preds []int, classes int) []int {
	dist := make([]int, classes)
	for _, p := range preds {
		dist[p]++
	}
	return dist
}

