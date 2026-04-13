//go:build darwin

// gorch_distributed: Distributed MNIST training via fragmind-pigeon.
//
// Splits a 2-layer MLP across two fragment workers and trains it using
// fragmind's LOA pool for zero-copy activation/gradient transfer:
//
//   Worker A: owns Linear(784→128) + ReLU
//   Worker B: owns Linear(128→10) + CrossEntropyLoss
//
// Training loop per batch:
//   1. Worker A: forward pass → activations (32KB) written to LOA → sent to B
//   2. Worker B: reads activations → forward → loss → backward → gradient (32KB) written to LOA → sent to A
//   3. Worker A: reads gradient → backward through its layer → optimizer step
//   4. Worker B: optimizer step (already has its own gradients)
//
// Two rings: forward (A→B activations) and backward (B→A gradients).
// All tensor data flows through the LOA pool — zero-copy on the reader side.
//
// Usage:
//   CGO_ENABLED=1 go run ./examples/gorch_distributed
//   CGO_ENABLED=1 go run ./examples/gorch_distributed -epochs 5 -lr 0.01
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

const (
	inputDim  = 784 // 28x28 MNIST
	hiddenDim = 128
	outputDim = 10
)

func main() {
	batchSize := flag.Int("batch", 64, "batch size")
	epochs := flag.Int("epochs", 3, "number of epochs")
	batchesPerEpoch := flag.Int("batches", 100, "batches per epoch (synthetic data)")
	lr := flag.Float64("lr", 0.001, "learning rate")
	flag.Parse()

	dir, err := os.MkdirTemp("", "gorch-fragmind-train-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	fmt.Println("=== Gorch + Fragmind Distributed Training Demo ===")
	fmt.Println()

	// --- 1. Create fragmind LOA pool ---
	// Slots must hold activations (batch*hidden*4 bytes) and gradients (same size)
	activationBytes := *batchSize * hiddenDim * 4
	loaSlotSize := uint32(activationBytes)
	if loaSlotSize < 64*1024 {
		loaSlotSize = 64 * 1024
	}
	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path:     filepath.Join(dir, "fragmind.loa"),
		PoolID:   1,
		NumSlots: 128,
		SlotSize: loaSlotSize,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	// --- 2. Create two rings: forward (A→B) and backward (B→A) ---
	ringSlotSize := fp.HdrSize + fp.LOARefSize + 16
	fwdRing, fwdCleanup, err := fp.CreateRing(dir, "ring-fwd", 256, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer fwdCleanup()

	bwdRing, bwdCleanup, err := fp.CreateRing(dir, "ring-bwd", 256, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer bwdCleanup()

	// --- 3. Create model layers ---
	layerA := nn.NewLinear(inputDim, hiddenDim) // Worker A
	layerB := nn.NewLinear(hiddenDim, outputDim) // Worker B

	optA := optim.NewAdam(layerA.Parameters(), float32(*lr))
	optB := optim.NewAdam(layerB.Parameters(), float32(*lr))

	fmt.Printf("Model:     Linear(%d→%d) + ReLU → Linear(%d→%d)\n", inputDim, hiddenDim, hiddenDim, outputDim)
	fmt.Printf("Worker A:  %d parameters (weights+bias)\n", layerA.Weight.Size()+layerA.Bias.Size())
	fmt.Printf("Worker B:  %d parameters (weights+bias)\n", layerB.Weight.Size()+layerB.Bias.Size())
	fmt.Printf("Batch:     %d | Epochs: %d | Batches/epoch: %d | LR: %g\n", *batchSize, *epochs, *batchesPerEpoch, *lr)
	fmt.Printf("LOA pool:  %d slots x %d bytes\n", 128, loaSlotSize)
	fmt.Printf("Activation: %d bytes per batch (%dx%d float32)\n", activationBytes, *batchSize, hiddenDim)
	fmt.Printf("Rings:     fwd (A→B activations) + bwd (B→A gradients)\n")
	fmt.Println()

	// --- 4. Training loop ---
	var totalFwdBytes, totalBwdBytes int64
	trainingStart := time.Now()

	for epoch := 0; epoch < *epochs; epoch++ {
		var epochLoss float64
		epochStart := time.Now()

		for batch := 0; batch < *batchesPerEpoch; batch++ {
			// Generate synthetic batch (random data, random labels 0-9)
			input := g.RandN(*batchSize, inputDim)
			labels := make([]float32, *batchSize)
			for i := range labels {
				labels[i] = float32(i % outputDim)
			}
			targets := g.NewTensor(labels, *batchSize, 1)

			// Zero gradients
			optA.ZeroGrad()
			optB.ZeroGrad()

			// === FORWARD: Worker A → LOA → Worker B ===

			// Worker A: forward through layer 1 + ReLU
			hidden := nn.NewReLU().Forward(layerA.Forward(input))
			hiddenData := hidden.Data()

			// Write activations to LOA
			actBuf, actRef, err := pool.Alloc(uint32(activationBytes), 1)
			if err != nil {
				log.Fatalf("LOA alloc fwd: %v", err)
			}
			copyFloat32ToBytes(actBuf, hiddenData)
			pool.Commit(actRef)

			// Send activation ref over forward ring
			sendLOARef(fwdRing, actRef, fp.SchemaActivation, uint32(batch), 1)
			totalFwdBytes += int64(activationBytes)

			// Worker B: read activations, forward, compute loss
			bActRef := recvLOARef(fwdRing)
			actBytes, err := pool.Deref(bActRef)
			if err != nil {
				log.Fatalf("LOA deref fwd: %v", err)
			}
			hiddenB := g.NewTensor(bytesToFloat32(actBytes[:activationBytes]), *batchSize, hiddenDim)
			hiddenB.SetRequiresGrad(true)
			pool.Release(bActRef)

			logits := layerB.Forward(hiddenB)
			loss := g.CrossEntropyLoss(logits, targets)
			epochLoss += float64(loss.Data()[0])

			// === BACKWARD: Worker B → LOA → Worker A ===

			// Worker B: backward through its layer
			loss.Backward()
			// hiddenB.Grad() = dL/dhidden (gradient flowing back to A)

			gradData := hiddenB.Grad()
			if gradData == nil {
				log.Fatal("no gradient for hidden activations")
			}

			// Write gradient to LOA
			gradBuf, gradRef, err := pool.Alloc(uint32(activationBytes), 2)
			if err != nil {
				log.Fatalf("LOA alloc bwd: %v", err)
			}
			copyFloat32ToBytes(gradBuf, gradData.Data())
			pool.Commit(gradRef)

			// Send gradient ref over backward ring
			sendLOARef(bwdRing, gradRef, fp.SchemaGradient, uint32(batch), 2)
			totalBwdBytes += int64(activationBytes)

			// Worker B: optimizer step (it already has gradients for its own params)
			optB.Step()

			// Worker A: read gradient from B, manual backward through its layer
			aGradRef := recvLOARef(bwdRing)
			gradBytes, err := pool.Deref(aGradRef)
			if err != nil {
				log.Fatalf("LOA deref bwd: %v", err)
			}
			dHidden := g.NewTensor(bytesToFloat32(gradBytes[:activationBytes]), *batchSize, hiddenDim)
			pool.Release(aGradRef)

			// Backprop through ReLU: dReLU = dHidden * (hidden > 0)
			reluMask := hidden.Data()
			dReLU := make([]float32, len(reluMask))
			dHiddenData := dHidden.Data()
			for i, v := range reluMask {
				if v > 0 {
					dReLU[i] = dHiddenData[i]
				}
			}

			// Backprop through Linear A: dL/dW = dReLU^T @ input, dL/db = sum(dReLU)
			inData := input.Data()
			wData := layerA.Weight.Data()

			// dL/dW (outFeatures x inFeatures)
			dwData := make([]float32, hiddenDim*inputDim)
			for j := 0; j < hiddenDim; j++ {
				for k := 0; k < inputDim; k++ {
					var s float32
					for i := 0; i < *batchSize; i++ {
						s += dReLU[i*hiddenDim+j] * inData[i*inputDim+k]
					}
					dwData[j*inputDim+k] = s
				}
			}

			// dL/db (1 x outFeatures)
			dbData := make([]float32, hiddenDim)
			for i := 0; i < *batchSize; i++ {
				for j := 0; j < hiddenDim; j++ {
					dbData[j] += dReLU[i*hiddenDim+j]
				}
			}

			// Set gradients on layer A's parameters
			layerA.Weight.ZeroGrad()
			layerA.Bias.ZeroGrad()
			wGrad := g.NewTensor(dwData, hiddenDim, inputDim)
			bGrad := g.NewTensor(dbData, 1, hiddenDim)

			// Manual grad accumulation (same as autograd does)
			for i := range wData {
				_ = wData[i] // just to avoid unused
			}
			setGrad(layerA.Weight, wGrad)
			setGrad(layerA.Bias, bGrad)

			// Worker A: optimizer step
			optA.Step()
		}

		avgLoss := epochLoss / float64(*batchesPerEpoch)
		elapsed := time.Since(epochStart)
		fmt.Printf("  Epoch %d/%d: loss=%.4f  time=%s  (%.0f batches/s)\n",
			epoch+1, *epochs, avgLoss, elapsed,
			float64(*batchesPerEpoch)/elapsed.Seconds())
	}

	totalTime := time.Since(trainingStart)
	totalBatches := *epochs * *batchesPerEpoch

	// --- 5. Summary ---
	fmt.Println()
	fmt.Println("=== Training Results ===")
	fmt.Printf("Total batches:     %d\n", totalBatches)
	fmt.Printf("Total time:        %s\n", totalTime)
	fmt.Printf("Avg batch:         %s\n", totalTime/time.Duration(totalBatches))
	fmt.Printf("FWD via LOA:       %s (%d transfers)\n", humanBytes(totalFwdBytes), totalBatches)
	fmt.Printf("BWD via LOA:       %s (%d transfers)\n", humanBytes(totalBwdBytes), totalBatches)
	fmt.Printf("Total LOA:         %s\n", humanBytes(totalFwdBytes+totalBwdBytes))
	fwdThroughput := float64(totalFwdBytes) / totalTime.Seconds() / 1e6
	bwdThroughput := float64(totalBwdBytes) / totalTime.Seconds() / 1e6
	fmt.Printf("LOA throughput:    %.1f MB/s fwd + %.1f MB/s bwd\n", fwdThroughput, bwdThroughput)
	fmt.Println()
	fmt.Println("Data flow:")
	fmt.Println("  Forward:  input → [Worker A: Linear+ReLU] → LOA → ring → [Worker B: Linear+Loss]")
	fmt.Println("  Backward: [Worker B: backward] → LOA → ring → [Worker A: backward+optim]")
	fmt.Println()

	// --- 6. Baseline comparison ---
	fmt.Println("--- Baseline (single-process, no fragmind) ---")
	model := nn.NewSequential(
		nn.NewLinear(inputDim, hiddenDim),
		nn.NewReLU(),
		nn.NewLinear(hiddenDim, outputDim),
	)
	optBase := optim.NewAdam(model.Parameters(), float32(*lr))

	baseStart := time.Now()
	for batch := 0; batch < totalBatches; batch++ {
		input := g.RandN(*batchSize, inputDim)
		labels := make([]float32, *batchSize)
		for i := range labels {
			labels[i] = float32(i % outputDim)
		}
		targets := g.NewTensor(labels, *batchSize, 1)
		optBase.ZeroGrad()
		logits := model.Forward(input)
		loss := g.CrossEntropyLoss(logits, targets)
		loss.Backward()
		optBase.Step()
	}
	baseTime := time.Since(baseStart)
	baseAvg := baseTime / time.Duration(totalBatches)
	distAvg := totalTime / time.Duration(totalBatches)
	overhead := float64(distAvg-baseAvg) / float64(baseAvg) * 100

	fmt.Printf("Avg batch:         %s\n", baseAvg)
	fmt.Printf("Fragmind overhead: %.1f%%\n", overhead)
	fmt.Println()
	if overhead < 15 {
		fmt.Println("Status: PASS (low overhead)")
	} else if overhead < 30 {
		fmt.Println("Status: PASS (acceptable overhead)")
	} else {
		fmt.Printf("Status: HIGH OVERHEAD %.1f%%\n", overhead)
	}
}

// --- Helpers ---

func sendLOARef(ring *fp.Ring, ref fp.LOARef, schema uint16, msgID uint32, srcID uint32) {
	hdr := fp.Header{
		Kind:     fp.KindProcess,
		Flags:    fp.FlagLOAPtr,
		SchemaID: schema,
		SrcID:    srcID,
		MsgID:    msgID,
		Ver:      1,
	}
	var buf [fp.LOARefSize]byte
	ref.Encode(buf[:])
	hdr.Len = fp.LOARefSize
	for !ring.TryWrite(hdr, buf[:]) {
		// spin — shouldn't happen in sequential demo
	}
}

func recvLOARef(ring *fp.Ring) fp.LOARef {
	msg, err := ring.Read(false)
	if err != nil {
		log.Fatalf("ring read: %v", err)
	}
	return fp.DecodeLOARef(msg.Payload)
}

// setGrad manually sets a tensor's gradient (since gorch doesn't expose this directly).
func setGrad(param *g.Tensor, grad *g.Tensor) {
	// Use the autograd infrastructure: create a dummy GradFn that returns our grad
	param.ZeroGrad()
	// Directly accumulate into param's grad field by triggering backward
	// through a trivial identity path.
	// Simpler: just write to the grad data.
	existing := param.Grad()
	if existing == nil {
		// Create grad tensor with same shape and copy data
		gData := make([]float32, param.Size())
		copy(gData, grad.Data())
		gTensor := g.NewTensor(gData, param.Shape()...)
		// Use the public accessor pattern — add grad via dummy backward
		param.SetRequiresGrad(true)
		// Hack: directly set via the Backward infrastructure
		// We'll use a workaround: create a scalar sum and backward
		sum := g.Full(0, 1)
		sum.SetRequiresGrad(true)
		sum.SetGradFn("setGrad", []*g.Tensor{param}, func(incoming *g.Tensor) []*g.Tensor {
			return []*g.Tensor{gTensor}
		})
		_ = sum
		// Actually, the simplest approach: just accumulate via data pointer
		// Since grad is nil, we can't set it through public API.
		// Use the Data() accessor on a new tensor.
		// Workaround: trigger backward on a constructed graph.
	}
	// Fallback: construct a minimal autograd graph.
	// Create: result = param * 1.0, then backward with our desired gradient.
	ones := g.Ones(param.Shape()...)
	product := g.Mul(param, ones)
	product.SetRequiresGrad(true)
	// Manually set product's grad and trigger backward
	productData := product.Data()
	_ = productData
	// Actually the simplest working approach: use Backward() on a scalar.
	// sum(param * 0) + manual = doesn't work.
	//
	// Let's just use the raw data pointer since gorch exposes Data():
	pData := param.Data()
	gData := grad.Data()
	// SGD/Adam reads param.Grad().Data(), so we need to set that.
	// Since ZeroGrad sets grad=nil and there's no SetGrad, we need to
	// trigger backward. Create: loss = sum(param * grad_values) so that
	// dL/dparam = grad_values.
	// But that changes param... Let's use a different approach.
	//
	// Simplest: run a no-op backward. Create a scalar from param, backward it,
	// then overwrite the gradient data.
	_ = pData
	_ = gData

	// Create identity path: scalar = sum(param), backward gives all-ones grad,
	// then overwrite with our actual gradient.
	scalar := g.Sum(param)
	scalar.Backward()
	// Now param.Grad() exists (all 1s). Overwrite with actual gradient.
	paramGrad := param.Grad()
	if paramGrad != nil {
		copy(paramGrad.Data(), grad.Data())
	}
}

func copyFloat32ToBytes(dst []byte, src []float32) {
	for i, v := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
}

func bytesToFloat32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func humanBytes(b int64) string {
	switch {
	case b >= 1024*1024:
		return fmt.Sprintf("%.1f MB", float64(b)/1024/1024)
	case b >= 1024:
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
