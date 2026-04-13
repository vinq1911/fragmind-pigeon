//go:build darwin

// gorch_distributed: Distributed MNIST training via fragmind-pigeon.
//
// Trains a 2-layer MLP on real MNIST data, split across two fragment workers:
//   Worker A: Linear(784→128) + ReLU  (first layer, produces hidden activations)
//   Worker B: Linear(128→10) + Loss   (second layer, produces logits + loss)
//
// Training loop per batch:
//   Forward:  A computes activations → LOA pool → ring → B computes logits + loss
//   Backward: B backprops → gradient via LOA pool → ring → A backprops + optimizer
//
// After training, evaluates on the MNIST test set to verify accuracy.
// Downloads MNIST data on first run (~11 MB).
//
// Usage:
//   CGO_ENABLED=1 go run ./examples/gorch_distributed
//   CGO_ENABLED=1 go run ./examples/gorch_distributed -epochs 5 -lr 0.001
//   CGO_ENABLED=1 go run ./examples/gorch_distributed -data /tmp/mnist  # reuse cached data
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
	"github.com/vinq1911/gorch/data"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/optim"
	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

const (
	inputDim  = 784
	hiddenDim = 128
	outputDim = 10
)

func main() {
	batchSize := flag.Int("batch", 64, "batch size")
	epochs := flag.Int("epochs", 3, "training epochs")
	lr := flag.Float64("lr", 0.001, "learning rate")
	dataDir := flag.String("data", "", "MNIST cache directory (auto if empty)")
	flag.Parse()

	if *dataDir == "" {
		d, err := os.MkdirTemp("", "fragmind-mnist-*")
		if err != nil {
			log.Fatal(err)
		}
		*dataDir = d
	}

	shmDir, err := os.MkdirTemp("", "fragmind-shm-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(shmDir)

	fmt.Println("=== Fragmind Distributed MNIST Training ===")
	fmt.Println()

	// --- 1. Load MNIST ---
	fmt.Println("Loading MNIST...")
	trainSet, err := data.LoadMNIST(*dataDir, true)
	if err != nil {
		log.Fatalf("load train: %v", err)
	}
	testSet, err := data.LoadMNIST(*dataDir, false)
	if err != nil {
		log.Fatalf("load test: %v", err)
	}
	fmt.Printf("Train: %d samples | Test: %d samples\n", trainSet.Len(), testSet.Len())

	// --- 2. Create fragmind infrastructure ---
	testBatchSize := 256
	maxBatch := *batchSize
	if testBatchSize > maxBatch {
		maxBatch = testBatchSize
	}
	maxActivationBytes := maxBatch * hiddenDim * 4
	loaSlotSize := uint32(maxActivationBytes)
	if loaSlotSize < 64*1024 {
		loaSlotSize = 64 * 1024
	}
	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path:     filepath.Join(shmDir, "fragmind.loa"),
		PoolID:   1,
		NumSlots: 128,
		SlotSize: loaSlotSize,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	ringSlotSize := fp.HdrSize + fp.LOARefSize + 16
	fwdRing, fwdCleanup, err := fp.CreateRing(shmDir, "ring-fwd", 256, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer fwdCleanup()
	bwdRing, bwdCleanup, err := fp.CreateRing(shmDir, "ring-bwd", 256, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer bwdCleanup()

	// --- 3. Create model ---
	layerA := nn.NewLinear(inputDim, hiddenDim)
	layerB := nn.NewLinear(hiddenDim, outputDim)
	optA := optim.NewAdam(layerA.Parameters(), float32(*lr))
	optB := optim.NewAdam(layerB.Parameters(), float32(*lr))

	fmt.Printf("Model:  Linear(%d→%d) + ReLU → Linear(%d→%d)\n", inputDim, hiddenDim, hiddenDim, outputDim)
	fmt.Printf("LR: %g | Batch: %d | Epochs: %d\n", *lr, *batchSize, *epochs)
	fmt.Printf("IPC:    LOA pool (%d slots × %d B) + 2 rings (fwd + bwd)\n", 128, loaSlotSize)
	fmt.Println()

	// --- 4. Distributed training ---
	trainLoader := data.NewDataLoader(trainSet, *batchSize, true)
	var totalFwdBytes, totalBwdBytes int64
	var totalBatches int
	trainingStart := time.Now()

	fmt.Println("--- Training (distributed via fragmind) ---")
	for epoch := 0; epoch < *epochs; epoch++ {
		trainLoader.Reset()
		var epochLoss float64
		batches := 0

		for {
			inputs, targets := trainLoader.Next()
			if inputs == nil {
				break
			}
			batch := inputs.Shape()[0]
			optA.ZeroGrad()
			optB.ZeroGrad()

			// == FORWARD: Worker A → LOA → Worker B ==
			hidden := nn.NewReLU().Forward(layerA.Forward(inputs))
			hiddenData := hidden.Data()
			actBytes := batch * hiddenDim * 4

			actBuf, actRef, err := pool.Alloc(uint32(actBytes), 1)
			if err != nil {
				log.Fatalf("LOA alloc fwd: %v", err)
			}
			copyF32ToBytes(actBuf, hiddenData)
			pool.Commit(actRef)
			sendRef(fwdRing, actRef, fp.SchemaActivation, uint32(batches), 1)
			totalFwdBytes += int64(actBytes)

			// Worker B reads, forward, loss
			bRef := recvRef(fwdRing)
			bActBytes, _ := pool.Deref(bRef)
			hiddenB := g.NewTensor(bytesToF32(bActBytes[:actBytes]), batch, hiddenDim)
			hiddenB.SetRequiresGrad(true)
			pool.Release(bRef)

			logits := layerB.Forward(hiddenB)
			loss := g.CrossEntropyLoss(logits, targets)
			epochLoss += float64(loss.Data()[0])

			// == BACKWARD: Worker B → LOA → Worker A ==
			loss.Backward()

			gradData := hiddenB.Grad()
			if gradData == nil {
				log.Fatal("no gradient for hidden")
			}
			gradBuf, gradRef, err := pool.Alloc(uint32(actBytes), 2)
			if err != nil {
				log.Fatalf("LOA alloc bwd: %v", err)
			}
			copyF32ToBytes(gradBuf, gradData.Data())
			pool.Commit(gradRef)
			sendRef(bwdRing, gradRef, fp.SchemaGradient, uint32(batches), 2)
			totalBwdBytes += int64(actBytes)

			optB.Step()

			// Worker A: read gradient, backprop, optimizer
			aGradRef := recvRef(bwdRing)
			aGradBytes, _ := pool.Deref(aGradRef)
			dHidden := bytesToF32(aGradBytes[:actBytes])
			pool.Release(aGradRef)

			backpropWorkerA(layerA, hidden, inputs, dHidden, batch)
			optA.Step()

			batches++
		}
		totalBatches += batches
		avgLoss := epochLoss / float64(batches)
		elapsed := time.Since(trainingStart)
		fmt.Printf("  Epoch %d/%d: loss=%.4f  batches=%d  elapsed=%s\n",
			epoch+1, *epochs, avgLoss, batches, elapsed.Round(time.Millisecond))
	}
	trainingTime := time.Since(trainingStart)

	// --- 5. Evaluate on test set ---
	fmt.Println()
	fmt.Println("--- Evaluation (test set, distributed forward) ---")
	testLoader := data.NewDataLoader(testSet, testBatchSize, false)
	testLoader.Reset()
	correct, total := 0, 0

	for {
		inputs, targets := testLoader.Next()
		if inputs == nil {
			break
		}
		batch := inputs.Shape()[0]

		// Distributed forward: A → LOA → B
		hidden := nn.NewReLU().Forward(layerA.Forward(inputs))
		actBytes := batch * hiddenDim * 4
		actBuf, actRef, err := pool.Alloc(uint32(actBytes), 1)
		if err != nil {
			log.Fatalf("eval LOA alloc: %v (actBytes=%d)", err, actBytes)
		}
		copyF32ToBytes(actBuf, hidden.Data())
		pool.Commit(actRef)
		sendRef(fwdRing, actRef, fp.SchemaActivation, 0, 1)

		bRef := recvRef(fwdRing)
		bActBytes, err := pool.Deref(bRef)
		if err != nil {
			log.Fatalf("eval LOA deref: %v", err)
		}
		hiddenB := g.NewTensor(bytesToF32(bActBytes[:actBytes]), batch, hiddenDim)
		pool.Release(bRef)

		logits := layerB.Forward(hiddenB)
		preds := logits.Data()
		tgts := targets.Data()

		for i := 0; i < batch; i++ {
			maxIdx := 0
			maxVal := preds[i*outputDim]
			for j := 1; j < outputDim; j++ {
				if preds[i*outputDim+j] > maxVal {
					maxVal = preds[i*outputDim+j]
					maxIdx = j
				}
			}
			if maxIdx == int(tgts[i]) {
				correct++
			}
			total++
		}
	}
	accuracy := float64(correct) / float64(total) * 100

	// --- 6. Baseline comparison ---
	fmt.Println()
	fmt.Println("--- Baseline (single-process, no fragmind) ---")
	model := nn.NewSequential(
		nn.NewLinear(inputDim, hiddenDim),
		nn.NewReLU(),
		nn.NewLinear(hiddenDim, outputDim),
	)
	optBase := optim.NewAdam(model.Parameters(), float32(*lr))
	baseLoader := data.NewDataLoader(trainSet, *batchSize, true)

	baseStart := time.Now()
	for epoch := 0; epoch < *epochs; epoch++ {
		baseLoader.Reset()
		for {
			inputs, targets := baseLoader.Next()
			if inputs == nil {
				break
			}
			optBase.ZeroGrad()
			logits := model.Forward(inputs)
			loss := g.CrossEntropyLoss(logits, targets)
			loss.Backward()
			optBase.Step()
		}
	}
	baseTime := time.Since(baseStart)

	// Baseline test accuracy
	baseTestLoader := data.NewDataLoader(testSet, testBatchSize, false)
	baseTestLoader.Reset()
	baseCorrect, baseTotal := 0, 0
	for {
		inputs, targets := baseTestLoader.Next()
		if inputs == nil {
			break
		}
		logits := model.Forward(inputs)
		preds := logits.Data()
		tgts := targets.Data()
		batch := inputs.Shape()[0]
		for i := 0; i < batch; i++ {
			maxIdx := 0
			maxVal := preds[i*outputDim]
			for j := 1; j < outputDim; j++ {
				if preds[i*outputDim+j] > maxVal {
					maxVal = preds[i*outputDim+j]
					maxIdx = j
				}
			}
			if maxIdx == int(tgts[i]) {
				baseCorrect++
			}
			baseTotal++
		}
	}
	baseAccuracy := float64(baseCorrect) / float64(baseTotal) * 100

	// --- 7. Summary ---
	distAvg := trainingTime / time.Duration(totalBatches)
	baseAvg := baseTime / time.Duration(totalBatches)
	overhead := float64(distAvg-baseAvg) / float64(baseAvg) * 100

	fmt.Printf("Baseline time:     %s (%s/batch)\n", baseTime.Round(time.Millisecond), baseAvg)
	fmt.Printf("Baseline accuracy: %.2f%%\n", baseAccuracy)
	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("Distributed:       %d epochs, %d batches, %s total\n",
		*epochs, totalBatches, trainingTime.Round(time.Millisecond))
	fmt.Printf("Batch latency:     %s (distributed) vs %s (baseline)\n", distAvg, baseAvg)
	fmt.Printf("Overhead:          %.1f%%\n", overhead)
	fmt.Printf("Test accuracy:     %.2f%% (distributed) vs %.2f%% (baseline)\n", accuracy, baseAccuracy)
	fmt.Printf("LOA transfers:     %s fwd + %s bwd = %s total\n",
		humanBytes(totalFwdBytes), humanBytes(totalBwdBytes), humanBytes(totalFwdBytes+totalBwdBytes))
	fmt.Println()

	if accuracy >= 90 {
		fmt.Printf("Status: PASS (%.2f%% accuracy, %.1f%% overhead)\n", accuracy, overhead)
	} else if accuracy >= 80 {
		fmt.Printf("Status: ACCEPTABLE (%.2f%% accuracy — more epochs may help)\n", accuracy)
	} else {
		fmt.Printf("Status: NEEDS WORK (%.2f%% accuracy)\n", accuracy)
	}
}

// --- Worker A backward ---

func backpropWorkerA(layer *nn.Linear, hidden, input *g.Tensor, dHiddenFlat []float32, batch int) {
	// ReLU backward: mask where hidden > 0
	hData := hidden.Data()
	dReLU := make([]float32, len(hData))
	for i, v := range hData {
		if v > 0 {
			dReLU[i] = dHiddenFlat[i]
		}
	}

	inData := input.Data()

	// dW = dReLU^T @ input
	dwData := make([]float32, hiddenDim*inputDim)
	for j := 0; j < hiddenDim; j++ {
		for k := 0; k < inputDim; k++ {
			var s float32
			for i := 0; i < batch; i++ {
				s += dReLU[i*hiddenDim+j] * inData[i*inputDim+k]
			}
			dwData[j*inputDim+k] = s
		}
	}

	// db = sum(dReLU, dim=0)
	dbData := make([]float32, hiddenDim)
	for i := 0; i < batch; i++ {
		for j := 0; j < hiddenDim; j++ {
			dbData[j] += dReLU[i*hiddenDim+j]
		}
	}

	// Set gradients via sum+backward trick
	setParamGrad(layer.Weight, dwData)
	setParamGrad(layer.Bias, dbData)
}

func setParamGrad(param *g.Tensor, gradData []float32) {
	// Trigger backward to create the grad tensor, then overwrite with actual gradient
	scalar := g.Sum(param)
	scalar.Backward()
	if param.Grad() != nil {
		copy(param.Grad().Data(), gradData)
	}
}

// --- Ring helpers ---

func sendRef(ring *fp.Ring, ref fp.LOARef, schema uint16, msgID, srcID uint32) {
	var buf [fp.LOARefSize]byte
	ref.Encode(buf[:])
	hdr := fp.Header{
		Len: fp.LOARefSize, Kind: fp.KindProcess, Flags: fp.FlagLOAPtr,
		SchemaID: schema, SrcID: srcID, MsgID: msgID, Ver: 1,
	}
	for !ring.TryWrite(hdr, buf[:]) {
	}
}

func recvRef(ring *fp.Ring) fp.LOARef {
	msg, err := ring.Read(false)
	if err != nil {
		log.Fatalf("ring read: %v", err)
	}
	return fp.DecodeLOARef(msg.Payload)
}

// --- Data helpers ---

func copyF32ToBytes(dst []byte, src []float32) {
	for i, v := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
}

func bytesToF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func humanBytes(b int64) string {
	switch {
	case b >= 1024*1024:
		return fmt.Sprintf("%.1f MB", float64(b)/1024/1024)
	default:
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	}
}
