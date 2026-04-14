//go:build darwin

// gorch_gpt_training: Distributed GPT training via fragmind pipeline parallelism.
//
// Splits a GPT model across two fragment workers and trains on text data:
//   Fragment A: token/pos embedding + transformer blocks 0..N/2-1
//   Fragment B: transformer blocks N/2..N-1 + final norm + LM head + loss
//
// Each training step:
//   Forward:  A → activations via LOA → B → logits → loss
//   Backward: B backward → gradient dL/dhidden via LOA → A backward
//   Update:   Both fragments run optimizer on their own parameters
//
// Downloads Shakespeare text (~1MB) and trains a character-level GPT.
//
// Usage:
//   CGO_ENABLED=1 go run ./examples/gorch_gpt_training
//   CGO_ENABLED=1 go run ./examples/gorch_gpt_training -epochs 3 -dim 128 -layers 4
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/model"
	"github.com/vinq1911/gorch/nn"
	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	dim := flag.Int("dim", 128, "model dimension")
	heads := flag.Int("heads", 4, "attention heads")
	layers := flag.Int("layers", 4, "transformer layers")
	maxSeq := flag.Int("maxseq", 64, "max sequence length")
	epochs := flag.Int("epochs", 2, "training epochs")
	stepsPerEpoch := flag.Int("steps", 200, "steps per epoch")
	lr := flag.Float64("lr", 1e-3, "learning rate")
	flag.Parse()

	dir, err := os.MkdirTemp("", "gorch-gpt-train-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	fmt.Println("=== Fragmind Distributed GPT Training ===")
	fmt.Println()

	// --- 1. Download and tokenize Shakespeare ---
	textPath := filepath.Join(dir, "shakespeare.txt")
	if err := downloadFile(textPath, "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"); err != nil {
		log.Fatalf("download: %v", err)
	}
	textBytes, _ := os.ReadFile(textPath)
	text := string(textBytes)
	tok := model.NewSimpleTokenizer(text)
	encoded := tok.Encode(text)
	vocabSize := tok.VocabSize()
	fmt.Printf("Data:      Shakespeare (%d chars, %d tokens, vocab=%d)\n", len(text), len(encoded), vocabSize)

	// --- 2. Create GPT model ---
	gpt := nn.NewGPT(vocabSize, *dim, *heads, *layers, *maxSeq)
	splitAt := *layers / 2
	fmt.Printf("Model:     GPT (dim=%d, heads=%d, layers=%d, maxseq=%d)\n", *dim, *heads, *layers, *maxSeq)
	fmt.Printf("Split:     A = embed + blocks[0:%d], B = blocks[%d:%d] + norm + head\n", splitAt, splitAt, *layers)
	fmt.Printf("Params:    %d total\n", gpt.CountParameters())

	// Collect parameters for each fragment
	var paramsA, paramsB []*g.Tensor
	paramsA = append(paramsA, gpt.TokenEmbed.Parameters()...)
	paramsA = append(paramsA, gpt.PosEmbed.Parameters()...)
	for i := 0; i < splitAt; i++ {
		paramsA = append(paramsA, gpt.Blocks[i].Parameters()...)
	}
	for i := splitAt; i < *layers; i++ {
		paramsB = append(paramsB, gpt.Blocks[i].Parameters()...)
	}
	paramsB = append(paramsB, gpt.FinalNorm.Parameters()...)
	paramsB = append(paramsB, gpt.LMHead.Parameters()...)

	fmt.Printf("Fragment A: %d params, Fragment B: %d params\n", countParams(paramsA), countParams(paramsB))

	// --- 3. Create fragmind IPC ---
	actSize := *maxSeq * *dim * 4
	loaSlotSize := uint32(actSize)
	if loaSlotSize < 64*1024 {
		loaSlotSize = 64 * 1024
	}
	pool, err := fp.CreateLOAPool(fp.LOAPoolOptions{
		Path: filepath.Join(dir, "fragmind.loa"), PoolID: 1,
		NumSlots: 128, SlotSize: loaSlotSize,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	ringSlotSize := fp.HdrSize + fp.LOARefSize + 16
	fwdRing, fwdCleanup, _ := fp.CreateRing(dir, "ring-fwd", 128, ringSlotSize)
	defer fwdCleanup()
	bwdRing, bwdCleanup, _ := fp.CreateRing(dir, "ring-bwd", 128, ringSlotSize)
	defer bwdCleanup()

	fmt.Printf("IPC:       LOA (%d slots × %d B) + 2 rings (fwd + bwd)\n", 128, loaSlotSize)
	fmt.Printf("Training:  %d epochs × %d steps, lr=%g\n", *epochs, *stepsPerEpoch, *lr)
	fmt.Println()

	// --- 4. Distributed training loop ---
	lrF := float32(*lr)
	var totalFwd, totalBwd int64
	trainStart := time.Now()

	fmt.Println("--- Distributed Training ---")
	for epoch := 0; epoch < *epochs; epoch++ {
		var epochLoss float64
		epochStart := time.Now()

		for step := 0; step < *stepsPerEpoch; step++ {
			// Random sequence from corpus
			start := rand.Intn(len(encoded) - *maxSeq - 1)
			input := encoded[start : start+*maxSeq]
			target := encoded[start+1 : start+*maxSeq+1]
			seqLen := *maxSeq

			// Zero all gradients
			zeroGrads(paramsA)
			zeroGrads(paramsB)

			// === FORWARD: Fragment A ===
			tokEmb := gpt.TokenEmbed.Forward(input)
			posIDs := makeRange(seqLen)
			posEmb := gpt.PosEmbed.Forward(posIDs)
			x := g.Add(tokEmb, posEmb)
			for i := 0; i < splitAt; i++ {
				x = gpt.Blocks[i].Forward(x, seqLen)
			}
			// x is (seqLen, dim) — the hidden state at the split point
			// Mark it as requiring grad so B's backward produces dL/dx
			x.SetRequiresGrad(true)

			// Write activations to LOA
			xData := x.Data()
			actBytes := seqLen * *dim * 4
			actBuf, actRef, err := pool.Alloc(uint32(actBytes), 1)
			if err != nil {
				log.Fatalf("LOA fwd: %v", err)
			}
			copyF32(actBuf, xData)
			pool.Commit(actRef)
			sendRef(fwdRing, actRef, fp.SchemaActivation, uint32(step), 1)
			totalFwd += int64(actBytes)

			// === FORWARD + BACKWARD: Fragment B ===
			bRef := recvRef(fwdRing)
			bData, _ := pool.Deref(bRef)
			xB := g.NewTensor(readF32(bData[:actBytes]), seqLen, *dim)
			xB.SetRequiresGrad(true)
			pool.Release(bRef)

			for i := splitAt; i < *layers; i++ {
				xB = gpt.Blocks[i].Forward(xB, seqLen)
			}
			xB = gpt.FinalNorm.Forward(xB)
			logits := gpt.LMHead.Forward(xB)

			targetF := make([]float32, seqLen)
			for i, t := range target {
				targetF[i] = float32(t)
			}
			loss := g.CrossEntropyLoss(logits, g.NewTensor(targetF, seqLen, 1))
			epochLoss += float64(loss.Data()[0])

			loss.Backward()

			// xB (the tensor B received) now has .Grad() = dL/dhidden
			gradB := xB.Grad()
			if gradB == nil {
				log.Fatal("no gradient at split point")
			}

			// Write gradient to LOA
			gradBuf, gradRef, err := pool.Alloc(uint32(actBytes), 2)
			if err != nil {
				log.Fatalf("LOA bwd: %v", err)
			}
			copyF32(gradBuf, gradB.Data())
			pool.Commit(gradRef)
			sendRef(bwdRing, gradRef, fp.SchemaGradient, uint32(step), 2)
			totalBwd += int64(actBytes)

			// Fragment B optimizer step
			sgdStep(paramsB, lrF)

			// === BACKWARD: Fragment A ===
			aGradRef := recvRef(bwdRing)
			aGradData, _ := pool.Deref(aGradRef)
			dHidden := g.NewTensor(readF32(aGradData[:actBytes]), seqLen, *dim)
			pool.Release(aGradRef)

			// Inject gradient into A's computation graph:
			// dummy_loss = sum(x * dHidden) => dL/d(A's params) via chain rule
			dummyLoss := g.Sum(g.Mul(x, dHidden))
			dummyLoss.Backward()

			// Fragment A optimizer step
			sgdStep(paramsA, lrF)
		}

		avgLoss := epochLoss / float64(*stepsPerEpoch)
		elapsed := time.Since(epochStart)
		fmt.Printf("  Epoch %d/%d: loss=%.4f  time=%s (%.0f steps/s)\n",
			epoch+1, *epochs, avgLoss, elapsed.Round(time.Millisecond),
			float64(*stepsPerEpoch)/elapsed.Seconds())
	}
	trainTime := time.Since(trainStart)
	totalSteps := *epochs * *stepsPerEpoch

	// --- 5. Generate sample text ---
	fmt.Println()
	fmt.Println("--- Sample Generation (distributed forward) ---")
	generated := tok.Encode("ROMEO:\n")
	for i := 0; i < 100; i++ {
		seq := generated
		if len(seq) > *maxSeq {
			seq = seq[len(seq)-*maxSeq:]
		}
		seqLen := len(seq)

		// Fragment A forward
		tokEmb := gpt.TokenEmbed.Forward(seq)
		posEmb := gpt.PosEmbed.Forward(makeRange(seqLen))
		x := g.Add(tokEmb, posEmb)
		for i := 0; i < splitAt; i++ {
			x = gpt.Blocks[i].Forward(x, seqLen)
		}

		// LOA transfer
		actBytes := seqLen * *dim * 4
		actBuf, actRef, _ := pool.Alloc(uint32(actBytes), 1)
		copyF32(actBuf, x.Data())
		pool.Commit(actRef)
		sendRef(fwdRing, actRef, fp.SchemaActivation, 0, 1)

		// Fragment B forward
		bRef := recvRef(fwdRing)
		bData, _ := pool.Deref(bRef)
		xB := g.NewTensor(readF32(bData[:actBytes]), seqLen, *dim)
		pool.Release(bRef)
		for i := splitAt; i < *layers; i++ {
			xB = gpt.Blocks[i].Forward(xB, seqLen)
		}
		xB = gpt.FinalNorm.Forward(xB)
		logits := gpt.LMHead.Forward(xB)

		last := logits.Data()[(seqLen-1)*vocabSize : seqLen*vocabSize]
		generated = append(generated, sampleTopK(last, vocabSize, 8))
	}
	fmt.Println(tok.Decode(generated))

	// --- 6. Baseline comparison ---
	fmt.Println("--- Baseline (single-process, no fragmind) ---")
	baseStart := time.Now()
	for step := 0; step < totalSteps; step++ {
		start := rand.Intn(len(encoded) - *maxSeq - 1)
		input := encoded[start : start+*maxSeq]
		target := encoded[start+1 : start+*maxSeq+1]

		zeroGrads(gpt.Parameters())
		logits := gpt.Forward(input)
		targetF := make([]float32, *maxSeq)
		for i, t := range target {
			targetF[i] = float32(t)
		}
		loss := g.CrossEntropyLoss(logits, g.NewTensor(targetF, *maxSeq, 1))
		loss.Backward()
		sgdStep(gpt.Parameters(), lrF)
	}
	baseTime := time.Since(baseStart)

	distAvg := trainTime / time.Duration(totalSteps)
	baseAvg := baseTime / time.Duration(totalSteps)
	overhead := float64(distAvg-baseAvg) / float64(baseAvg) * 100

	fmt.Printf("Baseline:  %s (%s/step)\n", baseTime.Round(time.Millisecond), baseAvg)
	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("Steps:     %d (%d epochs × %d steps)\n", totalSteps, *epochs, *stepsPerEpoch)
	fmt.Printf("Dist time: %s (%s/step)\n", trainTime.Round(time.Millisecond), distAvg)
	fmt.Printf("Base time: %s (%s/step)\n", baseTime.Round(time.Millisecond), baseAvg)
	fmt.Printf("Overhead:  %.1f%%\n", overhead)
	fmt.Printf("LOA data:  %s fwd + %s bwd = %s total\n",
		humanB(totalFwd), humanB(totalBwd), humanB(totalFwd+totalBwd))
	fmt.Printf("Pipeline:  embed+blocks[0:%d] → LOA → blocks[%d:%d]+norm+head\n", splitAt, splitAt, *layers)
	fmt.Println()
	if overhead < 15 {
		fmt.Println("Status: PASS (low overhead)")
	} else if overhead < 30 {
		fmt.Println("Status: PASS (acceptable)")
	} else {
		fmt.Printf("Status: HIGH OVERHEAD %.1f%%\n", overhead)
	}
}

// --- Helpers ---

func zeroGrads(params []*g.Tensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

func sgdStep(params []*g.Tensor, lr float32) {
	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		d := p.Data()
		gd := grad.Data()
		for i := range d {
			d[i] -= lr * gd[i]
		}
	}
}

func countParams(params []*g.Tensor) int {
	n := 0
	for _, p := range params {
		n += p.Size()
	}
	return n
}

func makeRange(n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = i
	}
	return r
}

func sampleTopK(logits []float32, vocab, k int) int {
	type kv struct{ i int; v float32 }
	top := make([]kv, k)
	for i := range top {
		top[i] = kv{-1, -math.MaxFloat32}
	}
	for i, v := range logits[:vocab] {
		mi := 0
		for j := 1; j < k; j++ {
			if top[j].v < top[mi].v {
				mi = j
			}
		}
		if v > top[mi].v {
			top[mi] = kv{i, v}
		}
	}
	var mx float32 = -math.MaxFloat32
	for _, t := range top {
		if t.v > mx {
			mx = t.v
		}
	}
	var s float32
	ps := make([]float32, k)
	for i, t := range top {
		ps[i] = float32(math.Exp(float64(t.v - mx)))
		s += ps[i]
	}
	r := rand.Float32() * s
	var c float32
	for i, p := range ps {
		c += p
		if r < c {
			return top[i].i
		}
	}
	return top[k-1].i
}

func sendRef(ring *fp.Ring, ref fp.LOARef, schema uint16, msgID, srcID uint32) {
	var buf [fp.LOARefSize]byte
	ref.Encode(buf[:])
	hdr := fp.Header{Len: fp.LOARefSize, Kind: fp.KindProcess, Flags: fp.FlagLOAPtr,
		SchemaID: schema, SrcID: srcID, MsgID: msgID, Ver: 1}
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

func copyF32(dst []byte, src []float32) {
	for i, v := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
}

func readF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func humanB(b int64) string {
	if b >= 1024*1024 {
		return fmt.Sprintf("%.1f MB", float64(b)/1024/1024)
	}
	return fmt.Sprintf("%.1f KB", float64(b)/1024)
}

func downloadFile(path, url string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	fmt.Printf("Downloading %s...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}
