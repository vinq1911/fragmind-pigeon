//go:build darwin

// gorch_gpt_distributed: Distributed GPT inference via fragmind pipeline parallelism.
//
// Splits a GPT model across two fragment workers:
//   Fragment A: token/pos embedding + transformer blocks 0..N/2-1
//   Fragment B: transformer blocks N/2..N-1 + final norm + LM head
//
// Activations flow A→B through fragmind's LOA pool (zero-copy shared memory).
// Text is generated autoregressively: each token prediction requires one
// forward pass through both fragments.
//
// Usage:
//   CGO_ENABLED=1 go run ./examples/gorch_gpt_distributed
//   CGO_ENABLED=1 go run ./examples/gorch_gpt_distributed -layers 8 -dim 256 -prompt "The"
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	g "github.com/vinq1911/gorch"
	"github.com/vinq1911/gorch/nn"
	"github.com/vinq1911/gorch/model"
	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	dim := flag.Int("dim", 128, "model dimension")
	heads := flag.Int("heads", 4, "number of attention heads")
	layers := flag.Int("layers", 4, "number of transformer layers")
	maxSeq := flag.Int("maxseq", 64, "max sequence length")
	genLen := flag.Int("gen", 32, "tokens to generate")
	prompt := flag.String("prompt", "The quick brown fox", "prompt text")
	trainSteps := flag.Int("train", 50, "pre-training steps on sample text")
	flag.Parse()

	dir, err := os.MkdirTemp("", "gorch-gpt-fragmind-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	fmt.Println("=== Fragmind Distributed GPT Demo ===")
	fmt.Println()

	// --- 1. Build tokenizer from sample text ---
	sampleText := "The quick brown fox jumps over the lazy dog. " +
		"A fast red car drives down the long winding road. " +
		"The sun sets behind the tall green mountains. " +
		"Birds fly high above the deep blue ocean waves."
	tok := model.NewSimpleTokenizer(sampleText + " " + *prompt)
	vocabSize := tok.VocabSize()
	fmt.Printf("Tokenizer: %d chars in vocab\n", vocabSize)

	// --- 2. Create GPT model ---
	gpt := nn.NewGPT(vocabSize, *dim, *heads, *layers, *maxSeq)
	splitAt := *layers / 2

	fmt.Printf("Model:     GPT (vocab=%d, dim=%d, heads=%d, layers=%d, maxseq=%d)\n",
		vocabSize, *dim, *heads, *layers, *maxSeq)
	fmt.Printf("Split:     Fragment A = embed + blocks[0:%d], Fragment B = blocks[%d:%d] + norm + head\n",
		splitAt, splitAt, *layers)
	fmt.Printf("Params:    %d total\n", gpt.CountParameters())
	fmt.Println()

	// --- 3. Quick pre-training so the model isn't pure random ---
	if *trainSteps > 0 {
		fmt.Printf("Pre-training on sample text (%d steps)...\n", *trainSteps)
		pretrain(gpt, tok, sampleText, *trainSteps)
		fmt.Println()
	}

	// --- 4. Create fragmind LOA pool + ring ---
	activationBytes := *maxSeq * *dim * 4 // worst case: maxSeq tokens × dim float32
	loaSlotSize := uint32(activationBytes)
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

	ringSlotSize := fp.HdrSize + fp.LOARefSize + 16
	fwdRing, fwdCleanup, err := fp.CreateRing(dir, "ring-fwd", 64, ringSlotSize)
	if err != nil {
		log.Fatal(err)
	}
	defer fwdCleanup()

	fmt.Printf("IPC:       LOA pool (%d slots × %d B) + ring\n", 64, loaSlotSize)
	fmt.Println()

	// --- 5. Distributed autoregressive generation ---
	fmt.Println("--- Distributed Generation (pipeline parallel) ---")
	fmt.Printf("Prompt:    %q\n", *prompt)

	promptIDs := tok.Encode(*prompt)
	generated := make([]int, len(promptIDs))
	copy(generated, promptIDs)

	var totalLOABytes int64
	genStart := time.Now()

	for step := 0; step < *genLen; step++ {
		seqLen := len(generated)
		if seqLen > *maxSeq {
			generated = generated[seqLen-*maxSeq:]
			seqLen = *maxSeq
		}

		// === Fragment A: embedding + first half of blocks ===
		tokEmb := gpt.TokenEmbed.Forward(generated)
		posIDs := make([]int, seqLen)
		for i := range posIDs {
			posIDs[i] = i
		}
		posEmb := gpt.PosEmbed.Forward(posIDs)
		x := g.Add(tokEmb, posEmb)

		for i := 0; i < splitAt; i++ {
			x = gpt.Blocks[i].Forward(x, seqLen)
		}

		// Write activations to LOA
		xData := x.Data()
		actBytes := seqLen * *dim * 4
		actBuf, actRef, err := pool.Alloc(uint32(actBytes), 1)
		if err != nil {
			log.Fatalf("LOA alloc: %v", err)
		}
		copyF32ToBytes(actBuf, xData)
		pool.Commit(actRef)
		sendRef(fwdRing, actRef, fp.SchemaActivation, uint32(step), 1)
		totalLOABytes += int64(actBytes)

		// === Fragment B: second half of blocks + norm + head ===
		bRef := recvRef(fwdRing)
		bData, err := pool.Deref(bRef)
		if err != nil {
			log.Fatalf("LOA deref: %v", err)
		}
		xB := g.NewTensor(bytesToF32(bData[:actBytes]), seqLen, *dim)
		pool.Release(bRef)

		for i := splitAt; i < *layers; i++ {
			xB = gpt.Blocks[i].Forward(xB, seqLen)
		}
		xB = gpt.FinalNorm.Forward(xB)
		logits := gpt.LMHead.Forward(xB) // (seqLen, vocabSize)

		// Sample next token from last position
		lastLogits := logits.Data()[(seqLen-1)*vocabSize : seqLen*vocabSize]
		nextToken := sampleTopK(lastLogits, vocabSize, 5)
		generated = append(generated, nextToken)
	}

	genTime := time.Since(genStart)
	outputText := tok.Decode(generated)

	fmt.Printf("Output:    %q\n", outputText)
	fmt.Printf("Tokens:    %d generated in %s (%.1f tok/s)\n",
		*genLen, genTime.Round(time.Millisecond), float64(*genLen)/genTime.Seconds())
	fmt.Printf("LOA data:  %s (%d transfers)\n", humanBytes(totalLOABytes), *genLen)
	fmt.Printf("Pipeline:  [embed+blocks 0:%d] → LOA → [blocks %d:%d + norm + head]\n",
		splitAt, splitAt, *layers)
	fmt.Println()

	// --- 6. Baseline comparison (no fragmind) ---
	fmt.Println("--- Baseline (single-process, no fragmind) ---")
	baseGenerated := make([]int, len(promptIDs))
	copy(baseGenerated, promptIDs)

	baseStart := time.Now()
	for step := 0; step < *genLen; step++ {
		seqLen := len(baseGenerated)
		if seqLen > *maxSeq {
			baseGenerated = baseGenerated[seqLen-*maxSeq:]
		}
		logits := gpt.Forward(baseGenerated)
		sl := len(baseGenerated)
		lastLogits := logits.Data()[(sl-1)*vocabSize : sl*vocabSize]
		nextToken := sampleTopK(lastLogits, vocabSize, 5)
		baseGenerated = append(baseGenerated, nextToken)
	}
	baseTime := time.Since(baseStart)

	overhead := float64(genTime-baseTime) / float64(baseTime) * 100
	fmt.Printf("Tokens:    %d in %s (%.1f tok/s)\n",
		*genLen, baseTime.Round(time.Millisecond), float64(*genLen)/baseTime.Seconds())
	fmt.Printf("Overhead:  %.1f%%\n", overhead)
	fmt.Println()

	if overhead < 15 {
		fmt.Println("Status: PASS (low overhead)")
	} else if overhead < 30 {
		fmt.Println("Status: PASS (acceptable overhead)")
	} else {
		fmt.Printf("Status: HIGH OVERHEAD %.1f%%\n", overhead)
	}
}

// --- Pre-training ---

func pretrain(gpt *nn.GPT, tok *model.SimpleTokenizer, text string, steps int) {
	ids := tok.Encode(text)
	seqLen := 32
	if seqLen > len(ids)-1 {
		seqLen = len(ids) - 1
	}

	// Simple SGD-like update
	lr := float32(0.01)
	for step := 0; step < steps; step++ {
		start := rand.Intn(len(ids) - seqLen - 1)
		input := ids[start : start+seqLen]
		target := ids[start+1 : start+seqLen+1]

		// Zero grads
		for _, p := range gpt.Parameters() {
			p.ZeroGrad()
		}

		logits := gpt.Forward(input)
		targetF := make([]float32, seqLen)
		for i, t := range target {
			targetF[i] = float32(t)
		}
		targetT := g.NewTensor(targetF, seqLen, 1)
		loss := g.CrossEntropyLoss(logits, targetT)
		loss.Backward()

		// SGD step
		for _, p := range gpt.Parameters() {
			grad := p.Grad()
			if grad == nil {
				continue
			}
			data := p.Data()
			gData := grad.Data()
			for i := range data {
				data[i] -= lr * gData[i]
			}
		}

		if step == 0 || step == steps-1 {
			fmt.Printf("  step %d/%d loss=%.4f\n", step+1, steps, loss.Data()[0])
		}
	}
}

// --- Helpers ---

func sampleTopK(logits []float32, vocabSize, k int) int {
	// Find top-k indices
	type kv struct {
		idx int
		val float32
	}
	topk := make([]kv, k)
	for i := 0; i < k; i++ {
		topk[i] = kv{-1, -math.MaxFloat32}
	}
	for i, v := range logits[:vocabSize] {
		minIdx := 0
		for j := 1; j < k; j++ {
			if topk[j].val < topk[minIdx].val {
				minIdx = j
			}
		}
		if v > topk[minIdx].val {
			topk[minIdx] = kv{i, v}
		}
	}

	// Softmax over top-k
	maxVal := float32(-math.MaxFloat32)
	for _, t := range topk {
		if t.val > maxVal {
			maxVal = t.val
		}
	}
	var sumExp float32
	probs := make([]float32, k)
	for i, t := range topk {
		probs[i] = float32(math.Exp(float64(t.val - maxVal)))
		sumExp += probs[i]
	}
	for i := range probs {
		probs[i] /= sumExp
	}

	// Sample
	r := rand.Float32()
	var cumProb float32
	for i, p := range probs {
		cumProb += p
		if r < cumProb {
			return topk[i].idx
		}
	}
	return topk[k-1].idx
}

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

// suppress unused
var _ = strings.TrimSpace
