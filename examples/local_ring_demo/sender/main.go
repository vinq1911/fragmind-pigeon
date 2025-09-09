package main

import (
	"log"
	"os"
	"strconv"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

const SchemaDemo = 42
const conceptDemo = uint64(0x8A7311CCDD55002A) // valid 64-bit hex

func mustAtoi(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func main() {
	inFD := mustAtoi(os.Getenv("FM_IN_FD"))
	outFD := mustAtoi(os.Getenv("FM_OUT_FD"))

	inRing, err := fp.OpenRingFromFD(inFD)
	if err != nil {
		log.Fatal(err)
	}
	outRing, err := fp.OpenRingFromFD(outFD)
	if err != nil {
		log.Fatal(err)
	}
	defer inRing.Close()
	defer outRing.Close()

	// Optional: COI register (only if your pigeon is running; otherwise StartCOI will error)
	if sock := os.Getenv("FM_PIGEON_SOCK"); sock != "" {
		cois := []fp.COI{{ConceptID: conceptDemo, Bits: 24, SchemaID: SchemaDemo}}
		h, err := fp.StartCOI(fp.COIOptions{SocketPath: sock}, cois)
		if err == nil {
			defer h.Close()
		} else {
			log.Printf("COI register skipped: %v", err)
		}
	}

	// Send pings and wait for replies using ReadWithin (polling, no eventfd needed)
	for i := 0; i < 5; i++ {
		payload := []byte("ping-" + strconv.Itoa(i))
		hdr := fp.Header{
			Len:         uint32(len(payload)),
			Kind:        fp.KindProcess,
			TSns:        uint64(time.Now().UnixNano()),
			ConceptID:   conceptDemo,
			ConceptBits: 24,
			SchemaID:    SchemaDemo,
			SrcID:       1001, MsgID: uint32(i + 1), Ver: 1,
		}
		for !outRing.TryWrite(hdr, payload) {
			time.Sleep(50 * time.Microsecond)
		}

		msg, err := inRing.ReadWithin(2 * time.Second)
		if err != nil {
			log.Fatalf("[sender] no reply: %v", err)
		}
		log.Printf("[sender] got reply: %q", string(msg.Payload))
	}
	log.Println("[sender] done")
}
