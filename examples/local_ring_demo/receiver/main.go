package main

import (
	"log"
	"os"
	"strconv"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

const SchemaDemo = 42
const conceptDemo = uint64(0x8A7311CCDD55002A)

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

	// Optional COI register (only if pigeon is running)
	if sock := os.Getenv("FM_PIGEON_SOCK"); sock != "" {
		cois := []fp.COI{{ConceptID: conceptDemo, Bits: 24, SchemaID: SchemaDemo}}
		h, err := fp.StartCOI(fp.COIOptions{SocketPath: sock}, cois)
		if err == nil {
			defer h.Close()
		} else {
			log.Printf("COI register skipped: %v", err)
		}
	}

	for i := 0; i < 5; i++ {
		msg, err := inRing.ReadWithin(10 * time.Second) // poll until message
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("[receiver] got %q", string(msg.Payload))

		reply := []byte("pong-for-" + string(msg.Payload))
		hdr := fp.Header{
			Len:         uint32(len(reply)),
			Kind:        fp.KindProcess,
			TSns:        uint64(time.Now().UnixNano()),
			ConceptID:   conceptDemo,
			ConceptBits: 24,
			SchemaID:    SchemaDemo,
			SrcID:       2002, MsgID: msg.Header.MsgID, Ver: 1,
		}
		for !outRing.TryWrite(hdr, reply) {
			time.Sleep(50 * time.Microsecond)
		}
	}
	log.Println("[receiver] done")
}
