// consumer: attaches to pigeon, subscribes to COI, receives weight shards via LOA.
package main

import (
	"flag"
	"hash/crc32"
	"log"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	sock := flag.String("sock", "/tmp/pigeon.sock", "pigeon socket")
	count := flag.Int("n", 5, "number of shards to receive")
	flag.Parse()

	// Attach to pigeon (gets ring pair via SCM_RIGHTS)
	att, err := fp.Attach(*sock)
	if err != nil {
		log.Fatalf("attach: %v", err)
	}
	defer att.Close()
	log.Printf("[consumer] attached fragID=%d", att.FragID)

	// Discover LOA pool
	pool, err := fp.DiscoverLOAPool(*sock)
	if err != nil {
		log.Fatalf("discover LOA: %v", err)
	}
	defer pool.Close()
	log.Printf("[consumer] LOA pool discovered")

	// Subscribe to model-1 weight shards
	conceptID := uint64(0x0001000000000000)
	coiHandle, err := fp.StartCOI(fp.COIOptions{
		SocketPath: *sock,
		FragID:     uint16(att.FragID),
	}, []fp.COI{{ConceptID: conceptID, Bits: 16, SchemaID: fp.SchemaWeightShard}})
	if err != nil {
		log.Fatalf("[consumer] COI register: %v", err)
	}
	defer coiHandle.Close()
	log.Printf("[consumer] subscribed to concept=%x bits=16", conceptID)

	// Receive weight shards (poll mode — rings don't have eventfd)
	verified := 0
	for i := 0; i < *count; i++ {
		var msg fp.Msg
		var data []byte
		var ref fp.LOARef
		var err error

		deadline := time.Now().Add(30 * time.Second)
		for time.Now().Before(deadline) {
			msg, data, ref, err = fp.ReadLOA(att.InRing, pool, false)
			if err == nil {
				break
			}
			time.Sleep(1 * time.Millisecond)
		}
		if err != nil {
			log.Fatalf("[consumer] read %d: timeout", i)
		}

		gotCRC := crc32.ChecksumIEEE(data)
		match := gotCRC == msg.Header.Checksum32
		if match {
			verified++
		}

		log.Printf("[consumer] received shard %d/%d (%d bytes, crc=%08x expected=%08x ok=%v from=%d)",
			i+1, *count, len(data), gotCRC, msg.Header.Checksum32, match, msg.Header.SrcID)

		if ref != (fp.LOARef{}) {
			pool.Release(ref)
		}
	}

	log.Printf("[consumer] done: %d/%d verified", verified, *count)
	if verified != *count {
		log.Fatal("[consumer] FAIL: data corruption detected")
	}
}
