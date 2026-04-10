// producer: attaches to pigeon, registers COI, writes weight shards via LOA.
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
	count := flag.Int("n", 5, "number of shards to send")
	size := flag.Int("size", 4096, "shard size in bytes")
	flag.Parse()

	// Attach to pigeon (gets ring pair via SCM_RIGHTS)
	att, err := fp.Attach(*sock)
	if err != nil {
		log.Fatalf("attach: %v", err)
	}
	defer att.Close()
	log.Printf("[producer] attached fragID=%d", att.FragID)

	// Discover LOA pool
	pool, err := fp.DiscoverLOAPool(*sock)
	if err != nil {
		log.Fatalf("discover LOA: %v", err)
	}
	defer pool.Close()
	log.Printf("[producer] LOA pool discovered")

	// Register our COI (we produce model-1 weight shards)
	conceptID := uint64(0x0001000000000000)
	_, err = fp.StartCOI(fp.COIOptions{
		SocketPath: *sock,
		FragID:     uint16(att.FragID),
	}, []fp.COI{{ConceptID: conceptID, Bits: 16, SchemaID: fp.SchemaWeightShard}})
	if err != nil {
		log.Printf("[producer] COI register (non-fatal): %v", err)
	}

	// Send weight shards
	for i := 0; i < *count; i++ {
		shard := makeShard(*size, i)
		shardCRC := crc32.ChecksumIEEE(shard)

		hdr := fp.Header{
			Kind:        fp.KindProcess,
			TSns:        uint64(time.Now().UnixNano()),
			ConceptID:   conceptID,
			ConceptBits: 16,
			SchemaID:    fp.SchemaWeightShard,
			SrcID:       att.FragID,
			MsgID:       uint32(i),
			Ver:         1,
			Checksum32:  shardCRC,
		}

		ref, err := fp.WriteLOAWithBackoff(att.OutRing, pool, hdr, shard, att.FragID, 5*time.Second)
		if err != nil {
			log.Fatalf("[producer] write %d: %v", i, err)
		}
		log.Printf("[producer] sent shard %d/%d (%d bytes, crc=%08x, slot=%d)",
			i+1, *count, *size, shardCRC, ref.SlotID)
		_ = ref
	}
	log.Printf("[producer] done, sent %d shards", *count)
}

func makeShard(size, seed int) []byte {
	buf := make([]byte, size)
	for i := range buf {
		buf[i] = byte((i*31 + seed*7) & 0xFF)
	}
	return buf
}
