package main

import (
	"log"
	"os"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	sock := os.Getenv("FM_PIGEON_SOCK")
	if sock == "" {
		sock = "/tmp/pigeon.sock"
	}

	cois := []fp.COI{{ConceptID: 0x8A7311CCDD55002A, Bits: 24, SchemaID: 42}}
	h, err := fp.StartCOI(fp.COIOptions{SocketPath: sock, RenewEvery: time.Second}, cois)
	if err != nil {
		log.Fatalf("register failed: %v", err)
	}
	defer h.Close()

	log.Println("registered; sleeping 20s so you can inspect...")
	time.Sleep(20 * time.Second)
}
