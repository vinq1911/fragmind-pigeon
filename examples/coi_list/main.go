package main

import (
	"fmt"
	"log"
	"os"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	sock := os.Getenv("FM_PIGEON_SOCK")
	if sock == "" {
		sock = "/tmp/pigeon.sock"
	}

	cc, err := func() (*fp.COIHandle, error) {
		// We just need the underlying client; StartCOI wires it up.
		return fp.StartCOI(fp.COIOptions{SocketPath: sock}, nil)
	}()
	if err != nil {
		log.Fatal(err)
	}
	defer cc.Close()

	// Reach inside (a bit hacky) or better add a public List on COIHandle.
	// For now, we reconstruct a client directly:
	cli, err := func() (interface{ List() ([]fp.COI, error) }, error) {
		// private helper: newCOIClient is unexported—feel free to export a List() on COIHandle instead.
		return nil, fmt.Errorf("export a List() on your handle to avoid this hack")
	}()
	_ = cli
	fmt.Println("👉 Quick follow-up: expose List() from COIHandle in fragpigeon to use this example.")
}
