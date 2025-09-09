package main

import (
	"fmt"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	fmt.Printf("Opening local COI table... %s\n", fp.DefaultCOIShmPath)
	t, err := fp.OpenLocalCOITable("") // uses default path (/dev/shm on Linux, temp-file on macOS)
	if err != nil {
		panic(err)
	}
	defer t.Close()

	ver, upd, entries := t.Snapshot()
	fmt.Printf("COI table v%d updated %s (%d entries):\n", ver, upd.Format(time.RFC3339), len(entries))
	for _, e := range entries {
		fmt.Printf("  concept=0x%x bits=%d schema=%d flags=0x%x\n", e.ConceptID, e.Bits, e.SchemaID, e.Flags)
	}
}
