package main

import (
	"fmt"
	"os"
	"time"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func main() {
	path := os.Getenv("FM_COI_SHM_PATH") // recommend setting this
	if path == "" {
		path = "/tmp/fragmind.coi.local"
	}

	t, err := fp.OpenLocalCOITable(path)
	if err != nil {
		panic(err)
	}
	defer t.Close()

	for i := 0; i < 10; i++ {
		ver, upd, es := t.Snapshot()
		fmt.Printf("v%d @%s (%d entries)\n", ver, upd.Format(time.RFC3339), len(es))
		for _, e := range es {
			fmt.Printf("  concept=0x%x bits=%d schema=%d flags=0x%x\n", e.ConceptID, e.Bits, e.SchemaID, e.Flags)
		}
		time.Sleep(2 * time.Second)
	}
}
