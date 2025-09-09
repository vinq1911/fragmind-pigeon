package main

import (
	"log"
	"os"
	"strconv"
	"strings"

	fp "github.com/vinq1911/fragmind-pigeon/pkg/fragpigeon"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	siteID := uint16(1)
	if s := os.Getenv("FM_SITE_ID"); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			siteID = uint16(v)
		}
	}

	uds := getenv("FM_PIGEON_SOCK", "/tmp/pigeon.sock")
	mode := getenv("FM_MODE", "none") // "quic" or "none"
	bind := getenv("FM_BIND", ":4433")
	peers := []string{}
	if ps := os.Getenv("FM_PEERS"); ps != "" {
		for _, p := range strings.Split(ps, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				peers = append(peers, p)
			}
		}
	}

	p := fp.NewPigeonWithNet(siteID, uds, mode, bind, peers)
	log.Printf("pigeon start site=%d uds=%s mode=%s bind=%s peers=%v", siteID, uds, mode, bind, peers)
	if err := p.Run(); err != nil {
		log.Fatal(err)
	}
}
