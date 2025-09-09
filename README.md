# fragmind-pigeon

COI-aware, ultra-fast IPC + cluster routing for fragmind fragments.
Local delivery via shared-memory SPSC rings; cross-host via pigeons.

## Install

```bash
go get github.com/vinq1911/fragmind-pigeon@latest
```

## Run the pigeon

```bash
FM_SITE_ID=1 FM_PIGEON_SOCK=/tmp/pigeon.sock ./pigeon
````

## Fragment usage

```go
import fp "github.com/yourname/fragmind-pigeon/pkg/fragpigeon"

cois := []fp.COI{{ConceptID: 0x8A7311CCDD55002A, Bits: 28, SchemaID: 1001}}
h, _ := fp.StartCOI(fp.COIOptions{}, cois)
defer h.Close()

in,  _ := fp.OpenRingFromFD(inFD)
out, _ := fp.OpenRingFromFD(outFD)

msg, _ := in.Read(true)
// process...
payload := []byte("ok")
hdr := fp.Header{Len:uint32(len(payload)), Kind:fp.KindProcess, ConceptID:0x1234, ConceptBits:24, SchemaID:1}
for !out.TryWrite(hdr, payload) {}
```

---

## Notes

- Pigeon exposes a local COI directory via /dev/shm/fragmind.coi.local.
- Fragments register/renew COIs over /tmp/pigeon.sock.
- Extend with QUIC peer links in gossip.go and routing in router.go.

---

### How to use in your fragment repo

1) Publish this repo (e.g., GitHub).  
2) In your fragment:

```bash
go get github.com/yourname/fragmind-pigeon@latest
````

### Wire your fragment to:

- receive ring FDs via env, OpenRingFromFD
- start auto-renew with StartCOI
- write/read messages with Header + Ring.
