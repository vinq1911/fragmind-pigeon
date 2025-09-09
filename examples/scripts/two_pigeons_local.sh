#!/usr/bin/env bash
set -euo pipefail

# Build QUIC pigeon
go build -tags=quic -o ./bin/pigeon ./cmd/pigeon

# Start A
FM_SITE_ID=1 FM_MODE=quic FM_BIND=:4433 FM_PEERS=127.0.0.1:4434 \
FM_PIGEON_SOCK=/tmp/pigeonA.sock FM_COI_SHM_PATH=/tmp/fragmind.coi.local.A \
./bin/pigeon 2>&1 | sed -e 's/^/[A] /' &
pidA=$!

# Start B
FM_SITE_ID=2 FM_MODE=quic FM_BIND=:4434 FM_PEERS=127.0.0.1:4433 \
FM_PIGEON_SOCK=/tmp/pigeonB.sock FM_COI_SHM_PATH=/tmp/fragmind.coi.local.B \
./bin/pigeon 2>&1 | sed -e 's/^/[B] /' &
pidB=$!

# Start C
FM_SITE_ID=3 FM_MODE=quic FM_BIND=:4435 FM_PEERS=127.0.0.1:4433 \
FM_PIGEON_SOCK=/tmp/pigeonC.sock FM_COI_SHM_PATH=/tmp/fragmind.coi.local.C \
./bin/pigeon 2>&1 | sed -e 's/^/[C] /' &
pidC=$!

sleep 1

# Register a COI on B
FM_PIGEON_SOCK=/tmp/pigeonA.sock go run ./examples/coi_register & reg=$!
# FM_PIGEON_SOCK=/tmp/pigeonB.sock go run ./examples/coi_register & reg=$!
# FM_PIGEON_SOCK=/tmp/pigeonC.sock go run ./examples/coi_register & reg=$!
# Watch A's local table for a bit (should not include remote—local only).
# To verify gossip, you can add a temporary log in ApplyRemote().
FM_COI_SHM_PATH=/tmp/fragmind.coi.local.A go run ./examples/coi_watch

killall -v pigeon