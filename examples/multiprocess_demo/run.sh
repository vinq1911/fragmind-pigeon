#!/usr/bin/env bash
set -euo pipefail

# Multi-process demo: pigeon daemon + producer + consumer as separate processes.
# Producer writes weight shards via LOA, consumer reads them zero-copy,
# all routed through the pigeon daemon via COI subscriptions.
#
# Usage:
#   bash examples/multiprocess_demo/run.sh
#   bash examples/multiprocess_demo/run.sh --count 20 --size 65536

cd "$(dirname "$0")/../.."

COUNT="${1:-5}"
SIZE="${2:-4096}"
SOCK="/tmp/fp-demo-$$.sock"
COI_SHM="/tmp/fp-demo-$$.coi"
LOA_SHM="/tmp/fp-demo-$$.loa"

cleanup() {
    kill "$PID_PIGEON" "$PID_CONSUMER" 2>/dev/null || true
    rm -f "$SOCK" "$COI_SHM" "$LOA_SHM"
}
trap cleanup EXIT

echo "=== Fragmind Multi-Process Demo ==="
echo "Pigeon socket: $SOCK"
echo "Shard count:   $COUNT"
echo "Shard size:    $SIZE bytes"
echo ""

# 1. Start pigeon daemon
FM_SITE_ID=1 FM_PIGEON_SOCK="$SOCK" FM_COI_SHM_PATH="$COI_SHM" FM_LOA_SHM_PATH="$LOA_SHM" \
    go run ./cmd/pigeon 2>&1 | sed 's/^/[pigeon] /' &
PID_PIGEON=$!
sleep 1

# 2. Start consumer (waits for messages)
go run ./examples/multiprocess_demo/consumer -sock "$SOCK" -n "$COUNT" 2>&1 | sed 's/^/[consumer] /' &
PID_CONSUMER=$!
sleep 0.5

# 3. Run producer (sends messages, then exits)
go run ./examples/multiprocess_demo/producer -sock "$SOCK" -n "$COUNT" -size "$SIZE" 2>&1 | sed 's/^/[producer] /'

# 4. Wait for consumer to finish
wait "$PID_CONSUMER"
CONSUMER_EXIT=$?

echo ""
if [ "$CONSUMER_EXIT" -eq 0 ]; then
    echo "=== DEMO PASSED ==="
else
    echo "=== DEMO FAILED (consumer exit=$CONSUMER_EXIT) ==="
    exit 1
fi
