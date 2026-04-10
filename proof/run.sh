#!/usr/bin/env bash
set -euo pipefail

# Fragmind Proof Benchmark Runner
# Runs the full comparative benchmark suite and saves results.
#
# Usage:
#   bash proof/run.sh              # full run
#   bash proof/run.sh --quick      # quick smoke test
#   bash proof/run.sh --bench-only # Go benchmarks only (no JSON report)

cd "$(dirname "$0")/.."

MODE="${1:-full}"

echo "=== Fragmind Proof Benchmark ==="
echo "Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Go:     $(go version)"
echo "OS:     $(uname -s)/$(uname -m)"
echo "CPU:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: //' || echo 'unknown')"
echo ""

case "$MODE" in
  --quick)
    echo "--- Quick smoke test ---"
    go test -v -run TestProofRun -short ./proof/ 2>&1 | tail -5
    echo ""
    echo "--- Quick benchmarks (1s each) ---"
    go test -bench=. -benchmem -benchtime=1x ./proof/ 2>&1
    ;;
  --bench-only)
    echo "--- Go benchmarks (2s each, 3 rounds) ---"
    go test -bench=. -benchmem -benchtime=1x -count=3 ./proof/ 2>&1
    ;;
  *)
    echo "--- Full proof run (JSON report + benchmarks) ---"
    echo ""
    # 1) Run TestProofRun which produces JSON + summary table
    go test -v -run TestProofRun -timeout 600s ./proof/ 2>&1
    echo ""
    # 2) Run Go benchmarks for ns/op and allocs/op
    echo "--- Go benchmark harness ---"
    go test -bench=. -benchmem -benchtime=1x -count=1 ./proof/ 2>&1
    echo ""
    # 3) Show saved results
    LATEST=$(ls -t proof/results/proof-*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
      echo "--- Results saved to: $LATEST ---"
      echo ""
      # Pretty-print key metrics
      python3 -c "
import json, sys
with open('$LATEST') as f:
    r = json.load(f)
print(f\"Timestamp: {r['timestamp']}\")
print(f\"Go: {r['go_version']} | {r['os']}/{r['arch']} | CPUs: {r['num_cpu']}\")
print()
print(f\"{'Transport':<20} {'Payload':<18} {'MB/s':>10} {'Msgs/s':>12} {'P50':>10} {'P99':>10} {'OK':>5}\")
print('─' * 87)
for b in r['results']:
    ok = 'yes' if b['verified'] else 'FAIL'
    print(f\"{b['transport']:<20} {b['payload_name']:<18} {b['throughput_mb_s']:>10.1f} {b['msgs_per_sec']:>12.0f} {b['latency_p50_ns']:>10} {b['latency_p99_ns']:>10} {ok:>5}\")
" 2>/dev/null || echo "(install python3 for pretty JSON output)"
    fi
    ;;
esac

echo ""
echo "Done."
