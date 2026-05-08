#!/usr/bin/env bash
# Pairwise InfiniBand bandwidth & latency between the two nodes.
# Tests every HCA (expect 8 NDR ConnectX-7 per HGX B200 node).
# Run from a login/control node; uses ssh to drive perftest on each B200 node.
#
# Prereqs: perftest package on both nodes; passwordless ssh between login and B200 nodes.
# Override NODE_A / NODE_B / HCAS as needed.

set -uo pipefail

NODE_A="${NODE_A:?set NODE_A=<hostname of node 1>}"
NODE_B="${NODE_B:?set NODE_B=<hostname of node 2>}"
# Default HCA list - adjust to match what 00_discovery showed:
HCAS="${HCAS:-mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7}"
PORT_BASE="${PORT_BASE:-18515}"
DURATION="${DURATION:-30}"

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/pairwise_ib_$(date -u +%Y%m%dT%H%M%SZ).log"

idx=0
for hca in $HCAS; do
  port=$((PORT_BASE + idx))

  for tool in ib_write_bw ib_write_lat; do
    echo "==== $tool on $hca (server=$NODE_A, client=$NODE_B), port $port ====" | tee -a "$OUT"

    ssh "$NODE_A" "$tool -d $hca -p $port -D $DURATION -F" >"$OUT.$NODE_A.$hca.$tool" 2>&1 &
    server_pid=$!
    sleep 2
    ssh "$NODE_B" "$tool -d $hca -p $port -D $DURATION -F $NODE_A" >"$OUT.$NODE_B.$hca.$tool" 2>&1
    wait "$server_pid" 2>/dev/null || true

    echo "--- server log ---" | tee -a "$OUT"
    cat "$OUT.$NODE_A.$hca.$tool" | tee -a "$OUT"
    echo "--- client log ---" | tee -a "$OUT"
    cat "$OUT.$NODE_B.$hca.$tool" | tee -a "$OUT"
    echo | tee -a "$OUT"
  done

  idx=$((idx + 1))
done

echo "Wrote $OUT"
echo
echo "Sanity check: each ib_write_bw run on NDR (400 Gb/s) should show ~46-49 GB/s sustained"
echo "ib_write_lat p50 should be < 2 microseconds for ConnectX-7 NDR within rack"
