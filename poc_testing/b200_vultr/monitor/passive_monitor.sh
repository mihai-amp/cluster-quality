#!/usr/bin/env bash
# Passive cluster monitor - runs continuously across the entire benchmark window.
# Captures dmesg XID/SXID, dcgm health snapshot every 5 min, and per-node nvidia-smi.
# Run on a login node or a dedicated control host; ssh into B200 nodes.
#
# Usage:
#   NODES="b200-01 b200-02" ./passive_monitor.sh &
#   # ... run benchmarks ...
#   # ctrl-c when done

set -uo pipefail
HERE="$(dirname "$0")"
OUT_DIR="$HERE/results/passive_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

NODES="${NODES:?set NODES to space-separated node list}"
INTERVAL="${INTERVAL:-300}"  # 5 min

echo "Passive monitor starting; output to $OUT_DIR"
echo "Interval: ${INTERVAL}s"
echo "Nodes: $NODES"
echo

trap 'echo "Stopping monitor"; exit 0' INT TERM

while true; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  for n in $NODES; do
    {
      echo "===== $ts on $n ====="
      ssh -n "$n" "
        echo '--- nvidia-smi snapshot ---'
        nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used --format=csv
        echo '--- dmesg recent (XID/SXID/throttle) ---'
        dmesg -T --since '5 minutes ago' 2>/dev/null | grep -iE 'xid|sxid|throttle|nvidia|nvlink|infiniband|mlx' || echo 'clean'
        echo '--- dcgmi health ---'
        dcgmi health -c 2>/dev/null | tail -20 || true
        echo '--- ib counters ---'
        for h in /sys/class/infiniband/*/ports/1/counters/*; do
          name=\$(basename \"\$h\")
          val=\$(cat \"\$h\")
          echo \"\$h \$val\"
        done 2>/dev/null | grep -E 'symbol_error|link_downed|port_xmit_discard|port_rcv_errors' | head -20
      "
    } >>"$OUT_DIR/passive_$n.log" 2>&1
  done
  sleep "$INTERVAL"
done
