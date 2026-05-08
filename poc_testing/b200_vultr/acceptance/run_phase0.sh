#!/usr/bin/env bash
# Orchestrates Phase 0 (steps 00-60). Step 70 (gpu-burn 24h) runs separately.
# Run from a login/control node.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/results"

NODES="${NODES:-}"   # space-separated; e.g. NODES="b200-01 b200-02"
if [ -z "$NODES" ]; then
  echo "Set NODES=\"<node1> <node2>\" before running." >&2
  exit 1
fi

echo "###### 00 Discovery ######"
for n in $NODES; do
  ssh "$n" "bash -s" <"$HERE/00_discovery.sh"
  scp "$n:$(pwd)/results/discovery_*.txt" "$HERE/results/" 2>/dev/null || true
done

echo "###### 10 DCGM diag (per node, parallel) ######"
for n in $NODES; do
  sbatch -w "$n" "$HERE/10_dcgm_diag.sh"
done

echo "###### 20 nvbandwidth (per node) ######"
for n in $NODES; do
  sbatch -w "$n" "$HERE/20_nvbandwidth.sh"
done

echo "###### 30 Pairwise IB ######"
set -- $NODES
NODE_A="$1" NODE_B="$2" bash "$HERE/30_pairwise_ib.sh"

echo "###### 40 NCCL collectives ######"
sbatch "$HERE/40_nccl_tests.sh"

echo "###### 50 iperf3 external (one node) ######"
ssh "$1" "bash -s" <"$HERE/50_iperf3_external.sh"

echo "###### 60 fio storage ######"
ssh "$1" "bash -s" <"$HERE/60_fio_storage.sh"

echo
echo "Phase 0 (steps 00-60) submitted/run. Check 'squeue' for DCGM/nvbw/NCCL job status."
echo "When all complete, review results/ and populate quirks.md."
echo "Then submit step 70 (gpu-burn 24h) separately:  sbatch $HERE/70_gpu_burn.sh"
