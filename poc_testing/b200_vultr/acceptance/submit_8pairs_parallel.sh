#!/usr/bin/env bash
# Submit 8 cross-node NCCL pair tests in parallel as independent sbatch jobs.
# Run on the controller. Outputs: /mnt/vfs/mihai/results/8pairs_parallel_<timestamp>/
set -uo pipefail

SCRIPT_DIR="$(dirname "$0")"
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR=/mnt/vfs/mihai/results/8pairs_parallel_${TS}
mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"

declare -a JOBIDS
for i in 0 1 2 3 4 5 6 7; do
  jid=$(PAIR_IDX=$i OUT=$OUT_DIR/pair${i}.log \
        sbatch --export=PAIR_IDX,OUT \
               --output=$OUT_DIR/sbatch_pair${i}_%j.out \
               "$SCRIPT_DIR/40_nccl_1pair_inter.sh" | awk '{print $NF}')
  JOBIDS+=("$jid")
  echo "submitted pair $i: job $jid"
done

echo
echo "JOBIDS: ${JOBIDS[*]}"
echo "Watch: squeue -u \$USER  |  tail -F $OUT_DIR/pair*.log"
