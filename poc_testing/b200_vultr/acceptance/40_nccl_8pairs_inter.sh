#!/usr/bin/env bash
#SBATCH --job-name=phase0_8pairs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_8pairs_%j.log
# 8 parallel 2-rank NCCL alltoall tests; each pair spans gpu01 ↔ gpu02 so all
# traffic is forced over IB by topology (no NVLink possible — only 1 GPU per
# node in each pair). 8 pairs run concurrently to stress all 8 NDR HCAs.

set -uo pipefail

IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
BIN=alltoall_perf   # on PATH inside the NeMo image
MIN_BYTES="${MIN_BYTES:-16777216}"
MAX_BYTES="${MAX_BYTES:-8589934592}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"

OUT_DIR=/mnt/vfs/mihai/results/8pairs_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

echo "==== 8 parallel cross-node alltoall pairs on $(scontrol show hostname "$SLURM_NODELIST" | tr '\n' ' ') at $(date -u) ===="
echo "Per pair: 2 ranks, 1 GPU on each node. Range $MIN_BYTES..$MAX_BYTES bytes."
echo "Each pair binds CUDA_VISIBLE_DEVICES=<pair_idx> on both nodes."
echo

for i in 0 1 2 3 4 5 6 7; do
  (
    srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 \
         --cpus-per-task=4 \
         --container-image="$IMAGE" \
         --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai \
         --mpi=pmix \
         --export=ALL,NCCL_IB_HCA=mlx5 \
         bash -c "export CUDA_VISIBLE_DEVICES=$i; exec \"\$@\"" _ \
            "$BIN" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 0 \
       > "$OUT_DIR/pair${i}.log" 2>&1
  ) &
done

wait
echo
echo "==== all 8 pairs complete ===="
ls -lh "$OUT_DIR"
echo
echo "==== per-pair peak busbw (max-size, out-of-place) ===="
for i in 0 1 2 3 4 5 6 7; do
  peak=$(awk '/^[[:space:]]+[0-9]+[[:space:]]+[0-9]+/ {if ($12+0 > p) p=$12} END {print p}' "$OUT_DIR/pair${i}.log")
  avg=$(awk '/Avg bus bandwidth/ {print $NF}' "$OUT_DIR/pair${i}.log")
  printf "  pair %d (CUDA_VISIBLE_DEVICES=%d): peak_busbw=%s GB/s  avg=%s GB/s\n" "$i" "$i" "${peak:-?}" "${avg:-?}"
done
