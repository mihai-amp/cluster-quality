#!/usr/bin/env bash
#SBATCH --job-name=phase0_a2a_c10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_alltoall_intra_4G_cycles10_%j.log
# Single-node 8-GPU NCCL alltoall, 4 GiB fixed, 10 cycles via -N (reuse comm).
# Avoids the per-rep bootstrap overhead that made the loop-based version slow.

set -uo pipefail

IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
BIN=alltoall_perf   # on PATH inside the NeMo image
SIZE_BYTES="${SIZE_BYTES:-4294967296}"   # 4 GiB
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"
CYCLES="${CYCLES:-10}"

echo "==== intra-node alltoall, 8 GPUs, 4 GiB, run_cycles=$CYCLES on $(hostname) at $(date -u) ===="
echo

srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 \
     --container-image="$IMAGE" \
     --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai \
     --mpi=pmix \
     --export=ALL \
     "$BIN" -b "$SIZE_BYTES" -e "$SIZE_BYTES" -f 2 -g 1 \
            -n "$ITERS" -w "$WARMUP" -N "$CYCLES" -c 0

echo
echo "==== per-cycle 4 GiB busbw (in-place algbw/busbw col 7/8, out-of-place 11/12) ===="
awk '
  /^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(uint8|float)/ {
    n++
    printf "cycle %2d:  in-place algbw=%.2f busbw=%.2f   out-of-place algbw=%.2f busbw=%.2f GB/s\n",
           n, $7, $8, $11, $12
  }
' /mnt/vfs/mihai/results/nccl_alltoall_intra_4G_cycles10_${SLURM_JOB_ID}.log

echo
echo "==== aggregate (out-of-place busbw across $CYCLES cycles) ===="
awk '
  /^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(uint8|float)/ {
    bw[++n] = $12
    sum += $12
    if (n == 1 || $12 < min) min = $12
    if (n == 1 || $12 > max) max = $12
  }
  END {
    if (n == 0) { print "no data"; exit }
    mean = sum / n
    for (i = 1; i <= n; i++) { d = bw[i] - mean; ssd += d * d }
    sd = sqrt(ssd / n)
    printf "  n=%d   mean=%.2f   stddev=%.3f (%.2f%%)   min=%.2f   max=%.2f   GB/s\n",
           n, mean, sd, 100*sd/mean, min, max
  }
' /mnt/vfs/mihai/results/nccl_alltoall_intra_4G_cycles10_${SLURM_JOB_ID}.log
