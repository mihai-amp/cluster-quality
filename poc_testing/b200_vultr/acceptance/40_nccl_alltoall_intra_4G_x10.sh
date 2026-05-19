#!/usr/bin/env bash
#SBATCH --job-name=phase0_a2a_intra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_alltoall_intra_4G_x10_%j.log
# Single-node 8-GPU NCCL alltoall at fixed 4 GiB message size, 10 separate
# invocations (each its own MPI bootstrap), for variance characterisation
# of intra-node NVLink5 alltoall bus bandwidth.

set -uo pipefail

IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
BIN=alltoall_perf   # on PATH inside the NeMo image
SIZE_BYTES="${SIZE_BYTES:-4294967296}"   # 4 GiB
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"
REPS="${REPS:-10}"

echo "==== intra-node alltoall, 8 GPUs, 4 GiB, $REPS repetitions on $(hostname) at $(date -u) ===="
echo "NCCL_DEBUG=INFO on rep 1 (algo selection), WARN thereafter."
echo

for rep in $(seq 1 "$REPS"); do
  echo "######## rep $rep ########"
  # Show algo/proto on first rep; quieter afterwards
  if [ "$rep" = "1" ]; then DEBUG="INFO"; else DEBUG="WARN"; fi
  srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 \
       --container-image="$IMAGE" \
       --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai \
       --mpi=pmix \
       --export=ALL,NCCL_DEBUG=$DEBUG,NCCL_DEBUG_SUBSYS=INIT \
       "$BIN" -b "$SIZE_BYTES" -e "$SIZE_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 0
  echo
done

echo "==== per-rep summary ===="
awk '
  /^######## rep / { rep=$2 }
  /^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(uint8|float)/ {
    # In-place: $6=time $7=algbw $8=busbw   Out-of-place: $10=time $11=algbw $12=busbw
    printf "rep %s  in-place algbw=%.2f busbw=%.2f   out-of-place algbw=%.2f busbw=%.2f GB/s\n",
           rep, $7, $8, $11, $12
  }
' /mnt/vfs/mihai/results/nccl_alltoall_intra_4G_x10_${SLURM_JOB_ID}.log

echo
echo "==== aggregate (out-of-place busbw across $REPS reps) ===="
awk '
  /^######## rep / { rep=$2 }
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
    printf "  n=%d   mean=%.2f   stddev=%.2f (%.1f%%)   min=%.2f   max=%.2f   GB/s\n",
           n, mean, sd, 100*sd/mean, min, max
  }
' /mnt/vfs/mihai/results/nccl_alltoall_intra_4G_x10_${SLURM_JOB_ID}.log
