#!/usr/bin/env bash
#SBATCH --job-name=phase0_pair
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
# One 2-rank cross-node NCCL alltoall, pinned to GPU index $PAIR_IDX.
# Submit 8 of these in parallel (different PAIR_IDX each) — slurm shares
# nodes across jobs at the GPU level so they run concurrently.
# Submitter must export PAIR_IDX (0..7) and OUT (path for log).

set -uo pipefail

PAIR_IDX="${PAIR_IDX:?must set PAIR_IDX 0..7}"
OUT="${OUT:-/mnt/vfs/mihai/results/nccl_pair${PAIR_IDX}_${SLURM_JOB_ID}.log}"

IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
BIN=alltoall_perf   # on PATH inside the NeMo image
MIN_BYTES="${MIN_BYTES:-16777216}"
MAX_BYTES="${MAX_BYTES:-8589934592}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"

{
  echo "==== pair $PAIR_IDX  (GPU index $PAIR_IDX on both nodes) at $(date -u) ===="
  echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
  srun --overlap --container-image="$IMAGE" \
       --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai \
       --mpi=pmix \
       --export=ALL,NCCL_IB_HCA=mlx5 \
       bash -c "export CUDA_VISIBLE_DEVICES=$PAIR_IDX; exec \"\$@\"" _ \
          "$BIN" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 0
  echo
  echo "==== pair $PAIR_IDX done ===="
} > "$OUT" 2>&1
