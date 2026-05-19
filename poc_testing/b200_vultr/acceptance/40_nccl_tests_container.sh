#!/usr/bin/env bash
#SBATCH --job-name=phase0_nccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_tests_%j.log
# NCCL collectives validation via pyxis+enroot. Intra (1 node, 8 GPUs) and inter (2 nodes, 16 GPUs).

set -uo pipefail
mkdir -p /mnt/vfs/mihai/results

IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
# NeMo image ships MPI-built nccl-tests in PATH with sm_100 SASS (avoids JIT delay
# we hit with stock pytorch:24.10 that has no Blackwell SASS).
BIN_DIR=""   # binaries on PATH inside the container
MIN_BYTES="${MIN_BYTES:-16777216}"   # 16 MiB
MAX_BYTES="${MAX_BYTES:-8589934592}" # 8 GiB
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"

COMMON_ENV=(
  --export=ALL,NCCL_IB_HCA=mlx5
)
PYXIS=(
  --container-image="$IMAGE"
  --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai
  --mpi=pmix
)

echo "==== NCCL tests on $(scontrol show hostname "$SLURM_NODELIST" | tr '\n' ' ') at $(date -u) ===="
srun -N1 -n1 -w "$(scontrol show hostname "$SLURM_NODELIST" | head -1)" "${PYXIS[@]}" nvidia-smi -L

echo
echo "######## INTRA-NODE (8 GPUs, 1 node) ########"
for op in all_reduce alltoall; do
  echo
  echo "==== $op  ranks=8  (intra-node) ===="
  srun -N1 -n8 --gpus-per-node=8 "${PYXIS[@]}" "${COMMON_ENV[@]}" \
    "${op}_perf" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 1
done

echo
echo "######## INTER-NODE (16 GPUs, 2 nodes) ########"
for op in all_reduce all_gather reduce_scatter alltoall; do
  echo
  echo "==== $op  ranks=16  (inter-node) ===="
  srun -N2 -n16 --gpus-per-node=8 "${PYXIS[@]}" "${COMMON_ENV[@]}" \
    "${op}_perf" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 1
done

echo
echo "Sanity check:"
echo "  B200 NVLink5 intra-node all_reduce  busbw > 350 GB/s at large sizes"
echo "  B200 NVLink5 intra-node alltoall    busbw > 350 GB/s at large sizes"
echo "  Inter-node 2-node all_reduce on 8x NDR  busbw > 60-80 GB/s at large sizes"
echo "  Inter-node 2-node alltoall on 8x NDR    busbw scaling-limited; expect ~40-60 GB/s"
