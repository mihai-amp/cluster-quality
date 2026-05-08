#!/usr/bin/env sh
#SBATCH --job-name=phase0_nccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=results/nccl_tests_%j.log
# NCCL collectives validation: intra-node (1 node, 8 GPUs) and inter-node (2 nodes, 16 GPUs).
# Sweeps message sizes and captures bus bandwidth.
#
# Prereq: nccl-tests built and available. Either:
#   - host: build from https://github.com/NVIDIA/nccl-tests
#   - container: most NeMo/Megatron-Bridge containers ship them under /opt/hpcx/nccl_rdma_sharp_plugin/...
# Set NCCL_TESTS_DIR to override.

set -uo pipefail
mkdir -p "$(dirname "$0")/results"

NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/opt/nccl-tests/build}"
MPIRUN="${MPIRUN:-mpirun}"

# Standard sweep: 16MiB to 8GiB
MIN_BYTES="${MIN_BYTES:-16777216}"
MAX_BYTES="${MAX_BYTES:-8589934592}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"

run_collective() {
  local op="$1"
  local nranks="$2"
  local label="$3"
  echo
  echo "==== $op  ranks=$nranks  ($label) ===="
  $MPIRUN -np "$nranks" --map-by ppr:8:node \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=ENV,COLL \
    "$NCCL_TESTS_DIR/${op}_perf" \
      -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 \
      -n "$ITERS" -w "$WARMUP" -c 1
}

echo "==== NCCL tests on $(scontrol show hostname "$SLURM_NODELIST" | tr '\n' ' ') at $(date -u) ===="
nvidia-smi -L

# Intra-node (single node, 8 GPUs)
echo
echo "######## INTRA-NODE (8 GPUs, 1 node) ########"
SLURM_NNODES_OVERRIDE=1 \
  $MPIRUN -np 8 -H "$(hostname):8" "$NCCL_TESTS_DIR/all_reduce_perf" \
    -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 1

# Inter-node (both nodes, 16 GPUs)
echo
echo "######## INTER-NODE (16 GPUs, 2 nodes) ########"
for op in all_reduce all_gather reduce_scatter; do
  run_collective "$op" 16 "inter-node"
done

echo
echo "Sanity check: B200 NVLink5 intra-node all_reduce should show busbw > 350 GB/s at large sizes."
echo "Inter-node 2-node all_reduce on 8x NDR should show busbw > 60-80 GB/s at large sizes."
