#!/usr/bin/env bash
#SBATCH --job-name=phase0_a2a_ibonly
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_alltoall_ibonly_%j.log
# Inter-node alltoall with NVLink disabled to remove NVL contribution to the bus-bw figure.
# NCCL_P2P_DISABLE=1 prevents CUDA P2P (NVLink); intra-node uses SHM (host memory) instead,
# inter-node uses IB. NCCL_SHM_DISABLE was also tried but the cluster's IB stack does not
# accept QPs for intra-node loopback (ibv_modify_qp EINVAL) — see quirks.md.

set -uo pipefail
mkdir -p /mnt/vfs/mihai/results

IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
BIN=alltoall_perf   # on PATH inside the NeMo image

MIN_BYTES="${MIN_BYTES:-16777216}"   # 16 MiB
MAX_BYTES="${MAX_BYTES:-8589934592}" # 8 GiB
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"

# No-NVLink env (intra-node falls back to SHM, inter-node stays on IB).
NO_NVL_ENV="ALL,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=ENV,NCCL_IB_HCA=mlx5,NCCL_P2P_DISABLE=1"

PYXIS=(
  --container-image="$IMAGE"
  --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai
  --mpi=pmix
)

echo "==== alltoall NVLink-disabled on $(scontrol show hostname "$SLURM_NODELIST" | tr '\n' ' ') at $(date -u) ===="
echo "Env: NCCL_P2P_DISABLE=1 NCCL_IB_HCA=mlx5"

echo
echo "==== alltoall  ranks=16  (inter-node, NO-NVLINK — SHM intra + IB inter) ===="
srun -N2 -n16 --gpus-per-node=8 "${PYXIS[@]}" \
  --export="$NO_NVL_ENV" \
  "$BIN" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -n "$ITERS" -w "$WARMUP" -c 1

echo
echo "Nominal: 8× ConnectX-7 NDR @ 400 Gbps = 50 GB/s per HCA, 400 GB/s per-node aggregate."
echo "Expected IB-only alltoall busbw on 2 nodes: scaling-limited, typically 30-50 GB/s per rank."
