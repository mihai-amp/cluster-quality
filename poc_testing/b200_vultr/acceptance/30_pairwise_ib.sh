#!/usr/bin/env bash
#SBATCH --job-name=phase0_pairib
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/pairwise_ib_%j.log
# Pairwise InfiniBand bandwidth & latency between the two compute nodes.
# Tests every HCA passed via NCCL_IB_HCA (or HCAS override).
# Submits as a Slurm job — uses srun, not ssh, so works on clusters without inter-node SSH.

set -uo pipefail

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"

# Discover the two nodes from Slurm's allocation
NODES=($(scontrol show hostname "$SLURM_NODELIST"))
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "ERROR: need at least 2 nodes; got: ${NODES[*]}" >&2
  exit 1
fi
NODE_A="${NODES[0]}"
NODE_B="${NODES[1]}"

# Default HCA list: prefer NCCL_IB_HCA (already set per-cluster), else fall back.
HCAS_DEFAULT="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"
HCAS="${HCAS:-$HCAS_DEFAULT}"
HCAS_LIST=$(echo "$HCAS" | tr ',' ' ')

PORT_BASE="${PORT_BASE:-18515}"
DURATION="${DURATION:-30}"

OUT="$OUT_DIR/pairwise_ib_$(date -u +%Y%m%dT%H%M%SZ).log"
{
  echo "==== pairwise IB tests at $(date -u) ===="
  echo "NODE_A=$NODE_A  NODE_B=$NODE_B"
  echo "HCAs: $HCAS_LIST"
  echo "Port base: $PORT_BASE   duration: ${DURATION}s"
  echo
} | tee "$OUT"

idx=0
for hca in $HCAS_LIST; do
  port=$((PORT_BASE + idx))

  for tool in ib_write_bw ib_write_lat; do
    echo "==== $tool on $hca (server=$NODE_A, client=$NODE_B), port $port ====" | tee -a "$OUT"

    SERVER_LOG="$OUT.$NODE_A.$hca.$tool"
    CLIENT_LOG="$OUT.$NODE_B.$hca.$tool"

    # Server on NODE_A, background.
    srun --nodelist="$NODE_A" -N1 -n1 \
      "$tool" -d "$hca" -p "$port" -D "$DURATION" -F >"$SERVER_LOG" 2>&1 &
    server_pid=$!

    sleep 3   # let server bind

    # Client on NODE_B, foreground.
    srun --nodelist="$NODE_B" -N1 -n1 \
      "$tool" -d "$hca" -p "$port" -D "$DURATION" -F "$NODE_A" >"$CLIENT_LOG" 2>&1 || true

    wait "$server_pid" 2>/dev/null || true

    echo "--- server log ---" | tee -a "$OUT"
    cat "$SERVER_LOG" | tee -a "$OUT"
    echo "--- client log ---" | tee -a "$OUT"
    cat "$CLIENT_LOG" | tee -a "$OUT"
    echo | tee -a "$OUT"
  done

  idx=$((idx + 1))
done

echo "Wrote $OUT"
echo
echo "Sanity check: each ib_write_bw on NDR (400 Gb/s) should show ~46-49 GB/s sustained"
echo "ib_write_lat p50 should be < 2 microseconds for ConnectX-7 NDR within rack"
