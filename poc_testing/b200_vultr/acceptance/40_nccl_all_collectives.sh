#!/usr/bin/env bash
#SBATCH --job-name=phase0_allcoll
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_%x_%j.log
# Run every NCCL collective sequentially inside the NeMo container, one srun
# step per collective, all output stitched into a single log.
#
# Resource allocation (override on sbatch command line):
#   sbatch 40_nccl_all_collectives.sh                       # 1 node (intra)
#   sbatch --nodes=2 40_nccl_all_collectives.sh             # 2 nodes (inter)
#
# Knobs via --export=VAR=val,... (commas split keys, so no comma inside values):
#   COLLECTIVES  space-separated; default: all_reduce all_gather reduce_scatter alltoall sendrecv broadcast reduce
#   MIN_BYTES    default 4G  (accepts K/M/G suffix)
#   MAX_BYTES    default 4G
#   STEP_FACTOR  default 2
#   ITERS        default 100
#   WARMUP       default 10
#   CYCLES       default 5
#   AVG          default 1   (-a: 0=RANK0, 1=AVG, 2=MIN, 3=MAX)
#   CHECK        default 0   (-c, 1=validate, slow)
#   IMAGE        default /mnt/vfs/mihai/nemo_26.02.sqsh
#   NCCL_ENV     extra NCCL env vars, COLON-separated   (default empty)
#                e.g. NCCL_ENV="NCCL_DEBUG=INFO:NCCL_ALGO=RING"
#
# Example:
#   sbatch --nodes=2 --export=MIN_BYTES=8G,MAX_BYTES=8G,ITERS=50 \
#          40_nccl_all_collectives.sh

set -uo pipefail

COLLECTIVES="${COLLECTIVES:-all_reduce all_gather reduce_scatter alltoall sendrecv broadcast reduce}"
MIN_BYTES="${MIN_BYTES:-4G}"
MAX_BYTES="${MAX_BYTES:-4G}"
STEP_FACTOR="${STEP_FACTOR:-2}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-10}"
CYCLES="${CYCLES:-5}"
AVG="${AVG:-1}"
CHECK="${CHECK:-0}"
IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
NCCL_ENV="${NCCL_ENV:-}"

NODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-8}}"

# K/M/G → bytes
suffix_to_bytes() {
  local v="$1"
  case "${v}" in
    *K|*k) echo $((${v%[Kk]} * 1024));;
    *M|*m) echo $((${v%[Mm]} * 1024 * 1024));;
    *G|*g) echo $((${v%[Gg]} * 1024 * 1024 * 1024));;
    *)     echo "$v";;
  esac
}
MIN_BYTES_RAW=$(suffix_to_bytes "$MIN_BYTES")
MAX_BYTES_RAW=$(suffix_to_bytes "$MAX_BYTES")

# Quiet the container's baked-in NCCL_DEBUG=INFO (override inside the container,
# since pyxis applies container env after slurm --export).
INSIDE='export NCCL_DEBUG=WARN; unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_TIMESTAMP_LEVELS;'

EXPORT="NONE"
if [ -n "$NCCL_ENV" ]; then
  EXPORT="$EXPORT,${NCCL_ENV//:/,}"
fi

echo "================================================================================"
echo "==== NCCL all-collectives sweep at $(date -u) ===="
echo "Host(s):     $(scontrol show hostname "$SLURM_NODELIST" | tr '\n' ' ')"
echo "Image:       $IMAGE"
echo "Nodes:       $NODES   GPUs/node: $GPUS_PER_NODE   total ranks: $(( NODES * GPUS_PER_NODE ))"
echo "Size range:  $MIN_BYTES → $MAX_BYTES   step×$STEP_FACTOR   iters=$ITERS  warmup=$WARMUP  cycles=$CYCLES  avg=$AVG  check=$CHECK"
echo "Collectives: $COLLECTIVES"
echo "Export:      $EXPORT"
echo "================================================================================"
echo

for col in $COLLECTIVES; do
  BIN="${col}_perf_mpi"
  echo
  echo "################################################################################"
  echo "######## $col   (nranks=$(( NODES * GPUS_PER_NODE )))"
  echo "################################################################################"
  srun --container-image="$IMAGE" \
       --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai \
       --mpi=pmix \
       --export="$EXPORT" \
       bash -c "$INSIDE"' exec "$@"' _ \
         "$BIN" -b "$MIN_BYTES_RAW" -e "$MAX_BYTES_RAW" -f "$STEP_FACTOR" \
                -g "$GPUS_PER_NODE" -n "$ITERS" -w "$WARMUP" -N "$CYCLES" \
                -c "$CHECK" -a "$AVG"
  echo
done

LOG=/mnt/vfs/mihai/results/nccl_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
echo "================================================================================"
echo "==== per-collective summary  (peak / mean / min / max busbw across all rows) ===="
echo "================================================================================"
awk '
  /^######## / { col=$2 }
  /^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(uint8|float)/ {
    # NeMo nccl-tests column layout (oop-first): $8=oop_busbw
    rows[col] = rows[col] " " $8
  }
  END {
    for (c in rows) {
      n=0; sum=0; min=0; max=0
      split(rows[c], a, " ")
      for (i in a) if (a[i] != "") {
        n++; v=a[i]+0; sum+=v
        if (n==1 || v<min) min=v
        if (n==1 || v>max) max=v
      }
      if (n>0) {
        printf "  %-16s n=%d  mean=%7.2f  min=%7.2f  max=%7.2f  GB/s\n",
               c, n, sum/n, min, max
      }
    }
  }
' "$LOG" | sort
