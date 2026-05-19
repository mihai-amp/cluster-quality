#!/usr/bin/env bash
#SBATCH --job-name=phase0_nccl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vfs/mihai/results/nccl_%x_%j.log
# Parametrised NCCL collective test.
#
# Resource allocation (override on sbatch command line):
#   sbatch 40_nccl_run.sh                       # 1 node, 8 GPUs (defaults)
#   sbatch --nodes=2 40_nccl_run.sh             # 2 nodes (16 GPUs)
#
# Knobs via --export=VAR=val,...:
#   COLLECTIVE   one of: all_reduce all_gather reduce_scatter alltoall sendrecv broadcast reduce
#                default: alltoall
#   MIN_BYTES    default 4G  (accepts K/M/G suffix)
#   MAX_BYTES    default 4G
#   STEP_FACTOR  default 2
#   ITERS        default 50  (-n)
#   WARMUP       default 20  (-w)
#   CYCLES       default 10  (-N run_cycles; repeats the sweep, same comm)
#   CHECK        default 0   (-c, slow correctness check)
#   IMAGE        container path (default: NeMo)
#   NCCL_ENV     extra env vars, COLON-separated  (default empty)
#                Colon is used so commas inside the slurm --export string don't
#                collide.  Internally translated back to commas for srun --export.
#                e.g.  NCCL_ENV="NCCL_ALGO=RING:NCCL_PROTO=Simple:NCCL_DEBUG=INFO"
#   EXTRA_ARGS   extra args appended to the binary  (default empty)
#
# Examples:
#   sbatch --export=COLLECTIVE=alltoall,MIN_BYTES=4G,MAX_BYTES=4G,CYCLES=10 40_nccl_run.sh
#   sbatch --nodes=2 --export=COLLECTIVE=all_reduce,MIN_BYTES=16M,MAX_BYTES=8G,ITERS=100 40_nccl_run.sh

set -uo pipefail

COLLECTIVE="${COLLECTIVE:-alltoall}"
NODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-8}}"
MIN_BYTES="${MIN_BYTES:-4G}"
MAX_BYTES="${MAX_BYTES:-4G}"
STEP_FACTOR="${STEP_FACTOR:-2}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"
CYCLES="${CYCLES:-10}"
CHECK="${CHECK:-0}"
NCCL_ENV="${NCCL_ENV:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Convert K/M/G suffixes to byte counts
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

NTASKS=$(( NODES * GPUS_PER_NODE ))
IMAGE="${IMAGE:-/mnt/vfs/mihai/nemo_26.02.sqsh}"
# NeMo container ships nccl-tests in PATH with sm_100 SASS baked in (avoids the JIT
# delay seen with stock pytorch:24.10). Two variants ship: `<op>_perf` (non-MPI,
# single-process multi-GPU) and `<op>_perf_mpi` (MPI build, multi-process). We use
# the _mpi build because (a) dgxc uses it, (b) one process per node is the right
# pattern for both intra and inter, (c) the MPI binary still works as a 1-rank
# world for single-node tests.  Override BIN= to use the non-MPI variant.
BIN="${BIN:-${COLLECTIVE}_perf_mpi}"

# Env handling notes:
#   - --export=NONE: slurm-side, don't inherit controller-shell env (prevents
#     stray NCCL_DEBUG=... in your bash session from leaking through)
#   - Container image (NeMo) sets its own env via Dockerfile ENV, including
#     NCCL_DEBUG=INFO. Pyxis applies these *after* slurm --export, so a slurm
#     -level override does NOT take effect.
#   - The reliable way to override container-baked env is a bash wrapper that
#     runs *inside* the container, after pyxis has finished. Below uses one.
EXPORT="NONE"
if [ -n "$NCCL_ENV" ]; then
  EXPORT="$EXPORT,${NCCL_ENV//:/,}"
fi
# In-container env we want as our quiet default (overrides container's NCCL_DEBUG=INFO):
INSIDE_CONTAINER_ENV='export NCCL_DEBUG=WARN; unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_TIMESTAMP_LEVELS;'

echo "==== $COLLECTIVE  ranks=$NTASKS  (nodes=$NODES, gpus/node=$GPUS_PER_NODE) ===="
echo "Range: $MIN_BYTES → $MAX_BYTES  step×$STEP_FACTOR  iters=$ITERS  warmup=$WARMUP  cycles=$CYCLES"
echo "Env:   $EXPORT"
echo "Extra: $EXTRA_ARGS"
echo "Host(s): $(scontrol show hostname "$SLURM_NODELIST" 2>/dev/null | tr '\n' ' ')"
echo "Time:  $(date -u)"
echo

# 1 task per node, each task drives all 8 local GPUs via -g $GPUS_PER_NODE.
# Avoids the 8-process-MPI bridging path entirely for intra-node; inter-node
# still uses MPI but only 1 rank per node (well-tested dgxc pattern).
# bash -c wrapper runs *inside* the container after pyxis sets its env, so
# we can quiet the container's baked-in NCCL_DEBUG=INFO.
srun --container-image="$IMAGE" \
     --container-mounts=/mnt/vfs/mihai:/mnt/vfs/mihai \
     --mpi=pmix \
     --export="$EXPORT" \
     bash -c "$INSIDE_CONTAINER_ENV"' exec "$@"' _ \
       "$BIN" -b "$MIN_BYTES_RAW" -e "$MAX_BYTES_RAW" -f "$STEP_FACTOR" \
              -g "$GPUS_PER_NODE" -n "$ITERS" -w "$WARMUP" -N "$CYCLES" -c "$CHECK" $EXTRA_ARGS

LOG=/mnt/vfs/mihai/results/nccl_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

echo
# Column layout (nccl-tests 2.17+): size count type redop root [oop: cputime algbw busbw #wrong] [ip: cputime algbw busbw #wrong]
# i.e. $7=oop_algbw  $8=oop_busbw  $11=ip_algbw  $12=ip_busbw
echo "==== per-cycle busbw at each message size (oop algbw/busbw / ip algbw/busbw) ===="
awk '
  /^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(uint8|float)/ {
    n++
    printf "  size=%12s  oop: algbw=%7.2f busbw=%7.2f   ip: algbw=%7.2f busbw=%7.2f GB/s\n",
           $1, $7, $8, $11, $12
  }
' "$LOG"

echo
echo "==== aggregate (out-of-place busbw across all rows) ===="
awk '
  /^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(uint8|float)/ {
    bw[++n] = $8
    sum += $8
    if (n == 1 || $8 < min) min = $8
    if (n == 1 || $8 > max) max = $8
  }
  END {
    if (n == 0) { print "no data"; exit }
    mean = sum / n
    for (i = 1; i <= n; i++) { d = bw[i] - mean; ssd += d * d }
    sd = sqrt(ssd / n)
    printf "  n=%d   mean=%.2f   stddev=%.3f (%.2f%%)   min=%.2f   max=%.2f   GB/s\n",
           n, mean, sd, 100*sd/mean, min, max
  }
' "$LOG"
