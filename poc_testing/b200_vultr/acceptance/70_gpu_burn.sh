#!/usr/bin/env bash
#SBATCH --job-name=phase0_gpuburn
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=25:00:00
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=results/gpu_burn_%N_%j.log
# Long-running gpu-burn on both nodes simultaneously.
# https://github.com/wilicc/gpu-burn
#
# Sized for cumulative >= 24h coverage across the 5-day window.
# Recommended: submit once early (Day 1 evening, post-acceptance) and again
# late (Day 4 evening, post-benchmarks). Each instance defaults to 12h via
# DURATION env var; override for a single 24h block.
#
# --nice=10000 + --requeue make this preemptible if your Slurm config supports
# preemption-by-priority; benchmark jobs (default nice) will preempt and the
# burn requeues automatically. If your cluster doesn't honor priority, just
# schedule this in known idle windows manually.
#
# Build prereq (run once on a node with nvcc):
#   git clone https://github.com/wilicc/gpu-burn.git
#   cd gpu-burn && make
# Then either copy the binary to a shared path or rebuild on each node.
# Set GPU_BURN to the binary path.
#
# Concurrent thermal/power capture is started in the background and stopped at the end.

set -uo pipefail
mkdir -p "$(dirname "$0")/results"

GPU_BURN="${GPU_BURN:-/shared/gpu-burn/gpu_burn}"
DURATION="${DURATION:-43200}"  # seconds; 43200 = 12h (run twice in 5-day plan for 24h cumulative)
HOST=$(hostname -s)
RESULTS_DIR="$(dirname "$0")/results"
DMON_LOG="$RESULTS_DIR/dmon_${HOST}_${SLURM_JOB_ID:-local}.log"
DMESG_LOG="$RESULTS_DIR/dmesg_${HOST}_${SLURM_JOB_ID:-local}.log"

if [ ! -x "$GPU_BURN" ]; then
  echo "ERROR: $GPU_BURN not executable. Build with: git clone https://github.com/wilicc/gpu-burn.git && cd gpu-burn && make" >&2
  exit 1
fi

# Start passive monitoring
nvidia-smi dmon -s puct -d 5 -o DT >"$DMON_LOG" 2>&1 &
DMON_PID=$!

# Capture XID/SXID from dmesg every minute
(while true; do
   date -u
   dmesg | tail -200 | grep -iE "xid|sxid|nvidia|nvlink|infiniband|mlx|throttle" || true
   sleep 60
 done) >"$DMESG_LOG" 2>&1 &
DMESG_PID=$!

trap 'kill $DMON_PID $DMESG_PID 2>/dev/null || true' EXIT

echo "==== gpu-burn for ${DURATION}s on $HOST starting at $(date -u) ===="
nvidia-smi --query-gpu=name,serial,uuid,vbios_version --format=csv

# Run gpu-burn (uses all visible GPUs by default)
"$GPU_BURN" "$DURATION"
RC=$?

echo "==== gpu-burn finished on $HOST at $(date -u) with exit code $RC ===="

# Final state snapshot
nvidia-smi
echo
echo "--- recent dmesg ---"
dmesg | tail -200 | grep -iE "xid|sxid|nvidia|throttle" || true

exit $RC
