#!/usr/bin/env bash
# Power/thermal/utilization capture for benchmark runs.
# Spawns nvidia-smi dmon in the background; intended to be sourced from a
# Slurm job's prolog or wrapped around `llmb-run submit` invocations.
#
# Usage:
#   source monitor/power_capture.sh
#   start_capture <experiment_label>
#   ... run benchmark ...
#   stop_capture
#
# Output: one CSV per node per benchmark under monitor/results/<label>/dmon_<host>.csv

start_capture() {
  local label="${1:-unlabeled}"
  local out_dir="$(dirname "${BASH_SOURCE[0]}")/results/${label}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$out_dir"

  # On every node in the allocation, start dmon
  if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    srun --ntasks-per-node=1 \
         bash -c "nvidia-smi dmon -s puct -d 5 -o DT -f $out_dir/dmon_\$(hostname -s).csv &
                  echo \$! > $out_dir/dmon_\$(hostname -s).pid" &
  else
    # Local fallback (single host)
    nvidia-smi dmon -s puct -d 5 -o DT -f "$out_dir/dmon_$(hostname -s).csv" &
    echo $! > "$out_dir/dmon_$(hostname -s).pid"
  fi

  export POWER_CAPTURE_DIR="$out_dir"
  echo "Power capture started: $out_dir"
}

stop_capture() {
  if [ -z "${POWER_CAPTURE_DIR:-}" ]; then
    echo "No capture in progress" >&2
    return 1
  fi

  for pidfile in "$POWER_CAPTURE_DIR"/*.pid; do
    [ -f "$pidfile" ] || continue
    local host="$(basename "$pidfile" .pid | sed 's/^dmon_//')"
    if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
      srun -w "$host" --ntasks=1 kill "$(cat "$pidfile")" 2>/dev/null || true
    else
      kill "$(cat "$pidfile")" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done

  echo "Power capture stopped: $POWER_CAPTURE_DIR"
  unset POWER_CAPTURE_DIR
}
