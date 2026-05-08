#!/usr/bin/env bash
# Multi-instance scheduling test: submit two small concurrent jobs on the
# same node, verify Slurm/Pyxis isolate them on disjoint GPU sets.
#
# Catches misconfigurations like:
#   - cgroups not enforcing GPU isolation
#   - Pyxis container collision on shared NCCL state
#   - Slurm not setting CUDA_VISIBLE_DEVICES per task
#
# Run after install + smoke; before main benchmarks.

set -uo pipefail
HERE="$(dirname "$0")"
mkdir -p "$HERE/results"

NODE="${NODE:?set NODE=<single B200 host name>}"
LOG_A="$HERE/results/multi_instance_A_$(date -u +%Y%m%dT%H%M%SZ).log"
LOG_B="$HERE/results/multi_instance_B_$(date -u +%Y%m%dT%H%M%SZ).log"

# Two small concurrent jobs on the same node, each requesting 4 GPUs.
# Pick a workload that runs fast: cpu_overhead microbenchmark (~5 min).
echo "Submitting two concurrent 4-GPU jobs on $NODE..."

JOB_A=$(sbatch --parsable -J phase0_multi_A -w "$NODE" --gpus=4 --output="$LOG_A" --wrap="
  echo \"=== JOB A on \$(hostname) GPUs=\$CUDA_VISIBLE_DEVICES ===\"
  llmb-run submit -w microbenchmark_cpu_overhead --scale 4 || true
  nvidia-smi --query-gpu=index --format=csv
")

JOB_B=$(sbatch --parsable -J phase0_multi_B -w "$NODE" --gpus=4 --output="$LOG_B" --wrap="
  echo \"=== JOB B on \$(hostname) GPUs=\$CUDA_VISIBLE_DEVICES ===\"
  llmb-run submit -w microbenchmark_cpu_overhead --scale 4 || true
  nvidia-smi --query-gpu=index --format=csv
")

echo "Job A: $JOB_A   log: $LOG_A"
echo "Job B: $JOB_B   log: $LOG_B"
echo
echo "Wait for both to complete (squeue), then verify:"
echo "  1. Both jobs ran simultaneously (overlapping start/end times)"
echo "  2. CUDA_VISIBLE_DEVICES disjoint between the two logs"
echo "  3. nvidia-smi indices in each log are disjoint subsets of 0-7"
echo "  4. Neither benchmark errored"
