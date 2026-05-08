#!/usr/bin/env bash
# Cluster discovery: capture every version, env var, and hardware identifier.
# Run on each node (or via srun across all nodes). Output goes to results/discovery_<host>.txt.

set -uo pipefail

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"
HOST=$(hostname -s)
OUT="$OUT_DIR/discovery_${HOST}.txt"

{
  echo "===== HOST ====="
  hostname -f
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  uname -a
  cat /etc/os-release

  echo; echo "===== CPU ====="
  lscpu
  echo; echo "--- NUMA ---"
  numactl --hardware 2>/dev/null || echo "numactl not installed"

  echo; echo "===== MEMORY ====="
  free -h
  echo; echo "--- DIMMs ---"
  sudo dmidecode -t memory 2>/dev/null | grep -E "Size|Speed|Type:" | head -50 || echo "dmidecode unavailable"

  echo; echo "===== GPU ====="
  nvidia-smi
  echo; echo "--- topology ---"
  nvidia-smi topo -m
  echo; echo "--- nvlink status ---"
  nvidia-smi nvlink --status

  echo; echo "===== CUDA / DRIVER ====="
  nvcc --version 2>/dev/null || echo "nvcc not on PATH"
  cat /proc/driver/nvidia/version 2>/dev/null

  echo; echo "===== NCCL ====="
  echo "(version printed by nccl-tests at runtime; capturing env)"
  env | grep -i ^NCCL_ || echo "no NCCL_ env vars set in this shell"
  ldconfig -p | grep -i nccl || true

  echo; echo "===== INFINIBAND ====="
  ibstatus 2>/dev/null || echo "ibstatus unavailable"
  echo; echo "--- ibstat ---"
  ibstat 2>/dev/null || true
  echo; echo "--- HCA list ---"
  ls /sys/class/infiniband/ 2>/dev/null
  echo; echo "--- OFED ---"
  ofed_info -s 2>/dev/null || echo "ofed_info unavailable"

  echo; echo "===== STORAGE ====="
  df -hT
  echo; echo "--- block devices ---"
  lsblk

  echo; echo "===== KERNEL MODULES ====="
  lsmod | grep -iE "nvidia|nv_peer|gdrdrv|ib_|mlx|rdma" || true

  echo; echo "===== ENROOT / PYXIS ====="
  enroot version 2>/dev/null || echo "enroot not on PATH"
  cat /etc/enroot/enroot.conf 2>/dev/null | grep -v '^\s*#' | grep -v '^\s*$' || true
  ls /etc/enroot/environ.d/ 2>/dev/null

  echo; echo "===== SLURM ====="
  sinfo --version 2>/dev/null
  scontrol show config 2>/dev/null | grep -E "SlurmctldHost|GresTypes|TaskPlugin|Prolog|Epilog" || true

  echo; echo "===== ENV ====="
  env | sort
} >"$OUT" 2>&1

echo "Wrote $OUT"
