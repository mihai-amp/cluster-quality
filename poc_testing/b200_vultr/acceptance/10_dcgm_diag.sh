#!/usr/bin/env bash
#SBATCH --job-name=phase0_dcgm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:30:00
#SBATCH --output=results/dcgm_diag_%N_%j.log
# DCGM Level 3 diagnostic: PCIe, NVLink, ECC, thermal, memory bandwidth, SM stress.
# Run once per node. Submit with: sbatch -w <node> 10_dcgm_diag.sh

set -uo pipefail
mkdir -p "$(dirname "$0")/results"

# dcgmi requires nv-hostengine running; on most setups it's already up via systemd
if ! pgrep -x nv-hostengine >/dev/null; then
  echo "nv-hostengine not running; attempting to start"
  sudo nv-hostengine || { echo "Failed to start nv-hostengine"; exit 1; }
fi

echo "==== dcgmi diag -r 3 on $(hostname) at $(date -u) ===="
dcgmi diag -r 3 --verbose

# Optional: also dump current health
echo
echo "==== dcgmi health on $(hostname) ===="
dcgmi health -c -j 2>/dev/null || true
