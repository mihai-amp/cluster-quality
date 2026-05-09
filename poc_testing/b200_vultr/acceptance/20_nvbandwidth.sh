#!/usr/bin/env bash
#SBATCH --job-name=phase0_nvbw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --output=results/nvbandwidth_%N_%j.log
# PCIe Gen5 H<->D and intra-node D<->D bandwidth via NVIDIA's nvbandwidth tool.
# https://github.com/NVIDIA/nvbandwidth
# Build once and copy the binary to a shared path, or use the container that ships with it.

set -uo pipefail
mkdir -p "$(dirname "$0")/results"

NVBW="${NVBW:-nvbandwidth}"
if ! command -v "$NVBW" >/dev/null; then
  echo "ERROR: nvbandwidth not on PATH. Build from https://github.com/NVIDIA/nvbandwidth or set NVBW=<path>." >&2
  exit 1
fi

echo "==== nvbandwidth on $(hostname) at $(date -u) ===="

# Run all default test cases (host-to-device, device-to-host, device-to-device)
"$NVBW" -t all

# Per-pair D<->D matrix (catches asymmetric NVLink performance)
"$NVBW" -t device_to_device_memcpy_read_ce
"$NVBW" -t device_to_device_memcpy_write_ce
