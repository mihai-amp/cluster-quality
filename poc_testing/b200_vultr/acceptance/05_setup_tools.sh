#!/usr/bin/env bash
# Build/fetch tools required by Phase 0 scripts that may not be pre-installed.
# Run once before the rest of acceptance; idempotent.
#
# What this installs to a shared TOOLS_DIR:
#   - gpu-burn (https://github.com/wilicc/gpu-burn) - compiled binary
#   - nvbandwidth (https://github.com/NVIDIA/nvbandwidth) - compiled binary
#   - nccl-tests (https://github.com/NVIDIA/nccl-tests) - compiled MPI binaries
#
# Most clusters ship dcgmi, perftest (ib_write_bw/lat), fio, iperf3 pre-installed.
# This script does NOT install those - flag them in quirks.md if missing.

set -euo pipefail

TOOLS_DIR="${TOOLS_DIR:-${HOME}/cluster-tools}"
mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

echo "Installing tools to $TOOLS_DIR"
echo

# --- gpu-burn ---
if [ ! -x "$TOOLS_DIR/gpu-burn/gpu_burn" ]; then
  echo "==== Cloning + building gpu-burn ===="
  rm -rf gpu-burn
  git clone https://github.com/wilicc/gpu-burn.git
  cd gpu-burn
  make
  cd ..
  echo "gpu-burn built at $TOOLS_DIR/gpu-burn/gpu_burn"
else
  echo "gpu-burn already built; skipping"
fi

# --- nvbandwidth ---
if [ ! -x "$TOOLS_DIR/nvbandwidth/build/nvbandwidth" ]; then
  echo "==== Cloning + building nvbandwidth ===="
  rm -rf nvbandwidth
  git clone https://github.com/NVIDIA/nvbandwidth.git
  cd nvbandwidth
  cmake -B build -S . && cmake --build build -j
  cd ..
  echo "nvbandwidth built at $TOOLS_DIR/nvbandwidth/build/nvbandwidth"
else
  echo "nvbandwidth already built; skipping"
fi

# --- nccl-tests ---
if [ ! -x "$TOOLS_DIR/nccl-tests/build/all_reduce_perf" ]; then
  echo "==== Cloning + building nccl-tests ===="
  rm -rf nccl-tests
  git clone https://github.com/NVIDIA/nccl-tests.git
  cd nccl-tests
  # MPI build for multi-node
  make MPI=1 MPI_HOME="${MPI_HOME:-/opt/hpcx/ompi}" -j
  cd ..
  echo "nccl-tests built at $TOOLS_DIR/nccl-tests/build/"
else
  echo "nccl-tests already built; skipping"
fi

echo
echo "All tools ready. Export these for the Phase 0 scripts:"
echo "  export GPU_BURN=$TOOLS_DIR/gpu-burn/gpu_burn"
echo "  export NVBW=$TOOLS_DIR/nvbandwidth/build/nvbandwidth"
echo "  export NCCL_TESTS_DIR=$TOOLS_DIR/nccl-tests/build"
