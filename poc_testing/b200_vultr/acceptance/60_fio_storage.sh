#!/usr/bin/env bash
# Storage benchmarks for the 30 TB shared FS.
# Three workloads:
#   1. Sequential 1 MiB read/write   - checkpoint streaming pattern
#   2. Random 4 KiB read/write       - metadata, small reads
#   3. Lots Of Small Files (LOSF)    - HF cache, dataset glob; flagged by Clustermax
#
# Override FIO_DIR to point at the shared FS path you want to test.

set -uo pipefail

FIO_DIR="${FIO_DIR:-${LLMB_INSTALL:-/shared/llmb}/fio_test}"
RUNTIME="${RUNTIME:-60}"
SIZE="${SIZE:-32G}"

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR" "$FIO_DIR"
OUT="$OUT_DIR/fio_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ).log"

{
  echo "==== fio storage tests on $FIO_DIR from $(hostname) ===="
  date -u
  df -hT "$FIO_DIR"

  echo; echo "--- 1. Sequential 1 MiB read ---"
  fio --name=seq_read --directory="$FIO_DIR" --rw=read --bs=1M --size="$SIZE" \
      --ioengine=libaio --iodepth=32 --direct=1 --runtime="$RUNTIME" --time_based --group_reporting

  echo; echo "--- 2. Sequential 1 MiB write ---"
  fio --name=seq_write --directory="$FIO_DIR" --rw=write --bs=1M --size="$SIZE" \
      --ioengine=libaio --iodepth=32 --direct=1 --runtime="$RUNTIME" --time_based --group_reporting

  echo; echo "--- 3. Random 4 KiB read ---"
  fio --name=rand_read --directory="$FIO_DIR" --rw=randread --bs=4k --size="$SIZE" \
      --ioengine=libaio --iodepth=64 --direct=1 --runtime="$RUNTIME" --time_based --group_reporting

  echo; echo "--- 4. Random 4 KiB write ---"
  fio --name=rand_write --directory="$FIO_DIR" --rw=randwrite --bs=4k --size="$SIZE" \
      --ioengine=libaio --iodepth=64 --direct=1 --runtime="$RUNTIME" --time_based --group_reporting

  echo; echo "--- 5. Lots Of Small Files: 100k * 16 KiB files ---"
  fio --name=losf --directory="$FIO_DIR" --rw=randwrite --bs=16k \
      --nrfiles=100000 --filesize=16k --openfiles=1024 \
      --ioengine=psync --iodepth=1 --direct=0 --runtime=120 --time_based --group_reporting
} | tee "$OUT"

# Clean up the fio test files
rm -rf "$FIO_DIR"

echo "Wrote $OUT"
