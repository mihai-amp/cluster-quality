#!/usr/bin/env bash
# Master orchestration script for B200 / Vultr POC.
# Run from $LLMB_INSTALL after install.sh has completed.
# Assumes train_batch.yaml and inference_batch.yaml are in the current directory
# (or adjust paths below).

set -euo pipefail

: "${LLMB_INSTALL:?LLMB_INSTALL must be set}"
cd "$LLMB_INSTALL"

BATCH_DIR="${BATCH_DIR:-$(dirname "$0")}"

echo "=== Smoke tests ==="
llmb-run submit -w microbenchmark_system_info --scale 8
llmb-run submit -w microbenchmark_cpu_overhead --scale 4

echo "=== Training matrix (dry-run preview) ==="
llmb-run submit -f "$BATCH_DIR/train_batch.yaml" --dry-run
read -p "Looks right? [enter to submit, ctrl-c to abort] "
llmb-run submit -f "$BATCH_DIR/train_batch.yaml"

echo "=== Inference (dry-run preview) ==="
llmb-run submit -f "$BATCH_DIR/inference_batch.yaml" --dry-run
read -p "Looks right? [enter to submit, ctrl-c to abort] "
llmb-run submit -f "$BATCH_DIR/inference_batch.yaml"

echo "=== Archive ==="
llmb-run archive --output "/shared/results/vultr-b200-$(date +%Y%m%d).tar.zst"
