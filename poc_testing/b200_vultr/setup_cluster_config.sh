#!/bin/bash
# Render $LLMB_INSTALL/cluster_config.yaml from current shell env.
# Reads MY/DGXC/LLMB_INSTALL/HF_TOKEN/NCCL_IB_HCA from env.sh; bails if any are missing.
# Idempotent — re-run any time the env or installed workload list changes.

set -euo pipefail

: "${MY:?source /mnt/vfs/<user>/env.sh first}"
: "${DGXC:?source env.sh first}"
: "${LLMB_INSTALL:?source env.sh first}"
: "${HF_TOKEN:?HF_TOKEN must be set in env.sh}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in env.sh — see bootstrap.md}"

OUT="$LLMB_INSTALL/cluster_config.yaml"

cat > "$OUT" <<EOF
schema_version: 2
llmb_repo: $DGXC
llmb_install: $LLMB_INSTALL
gpu_type: b200

install:
  node_architecture: x86_64

environment:
  HF_TOKEN: "$HF_TOKEN"
  NCCL_IB_HCA: "$NCCL_IB_HCA"

slurm:
  account: ""
  gpu:
    partition: batch
    gres: 8
  cpu:
    partition: batch
    gres: null

workloads:
  installed:
    - microbenchmark_system_info
    - microbenchmark_nccl
    - microbenchmark_cpu_overhead
    - pretrain_llama3.1
    - pretrain_nemotron4-15b
    - pretrain_qwen3
    - inference_deepseek-r1
    - inference_deepseek-r1-dynamo
    - inference_deepseek-r1-sglang
    - inference_llama3.3
    - inference_gpt-oss-dynamo
EOF

chmod 600 "$OUT"

echo "Wrote $OUT"
echo "Validate with:  cd $LLMB_INSTALL && llmb-run list"
