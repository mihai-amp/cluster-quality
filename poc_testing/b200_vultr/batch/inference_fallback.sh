#!/usr/bin/env bash
# Inference workload submission fallback.
# Use this if `llmb-run submit -f inference_batch.yaml` fails because the
# bulk YAML format isn't supported for inference workloads.
#
# Submits each inference workload explicitly with -r 3.
# Verify slugs first with: llmb-run list

set -euo pipefail
: "${LLMB_INSTALL:?LLMB_INSTALL must be set}"
cd "$LLMB_INSTALL"

REPEATS="${REPEATS:-3}"

echo "=== Llama 3.3 70B inference (1 GPU, NVFP4) ==="
llmb-run submit -w inference_llama3.3 -s 70b --scale 1 -r "$REPEATS"

echo "=== GPT-OSS 120B inference (4 GPUs, MXFP4) ==="
llmb-run submit -w inference_gpt_oss -s 120b --scale 4 -r "$REPEATS"

echo "=== DeepSeek R1 inference - TRT-LLM (4 GPUs, NVFP4) ==="
llmb-run submit -w inference_deepseek_r1_trtllm -s 671b --scale 4 -r "$REPEATS"

echo "=== DeepSeek R1 inference - SGLang (8 GPUs, NVFP4) ==="
llmb-run submit -w inference_deepseek_r1_sglang -s 671b --scale 8 -r "$REPEATS"

echo
echo "All inference jobs submitted. Monitor with: squeue -u $USER"
