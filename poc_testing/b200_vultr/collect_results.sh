#!/bin/bash
# Collect Phase 1 results: copy raw logs, run dgxc parsers, extract inference blocks,
# and (optionally) build the NVIDIA submission archive. Idempotent — safe to re-run.
#
# Output lands in $MY/results/phase1/<workload>/ and a top-level summary.txt.
#
# Usage: bash $PLAN/collect_results.sh [--archive]

set -euo pipefail

: "${MY:?source /mnt/vfs/<user>/env.sh first}"
: "${LLMB_INSTALL:?source env.sh first}"

OUT_DIR="$MY/results/phase1"
COMMON="$LLMB_INSTALL/llmb_repo/common"
MBRIDGE="$COMMON/parse_train_timing_mbridge.sh"
NEMO="$COMMON/parse_train_timing.sh"
DO_ARCHIVE=0
[[ "${1:-}" == "--archive" ]] && DO_ARCHIVE=1

mkdir -p "$OUT_DIR"

# Workload -> parser mapping
declare -A PARSER=(
    [pretrain_llama3.1]="$MBRIDGE"
    [pretrain_qwen3]="$MBRIDGE"
    [finetune_llama3]="$MBRIDGE"
    [pretrain_nemotron4-15b]="$NEMO"
)

INFERENCE=(
    inference_llama3.3
    inference_gpt-oss-dynamo
    inference_deepseek-r1
    inference_deepseek-r1-sglang
)

MICROBENCH=(
    microbenchmark_system_info
    microbenchmark_nccl
    microbenchmark_cpu_overhead
)

echo "==== Collecting Phase 1 results into $OUT_DIR ===="

# ---- Training / finetune: copy logs + run parsers ----
for wl in "${!PARSER[@]}"; do
    src="$LLMB_INSTALL/workloads/$wl/experiments"
    dst="$OUT_DIR/$wl"
    [[ ! -d "$src" ]] && { echo "skip $wl (no experiments dir)"; continue; }

    mkdir -p "$dst/logs"
    # Mirror per-experiment .out files (rename to include workload+config so they're scannable)
    find "$src" \( -name 'log-*.out' -o -name 'slurm-*.out' \) -not -name 'sbatch_*' | while read -r f; do
        cp "$f" "$dst/logs/$(basename "$(dirname "$(dirname "$f")")")_$(basename "$f")"
    done

    # Run the dgxc parser for this workload's framework
    parser="${PARSER[$wl]}"
    if [[ -x "$parser" ]]; then
        echo "  parsing $wl with $(basename "$parser")"
        "$parser" "$src" --format=csv >"$dst/parsed.csv" 2>"$dst/parser.log" || \
            echo "    parser exited non-zero — see $dst/parser.log"
        "$parser" "$src" --format=table >"$dst/parsed.txt" 2>/dev/null || true
    else
        echo "  skip parser for $wl (not at $parser)"
    fi
done

# ---- Inference: per-anatomy harvest (see quirks.md "dgxc workload anatomy") ----
# Anatomy B (TRT-LLM trtllm-bench, e.g. inference_llama3.3, inference_deepseek-r1):
#   experiments/<MODEL>_..._<USECASE>/<MODEL>_..._<USECASE>_streaming-on_<JOBID>.out
#   plus slurm-<JOBID>.out at workload root (same content, captured by outer Slurm).
# Anatomy C (Dynamo + AI Perf, e.g. inference_gpt-oss-dynamo, inference_deepseek-r1-dynamo):
#   experiments/<CONFIG>_<JOBID>/benchmark_logs/profile_export_aiperf.csv  <- METRICS
#   slurm-*.out at workload root contains NO metrics for Dynamo runs.
for wl in "${INFERENCE[@]}"; do
    src="$LLMB_INSTALL/workloads/$wl/experiments"
    root="$LLMB_INSTALL/workloads/$wl"
    dst="$OUT_DIR/$wl"
    [[ ! -d "$src" ]] && { echo "skip $wl (no experiments dir)"; continue; }

    mkdir -p "$dst/logs" "$dst/csv"

    # Anatomy B: per-use-case logs from inside experiment dirs + slurm wrappers at root
    find "$src" -name '*_streaming-*.out' 2>/dev/null | while read -r f; do
        cp "$f" "$dst/logs/$(basename "$(dirname "$f")")_$(basename "$f")"
    done
    find "$root" -maxdepth 1 -name 'slurm-*.out' 2>/dev/null | while read -r f; do
        cp "$f" "$dst/logs/$(basename "$f")"
    done

    # Anatomy C: AI Perf CSV output for Dynamo workloads
    find "$src" -name 'profile_export_aiperf.csv' 2>/dev/null | while read -r f; do
        # name as <experiment_dir>.csv so workload + jobid are recoverable
        cp "$f" "$dst/csv/$(basename "$(dirname "$(dirname "$f")")").csv"
    done
    # Also keep the JSON sibling — easier to programmatically parse than the CSV
    find "$src" -name 'profile_export_aiperf.json' 2>/dev/null | while read -r f; do
        cp "$f" "$dst/csv/$(basename "$(dirname "$(dirname "$f")")").json"
    done

    # Human-readable performance blocks extracted from TRT-LLM logs.
    # Skip logs that contain no metrics (failed/incomplete runs, or Dynamo's
    # slurm wrappers that never carry TRT-LLM output) so we don't emit empty
    # "--- filename ---" headers. Only write performance_blocks.txt if at
    # least one log produced content.
    content=""
    for f in $(find "$src" -name '*_streaming-*.out' 2>/dev/null; find "$root" -maxdepth 1 -name 'slurm-*.out' 2>/dev/null); do
        [ -f "$f" ] || continue
        block=$({ sed -n '/= PERFORMANCE OVERVIEW/,/Per User Output Speed/p' "$f"; sed -n '/Serving Benchmark Result/,/^={5,}/p' "$f"; })
        [ -z "$block" ] && continue
        content+="--- $(basename "$f") ---"$'\n'"$block"$'\n\n'
    done
    if [ -n "$content" ]; then
        {
            echo "==== $wl performance blocks ===="
            printf '%s' "$content"
        } >"$dst/performance_blocks.txt"
    else
        # Clean up any stale file from a previous run so the aggregator skips this workload
        rm -f "$dst/performance_blocks.txt"
    fi
done

# ---- Microbenchmarks: just copy logs ----
for wl in "${MICROBENCH[@]}"; do
    src="$LLMB_INSTALL/workloads/$wl/experiments"
    dst="$OUT_DIR/$wl"
    [[ ! -d "$src" ]] && continue

    mkdir -p "$dst/logs"
    find "$src" \( -name 'log-*.out' -o -name 'slurm-*.out' \) -not -name 'sbatch_*' | while read -r f; do
        cp "$f" "$dst/logs/$(basename "$(dirname "$(dirname "$f")")")_$(basename "$f")"
    done
done

# ---- NCCL bus BW (phase 0 collective sweep): copy the most recent test log ----
# 40_nccl_tests.sh writes to $PLAN/acceptance/results/nccl_tests_<jobid>.log
NCCL_SRC="$(dirname "$0")/acceptance/results"
if compgen -G "$NCCL_SRC/nccl_tests_*.log" >/dev/null 2>&1; then
    mkdir -p "$OUT_DIR/nccl_bus_bw"
    latest=$(ls -t "$NCCL_SRC"/nccl_tests_*.log | head -1)
    cp "$latest" "$OUT_DIR/nccl_bus_bw/$(basename "$latest")"
    echo "  copied NCCL bus BW log: $(basename "$latest")"
fi

# ---- Generate summary.md and summary.html ----
AGG="$(dirname "$0")/aggregate_results.py"
HTML="$(dirname "$0")/summary_to_html.py"
if [[ -x "$AGG" || -f "$AGG" ]]; then
    echo
    echo "==== Generating summary.md ===="
    if python3 "$AGG" "$OUT_DIR" >"$OUT_DIR/summary.md" 2>"$OUT_DIR/summary.log"; then
        cat "$OUT_DIR/summary.md"
        if [[ -f "$HTML" ]]; then
            python3 "$HTML" "$OUT_DIR/summary.md" "$OUT_DIR/summary.html" \
                && echo "  also wrote $OUT_DIR/summary.html"
        fi
    else
        echo "  aggregator failed — see $OUT_DIR/summary.log"
        cat "$OUT_DIR/summary.log"
    fi
fi

# ---- Optional: NVIDIA submission archive ----
if [[ "$DO_ARCHIVE" == "1" ]]; then
    echo
    echo "==== Building llmb-run archive ===="
    llmb-run archive --output "$OUT_DIR/llmb-archive-b200vultr.tar.zst"
    ls -lh "$OUT_DIR"/llmb-archive-*.tar.zst
fi

echo
echo "Done. Results at $OUT_DIR"
echo "Next: review summary.md, paste relevant tables into execution.md."
