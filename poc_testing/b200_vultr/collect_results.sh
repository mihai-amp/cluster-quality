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
    find "$src" -name 'slurm-*.out' -o -name '*.out' | while read -r f; do
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

# ---- Inference: copy logs + extract performance blocks ----
for wl in "${INFERENCE[@]}"; do
    src="$LLMB_INSTALL/workloads/$wl/experiments"
    dst="$OUT_DIR/$wl"
    [[ ! -d "$src" ]] && { echo "skip $wl (no experiments dir)"; continue; }

    mkdir -p "$dst/logs"
    find "$src" -name '*.out' | while read -r f; do
        cp "$f" "$dst/logs/$(basename "$(dirname "$(dirname "$f")")")_$(basename "$f")"
    done

    # Extract throughput / latency blocks
    {
        echo "==== $wl performance blocks ===="
        for f in $(find "$src" -name '*.out'); do
            echo "--- $(basename "$(dirname "$(dirname "$f")")") ---"
            awk '/PERFORMANCE OVERVIEW|Serving Benchmark Result|Throughput/,/^={5,}|^$/' "$f"
            echo
        done
    } >"$dst/performance_blocks.txt"
done

# ---- Microbenchmarks: just copy logs ----
for wl in "${MICROBENCH[@]}"; do
    src="$LLMB_INSTALL/workloads/$wl/experiments"
    dst="$OUT_DIR/$wl"
    [[ ! -d "$src" ]] && continue

    mkdir -p "$dst/logs"
    find "$src" -name '*.out' | while read -r f; do
        cp "$f" "$dst/logs/$(basename "$(dirname "$(dirname "$f")")")_$(basename "$f")"
    done
done

# ---- Status summary ----
{
    echo "# Phase 1 collection summary"
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Source:    $LLMB_INSTALL/workloads/"
    echo
    echo "## Per-workload status"
    echo
    printf "| Workload | Logs collected | Parser | Notes |\n"
    printf "|---|---:|---|---|\n"
    for wl in "${!PARSER[@]}" "${INFERENCE[@]}" "${MICROBENCH[@]}"; do
        dst="$OUT_DIR/$wl"
        [[ ! -d "$dst" ]] && continue
        n=$(find "$dst/logs" -name '*.out' 2>/dev/null | wc -l)
        parser="-"
        [[ -f "$dst/parsed.csv" ]] && parser="parsed.csv"
        [[ -f "$dst/performance_blocks.txt" ]] && parser="performance_blocks.txt"
        # Quick fail/ok scan
        fails=$(grep -lE 'Traceback|CANCELLED|ERROR' "$dst/logs/"*.out 2>/dev/null | wc -l)
        notes="$n logs, $fails failures"
        printf "| %s | %d | %s | %s |\n" "$wl" "$n" "$parser" "$notes"
    done
} >"$OUT_DIR/summary.md"

echo
echo "==== Summary ===="
cat "$OUT_DIR/summary.md"

# ---- Optional: NVIDIA submission archive ----
if [[ "$DO_ARCHIVE" == "1" ]]; then
    echo
    echo "==== Building llmb-run archive ===="
    llmb-run archive --output "$OUT_DIR/llmb-archive-b200vultr.tar.zst"
    ls -lh "$OUT_DIR"/llmb-archive-*.tar.zst
fi

echo
echo "Done. Results at $OUT_DIR"
echo "Next: review parsed.csv files, fill in execution.md result tables."
