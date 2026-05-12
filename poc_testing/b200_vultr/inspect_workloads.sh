#!/usr/bin/env bash
# Dump the directory anatomy of each $LLMB_INSTALL/workloads/<wl>/ so we can
# see exactly where logs, metric files, and configs land per workload type.
# Output goes to stdout — redirect to a file for sharing without copy-paste
# spacing pain:
#
#   bash $PLAN/inspect_workloads.sh > workload_anatomy.txt
#
# For each workload: shows the first 2 experiment subdirs (depth-5 file walk
# limited to *.out / *.log / *.csv / *.json / *.yaml) plus anything at the
# workload root.

set -uo pipefail

: "${LLMB_INSTALL:?source env.sh first}"

for wl_dir in "$LLMB_INSTALL"/workloads/*/; do
    wl=$(basename "$wl_dir")
    echo "=================================================="
    echo "=== $wl"
    echo "=================================================="

    if [ -d "$wl_dir/experiments" ]; then
        n=0
        for exp in $(ls -d "$wl_dir"/experiments/*/ 2>/dev/null); do
            n=$((n+1))
            [ "$n" -gt 2 ] && { echo "  ... ($(ls -d "$wl_dir"/experiments/*/ 2>/dev/null | wc -l) experiments total, showing first 2)"; break; }
            echo "EXPERIMENT: $(basename "$exp")"
            find "$exp" -maxdepth 5 -type f \
                \( -name '*.out' -o -name '*.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' \) \
                2>/dev/null | sed "s|${exp%/}/||" | sed 's|^|  |'
        done
    fi

    # Anything at the workload root
    local_files=$(ls -1 "$wl_dir" 2>/dev/null | grep -E '\.(out|log|csv|yaml)$')
    if [ -n "$local_files" ]; then
        echo "WORKLOAD ROOT (loose files):"
        echo "$local_files" | sed 's|^|  |'
    fi
    echo
done
