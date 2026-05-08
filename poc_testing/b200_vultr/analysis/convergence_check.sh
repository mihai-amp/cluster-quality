#!/usr/bin/env bash
# Convergence proxy for SDC (Silent Data Corruption) detection.
#
# Approach: run the same workload+precision multiple times. If the recipe
# uses a fixed seed, loss curves should be bit-identical (or within FP
# tolerance). Divergence between repeats indicates non-determinism, which
# at the very least means our variance numbers are noisy and at worst
# means the GPUs are computing different results from identical inputs
# (= classic SDC signal).
#
# Usage:
#   ./convergence_check.sh <experiment_dir_glob>
# e.g.:
#   ./convergence_check.sh "$LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/*8b*fp8*scale8*"
#
# Reads each run's training log, extracts (step, loss) pairs, and
# pairwise-diffs the loss values across runs.

set -uo pipefail
HERE="$(dirname "$0")"
mkdir -p "$HERE/results"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment_dir_glob>" >&2
  echo "Example: $0 '\$LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/*8b*fp8*scale8*'" >&2
  exit 2
fi

GLOB="$1"
OUT="$HERE/results/convergence_$(date -u +%Y%m%dT%H%M%SZ).txt"

# Adjust the regex below to match the log line format in your recipe's training log.
# Common patterns:
#   "global_step=N | loss=X.YYYY"            (NeMo)
#   "step: N, loss: X.YYYY"                  (Megatron-Bridge)
#   "{'iter': N, 'reduced_train_loss': X.YYYY}" (Megatron core)
LOSS_REGEX='(reduced_train_loss|loss)[^0-9]*([0-9]+)[^0-9]*([0-9]+\.[0-9]+)'

extract_losses() {
  local logfile="$1"
  python3 - "$logfile" "$LOSS_REGEX" <<'PY'
import re, sys
path, pattern = sys.argv[1], sys.argv[2]
rx = re.compile(pattern)
with open(path) as f:
  for line in f:
    m = rx.search(line)
    if m:
      groups = m.groups()
      step = next((g for g in groups if g.isdigit()), None)
      loss = next((g for g in groups if '.' in g), None)
      if step and loss:
        print(f"{step}\t{loss}")
PY
}

{
  echo "==== convergence check $(date -u) ===="
  echo "Glob: $GLOB"

  # shellcheck disable=SC2086
  runs=( $(ls -d $GLOB 2>/dev/null) )
  echo "Found ${#runs[@]} runs"
  for r in "${runs[@]}"; do echo "  $r"; done
  if [ "${#runs[@]}" -lt 2 ]; then
    echo "Need >=2 runs to compare." >&2
    exit 1
  fi

  # Extract losses to temp files
  tmp_dir=$(mktemp -d)
  i=0
  for r in "${runs[@]}"; do
    log=$(find "$r" -name '*.out' -o -name 'log-account*.out' -o -name 'training.log' 2>/dev/null | head -1)
    if [ -z "$log" ]; then
      echo "No log file in $r; skipping" >&2
      continue
    fi
    extract_losses "$log" >"$tmp_dir/run_$i.tsv"
    echo
    echo "Run $i (from $log): $(wc -l <"$tmp_dir/run_$i.tsv") loss points"
    head -5 "$tmp_dir/run_$i.tsv"
    i=$((i+1))
  done

  # Pairwise diff
  echo
  echo "==== pairwise loss differences ===="
  python3 - "$tmp_dir" <<'PY'
import os, sys, glob
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "run_*.tsv")))
data = []
for f in files:
  pts = {}
  with open(f) as fh:
    for line in fh:
      step, loss = line.strip().split("\t")
      pts[int(step)] = float(loss)
  data.append((f, pts))

for i in range(len(data)):
  for j in range(i+1, len(data)):
    fi, pi = data[i]
    fj, pj = data[j]
    common = sorted(set(pi) & set(pj))
    if not common:
      print(f"{os.path.basename(fi)} vs {os.path.basename(fj)}: no common steps")
      continue
    diffs = [abs(pi[s] - pj[s]) for s in common]
    mean = sum(diffs) / len(diffs)
    mx = max(diffs)
    bit_identical = mx == 0.0
    print(f"{os.path.basename(fi)} vs {os.path.basename(fj)}: "
          f"{len(common)} common steps, "
          f"mean|delta|={mean:.6e}, max|delta|={mx:.6e}, "
          f"bit_identical={bit_identical}")
PY

  rm -rf "$tmp_dir"
} | tee "$OUT"

echo
echo "Wrote $OUT"
echo
echo "Interpretation:"
echo "  bit_identical=True  -> deterministic; no SDC signal."
echo "  max|delta|<1e-4     -> within FP noise; almost certainly fine."
echo "  max|delta|>1e-3     -> investigate. Could be non-deterministic kernel,"
echo "                          different random init, or actual SDC."
