#!/usr/bin/env python3
"""Aggregate dgxc parser CSV outputs across repeats and emit a markdown
comparison table against the plan-expected MFU ranges.

Reads $MY/results/phase1/*/parsed.csv (produced by collect_results.sh which
itself wraps parse_train_timing[_mbridge].sh from the dgxc repo). Groups
successful runs by (workload, size, dtype, scale), computes mean/stddev of
step time and mean TFLOPS/GPU, computes MFU% using B200 peak TFLOPS per
dtype, and tags each row green/yellow/red against plan.md §4.1.

Usage: aggregate_results.py [results_dir]
       results_dir defaults to $MY/results/phase1.
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# B200 dense peak TFLOPS per dtype (plan.md §2)
PEAK_TFLOPS = {
    "bf16": 2250,
    "fp8": 4500,
    "nvfp4": 9000,
    "mxfp4": 9000,
}

# Plan-expected MFU% range by (workload, dtype) — from plan.md §4.1
EXPECTED_MFU = {
    ("pretrain_llama3.1", "fp8"): (45, 55),
    ("pretrain_llama3.1", "nvfp4"): (35, 50),
    ("pretrain_nemotron4-15b", "fp8"): (40, 50),
    ("pretrain_nemotron4-15b", "bf16"): (35, 45),
    ("pretrain_qwen3", "bf16"): (25, 40),
    ("finetune_llama3", "fp8"): (30, 45),
    ("finetune_llama3", "bf16"): (25, 40),
}

# Filename pattern produced by llmb-run; tolerant of pretrain/finetune/lora prefixes
FILENAME_RE = re.compile(
    r"(?P<dtype>fp8|bf16|nvfp4|mxfp4)(?:_cs)?_gpus(?P<scale>\d+)"
)
SIZE_RE = re.compile(r"_(?P<size>\d+[bm])(?:_a\d+b)?_(?:fp8|bf16|nvfp4|mxfp4)")


def parse_filename(fname: str):
    m = FILENAME_RE.search(fname)
    s = SIZE_RE.search(fname)
    if not m or not s:
        return None
    return {"dtype": m["dtype"], "scale": int(m["scale"]), "size": s["size"]}


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, var ** 0.5


def mfu_verdict(mfu: float, expected) -> str:
    if not expected:
        return "(no plan target)"
    low, high = expected
    if low <= mfu <= high:
        return f"in range [{low}-{high}%]"
    if mfu < low * 0.6:
        return f"RED — below {low}% by >40%"
    if mfu < low:
        return f"yellow — below plan [{low}-{high}%]"
    return f"yellow — above plan [{low}-{high}%]"


def aggregate(results_dir: Path):
    by_config = defaultdict(list)  # (workload, size, dtype, scale) -> [(time_ms, tflops), ...]
    fails_by_config = defaultdict(int)

    for parsed_csv in sorted(results_dir.glob("*/parsed.csv")):
        workload = parsed_csv.parent.name
        with open(parsed_csv) as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            meta = parse_filename(row.get("filename", ""))
            if not meta:
                continue
            key = (workload, meta["size"], meta["dtype"], meta["scale"])
            if row.get("status") != "Success":
                fails_by_config[key] += 1
                continue
            try:
                time_ms = float(row["time_mean_ms"])
                tflops = float(row["tflops_per_gpu_mean"])
            except (ValueError, KeyError):
                continue
            by_config[key].append((time_ms, tflops))

    return by_config, fails_by_config


def main():
    results_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else os.environ["MY"] + "/results/phase1"
    )
    if not results_dir.exists():
        print(f"# No results dir at {results_dir}", file=sys.stderr)
        sys.exit(1)

    by_config, fails = aggregate(results_dir)

    print("# Aggregated Phase 1 training results")
    print(f"Source: `{results_dir}`")
    print()
    if not by_config:
        print("_No successful runs parsed yet._")
        return

    print(
        "| Workload | Size | Dtype | Scale | n_ok | n_fail | "
        "Mean step (s) | Stddev step (s) | TFLOPS/GPU | MFU% | vs plan |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for key in sorted(by_config):
        workload, size, dtype, scale = key
        runs = by_config[key]
        n_fail = fails.get(key, 0)
        times = [r[0] for r in runs]
        tflops = [r[1] for r in runs]
        time_mean, time_std = mean_std(times)
        tflops_mean, _ = mean_std(tflops)
        peak = PEAK_TFLOPS.get(dtype, 0)
        mfu = tflops_mean / peak * 100 if peak else 0
        verdict = mfu_verdict(mfu, EXPECTED_MFU.get((workload, dtype)))
        print(
            f"| {workload} | {size} | {dtype} | {scale} | {len(runs)} | {n_fail} | "
            f"{time_mean / 1000:.3f} | {time_std / 1000:.3f} | "
            f"{tflops_mean:.0f} | {mfu:.1f}% | {verdict} |"
        )

    print()
    print("**Peak TFLOPS used (dense B200):**")
    for dtype, peak in PEAK_TFLOPS.items():
        print(f"- {dtype}: {peak}")


if __name__ == "__main__":
    main()
