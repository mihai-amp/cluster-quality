#!/usr/bin/env python3
"""Generate the Phase 1 summary.md from collected results.

Sections produced (summaries first, then per-run detail, then raw parser output):
  1. Runs per workload   — counts of logs, total runs, OK, Failed
  2. Training summary    — one row per (workload, size, dtype, scale)
  3. Finetune summary    — one row per (workload, size, dtype, scale)
  4. Inference summary   — one row per (workload, size, dtype, scale)
  5. NCCL bus bandwidth  — peak/avg busbw per collective (phase 0 NCCL sweep)
  6. Training full       — every successful run, per-run row
  7. Finetune full       — every successful run, per-run row
  8. Inference full      — every parsed log, per-run row
  9. Raw parser output   — verbatim dgxc parser tables + inference performance blocks

Reads:
  $MY/results/phase1/<workload>/parsed.csv          (training/finetune, from dgxc parsers)
  $MY/results/phase1/<workload>/logs/*.out          (any workload, scanned for inference metrics)

Usage: aggregate_results.py [results_dir]
       results_dir defaults to $MY/results/phase1.
"""

import csv
import datetime
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# B200 dense peak TFLOPS per dtype (used to compute MFU%, not for verdicts)
PEAK_TFLOPS = {"bf16": 2250, "fp8": 4500, "nvfp4": 9000, "mxfp4": 9000}

# Inference workload metadata — size / dtype / scale are properties of the workload,
# not encoded in per-experiment filenames. Sourced from `llmb-run list` and the
# dgxc README's per-workload config.
INFERENCE_META = {
    "inference_llama3.3":          {"size": "70b",  "dtype": "nvfp4", "scale": 1},
    "inference_deepseek-r1":       {"size": "671b", "dtype": "nvfp4", "scale": 4},
    "inference_deepseek-r1-dynamo": {"size": "671b", "dtype": "nvfp4", "scale": 32},
    "inference_deepseek-r1-sglang": {"size": "671b", "dtype": "nvfp4", "scale": 8},
    "inference_gpt-oss-dynamo":    {"size": "120b", "dtype": "mxfp4", "scale": 4},
}

FILENAME_RE = re.compile(r"(?P<dtype>fp8|bf16|nvfp4|mxfp4)(?:_cs)?_gpus(?P<scale>\d+)")
SIZE_RE = re.compile(r"_(?P<size>\d+[bm])(?:_a\d+b)?_(?:fp8|bf16|nvfp4|mxfp4)")


def parse_filename(fname):
    m = FILENAME_RE.search(fname)
    s = SIZE_RE.search(fname)
    if not m or not s:
        return None
    return {"dtype": m["dtype"], "scale": int(m["scale"]), "size": s["size"]}


def workload_type(wl):
    if wl.startswith("pretrain_"):
        return "training"
    if wl.startswith("finetune_"):
        return "finetune"
    if wl.startswith("inference_"):
        return "inference"
    if wl.startswith("microbenchmark_"):
        return "microbench"
    return "other"


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    return m, (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def collect_counts(results_dir: Path):
    """Per-workload counts: logs, runs (CSV rows or logs), ok, failed."""
    counts = {}
    for wl_dir in sorted(results_dir.iterdir()):
        if not wl_dir.is_dir():
            continue
        wl = wl_dir.name
        # Skip legacy empty bucket dirs left over from setup_workspace.sh
        if workload_type(wl) == "other":
            continue
        logs_dir = wl_dir / "logs"
        logs = list(logs_dir.glob("*.out")) if logs_dir.exists() else []
        n_logs = len(logs)

        parsed = wl_dir / "parsed.csv"
        n_ok = n_fail = n_total = 0

        if parsed.exists():
            # For training/finetune, the parser knows ground truth from its own iteration scan
            with open(parsed) as f:
                reader = csv.DictReader(line for line in f if not line.startswith("#"))
                for row in reader:
                    n_total += 1
                    status = (row.get("status") or "").strip()
                    if status == "Success":
                        n_ok += 1
                    else:
                        n_fail += 1
        else:
            # Inference / microbench — heuristic scan of slurm logs
            for log in logs:
                n_total += 1
                try:
                    content = log.read_text(errors="ignore")
                except Exception:
                    n_fail += 1
                    continue
                if re.search(r"Traceback|CANCELLED|FAIL|ERROR|Failed to|exited on signal", content):
                    n_fail += 1
                else:
                    n_ok += 1
        counts[wl] = (n_logs, n_total, n_ok, n_fail)
    return counts


def read_training_csv(wl_dir: Path):
    """Read parsed.csv from a training/finetune workload dir."""
    parsed = wl_dir / "parsed.csv"
    if not parsed.exists():
        return []
    rows = []
    with open(parsed) as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            meta = parse_filename(row.get("filename", ""))
            if not meta:
                continue
            # mbridge writes time_mean_ms, NeMo writes time_mean_seconds — accept either, normalize to ms
            step_ms = row.get("time_mean_ms", "").strip()
            step_std_ms = row.get("time_std_dev_ms", "").strip()
            if not step_ms and row.get("time_mean_seconds", "").strip():
                try:
                    step_ms = f"{float(row['time_mean_seconds']) * 1000:.3f}"
                except (ValueError, TypeError):
                    pass
            if not step_std_ms and row.get("time_std_dev_seconds", "").strip():
                try:
                    step_std_ms = f"{float(row['time_std_dev_seconds']) * 1000:.3f}"
                except (ValueError, TypeError):
                    pass
            rows.append(
                {
                    "workload": wl_dir.name,
                    **meta,
                    "status": (row.get("status") or "").strip(),
                    "step_ms": step_ms,
                    "step_std_ms": step_std_ms,
                    "tflops": row.get("tflops_per_gpu_mean", "").strip(),
                }
            )
    return rows


def parse_inference_log(log_path: Path):
    """Extract metrics from an inference workload's PERFORMANCE OVERVIEW block.

    Anchors on TRT-LLM's standard 10-line summary (per dgxc llama3.3/inference README):
      Request Throughput (req/sec)
      Total Output Throughput (tokens/sec)
      Total Token Throughput (tokens/sec)
      Total Latency (ms)
      Average request latency (ms)
      Per User Output Throughput [w/ ctx] (tps/user)
      Per GPU Output Throughput (tps/gpu)
      Average time-to-first-token [TTFT] (ms)
      Average time-per-output-token [TPOT] (ms)
      Per User Output Speed (tps/user)
    """
    try:
        content = log_path.read_text(errors="ignore")
    except Exception:
        return None

    fields = [
        ("req_per_s", r"Request Throughput \(req/sec\):\s*([\d.eE+-]+)"),
        ("total_output_tok_per_s", r"Total Output Throughput \(tokens/sec\):\s*([\d.eE+-]+)"),
        ("total_token_tok_per_s", r"Total Token Throughput \(tokens/sec\):\s*([\d.eE+-]+)"),
        ("total_latency_ms", r"Total Latency \(ms\):\s*([\d.eE+-]+)"),
        ("avg_req_latency_ms", r"Average request latency \(ms\):\s*([\d.eE+-]+)"),
        ("per_user_throughput", r"Per User Output Throughput \[w/ ctx\] \(tps/user\):\s*([\d.eE+-]+)"),
        ("per_gpu_throughput", r"Per GPU Output Throughput \(tps/gpu\):\s*([\d.eE+-]+)"),
        ("ttft_ms", r"Average time-to-first-token \[TTFT\] \(ms\):\s*([\d.eE+-]+)"),
        ("tpot_ms", r"Average time-per-output-token \[TPOT\] \(ms\):\s*([\d.eE+-]+)"),
        ("per_user_speed", r"Per User Output Speed \(tps/user\):\s*([\d.eE+-]+)"),
    ]
    metrics = {}
    for key, pattern in fields:
        m = re.search(pattern, content)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except (ValueError, TypeError):
                pass

    # Use case from path/filename (set by dgxc launcher convention)
    uc = re.search(r"_(reasoning|chat|summarization|generation)(?:_|$|\.)", str(log_path))
    if uc:
        metrics["use_case"] = uc.group(1)

    return metrics if metrics else None


def collect_inference(wl_dir: Path):
    """Parse every log for an inference workload."""
    logs_dir = wl_dir / "logs"
    if not logs_dir.exists():
        return []
    rows = []
    inf_meta = INFERENCE_META.get(wl_dir.name, {})
    for log in sorted(logs_dir.glob("*.out")):
        m = parse_inference_log(log)
        if m is None:
            continue
        # Require at least one perf metric — skip logs that parsed use_case but no
        # PERFORMANCE OVERVIEW (incomplete or failed runs)
        if not (set(m.keys()) - {"use_case"}):
            continue
        meta = parse_filename(log.name) or {}
        rows.append({"workload": wl_dir.name, "log": log.name, **inf_meta, **meta, **m})
    return rows


NCCL_HEADER_RE = re.compile(r"^====\s*(\S+)\s+ranks=(\d+)\s+\((.+?)\)\s*====")


def parse_nccl_log(log_path: Path):
    """Parse NCCL tests output into per-section busbw summaries.

    Each `==== <op> ranks=<n> (<scope>) ====` header starts a section.
    Within a section, locate the `# size ... busbw ...` header to find the
    out-of-place busbw column, then read numeric data rows for that column.
    The `# Avg bus bandwidth : X` footer is captured if present.
    """
    sections = []
    current = None
    busbw_col = None

    for raw in log_path.read_text().splitlines():
        m = NCCL_HEADER_RE.match(raw.strip())
        if m:
            if current is not None:
                sections.append(current)
            current = {
                "op": m.group(1),
                "ranks": int(m.group(2)),
                "scope": m.group(3),
                "peak_busbw": None,
                "peak_size": None,
                "avg_busbw": None,
                "rows": [],
            }
            busbw_col = None
            continue

        if current is None:
            continue

        # Column header — the first occurrence of `busbw` is the out-of-place column
        if raw.lstrip().startswith("#") and "size" in raw and "busbw" in raw:
            fields = raw.lstrip("#").split()
            try:
                busbw_col = fields.index("busbw")
            except ValueError:
                busbw_col = None
            continue

        # Avg footer
        if "Avg bus bandwidth" in raw:
            try:
                current["avg_busbw"] = float(raw.split(":")[-1].strip())
            except ValueError:
                pass
            continue

        # Data row: starts with whitespace + digit
        if busbw_col is not None and re.match(r"^\s*\d", raw):
            fields = raw.split()
            try:
                size = int(fields[0])
                busbw = float(fields[busbw_col])
            except (ValueError, IndexError):
                continue
            current["rows"].append((size, busbw))
            if current["peak_busbw"] is None or busbw > current["peak_busbw"]:
                current["peak_busbw"] = busbw
                current["peak_size"] = size

    if current is not None:
        sections.append(current)
    return sections


def collect_nccl(results_dir: Path):
    """Return parsed NCCL sections from the most recent log in nccl_bus_bw/."""
    nccl_dir = results_dir / "nccl_bus_bw"
    if not nccl_dir.is_dir():
        return [], None
    logs = sorted(nccl_dir.glob("nccl_tests_*.log"))
    if not logs:
        return [], None
    log = logs[-1]
    return parse_nccl_log(log), log.name


def fmt(v, spec=""):
    if v in ("", None):
        return "-"
    try:
        if spec:
            return format(float(v), spec)
        return str(v)
    except (ValueError, TypeError):
        return str(v)


def main():
    results_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else os.environ["MY"] + "/results/phase1"
    )
    if not results_dir.exists():
        print(f"# No results dir at {results_dir}", file=sys.stderr)
        sys.exit(1)

    print("# Phase 1 collection summary")
    print(f"Generated: {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z")
    print(f"Source: `{results_dir}`")
    print()

    # -------- Section 1: counts ----------------
    counts = collect_counts(results_dir)
    print("## 1. Successful runs per workload")
    print()
    print("Per-workload count of successful runs (workloads with zero successes are omitted).")
    print()
    print("| Workload | Successful runs |")
    print("|---|---:|")
    # Drop workloads with no successful runs — they'd be noise in the summary
    for wl, (_logs, _runs, ok, _fail) in sorted(counts.items()):
        if ok == 0:
            continue
        print(f"| {wl} | {ok} |")
    print()

    # -------- gather rows by type --------------
    training_rows, finetune_rows, inference_rows = [], [], []
    for wl_dir in sorted(results_dir.iterdir()):
        if not wl_dir.is_dir():
            continue
        wt = workload_type(wl_dir.name)
        if wt == "training":
            training_rows.extend(read_training_csv(wl_dir))
        elif wt == "finetune":
            finetune_rows.extend(read_training_csv(wl_dir))
        elif wt == "inference":
            inference_rows.extend(collect_inference(wl_dir))

    # -------- summarize helper for training/finetune --------
    def summarize_training(rows):
        by_config = defaultdict(list)
        for r in rows:
            if r["status"] != "Success":
                continue
            try:
                t = float(r["step_ms"])
                f = float(r["tflops"])
                wr_std = float(r["step_std_ms"]) if r.get("step_std_ms") else 0.0
            except (ValueError, TypeError):
                continue
            by_config[(r["workload"], r["size"], r["dtype"], r["scale"])].append((t, f, wr_std))
        return by_config

    def print_training_summary(header, by_config):
        print(header)
        print()
        print("Each row aggregates all successful runs grouped by (workload, size, dtype, scale).")
        print()
        print(
            "| Workload | Size | Dtype | Scale | n | "
            "Step mean (ms) | Step min (ms) | Step max (ms) | "
            "Within-run σ mean (ms) | σ across runs (ms) | "
            "TFLOPS mean | TFLOPS min | TFLOPS max | Peak TFLOPS | MFU% |"
        )
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        if not by_config:
            print("| _no successful runs yet_ | | | | | | | | | | | | | | | |")
        for key, runs in sorted(by_config.items()):
            wl, size, dtype, scale = key
            times = [r[0] for r in runs]
            tfs = [r[1] for r in runs]
            wr_stds = [r[2] for r in runs]
            tm, ts = mean_std(times)
            fm, _ = mean_std(tfs)
            t_min, t_max = min(times), max(times)
            f_min, f_max = min(tfs), max(tfs)
            wr_mean = sum(wr_stds) / len(wr_stds) if wr_stds else 0.0
            peak = PEAK_TFLOPS.get(dtype, 0)
            mfu = fm / peak * 100 if peak else 0
            print(
                f"| {wl} | {size} | {dtype} | {scale} | {len(runs)} | "
                f"{tm:.1f} | {t_min:.1f} | {t_max:.1f} | "
                f"{wr_mean:.1f} | {ts:.1f} | "
                f"{fm:.0f} | {f_min:.0f} | {f_max:.0f} | {peak} | {mfu:.1f}% |"
            )
        print()
        print("**Legend:**")
        print("- **Step mean/min/max** = per-step training time across runs in this config.")
        print("- **Within-run σ mean** = mean of per-run step-time std-dev (variance inside one run).")
        print("- **σ across runs** = std-dev of the per-run mean step time (variance between runs).")
        print("- **TFLOPS** = effective TFLOPS/GPU reported by the dgxc parser.")
        print("- **Peak TFLOPS** = B200 dense peak for this dtype (bf16: 2250, fp8: 4500, nvfp4/mxfp4: 9000).")
        print("- **MFU%** = TFLOPS mean / Peak TFLOPS × 100.")
        print()

    # -------- Section 2/3: training & finetune summaries --------
    print_training_summary("## 2. Training — summary per model", summarize_training(training_rows))
    print_training_summary("## 3. Finetune — summary per model", summarize_training(finetune_rows))

    # -------- Section 4: Inference summary --------
    print("## 4. Inference — summary per model (across use cases)")
    print()
    print("Each row aggregates all parsed use cases of an inference workload.")
    print()
    print(
        "| Workload | Size | Dtype | Scale | n use cases | "
        "TPS/GPU mean | TPS/GPU min | TPS/GPU max | "
        "TTFT mean (ms) | TTFT min | TTFT max | "
        "TPOT mean (ms) | TPOT min | TPOT max |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_inf = defaultdict(list)
    for r in inference_rows:
        by_inf[(r["workload"], r.get("size", ""), r.get("dtype", ""), r.get("scale", ""))].append(r)
    if not by_inf:
        print("| _no parsed inference results yet_ | | | | | | | | | | | | | |")
    for key, runs in sorted(by_inf.items()):
        wl, size, dtype, scale = key
        tps = [r.get("per_gpu_throughput") for r in runs if r.get("per_gpu_throughput") is not None]
        ttfts = [r.get("ttft_ms") for r in runs if r.get("ttft_ms") is not None]
        tpots = [r.get("tpot_ms") for r in runs if r.get("tpot_ms") is not None]
        tps_mean = sum(tps) / len(tps) if tps else None
        ttft_mean = sum(ttfts) / len(ttfts) if ttfts else None
        tpot_mean = sum(tpots) / len(tpots) if tpots else None
        print(
            f"| {wl} | {fmt(size)} | {fmt(dtype)} | {fmt(scale)} | {len(runs)} | "
            f"{fmt(tps_mean, '.1f')} | {fmt(min(tps) if tps else None, '.1f')} | {fmt(max(tps) if tps else None, '.1f')} | "
            f"{fmt(ttft_mean, '.1f')} | {fmt(min(ttfts) if ttfts else None, '.1f')} | {fmt(max(ttfts) if ttfts else None, '.1f')} | "
            f"{fmt(tpot_mean, '.2f')} | {fmt(min(tpots) if tpots else None, '.2f')} | {fmt(max(tpots) if tpots else None, '.2f')} |"
        )
    print()
    print("**Legend:**")
    print("- **TPS/GPU** = output tokens/sec per GPU (per-device throughput).")
    print("- **TTFT** = Time-to-First-Token, ms — latency from request submission to first streamed token.")
    print("- **TPOT** = Time-Per-Output-Token, ms — steady-state per-token latency after TTFT.")
    print()
    print(
        "**Note on high TTFT.** Some rows show very high TTFT (e.g. summarization ≈260s, "
        "reasoning ≈29s) — these are CON640 runs that use the dgxc default "
        "`max_num_tokens=2048`. At high concurrency on long inputs (summarization "
        "`max_seq_len=8513`), the prefill scheduler can only consume 2048 tokens/step, "
        "so queued requests wait thousands of steps before their first token is emitted. "
        "Re-running the same workload at CON128/256 with `max_num_tokens=8192` yields "
        "the same per-GPU throughput at 15–24× lower TTFT (see Section 8 for the breakdown)."
    )
    print()

    # -------- Section 5: NCCL bus bandwidth --------
    nccl_sections, nccl_logname = collect_nccl(results_dir)
    print("## 5. NCCL bus bandwidth")
    print()
    if nccl_sections:
        print(f"Per-collective busbw from `acceptance/40_nccl_tests.sh` (source: `{nccl_logname}`).")
        print("Sweep range: 16 MiB → 8 GiB. Peak is the maximum out-of-place busbw observed; ")
        print("Avg is NCCL's own `# Avg bus bandwidth` footer.")
        print()
        print("| Collective | Ranks | Scope | Peak busbw (GB/s) | At size | Avg busbw (GB/s) |")
        print("|---|---:|---|---:|---:|---:|")
        for s in nccl_sections:
            peak = s["peak_busbw"]
            peak_size = s["peak_size"]
            avg = s["avg_busbw"]
            if peak_size is not None:
                size_label = f"{peak_size / (1024 ** 3):.2f} GiB" if peak_size >= 1024 ** 3 else f"{peak_size // (1024 ** 2)} MiB"
            else:
                size_label = "-"
            print(
                f"| {s['op']} | {s['ranks']} | {s['scope']} | "
                f"{fmt(peak, '.2f')} | {size_label} | {fmt(avg, '.2f')} |"
            )
        print()
        print("**Legend:**")
        print("- **Collective** = NCCL op (`all_reduce`, `all_gather`, `reduce_scatter`, `alltoall`).")
        print("- **Ranks** = total GPUs in the test (8 = 1 node intra-node; 16 = 2 nodes inter-node).")
        print("- **Peak busbw** = max out-of-place busbw across the sweep, at the listed message size.")
        print("- **Avg busbw** = NCCL's per-run `# Avg bus bandwidth` footer (mean across the sweep).")
        print()
        print(
            "**Sanity envelope.** B200 NVLink5 intra-node `all_reduce`/`alltoall` ≳ 350 GB/s at large sizes; "
            "inter-node 2-node `all_reduce` on 8× NDR ≳ 60–80 GB/s; "
            "inter-node `alltoall` is scaling-limited, typically ~40–60 GB/s."
        )
    else:
        print(
            "_No NCCL bus BW log captured. Run `sbatch $PLAN/acceptance/40_nccl_tests.sh`, "
            "then re-run `collect_results.sh`._"
        )
    print()

    # -------- Section 6: Training full --------
    print("## 6. Training — full results (every successful run)")
    print()
    print("One row per successful training run, no aggregation.")
    print()
    print("| Workload | Size | Dtype | Scale | Step mean (ms) | Step σ (ms) | TFLOPS/GPU | Peak TFLOPS |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(training_rows, key=lambda x: (x["workload"], x["dtype"], x["scale"])):
        if r["status"] != "Success":
            continue
        peak = PEAK_TFLOPS.get(r["dtype"], 0)
        print(
            f"| {r['workload']} | {r['size']} | {r['dtype']} | {r['scale']} | "
            f"{fmt(r['step_ms'])} | {fmt(r['step_std_ms'])} | {fmt(r['tflops'])} | {peak} |"
        )
    if not any(r for r in training_rows if r["status"] == "Success"):
        print("| _no successful runs yet_ | | | | | | | |")
    print()
    print("**Legend:**")
    print("- **Step mean** = mean per-step training time within this run.")
    print("- **Step σ** = std-dev of step time within this run.")
    print("- **TFLOPS/GPU** = effective TFLOPS per GPU.")
    print("- **Peak TFLOPS** = B200 dense peak for this dtype (bf16: 2250, fp8: 4500, nvfp4/mxfp4: 9000).")
    print()

    # -------- Section 7: Finetune full --------
    print("## 7. Finetune — full results (every successful run)")
    print()
    print("One row per successful finetune run, no aggregation.")
    print()
    print("| Workload | Size | Dtype | Scale | Step mean (ms) | Step σ (ms) | TFLOPS/GPU | Peak TFLOPS |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(finetune_rows, key=lambda x: (x["workload"], x["dtype"], x["scale"])):
        if r["status"] != "Success":
            continue
        peak = PEAK_TFLOPS.get(r["dtype"], 0)
        print(
            f"| {r['workload']} | {r['size']} | {r['dtype']} | {r['scale']} | "
            f"{fmt(r['step_ms'])} | {fmt(r['step_std_ms'])} | {fmt(r['tflops'])} | {peak} |"
        )
    if not any(r for r in finetune_rows if r["status"] == "Success"):
        print("| _no successful runs yet_ | | | | | | | |")
    print()
    print("**Legend:**")
    print("- **Step mean** = mean per-step training time within this run.")
    print("- **Step σ** = std-dev of step time within this run.")
    print("- **TFLOPS/GPU** = effective TFLOPS per GPU.")
    print("- **Peak TFLOPS** = B200 dense peak for this dtype (bf16: 2250, fp8: 4500, nvfp4/mxfp4: 9000).")
    print()

    # -------- Section 8: Inference full --------
    print("## 8. Inference — full results (every use case)")
    print()
    print("One row per parsed inference use case, no aggregation.")
    print()
    print(
        "| Workload | Size | Dtype | Scale | Use case | "
        "Req/s | Total output tok/s | TPS/GPU | TPS/User | "
        "Avg req latency (ms) | TTFT (ms) | TPOT (ms) |"
    )
    print("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(inference_rows, key=lambda x: (x["workload"], x.get("use_case", ""))):
        print(
            f"| {r['workload']} | {fmt(r.get('size'))} | {fmt(r.get('dtype'))} | "
            f"{fmt(r.get('scale'))} | {fmt(r.get('use_case'))} | "
            f"{fmt(r.get('req_per_s'), '.2f')} | "
            f"{fmt(r.get('total_output_tok_per_s'), '.1f')} | "
            f"{fmt(r.get('per_gpu_throughput'), '.1f')} | "
            f"{fmt(r.get('per_user_throughput'), '.2f')} | "
            f"{fmt(r.get('avg_req_latency_ms'), '.1f')} | "
            f"{fmt(r.get('ttft_ms'), '.1f')} | "
            f"{fmt(r.get('tpot_ms'), '.2f')} |"
        )
    if not inference_rows:
        print("| _no parsed inference results yet_ | | | | | | | | | | | | |")
    print()
    print("**Legend:**")
    print("- **Req/s** = requests/sec served.")
    print("- **Total output tok/s** = aggregate output tokens/sec across all concurrent users.")
    print("- **TPS/GPU** = output tokens/sec per GPU (per-device throughput).")
    print("- **TPS/User** = output tokens/sec per concurrent user.")
    print("- **Avg req latency** = mean end-to-end request latency, ms.")
    print("- **TTFT** = Time-to-First-Token, ms.")
    print("- **TPOT** = Time-Per-Output-Token, ms (steady-state per-token latency).")
    print()

    # -------- Section 9: Raw parser output --------
    print("## 9. Raw parser output")
    print()
    print("Verbatim output of the dgxc training parser (`parse_train_timing*.sh --format=table`)")
    print("and the inference performance blocks extracted from each workload's logs.")
    print()

    raw_emitted = False
    for wl_dir in sorted(results_dir.iterdir()):
        if not wl_dir.is_dir():
            continue
        wl = wl_dir.name
        parsed_txt = wl_dir / "parsed.txt"
        perf_blocks = wl_dir / "performance_blocks.txt"
        if parsed_txt.exists() and parsed_txt.stat().st_size > 0:
            # Filter raw parser output to successful runs only — drop Failed rows
            # and the Summary footer lines that mention failure
            kept = []
            for line in parsed_txt.read_text().splitlines():
                if "Failed" in line:
                    continue
                if line.lstrip().startswith(("Failed experiments:", "Success rate:")):
                    continue
                kept.append(line)
            content = "\n".join(kept).rstrip()
            if content:
                print(f"### {wl} (training parser, successful runs only)")
                print()
                print("```")
                print(content)
                print("```")
                print()
                raw_emitted = True
        elif perf_blocks.exists() and perf_blocks.stat().st_size > 0:
            print(f"### {wl} (inference performance blocks)")
            print()
            print("```")
            print(perf_blocks.read_text().rstrip())
            print("```")
            print()
            raw_emitted = True

    if not raw_emitted:
        print("_No raw parser output captured yet — re-run `collect_results.sh` after experiments complete._")


if __name__ == "__main__":
    main()
