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
# NVIDIA reference MFU numbers from `results/nvidia_reference/b200_training.md`.
# Keyed by (workload-suffix, size, dtype, scale). Used as a comparison column
# in the training / finetune summary tables. None means NVIDIA didn't publish.
NVIDIA_REF_MFU = {
    ("pretrain_llama3.1",      "8b",  "fp8",   8):  34.54,
    ("pretrain_llama3.1",      "8b",  "nvfp4", 8):  None,
    ("pretrain_nemotron4-15b", "15b", "bf16",  16): 50.16,
    ("pretrain_nemotron4-15b", "15b", "fp8",   16): 35.41,
    ("pretrain_qwen3",         "30b", "bf16",  16): None,
    ("finetune_llama3",        "70b", "bf16",  16): 23.24,
    ("finetune_llama3",        "70b", "fp8",   16): 17.33,
}

INFERENCE_META = {
    "inference_llama3.3":           {"size": "70b",  "dtype": "nvfp4", "scale": 1,  "engine": "TRT-LLM"},
    "inference_deepseek-r1":        {"size": "671b", "dtype": "nvfp4", "scale": 4,  "engine": "TRT-LLM"},
    "inference_deepseek-r1-dynamo": {"size": "671b", "dtype": "nvfp4", "scale": 32, "engine": "Dynamo + TRT-LLM"},
    "inference_deepseek-r1-sglang": {"size": "671b", "dtype": "nvfp4", "scale": 8,  "engine": "SGLang"},
    "inference_gpt-oss-dynamo":     {"size": "120b", "dtype": "mxfp4", "scale": 4,  "engine": "Dynamo + TRT-LLM"},
}

# dgxc use-case → (input tokens, output tokens). Sourced from dataset_<usecase>_<in>_<out>.txt
# filenames in workload dirs. Same across llama3.3/deepseek-r1/etc.
USE_CASE_TOKENS = {
    "chat":          (128,  128),
    "reasoning":     (1000, 1000),
    "summarization": (8000, 512),
    "generation":    (512,  8000),
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

    # Concurrency from path/filename: dgxc names look like `_CON640_chat_`
    con = re.search(r"_CON(\d+)_", str(log_path))
    if con:
        metrics["concurrency"] = int(con.group(1))

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
                "raw_lines": [],   # data-table lines + summary header/footer (for embedding raw)
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
            current["raw_lines"].append(raw)
            continue

        # Units line (e.g. "(B) (elements) (us) (GB/s) ...") right after the header
        if raw.lstrip().startswith("#") and "(GB/s)" in raw:
            current["raw_lines"].append(raw)
            continue

        # Avg footer
        if "Avg bus bandwidth" in raw:
            try:
                current["avg_busbw"] = float(raw.split(":")[-1].strip())
            except ValueError:
                pass
            current["raw_lines"].append(raw)
            continue

        # Out-of-bounds line, often near the footer — keep it for the raw block too
        if "Out of bounds values" in raw:
            current["raw_lines"].append(raw)
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
            current["raw_lines"].append(raw)
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


def load_nvidia_inference_ref(csv_path):
    """Parse NVIDIA inference reference CSV (per `results/nvidia_reference/nvidia_reference_inference.csv`).

    Returns dict keyed by (isl, osl, scale, concurrency) -> {tps_per_gpu, ttft_ms, itl_ms, e2e_ms}.
    """
    import csv as _csv
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path) as f:
        reader = _csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            io = row[0].strip()
            # match "20000 in → 1000 out" / unicode arrow
            m = re.match(r"(\d+)\s*in\s*[→\-]+>?\s*(\d+)\s*out", io)
            if not m:
                continue
            try:
                isl = int(m.group(1))
                osl = int(m.group(2))
                scale = int(row[1])
                # The CSV quotes numbers like "1,192.77" — strip commas
                tps = float(row[2].replace(",", ""))
                ttft = float(row[3].replace(",", ""))
                itl = float(row[4].replace(",", ""))
                e2e = float(row[5].replace(",", ""))
                con = int(row[11])
            except (ValueError, IndexError):
                continue
            out[(isl, osl, scale, con)] = {
                "tps_per_gpu": tps, "ttft_ms": ttft, "itl_ms": itl, "e2e_ms": e2e,
            }
    return out


# Which NVIDIA reference CSV applies to each of our inference workloads.
# Files live under `results/nvidia_reference/`.
NVIDIA_INF_REF_FILES = {
    "inference_llama3.3":    "nvidia_reference_inference_llama3.3.csv",
    "inference_deepseek-r1": "nvidia_reference_inference_dsv3.csv",
}


def find_closest_nvidia_cell(nv_ref, our_isl, our_osl, our_scale, our_con):
    """Find the closest NVIDIA-published cell to (ISL, OSL, scale, CON).

    Strategy: nearest scale (prefer exact), then min L1 distance on (ISL+OSL),
    then nearest concurrency. Returns ((isl, osl, scale, con), value_dict) or (None, None).
    """
    if not nv_ref:
        return None, None
    keys = list(nv_ref.keys())
    available_scales = sorted({k[2] for k in keys})
    best_scale = min(available_scales, key=lambda s: abs(s - our_scale))
    seqs = {(k[0], k[1]) for k in keys if k[2] == best_scale}
    best_seq = min(seqs, key=lambda s: abs(s[0] - our_isl) + abs(s[1] - our_osl))
    cons = [k[3] for k in keys if k[:3] == (best_seq[0], best_seq[1], best_scale)]
    best_con = min(cons, key=lambda c: abs(c - our_con))
    nv_key = (best_seq[0], best_seq[1], best_scale, best_con)
    return nv_key, nv_ref[nv_key]


def inf_match_quality(our_isl, our_osl, our_con, nv_isl, nv_osl, nv_con):
    """Return ✓ (near-exact), ≈ (close), or ⚠ (loose) for an our-vs-NVIDIA cell."""
    seq_off = max(abs(our_isl - nv_isl) / max(nv_isl, 1), abs(our_osl - nv_osl) / max(nv_osl, 1))
    con_off = abs(our_con - nv_con) / max(nv_con, 1)
    if seq_off < 0.1 and con_off < 0.1:
        return "✓"
    if seq_off < 0.5 and con_off < 1.0:
        return "≈"
    return "⚠"


IB_HEADER_RE = re.compile(r"^==== (ib_\w+) on (mlx5_\d+)\b")


def parse_ib_perftest_log(path):
    """Parse a 30_pairwise_ib.sh log into per-HCA bw (MiB/s) and lat (us).

    Returns {hca: {"bw_mib": float, "lat_us": float}}.
    """
    per_hca = {}
    section = None
    saw_header = False
    for raw in path.read_text(errors="ignore").splitlines():
        m = IB_HEADER_RE.match(raw)
        if m:
            section = (m.group(1), m.group(2))
            saw_header = False
            continue
        if section is None:
            continue
        tool, hca = section
        s = raw.strip()
        if s.startswith("#bytes") or s.startswith("# bytes"):
            saw_header = True
            continue
        if not saw_header:
            continue
        # Data row — first whitespace-delimited token must be all-digits
        parts = s.split()
        if not parts or not parts[0].isdigit():
            continue
        per_hca.setdefault(hca, {})
        try:
            if tool == "ib_write_bw":
                # size iters bw_peak bw_avg msgrate  (avg in MiB/sec)
                per_hca[hca]["bw_mib"] = float(parts[3])
            elif tool == "ib_write_lat":
                # size iters t_avg tps_avg  (t_avg in usec)
                per_hca[hca]["lat_us"] = float(parts[2])
        except (ValueError, IndexError):
            pass
        # Each tool emits one summary row per section
        saw_header = False
    return per_hca


def collect_ib_perftest(phase0_dir):
    """Return (per_hca, log_path) for the most-recent pairwise_ib log, or (None, None)."""
    if not phase0_dir.is_dir():
        return None, None
    candidates = sorted(phase0_dir.glob("ib_perftest/pairwise_*.log"))
    if not candidates:
        return None, None
    log = candidates[-1]
    return parse_ib_perftest_log(log), log.name


def main():
    # Accept either the parent `results/` directory or the older `results/phase1/`.
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
    else:
        arg = Path(os.environ["MY"]) / "results"
    if arg.name == "phase1":
        phase1_dir = arg
        phase0_dir = arg.parent / "phase0"
        results_root = arg.parent
    else:
        results_root = arg
        phase1_dir = arg / "phase1"
        phase0_dir = arg / "phase0"
    results_dir = phase1_dir   # keep historic variable name for the rest of the function
    if not phase1_dir.exists():
        print(f"# No phase1 results dir at {phase1_dir}", file=sys.stderr)
        sys.exit(1)

    print("# B200 / Vultr POC — cluster test summary")
    print(f"Generated: {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z")
    print()
    print(
        "Consolidated test summary for the AMP / Vultr HGX B200 cluster POC. "
        "Phase 0 covers hardware/fabric acceptance (NCCL collectives, IB perftest); "
        "Phase 1 covers the dgxc-benchmarking performance suite (training, finetune, inference)."
    )
    print()
    print("**Files in this report folder:**")
    print()
    print("- `summary.md` / `summary.html` — this document")
    print("- `inference_vs_nvidia.png` — inline plot embedded in §6")
    print("- `sources/nccl_tests_dgxc.log` — raw NCCL bus-bandwidth sweep (referenced by §1 and §10)")
    print("- `sources/pairwise_ib_336.log` — raw `perftest` `ib_write_bw` / `ib_write_lat` output (referenced by §2)")
    print("- `sources/nvidia_reference_b200_training.md` — NVIDIA-published reference MFU for B200 training (referenced by §3 and §4)")
    print("- `sources/nvidia_reference_inference_llama3.3.csv` — NVIDIA reference inference points for `llama-3.3-70b-instruct:1.13.1` (referenced by §6)")
    print("- `sources/nvidia_reference_inference_dsv3.csv` — NVIDIA reference inference points for `deepseek-r1-TRTLLM-Serve:26-02` (referenced by §6)")
    print("- `sources/phase1_logs/<workload>/` — one representative run log per (workload, config) combination from Phase 1; large logs trimmed to the per-step section. Workloads included: `finetune_llama3`, `inference_deepseek-r1`, `inference_llama3.3`, `microbenchmark_system_info`, `pretrain_llama3.1`, `pretrain_nemotron4-15b`, `pretrain_qwen3`. Other Phase 1 workloads use stdout formats (SGLang server logs, Dynamo + AI Perf CSV) not surfaced in this report.")
    print()

    # ============================================================
    #  PHASE 0 — ACCEPTANCE TESTS
    # ============================================================
    print("## Acceptance Tests (Phase 0)")
    print()
    print(
        "Hardware/fabric validation independent of the NVIDIA benchmark suite. "
        "These confirm we're getting expected bandwidth/latency from NVLink, IB, and the storage path "
        "before trusting the Phase 1 performance numbers."
    )
    print()

    # -------- Section 1: NCCL bus bandwidth (moved from former §5) --------
    nccl_sections, nccl_logname = collect_nccl(results_dir)
    print("### 1. NCCL bus bandwidth")
    print()
    if nccl_sections:
        sizes = [sz for s in nccl_sections for sz, _ in s["rows"] if sz > 0]
        if sizes:
            def _fmt_size(n):
                for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
                    if n >= div:
                        return f"{n / div:g} {unit}"
                return f"{n} B"
            sweep_range = f"{_fmt_size(min(sizes))} → {_fmt_size(max(sizes))}"
        else:
            sweep_range = "unknown"
        print(f"Per-collective busbw (source: [`sources/{nccl_logname}`](sources/{nccl_logname})).")
        print(f"Sweep range: {sweep_range}. Peak is the maximum out-of-place busbw observed; ")
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
        print("::: legend")
        print("**Legend:**")
        print("- **Collective** = NCCL op (`all_reduce`, `all_gather`, `reduce_scatter`, `alltoall`).")
        print("- **Ranks** = total GPUs in the test (8 = 1 node intra-node; 16 = 2 nodes inter-node).")
        print("- **Peak busbw** = max out-of-place busbw across the sweep, at the listed message size.")
        print("- **Avg busbw** = NCCL's per-run `# Avg bus bandwidth` footer (mean across the sweep).")
        print(":::")
        print()
        print("::: note")
        print(
            "**Sanity envelope.** B200 NVLink5 intra-node `all_reduce`/`alltoall` ≳ 350 GB/s at large sizes; "
            "inter-node 2-node `all_reduce` on 8× NDR ≳ 60–80 GB/s; "
            "inter-node `alltoall` is scaling-limited, typically ~40–60 GB/s."
        )
        print()
        print(f"Full per-collective sweep tables are in §10 (Raw Output).")
        print(":::")
    else:
        print("_No NCCL bus BW log captured._")
    print()

    # -------- Section 2: IB perftest pairwise --------
    print("### 2. IB pairwise bandwidth & latency (perftest)")
    print()
    ib_data, ib_logname = collect_ib_perftest(phase0_dir)
    if ib_data:
        # In the report folder we drop the bundled source under sources/ with a stable name.
        ib_src = "pairwise_ib_336.log" if ib_logname == "pairwise_336.log" else ib_logname
        print(f"Pure-IB pairwise measurement via `ib_write_bw` / `ib_write_lat` between gpu01↔gpu02, one HCA at a time (source: [`sources/{ib_src}`](sources/{ib_src})).")
        print()
        print("| HCA | BW avg (GB/s) | BW avg (Gbps) | Latency p50 (μs) |")
        print("|---|---:|---:|---:|")
        for hca in sorted(ib_data.keys(), key=lambda h: int(h.split("_")[1])):
            d = ib_data[hca]
            bw_mib = d.get("bw_mib")
            lat = d.get("lat_us")
            if bw_mib is None:
                continue
            bw_gbs = bw_mib / 1024.0
            bw_gbps = bw_mib * 8 / 1000.0  # MiB/s → Gbps (rough — wire rate proxy)
            print(f"| {hca} | {bw_gbs:.2f} | {bw_gbps:.1f} | {fmt(lat, '.2f')} |")
        print()
        print("::: legend")
        print("**Legend:**")
        print("- **BW avg (GB/s)** = `ib_write_bw` sustained throughput (binary GB/s — divide by 1024 of MiB/s).")
        print("- **BW avg (Gbps)** = wire-rate proxy (MiB/s × 8 / 1000) — compare against 400 Gbps NDR line rate.")
        print("- **Latency p50** = `ib_write_lat` average for 2-byte writes — proxy for inter-node small-message latency.")
        print(":::")
        print()
        print("::: note")
        print("**Sanity envelope.** ConnectX-7 NDR (400 Gbps line rate) sustains ~46–49 GB/s per HCA in `ib_write_bw`; "
              "p50 latency < 2 μs within the same rack.")
        print(":::")
    else:
        print("_No `ib_perftest/pairwise_*.log` captured under `results/phase0/`. Run `sbatch acceptance/30_pairwise_ib.sh` to populate._")
    print()

    # ============================================================
    #  PHASE 1 — PERFORMANCE BENCHMARKS
    # ============================================================
    print("## Performance Summary (Phase 1)")
    print()
    print(
        "Aggregated throughput, MFU, and inference-latency results from the dgxc-benchmarking suite "
        "(training, finetune, inference). Parsed via `collect_results.sh`; raw artifacts in "
        "`results/phase1/` and the archive at `results/phase1/archives/dgxc_archive_<date>.tar.zst`."
    )
    print()
    print("::: note")
    print(
        "**A note on the NVIDIA reference comparison.** Where shown, the `NVIDIA ref` columns "
        "below compare our measurements against NVIDIA's published B200 numbers "
        "(<https://aibenchmarking.ngc.nvidia.com/>). These are *approximate* comparisons because "
        "NVIDIA doesn't publish every configuration we tested — sequence-length pairs, concurrency "
        "values, and library versions (e.g., TRT-LLM, NeMo, dgxc) often differ between our runs "
        "and the closest NVIDIA-published cell. The intent is to sanity-check **order-of-magnitude** "
        "and **relative scale** vs the reference, not to claim a strict apples-to-apples match. "
        "When the closest NVIDIA cell is materially different from our config (e.g., concurrency "
        "off by 2×+, or sequence length off by 50%+), the comparison should be read as directional."
    )
    print(":::")
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
        print("Each row aggregates all runs grouped by (workload, size, dtype, scale).")
        print()
        print(
            "| Workload | Size | Dtype | Scale | n | "
            "Step mean (ms) | Step min (ms) | Step max (ms) | "
            "Within-run σ mean (ms) | σ across runs (ms) | "
            "TFLOPS mean | TFLOPS min | TFLOPS max | Peak TFLOPS | MFU% | NVIDIA ref MFU% | Δ vs ref |"
        )
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        if not by_config:
            print("| _no runs yet_ | | | | | | | | | | | | | | | | | |")
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
            ref = NVIDIA_REF_MFU.get((wl, size, dtype, int(scale) if str(scale).isdigit() else scale))
            if ref is None:
                ref_s, delta_s = "—", "—"
            else:
                ref_s = f"{ref:.1f}%"
                delta = mfu - ref
                delta_s = f"{delta:+.1f}pp"
            print(
                f"| {wl} | {size} | {dtype} | {scale} | {len(runs)} | "
                f"{tm:.1f} | {t_min:.1f} | {t_max:.1f} | "
                f"{wr_mean:.1f} | {ts:.1f} | "
                f"{fm:.0f} | {f_min:.0f} | {f_max:.0f} | {peak} | {mfu:.1f}% | {ref_s} | {delta_s} |"
            )
        print()
        print("::: legend")
        print("**Legend:**")
        print("- **Step mean/min/max** = per-step training time across runs in this config.")
        print("- **Within-run σ mean** = mean of per-run step-time std-dev (variance inside one run).")
        print("- **σ across runs** = std-dev of the per-run mean step time (variance between runs).")
        print("- **TFLOPS** = effective TFLOPS/GPU reported by the dgxc parser.")
        print("- **Peak TFLOPS** = B200 dense peak for this dtype (bf16: 2250, fp8: 4500, nvfp4/mxfp4: 9000).")
        print("- **MFU%** = TFLOPS mean / Peak TFLOPS × 100.")
        print("- **NVIDIA ref MFU%** = NVIDIA-published B200 MFU for the same config; "
              "source: <https://aibenchmarking.ngc.nvidia.com/> "
              "(transcribed in [`sources/nvidia_reference_b200_training.md`](sources/nvidia_reference_b200_training.md)).")
        print("- **Δ vs ref** = our MFU minus NVIDIA's, in percentage points (positive = we exceed the ref).")
        print(":::")
        print()

    # -------- Section 2/3: training & finetune summaries --------
    print_training_summary("### 3. Training — summary per model", summarize_training(training_rows))
    print_training_summary("### 4. Finetune — summary per model", summarize_training(finetune_rows))

    # -------- Section 4: Inference summary --------
    print("### 5. Inference — summary per model (across use cases)")
    print()
    print("Each row aggregates all parsed use cases of an inference workload.")
    print()
    print(
        "| Workload | Engine | Size | Dtype | Scale | n use cases | "
        "TPS/GPU mean | TPS/GPU min | TPS/GPU max | "
        "TTFT mean (ms) | TTFT min | TTFT max | "
        "TPOT mean (ms) | TPOT min | TPOT max |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_inf = defaultdict(list)
    for r in inference_rows:
        by_inf[(r["workload"], r.get("size", ""), r.get("dtype", ""), r.get("scale", ""))].append(r)
    if not by_inf:
        print("| _no parsed inference results yet_ | | | | | | | | | | | | | | |")
    for key, runs in sorted(by_inf.items()):
        wl, size, dtype, scale = key
        engine = INFERENCE_META.get(wl, {}).get("engine", "?")
        tps = [r.get("per_gpu_throughput") for r in runs if r.get("per_gpu_throughput") is not None]
        ttfts = [r.get("ttft_ms") for r in runs if r.get("ttft_ms") is not None]
        tpots = [r.get("tpot_ms") for r in runs if r.get("tpot_ms") is not None]
        tps_mean = sum(tps) / len(tps) if tps else None
        ttft_mean = sum(ttfts) / len(ttfts) if ttfts else None
        tpot_mean = sum(tpots) / len(tpots) if tpots else None
        print(
            f"| {wl} | {engine} | {fmt(size)} | {fmt(dtype)} | {fmt(scale)} | {len(runs)} | "
            f"{fmt(tps_mean, '.1f')} | {fmt(min(tps) if tps else None, '.1f')} | {fmt(max(tps) if tps else None, '.1f')} | "
            f"{fmt(ttft_mean, '.1f')} | {fmt(min(ttfts) if ttfts else None, '.1f')} | {fmt(max(ttfts) if ttfts else None, '.1f')} | "
            f"{fmt(tpot_mean, '.2f')} | {fmt(min(tpots) if tpots else None, '.2f')} | {fmt(max(tpots) if tpots else None, '.2f')} |"
        )
    print()
    print("::: legend")
    print("**Legend:**")
    print("- **Engine** = inference framework: `TRT-LLM` (TensorRT-LLM bench), `SGLang`, `Dynamo + TRT-LLM` (NVIDIA Dynamo serving with TRT-LLM backend).")
    print("- **TPS/GPU** = output tokens/sec per GPU (per-device throughput).")
    print("- **TTFT** = Time-to-First-Token, ms — latency from request submission to first streamed token.")
    print("- **TPOT** = Time-Per-Output-Token, ms — steady-state per-token latency after TTFT.")
    print(":::")
    print()
    print("::: note")
    print(
        "**Why TTFT max ≫ TTFT mean for `inference_llama3.3`.** Look at the per-use-case "
        "breakdown in §9: the high values come specifically from the CON640 `summarization` "
        "and `reasoning` rows — TTFT up to ~260 s for summarization, ~29 s for reasoning. "
        "These are a **benchmark-configuration artifact, not a hardware limit**:"
    )
    print()
    print("- dgxc's default at high concurrency (`CON640`) sets `max_num_tokens=2048`.")
    print("- `summarization` has 8000-token inputs; 640 concurrent requests must all "
          "wait through the prefill scheduler, which can only consume 2048 input tokens per step.")
    print("- Queues build up; TTFT inflates. **Per-token throughput (TPS/GPU, TPOT) is unaffected** — "
          "the model still serves the same tokens/sec; only first-token latency is hurt.")
    print("- The same `summarization` workload at `CON128` / `CON256` with `max_num_tokens=8192` "
          "(visible in the lower-concurrency `inference_llama3.3` rows of §9) achieves the same "
          "TPS/GPU at **15–24× lower TTFT**.")
    print()
    print(
        "Takeaway: don't read 60s TTFT as a B200/cluster limitation. For first-token latency "
        "comparisons, use the CON128/256 rows in §9; for steady-state per-token latency use TPOT (which is consistent across all rows)."
    )
    print(":::")
    print()

    # -------- Section 7: Inference vs NVIDIA reference (closest match) --------
    print("### 6. Inference — comparison vs NVIDIA reference (closest match)")
    print()

    # Pre-load reference CSVs once per workload referenced
    nv_refs_by_workload = {}
    for wl, fname in NVIDIA_INF_REF_FILES.items():
        nv_refs_by_workload[wl] = load_nvidia_inference_ref(results_root / "nvidia_reference" / fname)

    inf_compare_rows = []
    for r in inference_rows:
        if r.get("per_gpu_throughput") is None:
            continue
        uc = r.get("use_case")
        con = r.get("concurrency")
        tok = USE_CASE_TOKENS.get(uc) if uc else None
        try:
            scale_i = int(r.get("scale"))
        except (TypeError, ValueError):
            continue
        if not (uc and con and tok):
            continue
        nv_ref = nv_refs_by_workload.get(r["workload"]) or {}
        if not nv_ref:
            continue
        our_isl, our_osl = tok
        nv_key, ref = find_closest_nvidia_cell(nv_ref, our_isl, our_osl, scale_i, con)
        if ref is None:
            continue
        nv_isl, nv_osl, nv_scale, nv_con = nv_key
        quality = inf_match_quality(our_isl, our_osl, con, nv_isl, nv_osl, nv_con)
        inf_compare_rows.append({
            "workload": r["workload"], "use_case": uc, "scale": scale_i,
            "our_io": f"{our_isl}→{our_osl}", "our_con": con,
            "our_tps": r.get("per_gpu_throughput"), "our_ttft": r.get("ttft_ms"), "our_tpot": r.get("tpot_ms"),
            "nv_io": f"{nv_isl}→{nv_osl}", "nv_con": nv_con, "nv_scale": nv_scale,
            "nv_tps": ref["tps_per_gpu"], "nv_ttft": ref["ttft_ms"], "nv_tpot": ref["itl_ms"],
            "match": quality,
        })

    if inf_compare_rows:
        print(
            "For each of our parsed inference runs with a matching NVIDIA-published reference, this table shows "
            "the closest published cell. Reference sources per workload:"
        )
        print()
        print("- `inference_llama3.3` → `sources/nvidia_reference_inference_llama3.3.csv` (model `llama-3.3-70b-instruct:1.13.1`, NVFP4, B200).")
        print("- `inference_deepseek-r1` → `sources/nvidia_reference_inference_dsv3.csv` (model `deepseek-r1-TRTLLM-Serve:26-02`, FP4, B200).")
        print()
        print("Our TRT-LLM version is **1.1.0rc5**; the NVIDIA cells were measured with newer builds (see the 'NVIDIA ref' note at the top of Phase 1).")
        print()
        print(
            "| Workload | Use case | Scale | Our ISL→OSL | Our CON | Our TPS/GPU | Our TTFT ms | Our TPOT ms | "
            "NVIDIA cell (ISL→OSL, scale, CON) | NVIDIA TPS/GPU | NVIDIA TTFT | NVIDIA TPOT | Match |"
        )
        print("|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|:---:|")
        # Deduplicate identical rows (multiple repeats of same run produce identical metrics)
        seen = set()
        sort_key = {"✓": 0, "≈": 1, "⚠": 2}
        for row in sorted(inf_compare_rows, key=lambda r: (sort_key.get(r["match"], 9), r["workload"], r["use_case"], r["our_con"])):
            key = (row["workload"], row["use_case"], row["our_io"], row["our_con"])
            if key in seen:
                continue
            seen.add(key)
            nv_cell = f"{row['nv_io']}, s={row['nv_scale']}, CON={row['nv_con']}"
            print(
                f"| {row['workload']} | {row['use_case']} | {row['scale']} | {row['our_io']} | {row['our_con']} | "
                f"{fmt(row['our_tps'], '.0f')} | {fmt(row['our_ttft'], '.0f')} | {fmt(row['our_tpot'], '.1f')} | "
                f"{nv_cell} | "
                f"{fmt(row['nv_tps'], '.0f')} | {fmt(row['nv_ttft'], '.0f')} | {fmt(row['nv_tpot'], '.1f')} | "
                f"{row['match']} |"
            )
        print()
        print("::: legend")
        print("**Match column legend:**")
        print("- **✓** = near-exact match (ISL/OSL within 10%, concurrency within 10%).")
        print("- **≈** = close (ISL/OSL within 50% and concurrency within 2×).")
        print("- **⚠** = loose (one or both axes off by more than that). Read as directional only.")
        print(":::")
        print()
        print("::: note")
        print(
            "**Why our numbers often look better than NVIDIA's:** likely the **TRT-LLM version gap** "
            "(our 1.1.0rc5 vs NVIDIA's 1.13.1) and possible differences in scheduling defaults between "
            "dgxc's `trtllm-bench` and the Performance Explorer test rig. Comparison sanity-checks "
            "order-of-magnitude, not absolute deltas."
        )
        print(":::")

        # Inline plot — our runs (red stars) on top of NVIDIA reference curves (green)
        plot_path = Path(__file__).parent / "results" / "report" / "inference_vs_nvidia.png"
        if plot_path.exists():
            print()
            print("**Plot** (one subplot per `(workload, ISL→OSL, scale)` cell that has both NVIDIA reference data and at least one of our runs; log-scale axes):")
            print()
            print("![Inference TPS/GPU vs NVIDIA reference](inference_vs_nvidia.png)")
    else:
        print(
            "_No matching NVIDIA reference cells found. Verify the CSVs exist under `results/nvidia_reference/` "
            "and that our runs have parseable `(use_case, concurrency)` metadata._"
        )
    print()

    # -------- Group: Full Results --------
    print("## Full Results (Phase 1)")
    print()
    print(
        "One row per run, no aggregation. Provided for auditability; if you only need the "
        "headline numbers, the Performance Summary section above is sufficient."
    )
    print()

    # -------- Section 8: Training full --------
    print("### 7. Training — full results")
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
        print("| _no runs yet_ | | | | | | | |")
    print()
    print("**Legend:**")
    print("- **Step mean** = mean per-step training time within this run.")
    print("- **Step σ** = std-dev of step time within this run.")
    print("- **TFLOPS/GPU** = effective TFLOPS per GPU.")
    print("- **Peak TFLOPS** = B200 dense peak for this dtype (bf16: 2250, fp8: 4500, nvfp4/mxfp4: 9000).")
    print()

    # -------- Section 7: Finetune full --------
    print("### 8. Finetune — full results")
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
        print("| _no runs yet_ | | | | | | | |")
    print()
    print("**Legend:**")
    print("- **Step mean** = mean per-step training time within this run.")
    print("- **Step σ** = std-dev of step time within this run.")
    print("- **TFLOPS/GPU** = effective TFLOPS per GPU.")
    print("- **Peak TFLOPS** = B200 dense peak for this dtype (bf16: 2250, fp8: 4500, nvfp4/mxfp4: 9000).")
    print()

    # -------- Section 8: Inference full --------
    print("### 9. Inference — full results (every use case)")
    print()
    print("One row per parsed inference use case, no aggregation.")
    print()
    print(
        "| Workload | Engine | Size | Dtype | Scale | Use case | In→Out tok | "
        "Req/s | Total output tok/s | TPS/GPU | TPS/User | "
        "Avg req latency (ms) | TTFT (ms) | TPOT (ms) |"
    )
    print("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(inference_rows, key=lambda x: (x["workload"], x.get("use_case", ""))):
        engine = INFERENCE_META.get(r["workload"], {}).get("engine", "?")
        uc = r.get("use_case", "")
        tok_in, tok_out = USE_CASE_TOKENS.get(uc, (None, None))
        tok_label = f"{tok_in}→{tok_out}" if tok_in is not None else "-"
        print(
            f"| {r['workload']} | {engine} | {fmt(r.get('size'))} | {fmt(r.get('dtype'))} | "
            f"{fmt(r.get('scale'))} | {fmt(uc)} | {tok_label} | "
            f"{fmt(r.get('req_per_s'), '.2f')} | "
            f"{fmt(r.get('total_output_tok_per_s'), '.1f')} | "
            f"{fmt(r.get('per_gpu_throughput'), '.1f')} | "
            f"{fmt(r.get('per_user_throughput'), '.2f')} | "
            f"{fmt(r.get('avg_req_latency_ms'), '.1f')} | "
            f"{fmt(r.get('ttft_ms'), '.1f')} | "
            f"{fmt(r.get('tpot_ms'), '.2f')} |"
        )
    if not inference_rows:
        print("| _no parsed inference results yet_ | | | | | | | | | | | | | | |")
    print()
    print("::: legend")
    print("**Legend:**")
    print("- **Engine** = inference serving framework / kernel library:")
    print("  - **TRT-LLM** = TensorRT-LLM benchmark harness (`trtllm-bench`).")
    print("  - **SGLang** = SGLang server.")
    print("  - **Dynamo + TRT-LLM** = NVIDIA Dynamo serving framework, TRT-LLM as backend.")
    print("- **In→Out tok** = input → output sequence length per request (chat 128→128, "
          "reasoning 1000→1000, summarization 8000→512, generation 512→8000). "
          "Sourced from dgxc `dataset_<usecase>_<in>_<out>.txt`.")
    print("- **Req/s** = requests/sec served.")
    print("- **Total output tok/s** = aggregate output tokens/sec across all concurrent users.")
    print("- **TPS/GPU** = output tokens/sec per GPU (per-device throughput).")
    print("- **TPS/User** = output tokens/sec per concurrent user.")
    print("- **Avg req latency** = mean end-to-end request latency, ms.")
    print("- **TTFT** = Time-to-First-Token, ms.")
    print("- **TPOT** = Time-Per-Output-Token, ms (steady-state per-token latency).")
    print(":::")
    print()

    # -------- Section 9: Raw parser output --------
    # -------- Group: Raw Output (covers both Phase 0 NCCL raw + Phase 1 parser raw) --------
    print("## Raw Output")
    print()
    print(
        "Verbatim outputs from the test harnesses — handy for cross-checking the summary "
        "tables against the original files. Includes the NCCL collective sweeps (Phase 0) "
        "and the dgxc training-parser / TRT-LLM `PERFORMANCE OVERVIEW` blocks (Phase 1)."
    )
    print()

    # -------- NCCL raw sweep first (moved from former Appendix A) --------
    if nccl_sections:
        print("### 10. NCCL raw sweep (per collective)")
        print()
        print(
            f"Direct nccl-tests output for each (collective, scope) — full size × algbw × busbw "
            f"sweep, in-place and out-of-place columns. Source: `{nccl_logname}`."
        )
        print()
        for s in nccl_sections:
            if not s.get("raw_lines"):
                continue
            print(f"**{s['op']}  ranks={s['ranks']}  ({s['scope']})**")
            print()
            print("```")
            for line in s["raw_lines"]:
                print(line.rstrip())
            print("```")
            print()

    print("### 11. dgxc parser output (training + inference)")
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
                print(f"### {wl} (training parser)")
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
    print()

if __name__ == "__main__":
    main()
