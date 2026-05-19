#!/usr/bin/env python3
"""Copy one representative log per (workload, config) combination from
results/phase1/<workload>/logs/ to results/report/sources/phase1_logs/<workload>/.

Combination key is derived by stripping the run-specific suffix:
  - Training/finetune: drop `_<10+ digit timestamp>_(log|sbatch)-default*.out`
  - Inference: drop the trailing `_<jobid>.out`

Usage:  python3 copy_phase1_sample_logs.py results/
"""
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict


def combo_key(name):
    """Best-effort: identify the unique combination this filename represents."""
    # 1) Training/finetune: strip from first long timestamp onwards
    m = re.match(r"^(.+?)_\d{10,}_(log|sbatch)-default", name)
    if m:
        return m.group(1)
    # 2) Inference: drop trailing `_<jobid>.out`
    m = re.match(r"^(.+?)_(\d+)\.out$", name)
    if m:
        return m.group(1)
    # Fallback: filename without extension
    return name.rsplit(".", 1)[0]


# Markers that indicate a successful run with useful content
TRAINING_STEP_RE = re.compile(r"^\s*Step Time\s*:\s*[\d.]+s\s+GPU utilization", re.M)        # Megatron-Bridge
TRAINING_ITER_RE = re.compile(r"\] iteration\s+\d+/\s*\d+\s*\|", re.M)                       # Megatron-Bridge
NEMO_STEP_RE = re.compile(r"train_step_timing in s:\s*[\d.]+", re.M)                          # NeMo (nemotron4)
NEMO_TFLOPS_RE = re.compile(r"TFLOPS_per_GPU:\s*[\d.e+\-]+", re.M)                            # NeMo
INFERENCE_PERF_RE = re.compile(r"PERFORMANCE OVERVIEW", re.M)                                 # TRT-LLM bench
NCCL_AVG_RE = re.compile(r"^# Avg bus bandwidth", re.M)                                       # nccl-tests
SYSINFO_RE = re.compile(r"^(NCCL_IB_HCA|NCCL_SOCKET_IFNAME|nccl-version|NUMA)", re.M)         # system_info dumps

# Trim threshold + how much to keep when a file is too large
TRIM_THRESHOLD = 200_000   # 200 KB
TRIM_TAIL_LINES = 400


def score_file(path):
    """Higher = more likely to contain the metrics we care about."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return -1
    score = 0
    if TRAINING_STEP_RE.search(text):  score += 100   # Megatron-Bridge "Step Time"
    if TRAINING_ITER_RE.search(text):  score += 50
    if NEMO_STEP_RE.search(text):      score += 100   # NeMo train_step_timing
    if NEMO_TFLOPS_RE.search(text):    score += 50
    if INFERENCE_PERF_RE.search(text): score += 100   # TRT-LLM bench
    if NCCL_AVG_RE.search(text):       score += 100   # nccl-tests
    if SYSINFO_RE.search(text):        score += 100   # microbenchmark_system_info
    # Bonus for non-trivial size (suggests real run vs immediate fail)
    score += min(path.stat().st_size // 10_000, 30)
    return score


# Patterns of filenames we should skip outright (no useful content):
SKIP_FILENAME_PATTERNS = (
    "_sbatch_default-default",      # slurm wrapper scaffolding
    "experiments_",                 # short copies of streaming-on .out without full content
)


def should_skip_filename(name):
    # Slurm controller stdout files at workload root (env-export + srun + exit code only)
    if re.match(r"^slurm-\d+\.out$", name):
        return True
    if name.endswith(".err"):
        return True
    for pat in SKIP_FILENAME_PATTERNS:
        if pat in name:
            return True
    return False


def find_step_section_start(lines):
    """Return the index of the line where the per-step output starts, or 0.

    Heuristic: the first line matching either training step pattern.
    """
    pat = re.compile(
        r"(\] iteration\s+\d+/\s*\d+\s*\|)|(^\s*Step Time\s*:\s*[\d.]+s\s+GPU utilization)"
    )
    for i, line in enumerate(lines):
        if pat.search(line):
            return i
    return 0


def write_trimmed(src, dst, tail_lines=TRIM_TAIL_LINES):
    """Copy `src` to `dst`. If oversized, trim to the relevant section:
    - training: from first per-step line to EOF
    - inference: last `tail_lines` (PERFORMANCE OVERVIEW lives near the end)
    """
    if src.stat().st_size <= TRIM_THRESHOLD:
        shutil.copy2(src, dst)
        return False
    lines = src.read_text(errors="replace").splitlines()
    start = find_step_section_start(lines)
    if start > 0:
        # Training-style: keep from first step line. Cap to keep things tidy.
        keep = lines[start:]
        kind = "training step section"
    else:
        keep = lines[-tail_lines:]
        kind = f"last {tail_lines} lines"
    header = [
        "# ===== TRIMMED SAMPLE =====",
        f"# Original: {src.name}",
        f"# Original size: {src.stat().st_size:,} bytes ({len(lines):,} lines)",
        f"# This file: {kind} ({len(keep):,} lines).",
        "# ==========================",
        "",
    ]
    with open(dst, "w") as f:
        f.write("\n".join(header))
        f.write("\n".join(keep))
        f.write("\n")
    return True


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    phase1 = root / "phase1"
    dst_root = root / "report" / "sources" / "phase1_logs"

    if not phase1.is_dir():
        sys.stderr.write(f"no phase1 dir at {phase1}\n")
        return 1

    # Wipe any prior content so a re-run produces a clean snapshot
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    trimmed = 0
    skipped = 0
    for wl_dir in sorted(phase1.iterdir()):
        if not wl_dir.is_dir():
            continue
        logs = sorted((wl_dir / "logs").glob("*.out")) if (wl_dir / "logs").is_dir() else []
        if not logs:
            continue
        # Drop slurm-wrapper logs / .err / slurm-<jobid>.out / experiments_* (no run content)
        logs = [p for p in logs if not should_skip_filename(p.name)]
        if not logs:
            continue
        groups = defaultdict(list)
        for p in logs:
            groups[combo_key(p.name)].append(p)
        wl_dst = dst_root / wl_dir.name
        wl_dst.mkdir(parents=True, exist_ok=True)
        for key, files in sorted(groups.items()):
            # Prefer files with actual step/perf markers; tiebreak by size.
            chosen = max(files, key=score_file)
            # If even the best file has no real markers, skip the group rather than
            # ship a failed/empty run as a "sample".
            if score_file(chosen) < 50:
                skipped += len(files)
                continue
            was_trimmed = write_trimmed(chosen, wl_dst / chosen.name)
            copied += 1
            if was_trimmed:
                trimmed += 1
            skipped += len(files) - 1
        # If we ended up with an empty workload dir, remove it
        if wl_dst.is_dir() and not any(wl_dst.iterdir()):
            wl_dst.rmdir()

    print(f"copied {copied} representative logs ({trimmed} trimmed); skipped {skipped} additional runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
