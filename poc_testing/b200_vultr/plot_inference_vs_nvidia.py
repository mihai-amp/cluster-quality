#!/usr/bin/env python3
"""Plot our inference TPS/GPU points on top of NVIDIA's published curves.

One subplot per (workload, ISL→OSL) cell. X-axis: concurrency (log). Y-axis: TPS/GPU.
Saves PNG into `results/report/inference_vs_nvidia.png`.

Usage:  python3 plot_inference_vs_nvidia.py results/
"""
import sys, csv, re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_results import (
    load_nvidia_inference_ref,
    collect_inference,
    NVIDIA_INF_REF_FILES,
    USE_CASE_TOKENS,
    workload_type,
)


def main():
    results_root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    phase1 = results_root / "phase1"
    nv_dir = results_root / "nvidia_reference"

    # Collect our inference rows for the two workloads with refs
    our_rows = []
    for wl_name in NVIDIA_INF_REF_FILES:
        wl_dir = phase1 / wl_name
        if wl_dir.is_dir():
            our_rows.extend(collect_inference(wl_dir))

    # Group references by (workload, isl, osl, scale) -> list of (con, tps_per_gpu)
    by_cell_nv = defaultdict(list)
    for wl, fname in NVIDIA_INF_REF_FILES.items():
        ref = load_nvidia_inference_ref(nv_dir / fname)
        for (isl, osl, scale, con), v in ref.items():
            by_cell_nv[(wl, isl, osl, scale)].append((con, v["tps_per_gpu"]))

    # Determine which (workload, isl, osl, scale) cells to plot: anywhere we have either NVIDIA or our data
    cells_with_our = defaultdict(list)
    for r in our_rows:
        if r.get("per_gpu_throughput") is None:
            continue
        uc = r.get("use_case")
        con = r.get("concurrency")
        tok = USE_CASE_TOKENS.get(uc)
        try:
            scale = int(r.get("scale"))
        except (TypeError, ValueError):
            continue
        if not (uc and con and tok):
            continue
        # Map our scale to the nearest available NVIDIA scale for plot alignment
        nv_scales_for_wl = sorted({k[3] for k in by_cell_nv if k[0] == r["workload"]})
        if not nv_scales_for_wl:
            continue
        plot_scale = min(nv_scales_for_wl, key=lambda s: abs(s - scale))
        # Map our ISL/OSL to nearest NVIDIA pair within plot_scale
        nv_seqs = sorted({(k[1], k[2]) for k in by_cell_nv if k[0] == r["workload"] and k[3] == plot_scale})
        if not nv_seqs:
            continue
        best_seq = min(nv_seqs, key=lambda s: abs(s[0] - tok[0]) + abs(s[1] - tok[1]))
        cells_with_our[(r["workload"], best_seq[0], best_seq[1], plot_scale)].append({
            "our_con": con, "our_tps": r["per_gpu_throughput"],
            "our_io": f"{tok[0]}→{tok[1]}", "our_scale": scale,
        })

    # Only plot cells that have BOTH NVIDIA reference data AND at least one of our runs.
    cells = sorted(set(by_cell_nv.keys()) & set(cells_with_our.keys()))

    if not cells:
        print("No data to plot.", file=sys.stderr)
        return 1

    n = len(cells)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), squeeze=False)
    fig.suptitle("Inference TPS/GPU — ours vs NVIDIA reference (B200, NVFP4/FP4)", fontsize=13)

    for i, key in enumerate(cells):
        wl, isl, osl, scale = key
        ax = axes[i // ncols][i % ncols]
        # NVIDIA reference curve
        if key in by_cell_nv:
            data = sorted(by_cell_nv[key])
            xs = [c for c, _ in data]
            ys = [t for _, t in data]
            ax.plot(xs, ys, "o-", color="#76b900", label="NVIDIA reference", markersize=5)
        # Our points (may have different precise ISL/OSL — annotate)
        if key in cells_with_our:
            for our in cells_with_our[key]:
                ax.plot(our["our_con"], our["our_tps"], "*", color="#cc3344", markersize=14,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=f"ours (ISL/OSL={our['our_io']}, scale={our['our_scale']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Concurrency (log)")
        ax.set_ylabel("TPS / GPU (log)")
        ax.set_title(f"{wl}\n{isl}→{osl}, scale={scale}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        # Dedup legend
        h, l = ax.get_legend_handles_labels()
        seen = set(); h2 = []; l2 = []
        for hi, li in zip(h, l):
            if li in seen: continue
            seen.add(li); h2.append(hi); l2.append(li)
        ax.legend(h2, l2, fontsize=8, loc="best")

    # Hide any unused subplots
    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = results_root / "report" / "inference_vs_nvidia.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
