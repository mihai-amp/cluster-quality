# B200 / Vultr — Cluster Quirks & Reality Notes

A live journal of cluster-specific findings discovered during the run. Append entries as they happen — don't wait until the end.

> **Plan:** `plan.md` — the design and expectations.
> **Execution:** `execution.md` — the runbook and results tables.

---

## Discovered configuration

Filled in from `microbenchmark_system_info` output and ad-hoc inspection on Day 1.

| Item | Value | Source / Notes |
|---|---|---|
| Slurm version | TBD | `sinfo --version` |
| Slurm partition(s) used | TBD | |
| Enroot version | TBD | `enroot version` |
| Pyxis version | TBD | |
| Node hostnames | TBD | |
| InfiniBand HCAs detected (per node) | TBD | expect 8× ConnectX-7 NDR |
| `NCCL_IB_HCA` setting | TBD | from `/etc/enroot/environ.d/*.env` |
| NCCL version | TBD | from container or env |
| Driver / CUDA version | TBD | `nvidia-smi` |
| Shared FS type | TBD | NFS / Lustre / Weka / GPFS |
| Shared FS mount path | TBD | path used for `LLMB_INSTALL` |
| Shared FS read BW (rough) | TBD | optional `dd`/`fio` check |
| GPU TDP / clocks observed | TBD | `nvidia-smi --query-gpu=power.draw,clocks.current.sm` |

---

## Workarounds applied

| Date | Workload affected | Issue observed | Workaround | Reference |
|---|---|---|---|---|
| | | | | |

---

## Anomalies in results

Use this section to flag deltas from `plan.md` §4 expectations and your hypothesis. One row per anomaly; expand into prose if needed.

| Date | Workload | Precision | Observed | Expected | Δ | Hypothesis | Status |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## NCCL / IB diagnostics

Capture useful one-shot diagnostics here for the deliverable.

```
# Add output of: nccl-tests all_reduce_perf, nvidia-smi nvlink, mlnx_perf snapshots
```

---

## Vultr-specific findings

Anything that's specifically a property of Vultr's HGX B200 offering (vs. generic IB-on-bare-metal). Write it up so a future POC on a different neocloud has a baseline to compare against.

- _(empty — populate as you go)_

---

## Lessons learned

Populate at the end of the run. These are the takeaways worth carrying to the next POC.

- _(empty — populate at the end)_
