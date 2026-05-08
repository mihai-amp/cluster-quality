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
| Slurm partition(s) used | `batch` | `sinfo` |
| Enroot version | TBD | `enroot version` |
| Pyxis version | TBD | |
| Node hostnames | `gpu01`, `gpu02` (controller is `controller`) | `scontrol show nodes` |
| Inter-node SSH | **Disabled** — `ssh gpu0X` from controller does not work | use `srun`/`sbatch` for compute-node access |
| InfiniBand HCAs detected (per node) | TBD | expect 8× ConnectX-7 NDR |
| `NCCL_IB_HCA` setting | TBD | from `/etc/enroot/environ.d/*.env` |
| NCCL version | TBD | from container or env |
| Driver / CUDA version | TBD | `nvidia-smi` |
| Controller CUDA toolkit | **Not installed** | head node is for control + Claude Code only |
| Shared FS type | NFSv4.2, no `nconnect` | `mount \| grep vfs` — single TCP stream cap |
| Shared FS mount path | `/mnt/vfs` (30 TB) | `LLMB_INSTALL=/mnt/vfs/mihai/llmb-install` |
| Shared FS read BW (rough) | TBD | optional `dd`/`fio` check |
| Local NVMe per compute node | **1.7 TB visible** (vs. 38 TB expected) | `df -h` on compute node — Vultr ticket pending |
| `/dev/shm` per compute node | 1.5 TB tmpfs | used for `TRITON_CACHE_DIR` |
| GPU TDP / clocks observed | TBD | `nvidia-smi --query-gpu=power.draw,clocks.current.sm` |

---

## Workarounds applied

| Date | Workload affected | Issue observed | Workaround | Reference |
|---|---|---|---|---|
| 2026-05-08 | Phase 0 — `05_setup_tools.sh` | `gpu-burn` build fails on controller (`cublas_v2.h: No such file`) — head node has no CUDA toolkit | Run `05_setup_tools.sh` on a compute node via `srun -w gpu01 ...`; place binaries on shared NFS via `TOOLS_DIR=/mnt/vfs/<user>/tools` | `bootstrap.md` §4 B1 |
| 2026-05-08 | Phase 0 — `nccl-tests` build | Default `MPI_HOME=/opt/hpcx/ompi` doesn't exist; `mpi.h` not found | Diagnose actual MPI location on compute nodes, set `MPI_HOME` accordingly; or `apt-get install libopenmpi-dev` and use `/usr/lib/x86_64-linux-gnu/openmpi` | `bootstrap.md` §4 B1 |
| 2026-05-08 | Phase 0 — `run_phase0.sh` orchestrator | Uses `ssh`/`scp` between controller and compute nodes; SSH not enabled on this cluster | Run Phase 0 steps manually via `srun`/`sbatch`; rewrite orchestrator to use `srun` and shared-NFS scripts | `bootstrap.md` §4 B3 |
| 2026-05-08 | `dgxc-benchmarking/install.sh` | Fails with `Git LFS is not installed` | `apt-get install -y git-lfs && git lfs install --system`, then resume via `source llmb_venv/bin/activate && llmb-install` (skips uv reinstall) | `bootstrap.md` §4 B2 |
| 2026-05-08 | Slurm — controller↔compute comms | Both compute nodes timed out on slurmd port 6818 from controller; both stuck in `comp` state | Vultr restarted slurmd on compute nodes; resolved. Watch for recurrence after reboots. | — |

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
