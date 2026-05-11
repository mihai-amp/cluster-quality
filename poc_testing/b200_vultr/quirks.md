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
| 2026-05-09 | All multi-node workloads | `/etc/hosts` had each compute node mapped twice to the same IP (cloud-init template bug). `hostname --ip-address` returned the IP duplicated, malforming `head_node_ip` in launcher | Dedupe `/etc/hosts` via `awk '!seen[$0]++'`; disable `manage_etc_hosts` in `/etc/cloud/cloud.cfg` so it survives reboot | `acceptance/35_ib_health_check.sh` |
| 2026-05-09 | `dgxc-benchmarking` launcher | `HF_HUB_OFFLINE=1` hard-coded in generated sbatch — container can't reach HF even when host has internet | Pre-populate HF cache at install time (already done by `llmb-install`); mount `$LLMB_INSTALL/.cache/huggingface` into container via `/etc/enroot/mounts.d/00-vfs.fstab`; set `HF_HOME` + `HF_HUB_CACHE` in `/etc/enroot/environ.d/40-hf-cache.env` | — |
| 2026-05-09 | All containerized workloads | `/etc/enroot/environ.d/` only has `10-terminal.env`; cluster env (HF token, NCCL_IB_HCA) doesn't propagate into containers | Write `30-hf.env`, `40-hf-cache.env`, `50-nccl.env`, `55-triton.env` on each compute node to inject the right env vars at container start | — |
| 2026-05-09 | All containerized workloads | `/mnt/vfs` not bind-mounted into enroot containers by default — HF cache and workload code at `/mnt/vfs/...` invisible inside container | Add `/mnt/vfs /mnt/vfs none x-create=auto,rbind,rw` to `/etc/enroot/mounts.d/00-vfs.fstab` on each compute node | — |
| 2026-05-09 | All Megatron-Bridge / Triton workloads | Container's `/dev/shm` is remounted with `noexec` — Triton fails to mmap its compiled `.so` from there: `ImportError: failed to map segment from shared object` | Set `TRITON_CACHE_DIR=/tmp/triton-cache` and `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache` in `/etc/enroot/environ.d/55-triton.env` (container's `/tmp` is a clean tmpfs without noexec) | — |
| 2026-05-09 | All sbatch-launched dgxc workloads | `#SBATCH --time=01:00:00` hardcoded in NeMo Run-generated sbatch scripts; Qwen3 MoE startup alone exceeds 1h before training | No working override found: `SBATCH_TIMELIMIT` env var ignored, no CLI flag in llmb-run. Workaround: patch the workload's NeMo Run launcher to default to a longer time | — |
| 2026-05-09 | `setup_cluster_config.sh` (this repo) | First-pass version overwrote `cluster_config.yaml` that `llmb-install` had populated with venv mappings; subsequent `llmb-run submit` calls failed with "venv_path not found" | Re-ran `llmb-install` (preserved big downloads) to regenerate cluster_config.yaml properly; future: the script should *merge with* the installer's generated config, not replace it | — |
| 2026-05-10 | **All multi-node workloads — BLOCKING** | Inter-node InfiniBand fabric is non-functional. All 8 NDR ConnectX-7 rails fail RDMA between gpu01 and gpu02: `ib_write_bw` reports `Failed status 12 (IBV_WC_RETRY_EXC_ERR)`, 128 sends / 0 completions. UCX: `Destination is unreachable` on `rc_verbs`, `ud_verbs`, `dc_mlx5`. Ethernet between nodes works fine; single-node workloads run normally. | **Vultr ticket required** — see `vultr_ticket.md`. Single-node workloads (`pretrain_llama3.1 @ 8`, all inference at ≤4 GPU on one node) continue running; multi-node workloads (`pretrain_qwen3 @ 16`, `pretrain_nemotron4-15b @ 16`, `finetune_llama3 @ 16`, `inference_deepseek-r1-sglang @ 8` spanning nodes) blocked until fixed | `vultr_ticket.md`, `acceptance/35_ib_health_check.sh` |
| 2026-05-11 | Multi-node IB fabric — RESOLVED | Vultr fixed inter-node InfiniBand routing; `ib_write_bw` over `mlx5_0` now sustains 46150 MB/s (≈46.15 GB/s) between gpu01 and gpu02 — at expected NDR rate. All 8 NDR rails recovered. | None needed — fabric is now functional. Resume multi-node workload submission. | `vultr_ticket.md` |
| 2026-05-11 | **All multi-node NCCL workloads** | NCCL bootstrap picked the public-IP NIC (`enp193s0f0np0` → 66.42.81.161) as its rendezvous socket instead of the internal 10.6.96.x interface, then froze with `socketPollConnect: connect to 66.42.81.161 returned Connection timed out`. Manifested as a silent hang right after `NCCL version` printed — looked identical to the IB outage but was actually a bootstrap-NIC selection bug. IB itself was fine throughout. | Set `NCCL_SOCKET_IFNAME=enp193s0f1np1` (the dual-port NIC's port-1 internal-net side; port-0 is public) in `/etc/enroot/environ.d/50-nccl.env` on each compute node. Look for `NCCL INFO Bootstrap : Using enp193s0f1np1:10.6.96.4<0>` in the log to confirm. | — |
| 2026-05-11 | Tail/parse pattern in `collect_results.sh` | NeMo Run emits two `.out` files per experiment — `log-default-default.<config>_<jobid>_0.out` (training output) and `sbatch_default-default.<config>_<jobid>.out` (wrapper script trace). Default `*.out` globbing picked up both and confused the latest-log heuristic. | Restrict `find` to `log-*.out` / `slurm-*.out` and explicitly `-not -name 'sbatch_*'` in `collect_results.sh`; same pattern when tailing live runs from the controller. | `collect_results.sh` |

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
