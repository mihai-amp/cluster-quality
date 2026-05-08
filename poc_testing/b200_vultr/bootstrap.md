# B200 / Vultr — Bootstrap & Resume Runbook

Step-by-step path from a freshly provisioned Vultr B200 cluster to a state where Phase 0 acceptance and `llmb-install` can run. Captures the workspace setup, the discoveries made during initial bootstrap, and the unresolved blockers as of the latest checkpoint.

> **Plan:** `plan.md` — design intent, workload selection, expectations.
> **Execution:** `execution.md` — full runbook for Phase 0 + Phase 1.
> **Quirks:** `quirks.md` — live journal of cluster-specific findings.

This doc answers: *what do I run right now to keep moving?*

---

## 1. Cluster facts (Vultr HGX B200, 16 GPU)

- **Controller (head node):** Intel Xeon E-2388G, 128 GB RAM, no GPUs, **no CUDA toolkit**. Used for Slurm submission and Claude Code only.
- **Compute nodes:** `gpu01`, `gpu02`. 8× B200 each, AMD EPYC 9575F (x86_64), 3 TB RAM. CUDA driver present; toolkit headers presence TBD per node.
- **Login model:** root SSH to controller only. **`ssh gpu0X` from controller does not work** — interact with compute nodes via `srun`/`sbatch` only.
- **Shared FS:** `10.6.96.6:/` mounted at `/mnt/vfs`, NFSv4.2, 30 TB. Mount opts lack `nconnect`, capping bandwidth at one TCP stream (~2–3 GB/s). Acceptable for now; revisit if FS becomes the bottleneck.
- **Local NVMe per compute node:** **only `/dev/sda2` at 1.7 TB is mounted**, not the 10× 3.84 TB the spec implied. The remaining ~36 TB is either unassembled or unmounted — Vultr ticket pending. `/dev/shm` tmpfs is 1.5 TB and is the right place for Triton/Inductor caches.

## 2. Workspace layout

Created by `setup_workspace.sh` (run once on the controller as root). Lives entirely under `/mnt/vfs/mihai/`:

```
/mnt/vfs/mihai/
├── env.sh                  # source this in any shell
├── workspace/
│   ├── dgxc-benchmarking/  # NVIDIA suite
│   ├── cluster-quality/    # this repo
│   └── llmb_venv/          # Python 3.12 venv created by ./install.sh
├── llmb-install/           # $LLMB_INSTALL — containers, datasets, checkpoints
├── hf-cache/               # $HF_HOME
├── tools/                  # gpu-burn, nvbandwidth, nccl-tests built binaries
├── results/
│   ├── phase0/             # acceptance test outputs (numbered subdirs)
│   └── phase1/             # workload benchmark outputs
├── batch/                  # batch YAMLs (not yet populated from $PLAN/batch)
├── scratch/
└── logs/
```

Env vars (defined in `/mnt/vfs/mihai/env.sh`, auto-sourced from `~/.bashrc`):

```
MY=/mnt/vfs/mihai
LLMB_INSTALL=$MY/llmb-install
HF_HOME=$MY/hf-cache
TRITON_CACHE_DIR=/dev/shm/$USER/triton          # local NVMe alternative
TORCHINDUCTOR_CACHE_DIR=/dev/shm/$USER/inductor
PLAN=$MY/workspace/cluster-quality/poc_testing/b200_vultr
DGXC=$MY/workspace/dgxc-benchmarking
PATH includes ~/.local/bin                      # for Claude Code
```

## 3. State at last checkpoint

| Item | Status |
|---|---|
| `setup_workspace.sh` | Done — directories, repos, `env.sh` all created |
| Claude Code installed on controller | Done (`~/.local/bin/claude`) |
| Slurm comms (controller ↔ compute nodes) | **Working** after Vultr restart of slurmd |
| Phase 0 — `00_discovery.sh` | Not yet run |
| Phase 0 — `05_setup_tools.sh` | Partial — `gpu-burn` and `nvbandwidth` likely built; **`nccl-tests` failed** (mpi.h not found) |
| Phase 0 — steps 10/20/30/40/45/50/60 | Not yet run |
| Phase 0 — `70_gpu_burn.sh` (24 h) | Not yet started |
| `llmb-install` | **Failed at git-lfs missing**; venv at `$MY/workspace/llmb_venv` is ready for retry |
| `git-lfs` installed | Not yet |
| `HF_TOKEN` set | Not yet — required before `llmb-install` proceeds |

## 4. Open blockers (in priority order)

### B1. nccl-tests build — wrong `MPI_HOME`

`05_setup_tools.sh` defaults `MPI_HOME=/opt/hpcx/ompi`, which doesn't have `mpi.h` on these nodes. Diagnose where MPI actually lives:

```bash
srun -w gpu01 -N1 -n1 bash -c '
  echo "=== HPC-X variants ==="
  ls -d /opt/hpcx* 2>/dev/null
  echo "=== other openmpi ==="
  ls -d /opt/openmpi* /opt/ompi* 2>/dev/null
  echo "=== mpi.h locations ==="
  find /usr/include /usr/lib /opt -name "mpi.h" 2>/dev/null | head
  echo "=== mpicc ==="
  which mpicc 2>/dev/null
  echo "=== dpkg ==="
  dpkg -l 2>/dev/null | grep -iE "openmpi|libmpi|hpcx" | head
'
```

Decision based on output:

| If found | Set | Then |
|---|---|---|
| `/opt/hpcx-2.X.Y/ompi/include/mpi.h` | `MPI_HOME=/opt/hpcx-2.X.Y/ompi` | rebuild nccl-tests |
| `/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h` (apt) | `MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi` | rebuild nccl-tests |
| Nothing found | `apt-get install -y libopenmpi-dev` on each compute node | then rebuild |

Rebuild command (replace `MPI_HOME` accordingly):

```bash
srun -w gpu01 -N1 -n1 bash -c '
  cd /mnt/vfs/mihai/tools/nccl-tests &&
  rm -rf build &&
  make MPI=1 MPI_HOME=<resolved-path> -j
'
```

If apt install is needed:

```bash
srun -w gpu01 -N1 -n1 bash -c "apt-get update && apt-get install -y libopenmpi-dev"
srun -w gpu02 -N1 -n1 bash -c "apt-get update && apt-get install -y libopenmpi-dev"
```

### B2. `llmb-install` — git-lfs not installed

```bash
apt-get update && apt-get install -y git-lfs
git lfs install --system
```

Then take the **faster retry** path (skip re-installing uv + tools):

```bash
export HF_TOKEN=hf_xxx                   # required
source $MY/workspace/llmb_venv/bin/activate
llmb-install
```

When prompted: pick **`x86_64`**, and select these 10 workloads (per `execution.md §1`):

```
pretrain_llama3.1
pretrain_qwen3
pretrain_nemotron4
finetune_llama3
inference_deepseek_r1_trtllm
inference_deepseek_r1_sglang
inference_llama3.3
inference_gpt_oss
microbenchmark_cpu_overhead
microbenchmark_system_info
```

This is a **3–4 h interactive process** plus hours of background downloads. **Run inside tmux** (`tmux new -s install`).

### B3. `run_phase0.sh` orchestrator uses ssh

The orchestrator at `acceptance/run_phase0.sh` calls `ssh "$n"` and `scp` for steps 00, 50, 60. These won't work on this cluster (no SSH between controller and compute). Options:

- **Workaround (now):** run Phase 0 steps manually via `srun`/`sbatch` (sequence in §5).
- **Permanent fix:** rewrite the orchestrator to use `srun` for execution and read scripts directly from `$PLAN/acceptance/` (already on shared NFS, eliminating `scp`).

## 5. Resume sequence

Use two tmux sessions in parallel.

### Session 1: `tmux new -s install`

```bash
source $MY/env.sh
apt-get update && apt-get install -y git-lfs
git lfs install --system
export HF_TOKEN=hf_xxx                          # set this
source $MY/workspace/llmb_venv/bin/activate
llmb-install                                     # interactive; then detach (Ctrl-b d)
```

### Session 2: `tmux new -s phase0`

```bash
source $MY/env.sh

# Resolve B1 (MPI_HOME) — see diagnostic block above
srun -w gpu01 -N1 -n1 bash -c '<diagnostic block>'
# Then rebuild nccl-tests with correct MPI_HOME

# Discovery
srun -w gpu01 -N1 -n1 bash $PLAN/acceptance/00_discovery.sh
srun -w gpu02 -N1 -n1 bash $PLAN/acceptance/00_discovery.sh

# DCGM diag (parallel)
sbatch -w gpu01 $PLAN/acceptance/10_dcgm_diag.sh
sbatch -w gpu02 $PLAN/acceptance/10_dcgm_diag.sh

# nvbandwidth (per node)
sbatch -w gpu01 $PLAN/acceptance/20_nvbandwidth.sh
sbatch -w gpu02 $PLAN/acceptance/20_nvbandwidth.sh

# Pairwise IB — needs both nodes; 30_pairwise_ib.sh uses ssh internally so review/edit first
# NCCL collectives — both nodes, sbatch
sbatch $PLAN/acceptance/40_nccl_tests.sh

# Multi-instance scheduling
sbatch $PLAN/acceptance/45_multi_instance.sh

# fio storage
srun -w gpu01 -N1 -n1 bash $PLAN/acceptance/60_fio_storage.sh

# (50 iperf3 needs IPERF_SERVER set externally; can be skipped if no external endpoint)
```

Watch progress: `squeue` and `ls $PLAN/acceptance/results/`.

After 00–60 are clean, kick off step 70 separately:

```bash
sbatch $PLAN/acceptance/70_gpu_burn.sh
```

## 6. Validate after `llmb-install` finishes

```bash
source $MY/workspace/llmb_venv/bin/activate
llmb-run list                                              # confirm slugs match batch/*.yaml
llmb-run submit -f $PLAN/batch/train_batch.yaml --dry-run  # preview without submitting
```

If `llmb-run list` slugs don't match the keys in `train_batch.yaml`, edit the YAML before real submission — `plan.md` flags this risk explicitly.

## 7. Cluster-side scripts that need updating

These changes should land back in this repo via PR after the run, so future POCs benefit:

- `setup_workspace.sh` — already includes Claude PATH and Claude install. Consider adding git-lfs install.
- `acceptance/05_setup_tools.sh` — should detect head-node (no CUDA) and refuse, or auto-route to compute via `srun`. Should default `TOOLS_DIR=/mnt/vfs/<user>/tools`. Should probe for `MPI_HOME` instead of hardcoding `/opt/hpcx/ompi`.
- `acceptance/run_phase0.sh` — needs `srun`-based version for clusters without inter-node SSH.
- `acceptance/30_pairwise_ib.sh` and `60_fio_storage.sh` — verify whether they assume ssh; rewrite if so.

## 8. Anything to send to Vultr

- Local NVMe: only 1.7 TB visible. Confirm whether the other ~36 TB (10× 3.84 TB) should be mounted, and how (RAID0 vs. per-drive).
- NFS `nconnect`: confirm appliance recommendation; current single TCP stream is the cap.
- Slurm comms required restart once already (root cause unclear). Note in case it recurs.
