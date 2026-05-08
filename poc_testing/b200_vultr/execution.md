# B200 Benchmark — Execution Runbook

> **Plan:** `plan.md` — goals, workload selection, expectations.
> **Quirks:** `quirks.md` — cluster-specific findings to apply.

This runbook has two phases:

- **Phase 0 — Cluster acceptance** (§A): bare-cluster validation before any NVIDIA workloads run. Lives in `acceptance/`.
- **Phase 1 — Workload benchmarks** (§1–§7 below): dgxc-benchmarking suite. Batch artifacts in `batch/`.

---

## A. Phase 0 — Cluster acceptance

Run these in numbered order. Most produce a log file in `acceptance/results/`. See script headers for tool prerequisites (perftest, fio, iperf3, nccl-tests, dcgmi, nvbandwidth, gpu-burn) — most are pre-installed on Vultr nodes; gpu-burn requires `git clone && make`.

| Step | Script | Wall clock |
|---|---|---:|
| Discovery | `acceptance/00_discovery.sh` | ~5 min |
| Build helper tools | `acceptance/05_setup_tools.sh` | ~10 min |
| DCGM diag (per node) | `acceptance/10_dcgm_diag.sh` | ~30-60 min/node |
| PCIe / NVLink bandwidth | `acceptance/20_nvbandwidth.sh` | ~5 min |
| Pairwise IB perf | `acceptance/30_pairwise_ib.sh` | ~30 min |
| NCCL collectives | `acceptance/40_nccl_tests.sh` | ~30 min |
| Multi-instance scheduling | `acceptance/45_multi_instance.sh` | ~10 min |
| iperf3 external (GCP endpoint) | `acceptance/50_iperf3_external.sh` | ~10 min |
| fio storage | `acceptance/60_fio_storage.sh` | ~30 min |
| gpu-burn (2× 12h windows) | `acceptance/70_gpu_burn.sh` | 12 h × 2 = 24 h cumulative |

`acceptance/run_phase0.sh` orchestrates steps 00–60 sequentially. gpu-burn runs in two windows (Day 1 evening + Day 4 evening) for 24 h cumulative. Set `IPERF_SERVER=<GCP endpoint>` before running step 50.

After Phase 0, copy the log directory into `quirks.md` discoveries and flag any anomalies before kicking off Phase 1.

---

## 1. One-time install (interactive, ~3-4 hrs)

```bash
git clone https://github.com/NVIDIA/dgxc-benchmarking.git
cd dgxc-benchmarking
export HF_TOKEN=<your_token>
export LLMB_INSTALL=<path on the 30TB shared FS>

./install.sh
```

When prompted, select these workloads:

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

> **Verify slugs first:** run `llmb-run list` after install to confirm names. Fix any mismatches in the YAML files under `batch/` before submitting.

After this first interactive install, future workload additions are scripted:

```bash
llmb-install express $LLMB_INSTALL --workloads <comma-list>
```

## 2. Validate

```bash
cd $LLMB_INSTALL
llmb-run list                                        # list installed workloads
llmb-run submit -f batch/train_batch.yaml --dry-run  # preview without submitting
```

## 3. Batch files & wrappers

Files live in `batch/`:

- `batch/train_batch.yaml` — pretrain + finetune (23 jobs)
- `batch/inference_batch.yaml` — inference workloads (YAML format unverified for inference)
- `batch/inference_fallback.sh` — explicit per-workload fallback if YAML batch fails
- `batch/run_all.sh` — orchestration wrapper

Power/thermal capture lives in `monitor/`:
- `monitor/power_capture.sh` — sourceable helpers; spawns `nvidia-smi dmon` per benchmark
- `monitor/passive_monitor.sh` — long-running cluster-wide health snapshot (run continuously across the benchmark window)

Convergence (SDC proxy) lives in `analysis/`:
- `analysis/convergence_check.sh` — pairwise loss-curve diff across the 3 Llama 3.1 8B FP8 repeats. Run after Day 3.

Submit batch files from `$LLMB_INSTALL` (where `cluster_config.yaml` lives). Either copy the batch directory into `$LLMB_INSTALL` or reference it by absolute path.

## 4. Day-by-day timeline

| Day | Phase | Activity | Active time |
|---|---|---|---|
| 1 AM | 0 | Run `acceptance/run_phase0.sh` (discovery → fio); start `./install.sh` in parallel | ~6 hr (mostly idle) |
| 1 PM | 0 | Submit gpu-burn window 1 (12 h, runs overnight); start `monitor/passive_monitor.sh &` for the rest of the test | ~30 min active |
| 2 AM | 0 | Burn window 1 finishes; review Phase 0 logs; populate `quirks.md` | ~1 hr |
| 2 PM | 1 | Smoke (`microbenchmark_system_info`, `microbenchmark_cpu_overhead`); dry-run + submit `train_batch.yaml` | ~30 min |
| 3 | 1 | Slurm processes 23 training jobs; passive monitor + power capture run alongside | monitor occasionally |
| 4 AM | 1 | Submit `inference_batch.yaml` (or `inference_fallback.sh` if YAML errors); capture IB perf during DeepSeek SGLang | ~30 min |
| 4 PM | 0 | Submit gpu-burn window 2 (12 h overnight) once benchmark queue drains | ~10 min |
| 5 AM | 1 | Burn window 2 ends; run `analysis/convergence_check.sh` over Llama 3.1 8B FP8 repeats; reruns for high-variance workloads | ~half day |
| 5 PM | 1 | `llmb-run archive`; parse logs; finalize results tables; stop passive monitor | ~2 hr |

**Cumulative gpu-burn coverage:** 24 h across two 12 h windows. Less than Clustermax's 3-4 weeks, but as much as the 5-day budget allows without preempting benchmarks.

## 5. Where logs land

- Per-job: `$LLMB_INSTALL/workloads/<workload>/experiments/<run_id>/`
- Each run also drops `llmb-config_<JOBID>.yaml` capturing parameters
- Archive everything at the end with `llmb-run archive`
- Parsing helpers: `$LLMB_REPO/common/parse_train_timing.sh` (and `_mbridge.sh` variant)

---

## 6. Results — to fill in after experimentation

### 6.1 Pretrain & Finetune

| Workload | Size | Precision | GPUs | Step time mean (s) | Step time stddev | Tokens/s/GPU | MFU % | Container | Run count |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama 3.1 8B | 8B | FP8 | 8 |  |  |  |  | 26.02.00 | 3 |
| Llama 3.1 8B | 8B | NVFP4 | 8 |  |  |  |  | 26.02.00 | 3 |
| Nemotron4 15B | 15B | FP8 | 16 |  |  |  |  | 25.09.00 | 3 |
| Nemotron4 15B | 15B | BF16 | 16 |  |  |  |  | 25.09.00 | 3 |
| Qwen3 30B | 30B | BF16 | 16 |  |  |  |  | 26.02.01 | 3 |
| Llama 3 70B FT | 70B | FP8 | 16 |  |  |  |  | 26.02.01 | 3 |
| Llama 3 70B FT | 70B | BF16 | 16 |  |  |  |  | 26.02.01 | 3 |

### 6.2 Inference

| Workload | Engine | Size | GPUs | Precision | Throughput (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | Concurrency |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Llama 3.3 70B | TRT-LLM | 70B | 1 | NVFP4 |  |  |  |  |  |  |
| GPT-OSS 120B | Dynamo+TRT-LLM | 120B | 4 | MXFP4 |  |  |  |  |  |  |
| DeepSeek R1 | TRT-LLM | 671B | 4 | NVFP4 |  |  |  |  |  |  |
| DeepSeek R1 | SGLang | 671B | 8 | NVFP4 |  |  |  |  |  |  |

### 6.3 Microbenchmark observations

| Test | Result | Notes |
|---|---|---|
| `microbenchmark_system_info` |  | NCCL_IB_HCA, NUMA binding, etc. |
| `microbenchmark_cpu_overhead` (GPT-OSS) |  | Per-GPU CPU overhead (μs) |

---

## 7. Archive & handoff

```bash
# At end of run
llmb-run archive --output /shared/results/vultr-b200-$(date +%Y%m%d).tar.zst
```

Archive contains experiment logs and `llmb-config_*.yaml` per job. Profile data (`.nsys-rep`) is excluded — capture separately if needed.
