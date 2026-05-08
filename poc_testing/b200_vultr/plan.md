# B200 Benchmark Plan — AMP / Vultr HGX B200

**Owner (AMP):** Mihai Tiuca
**Owner (BFL):** TBD
**Cluster:** 2× HGX B200 (16 GPUs total) on Vultr; head node + 2 compute nodes via Slurm
**Suite:** [NVIDIA dgxc-benchmarking](https://github.com/NVIDIA/dgxc-benchmarking) (`llmb_version: 26.02`)
**Wall-clock budget:** 5 days
**Storage:** 30 TB shared FS (type TBD)

> **Companion docs:**
> - `execution.md` — setup commands, batch files, run sequence, results tables
> - `quirks.md` — cluster-specific findings discovered during the run

---

## 1. Executive Summary

### Goal

Characterize the HGX B200 platform offered by Vultr across a representative slice of NVIDIA's `dgxc-benchmarking` suite, producing defensible throughput and Model-FLOPS-Utilization (MFU) measurements for an internal closed-audience report.

### Why this matters

- **Validate Vultr's HGX B200 offering** against NVIDIA reference numbers — confirm we're getting the silicon performance we're paying for.
- **Inform AMP infrastructure decisions** about which neocloud provides the best price/performance for B200 capacity.
- **Build internal expertise** with NVFP4 — the new Blackwell precision path most worth exercising before it becomes load-bearing in production training.

### Scope and constraints

- 16 GPUs is well below the suite's exemplar scale (256–512 GPUs), so most flagship workloads (Llama 3.1 405B, DeepSeek V3 671B pretrain, Grok1 314B, Nemotron4 340B, Qwen3 235B) are out of reach.
- 2× HGX B200 nodes connect via InfiniBand, **not** NVL72 NVLink fabric — measurements will not reflect GB200/GB300 NVL72 collective-communication behavior.
- Closed audience means no external publication rules to follow, but internal credibility still requires multiple repeats and standard metrics.

---

## 2. Hardware & Network Reference

### Per-GPU (B200)

| Spec | Value |
|---|---|
| Memory | 180 GB HBM3e |
| Memory bandwidth | 7.7 TB/s |
| TDP | 1000 W |
| NVLink (5th gen) | 1.8 TB/s per GPU, 18 links |

### Peak tensor throughput (TFLOPS, dense, B200)

| Precision | Per GPU | Per Node (×8) | Cluster (×16) |
|---|---:|---:|---:|
| BF16 | 2,250 | 18,000 | 36,000 |
| FP8 | 4,500 | 36,000 | 72,000 |
| NVFP4 | 9,000 | 72,000 | 144,000 |
| MXFP4 (inference) | 9,000 | 72,000 | 144,000 |

### Cluster nodes (actuals from Vultr)

| Component | Compute node (×2) | Head node |
|---|---|---|
| GPU | 8× HGX B200 (180 GB HBM3e each) | none |
| CPU | 2× AMD EPYC 9575F @ 3.30 GHz (Zen 4; 64 cores per socket) | Intel Xeon E-2388G (8C/16T) |
| RAM | ~3 TB (3,095,916 MB) | 128 GB |
| Local NVMe | 10× 3.84 TB ≈ 38 TB | 2× 1.92 TB |
| Inter-node fabric | 8× NDR 400 Gb/s ConnectX-7 (≈ 400 GB/s/node) | management only |

Implications:
- AMD EPYC = **x86_64**. Choose `x86_64` when prompted by `llmb-install` (wrong arch = "Exec format error" inside containers).
- 3 TB host RAM per compute node is generous for offload/staging.
- ~38 TB local NVMe per compute node — consider for fast scratch (HF cache, container images, fio testing) where the recipe doesn't require shared FS.
- Head node (8 cores, 128 GB) is sized for control + Claude Code, not heavy installs. Run `./install.sh` from the head node, but the installer will offload large container pulls to compute nodes via Slurm.

### Cluster aggregate

| Resource | Value |
|---|---|
| Total GPU HBM | 2.88 TB |
| Total HBM bandwidth | 123 TB/s |
| Total host RAM | ~6 TB (compute) + 128 GB (head) |
| Total local NVMe | ~76 TB (compute) + ~3.84 TB (head) |
| Total CPU cores | 256 (compute, AMD Zen 4) + 8 (head, Intel) |
| Intra-node NVLink (per node, 8 GPUs) | NVLink 5 fabric, 1.8 TB/s/GPU |
| Inter-node InfiniBand | 8× NDR 400 Gb/s ConnectX-7 ≈ 3.2 Tb/s ≈ 400 GB/s/node |
| Inter-node bisection (2 nodes) | ~400 GB/s aggregate |

### Software stack

| Component | Version |
|---|---|
| LLMB suite | 26.02 |
| Containers | mixed: NeMo 25.09.00 / 26.02.00 / 26.02.01, TRT-LLM 1.1.0rc5, SGLang v0.5.3rc0-cu128-b200, Dynamo 0.6.1, PyTorch 25.12-py3 |
| Scheduler | Slurm 22.x+ (Vultr-managed) |
| Container runtime | Enroot 4.0+ + Pyxis |

---

## 3. Test Plan — Workloads & Coverage

Two phases:

- **Phase 0 — Cluster acceptance** (Day 1–2): hardware/network/storage validation independent of NVIDIA's suite. Catches misconfiguration before we trust benchmark numbers.
- **Phase 1 — Workload benchmarks** (Day 3–5): the dgxc-benchmarking subset.

### 3.0 Cluster acceptance & validation (Phase 0)

Inspired in part by [SemiAnalysis Clustermax 2.0 acceptance criteria](https://newsletter.semianalysis.com/p/the-gpu-cloud-clustermax-rating-system-how-to-rent-gpus). These tests run on bare nodes, before the NVIDIA suite. Scripts live in `acceptance/`.

| Test | Tool | Scale | Purpose |
|---|---|---|---|
| Cluster discovery | bash + `nvidia-smi`, `lscpu`, `ibstat`, `ofed_info`, `enroot version`, `lsmod`, `env` | both nodes | Capture every version, env var, IB HCA layout, kernel module state. Snapshot for the report. |
| GPU health diagnostic | `dcgmi diag -r 3` | per node | Level-3 NVIDIA diagnostic: PCIe, ECC, thermal, NVLink, memory. ~30-60 min. |
| PCIe / NVLink bandwidth | `nvbandwidth` | per node | H↔D and D↔D bandwidth. Catches Gen5 lane drops and NVLink topology bugs. |
| Pairwise IB perf | `ib_write_bw`, `ib_write_lat` (perftest) | every NIC pair across nodes | Validates each of the 8 HCAs delivers ~400 Gb/s; catches one bad NIC NCCL would hide. |
| NCCL collectives — intra-node | nccl-tests `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf` | 8 GPUs / 1 node | NVLink fabric performance, msg sizes 16 MiB → 8 GiB |
| NCCL collectives — inter-node | nccl-tests `all_reduce_perf` etc. | 16 GPUs / 2 nodes | Inter-node IB collective performance — the most important number for distributed training |
| External bandwidth | `iperf3` | 1 node ↔ external endpoint | Egress/ingress to the public internet. Useful for HF downloads, dataset pulls, observability uplinks. |
| Storage — sequential & random | `fio` | shared FS | Read/write at 1 MiB blocks (sequential) and 4 KiB random. Establishes ceiling for checkpoint IO. |
| Storage — Lots Of Small Files | `fio` w/ many-small-file workload | shared FS | Clustermax flagged LOSF as a top pain point — exercise it before HF-cache hits the FS hard. |
| GPU burn-in | [`gpu-burn`](https://github.com/wilicc/gpu-burn) | both nodes simultaneously, **24 h** | Sustained thermal/power/compute stress. Catches infant mortality, throttling, silent compute errors. |
| Multi-instance scheduling | `sbatch` 2× concurrent | 1 node, 4+4 GPUs | Validates Slurm/Pyxis cgroup isolation; catches misconfigured GPU binding |
| Continuous monitoring | `nvidia-smi dmon`, `dcgm-exporter`, `dmesg` XID/SXID watch | passive, all phases | Captured via `monitor/passive_monitor.sh`; reviewed in `quirks.md`. |
| Per-benchmark power capture | `nvidia-smi dmon -s puct` | every Phase 1 job | Output via `monitor/power_capture.sh`; enables perf-per-watt and throttling correlation. |
| SDC convergence proxy | `analysis/convergence_check.sh` | 3× Llama 3.1 8B FP8 repeats | Pairwise loss-curve diff; non-zero divergence beyond FP noise indicates non-determinism or SDC. |

> **Caveat on burn duration:** Clustermax recommends 3-4 weeks of cluster-wide burn. Our 24 h catches gross failures and obvious thermals, but cannot detect rare infant-mortality faults. Document this as a known gap.

### 3.1 Pretrain (3 workloads)

| Workload | Framework | Size | GPUs | Precisions | Why |
|---|---|---|---:|---|---|
| Llama 3.1 8B | Megatron-Bridge | 8B | 8 | FP8, NVFP4 | Dense Transformer reference; only NVFP4 pretrain we can run |
| Nemotron4 15B | NeMo | 15B | 16 | FP8, BF16 | NeMo-framework coverage (rest of suite is Megatron-Bridge) |
| Qwen3 30B | Megatron-Bridge | 30B | 16 | BF16 | Only MoE pretrain that fits 16 GPUs; exercises expert all-to-all |

### 3.2 Finetune (1 workload)

| Workload | Framework | Size | GPUs | Precisions | Why |
|---|---|---|---:|---|---|
| Llama 3 70B finetune | Megatron-Bridge | 70B | 16 | FP8, BF16 | Only finetune in the suite; lands exactly in our scale range |

### 3.3 Inference (4 workloads)

| Workload | Engine | Size | GPUs | Precision | Why |
|---|---|---|---:|---|---|
| Llama 3.3 70B | TRT-LLM | 70B | 1 | NVFP4 | Single-GPU latency reference |
| GPT-OSS 120B | Dynamo + TRT-LLM | 120B | 4 | MXFP4 | MXFP4 path; not exercised by training workloads |
| DeepSeek R1 | TRT-LLM | 671B | 4 | NVFP4 | Engine comparison (same model, two engines) |
| DeepSeek R1 | SGLang | 671B | 8 | NVFP4 | Engine comparison; spans both nodes — IB hop characterization |

### 3.4 Microbenchmarks

| Workload | GPUs | Why |
|---|---:|---|
| `microbenchmark_system_info` | 8 | Validates NCCL/IB/Enroot config; first thing to run |
| `microbenchmark_cpu_overhead` (GPT-OSS) | 1–4 | Smoke test for TRT-LLM container |

### 3.5 Coverage matrix

|  | BF16 | FP8 | NVFP4 | MXFP4 |
|---|:---:|:---:|:---:|:---:|
| Dense pretrain | ✓ (Nemotron4) | ✓ (Llama 8B, Nemotron4) | ✓ (Llama 8B) | – |
| MoE pretrain | ✓ (Qwen3) | – | – | – |
| Finetune | ✓ (Llama 3 70B) | ✓ (Llama 3 70B) | – | – |
| Dense inference | – | – | ✓ (Llama 3.3, DeepSeek R1) | ✓ (GPT-OSS) |

| Framework | Coverage |
|---|---|
| Megatron-Bridge | Llama 3.1 8B, Qwen3 30B, Llama 3 70B finetune |
| NeMo | Nemotron4 15B |
| TRT-LLM | Llama 3.3 70B inf, DeepSeek R1 inf, GPT-OSS micro |
| SGLang | DeepSeek R1 inf |
| Dynamo | GPT-OSS inf |

---

## 4. Expectations

> **Important:** NVIDIA's per-workload READMEs in `dgxc-benchmarking` publish **recipe configs and computation methodology, not target numbers**. They do not include reference step time, TPS/GPU, MFU%, TTFT, or TPOT. Anything that looks like a target in this section is our experience-based estimate. For NVIDIA's actual published B200 numbers we'd need to look elsewhere (MLPerf submissions, NVIDIA blog posts) — out of scope for this POC unless we want them.

### 4.1 Pretrain & Finetune — recipe configs, formulas, MFU ranges

For each workload at our run scale, the recipe's parallelism config and global batch size are fixed (extracted from each workload's README). After each run, compute MFU from the logged step time:

```
tokens_per_step = SeqLen × GBS
tokens_per_sec  = tokens_per_step / step_time
MFU             = (GBS × FLOPs_per_sample) / (step_time × GPUs × peak_FLOPS)
```

`peak_FLOPS` for B200: BF16 = 2.25e15, FP8 = 4.50e15, NVFP4 = 9.00e15.

| Workload | Precision | GPUs | SeqLen | GBS | TP | PP | EP | MBS | Tokens/step | Realistic MFU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.1 8B | FP8 | 8 | 8192 | 128 | 1 | 1 | 1 | 2 | 1,048,576 | **45–55%** |
| Llama 3.1 8B | NVFP4 | 8 | 8192 | 128 | 1 | 1 | 1 | 4 | 1,048,576 | **35–50%** |
| Nemotron4 15B | FP8 | 16 | 4096 | 64 | 1 | 1 | – | 2 | 262,144 | **40–50%** |
| Nemotron4 15B | BF16 | 16 | 4096 | 64 | 1 | 1 | – | 2 | 262,144 | **35–45%** |
| Qwen3 30B | BF16 | 16 | 4096 | 1024 | 1 | 1 | 8 | 4 | 4,194,304 | **25–40%** |
| Llama 3 70B FT (LoRA) | FP8 | 16 | 4096 | 64 | 1 | 2 | – | 1 | 262,144 | **30–45%** |
| Llama 3 70B FT (LoRA) | BF16 | 16 | 4096 | 64 | 1 | 2 | – | 1 | 262,144 | **25–40%** |

**Step-time stddev target across repeats:** <3% (small dense), <5% (MoE / 70B FT). >10% is a red flag — investigate hot-node, network noise, or thermal throttling.

**Workload-specific notes:**
- **Llama 3.1 8B**: smallest dense workload; tight in NVLink domain. Highest MFU expected in the suite.
- **Nemotron4 15B**: FLOPs constant published as 3.85e14/sample. NeMo framework adds ~5% launch overhead vs Megatron-Bridge.
- **Qwen3 30B**: 3B active parameters of 30B total (MoE). EP=8 means experts span all 8 GPUs of a node — exercises intra-node NVLink all-to-all heavily, no inter-node EP comm at this scale.
- **Llama 3 70B finetune**: LoRA fine-tune on SQUAD. **Use `parse_train_timing_mbridge.sh`** — the framework's reported `MODEL_TFLOP/s/GPU` is incorrect for LoRA in this release.

> **Aggregate sanity check:** at 16 GPUs FP8, peak compute is 72 PFLOPS dense. A 50% MFU pretrain run is sustaining 36 PFLOPS — equivalent to ~18 H100s at peak FP8. If we're seeing single-digit PFLOPS sustained, something is broken (likely NCCL fallback to socket transport).

### 4.2 Inference — recipe configs and use cases

Each inference workload runs **4 use cases** with different ISL/OSL pairs. Capture metrics per use case.

| Use case | ISL | OSL | What it stresses |
|---|---:|---:|---|
| reasoning | 1000 | 1000 | balanced prefill/decode |
| chat | 128 | 128 | short-form, decode latency |
| summarization | 8000 | 512 | prefill-heavy, KV bandwidth |
| generation | 512 | 8000 | decode-heavy, sustained TPOT |

Recipe configs at our scales:

| Workload | Engine | GPUs | TP | PP | EP/DP | attn-DP | KV frac | Notes |
|---|---|---:|---:|---:|---:|---|---:|---|
| Llama 3.3 70B | TRT-LLM | 1 | 1 | 1 | 1 | yes | 0.95 | Only `max_throughput` mode on B200 (`min_latency` requires GB200 + 4 GPUs). Model: nvidia/Llama-3.3-70B-Instruct-FP4 (~40 GB). |
| GPT-OSS 120B | Dynamo+TRT-LLM | 4 | 4 | 1 | DP=4 | yes (dp-attention) | 0.8 | Only `generation` use case (128/1000) published. Model: openai/gpt-oss-120b (~61 GB). |
| DeepSeek R1 | TRT-LLM | 4 | 4 | 1 | EP=4 | yes | 0.50–0.85 (varies by use case) | All 4 use cases. No chunked-prefill on B200 (vs GB200). Model: nvidia/DeepSeek-R1-FP4 (~395 GB). |
| DeepSeek R1 | SGLang | 8 | 8 | 1 | DP=8 | yes (dp-attention) | 0.7–0.8 (varies) | All 4 use cases. Spans 2 nodes — captures inter-node IB latency in TPOT tail. |

**Realistic latency ranges (experience-based, not NVIDIA-published):**

| Workload | Throughput (tok/s, aggregate) | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---:|---:|
| Llama 3.3 70B (TRT-LLM, 1 GPU) | 80–150 (single-stream) / 1k–3k (concurrency=2k) | 80–200 | 8–18 |
| GPT-OSS 120B (Dynamo+TRT-LLM, 4 GPUs) | 1,500–4,000 | 100–250 | 10–25 |
| DeepSeek R1 (TRT-LLM, 4 GPUs) | 1,000–3,000 | 200–500 | 15–30 |
| DeepSeek R1 (SGLang, 8 GPUs) | 2,000–5,000 | 250–600 | 15–35 |

> **Engine comparison signal:** the most informative comparison in the inference set is TRT-LLM-@4 vs SGLang-@8 on DeepSeek R1. SGLang's 8-GPU config has 2× the compute but pays the inter-node IB cost. Whether throughput scales above 1.5× and how much TTFT p99 grows tells us something concrete about how useful inter-node inference is on this fabric.

### 4.3 Aggregate cluster theoretical peaks

| Quantity | Value |
|---|---|
| Cluster peak FP8 throughput (16 GPUs, dense) | 72 PFLOPS |
| Cluster peak NVFP4 throughput (16 GPUs, dense) | 144 PFLOPS |
| Cluster peak BF16 throughput (16 GPUs, dense) | 36 PFLOPS |
| Cluster aggregate HBM | 2.88 TB |
| Cluster aggregate HBM BW | 123 TB/s |
| Intra-node NVLink BW (per node) | 1.8 TB/s/GPU × 8 ≈ 14.4 TB/s aggregate |
| Inter-node IB BW (Vultr typical) | 400 GB/s per node, ~400 GB/s bisection |

---

## 5. Pass/Fail Criteria

- **Green:** All workloads complete; mean MFU within realistic range; step-time stddev within target.
- **Yellow:** 1-2 workloads outside range or with >10% step-time stddev — investigate but don't block report.
- **Red:** Multiple workloads at <60% of expected MFU, or NCCL bandwidth at <50% of nominal — likely cluster misconfiguration; pause and triage before continuing.

---

## 6. Anticipated Risks

These are forward-looking risks identified during planning. Issues actually encountered should be logged in `quirks.md`.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wrong `NCCL_IB_HCA` on Vultr → bandwidth halved | Medium | High | Run `microbenchmark_system_info` first; confirm 8 HCAs detected |
| NeMo 26.02.00 EFA library conflict (Llama 3.1 8B uses this container) | Low (Vultr is IB, not EFA) | Medium | Apply `rm -rf /opt/rdma-core/build/lib/` patch only if NCCL falls back to socket |
| `uv 0.9.29+` breaks recipes installing `nemo_run` | Low | High | `install.sh` enforces `uv <=0.9.28`; do not override |
| HF rate-limit during install | Low | Low | HF_TOKEN set; retry sequentially if 429s |
| SGLang 8-GPU run has high tail latency | Medium | Medium | Expected — IB hop visible. Capture and report rather than treat as failure |
| Step-time variance >10% across repeats | Medium | Medium | Reserve Day 5 for reruns on different node pairs via `--nodelist` |
| Inference batch YAML format unsupported | Medium | Low | Fall back to explicit per-workload `llmb-run submit` invocations |

---

## 7. Deliverables

After the run completes:

1. `llmb-archive-<timestamp>.tar.zst` — full experiment archive (logs + configs).
2. Filled-in results tables in `execution.md`.
3. Comparison commentary against §4 Expectations — flag deltas and hypotheses (in `quirks.md`).
4. NCCL/IB diagnostic snapshot from `microbenchmark_system_info`.
5. (Optional) Nsight Systems profile of one representative pretrain run for the report appendix.

---

## 8. References

- [NVIDIA dgxc-benchmarking repo](https://github.com/NVIDIA/dgxc-benchmarking)
- [Workload READMEs](https://github.com/NVIDIA/dgxc-benchmarking/tree/main) — each contains the workload-specific NVIDIA reference numbers
- [llmb-run docs](https://github.com/NVIDIA/dgxc-benchmarking/blob/main/cli/llmb-run/README.md)
- [Bulk submission examples](https://github.com/NVIDIA/dgxc-benchmarking/blob/main/cli/llmb-run/Bulk_Examples.md)
- [Exemplar validation guide](https://github.com/NVIDIA/dgxc-benchmarking/blob/main/Exemplar_validation.md)
- Contact: `LLMBenchmarks@nvidia.com`
