# B200 / Vultr — Test Inventory

Quick reference of every test in the POC, grouped by phase and workload type. For motivation and expected ranges see `plan.md`; for run commands see `execution.md`.

---

## Phase 0 — Cluster acceptance (bare-hardware validation)

- **00 — Discovery** — `nvidia-smi`, `lscpu`, `ibstat`, `ofed_info`, `enroot version`, kernel modules, env snapshot
- **05 — Setup tools** — build `gpu-burn`, `nvbandwidth`, `nccl-tests` once into shared NFS
- **10 — DCGM diag** (per node) — `dcgmi diag -r 3`: PCIe, ECC, thermal, NVLink, memory (~30–60 min/node)
- **20 — nvbandwidth** (per node) — H↔D and D↔D PCIe Gen5 + NVLink throughput
- **30 — Pairwise IB perf** — `ib_write_bw` / `ib_write_lat` on every NIC pair across nodes (validates each of 8 HCAs at ~400 Gb/s)
- **40 — NCCL collectives**
  - Intra-node — `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf` on 8 GPUs / 1 node
  - Inter-node — same on 16 GPUs / 2 nodes
- **45 — Multi-instance scheduling** — 2× concurrent `sbatch` (4+4 GPUs) — validates Slurm/Pyxis cgroup isolation
- **50 — iperf3 external** — egress/ingress to public internet endpoint
- **60 — fio storage** — sequential 1 MiB + random 4 KiB + lots-of-small-files on `/mnt/vfs`
- **70 — gpu-burn** — both nodes simultaneous, 24 h cumulative (2× 12 h windows)

## Phase 0 — Continuous monitoring (passive)

- **`nvidia-smi dmon`** + **`dcgm-exporter`** + **`dmesg` XID/SXID watch** — runs through all phases
- **Per-benchmark power capture** — `nvidia-smi dmon -s puct` on every Phase 1 job
- **SDC convergence proxy** — pairwise loss-curve diff across 3× Llama 3.1 8B FP8 repeats

---

## Phase 1 — Pretrain (3 workloads)

- **Llama 3.1 8B** — Megatron-Bridge, 8 GPU, **FP8 + NVFP4** (only NVFP4 pretrain we can run)
- **Nemotron4 15B** — NeMo framework, 16 GPU, **FP8 + BF16** (NeMo coverage)
- **Qwen3 30B MoE** — Megatron-Bridge, 16 GPU, **BF16** (only MoE pretrain that fits; expert all-to-all)

## Phase 1 — Finetune (1 workload)

- **Llama 3 70B (LoRA)** — Megatron-Bridge, 16 GPU, **FP8 + BF16**

## Phase 1 — Inference (4 workloads × 4 use cases each)

Use cases: **reasoning** (1k/1k), **chat** (128/128), **summarization** (8k/512), **generation** (512/8k).

- **Llama 3.3 70B** — TRT-LLM, 1 GPU, NVFP4 — single-GPU latency reference
- **GPT-OSS 120B** — Dynamo + TRT-LLM, 4 GPU, **MXFP4** — only `generation` use case
- **DeepSeek R1** — TRT-LLM, 4 GPU, NVFP4 — engine comparison A
- **DeepSeek R1** — SGLang, 8 GPU, NVFP4 — engine comparison B (spans both nodes, IB hop)

## Phase 1 — Microbenchmarks

- **`microbenchmark_system_info`** — 8 GPU — validates NCCL/IB/Enroot config; first thing to run
- **`microbenchmark_cpu_overhead`** (GPT-OSS) — 1–4 GPU — TRT-LLM container smoke test

---

## Coverage matrix

| | BF16 | FP8 | NVFP4 | MXFP4 |
|---|:---:|:---:|:---:|:---:|
| Dense pretrain | Nemotron4 | Llama 8B, Nemotron4 | Llama 8B | – |
| MoE pretrain | Qwen3 | – | – | – |
| Finetune | Llama 3 70B | Llama 3 70B | – | – |
| Dense inference | – | – | Llama 3.3, DeepSeek R1 | GPT-OSS |

## Frameworks exercised

- **Megatron-Bridge** — Llama 3.1 8B, Qwen3 30B, Llama 3 70B FT
- **NeMo** — Nemotron4 15B
- **TRT-LLM** — Llama 3.3 70B inf, DeepSeek R1 inf, GPT-OSS micro
- **SGLang** — DeepSeek R1 inf
- **Dynamo** — GPT-OSS inf
