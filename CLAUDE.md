# cluster-quality — guidance for Claude

This repo holds POCs validating GPU clusters before AMP commits to them. Each POC lives at `poc_testing/<gpu>_<provider>/` and follows the same 3-doc + script-dirs layout.

## When asked to work in a POC

Read these in order:
1. `<poc>/plan.md` — design intent, hardware/network reference, expectations. Treat as read-mostly; only edit if scope changes.
2. `<poc>/execution.md` — runbook with day-by-day, batch refs, results tables. **Fill in the result tables** in §6 as runs complete.
3. `<poc>/quirks.md` — live journal. **Append findings as they happen** — don't batch up at end of run.

Then look at the script dirs:
- `<poc>/acceptance/` — Phase 0 cluster validation (numbered 00–70). Run before any benchmark.
- `<poc>/batch/` — Phase 1 dgxc-benchmarking submissions.
- `<poc>/monitor/` — continuous power/health capture during benchmarks.
- `<poc>/analysis/` — post-hoc parsing and SDC convergence proxy.

## Common operations

- Submit a benchmark batch: `cd $LLMB_INSTALL && llmb-run submit -f <path>/train_batch.yaml`
- Dry-run first: append `--dry-run`. **Always.**
- Verify workload slugs: `llmb-run list`
- Check Slurm queue: `squeue -u $USER`
- Cancel a job: `scancel <jobid>`
- Find logs: `$LLMB_INSTALL/workloads/<workload>/experiments/<run_id>/`
- Parse step times: `$LLMB_REPO/common/parse_train_timing*.sh`

## Working norms

- **Confirm before destructive actions.** gpu-burn occupies all GPUs; fio writes large files to shared FS; `scancel` kills running work. State the intent and ask before running.
- **Don't `git add -A`.** Be explicit about what you're staging — runtime output goes under `*/results/` which is gitignored, but log files in odd locations should not get committed.
- **Update `quirks.md` continuously**, not at the end. Cluster findings rot if not captured immediately.
- **The plan vs reality split is intentional.** When something is broken or surprising, that's a quirks.md entry, not a plan.md edit. Edit plan.md only if we're changing what we set out to do.
- Most acceptance scripts assume tools pre-installed on the cluster (`dcgmi`, `nccl-tests`, `perftest`, `fio`, `iperf3`). Check before running; flag missing prereqs as a quirks entry.

## Cluster-specific (B200/Vultr)

- Architecture is **x86_64** (AMD EPYC 9575F). Choose accordingly in `llmb-install`.
- 30 TB shared FS lives at a path TBD on Day 1 — set `LLMB_INSTALL` there.
- 38 TB local NVMe per node available for scratch (HF cache, container images).
- Inter-node fabric is InfiniBand (8× NDR ConnectX-7), not EFA. None of the EFA workarounds apply.
- iperf3 external uses an AMP-controlled GCP endpoint — set `IPERF_SERVER` before running step 50.

## What NOT to do without asking

- Don't change `plan.md` §3 or §4 (workloads or expectations) — those are decisions, not documentation.
- Don't `rm -rf` anywhere outside `*/results/`.
- Don't push to remote without explicit user request.
- Don't run gpu-burn during active benchmarks — it'll preempt and ruin the run.
