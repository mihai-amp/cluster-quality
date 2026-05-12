#!/bin/bash
# Set up Mihai's per-user workspace on the B200 cluster shared NFS.
# Run on the head node as root (or as the target user).
# Idempotent: safe to re-run.

set -euo pipefail

export MY=/mnt/vfs/mihai
export LLMB_INSTALL=$MY/llmb-install

# ---------- Layout ----------
mkdir -p $MY/{workspace,logs,scratch,batch,tools,hf-cache}
mkdir -p $LLMB_INSTALL
mkdir -p $MY/results/phase0/{00_discovery,10_dcgm_diag,20_nvbandwidth,30_pairwise_ib,40_nccl_tests,50_iperf3_external,60_fio_storage,70_gpu_burn,monitor,analysis}
mkdir -p $MY/results/phase1/{pretrain,finetune,inference,microbench}

chmod -R 755 $MY
chmod 700 $MY/scratch

# ---------- Repos ----------
cd $MY/workspace
[ -d dgxc-benchmarking ] || git clone https://github.com/NVIDIA/dgxc-benchmarking.git

if [ ! -d cluster-quality ]; then
    if git clone git@github.com:mihai-amp/cluster-quality.git 2>/dev/null; then
        :
    else
        echo "SSH clone failed, falling back to HTTPS (will require PAT for push later)"
        git clone https://github.com/mihai-amp/cluster-quality.git
    fi
fi

# ---------- Sourceable env file ----------
cat > $MY/env.sh <<'EOF'
# Source in any shell that drives benchmarks:  source /mnt/vfs/mihai/env.sh
export MY=/mnt/vfs/mihai
export LLMB_INSTALL=$MY/llmb-install
export HF_HOME=$MY/hf-cache
export HF_HUB_CACHE=$MY/hf-cache
# export HF_TOKEN=...   # set this; plan flags HF rate-limit risk

# Per-node local caches — NEVER on NFS. /dev/shm is ~1.5T tmpfs on compute nodes.
export TRITON_CACHE_DIR=/dev/shm/$USER/triton
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/$USER/inductor
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR

export PLAN=$MY/workspace/cluster-quality/poc_testing/b200_vultr
export DGXC=$MY/workspace/dgxc-benchmarking

# Tail the live logs of a specific Slurm job by numeric ID.
# Usage: tail_job 256
tail_job() {
    local job="${1:?usage: tail_job <jobid>}"
    local logdir=$(scontrol show job "$job" -o 2>/dev/null | grep -oP 'StdOut=\K\S+' | xargs dirname 2>/dev/null)
    if [ -z "$logdir" ]; then echo "Job $job: no log dir found (job complete or unknown)" >&2; return 1; fi
    echo "Job $job log dir: $logdir" >&2
    tail -f "$logdir"/output_workers.log "$logdir"/log*.out 2>/dev/null
}
EOF

# Auto-source on login
grep -q 'mihai/env.sh' ~/.bashrc || echo "source $MY/env.sh" >> ~/.bashrc

echo
echo "Workspace ready at $MY"
echo "Next:  source $MY/env.sh"
echo "Verify: ls \$LLMB_INSTALL && ls \$PLAN && ls \$DGXC"
