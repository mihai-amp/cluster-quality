#!/bin/bash
# Set up Mihai's per-user workspace on the B200 cluster shared NFS.
# Run on the head node as root (or as the target user).
# Idempotent: safe to re-run.

set -euo pipefail

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

export MY=/mnt/vfs/mihai
export LLMB_INSTALL=$MY/llmb-install

log "Starting setup. MY=$MY  LLMB_INSTALL=$LLMB_INSTALL"

# ---------- Layout ----------
log "[1/4] Creating directory layout under $MY"
mkdir -p $MY/{workspace,logs,scratch,batch,tools,hf-cache}
mkdir -p $LLMB_INSTALL
mkdir -p $MY/results/phase0/{00_discovery,10_dcgm_diag,20_nvbandwidth,30_pairwise_ib,40_nccl_tests,50_iperf3_external,60_fio_storage,70_gpu_burn,monitor,analysis}
mkdir -p $MY/results/phase1/{pretrain,finetune,inference,microbench}

log "       Setting permissions (755 on $MY, 700 on scratch)"
chmod -R 755 $MY
chmod 700 $MY/scratch
log "       Layout done"

# ---------- Repos ----------
log "[2/4] Cloning / verifying repos under $MY/workspace"
cd $MY/workspace
if [ -d dgxc-benchmarking ]; then
    log "       dgxc-benchmarking: already present, skipping clone"
else
    log "       Cloning NVIDIA/dgxc-benchmarking (this may take a minute)..."
    git clone https://github.com/NVIDIA/dgxc-benchmarking.git
    log "       dgxc-benchmarking cloned"
fi

if [ -d cluster-quality ]; then
    log "       cluster-quality: already present, skipping clone"
else
    log "       Cloning mihai-amp/cluster-quality (SSH first, HTTPS fallback)..."
    if git clone git@github.com:mihai-amp/cluster-quality.git 2>/dev/null; then
        log "       cluster-quality cloned (SSH)"
    else
        log "       SSH clone failed, falling back to HTTPS (push will require PAT)"
        git clone https://github.com/mihai-amp/cluster-quality.git
        log "       cluster-quality cloned (HTTPS)"
    fi
fi

# ---------- Sourceable env file ----------
log "[3/4] Writing $MY/env.sh"
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

log "       env.sh written"

# Auto-source on login
log "[4/4] Ensuring ~/.bashrc sources env.sh"
if grep -q 'mihai/env.sh' ~/.bashrc; then
    log "       Already in ~/.bashrc"
else
    echo "source $MY/env.sh" >> ~/.bashrc
    log "       Appended source line to ~/.bashrc"
fi

log "Done. Workspace ready at $MY"
echo
echo "Next:  source $MY/env.sh"
echo "Verify: ls \$LLMB_INSTALL && ls \$PLAN && ls \$DGXC"
