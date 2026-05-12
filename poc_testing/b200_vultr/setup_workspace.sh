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

log "       Setting permissions on top-level dirs (non-recursive)"
chmod 755 $MY $MY/workspace $MY/logs $MY/batch $MY/tools $MY/hf-cache $LLMB_INSTALL 2>/dev/null || true
chmod 700 $MY/scratch 2>/dev/null || true
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

# List the live log files for a Slurm job by numeric ID, and recommend one
# to tail. Doesn't tail automatically — for Dynamo jobs with many workers,
# auto-tailing every matched file exhausts the shell's open-fd limit.
# Walks $LLMB_INSTALL/workloads for files modified recently whose path
# or name contains the job ID. Recommended file (Slurm stdout) is in magenta.
# Usage: tail_job 256
tail_job() {
    local job="${1:?usage: tail_job <jobid>}"
    local stdout=$(scontrol show job "$job" -o 2>/dev/null | grep -oP 'StdOut=\K\S+')
    local extras
    extras=$(find "$LLMB_INSTALL/workloads" -mmin -120 -type f \
        \( -name "*${job}*.out" -o -name "*${job}*.log" -o -path "*_${job}/server_logs/*" -o -path "*_${job}_*/server_logs/*" \) \
        2>/dev/null | sort -u)
    if [ -z "$stdout" ] && [ -z "$extras" ]; then
        echo "Job $job: no log file found (job may have just started, or is complete and out of MinJobAge)" >&2
        return 1
    fi
    # ANSI colors (only if stderr is a tty)
    local Y='' G='' M='' R=''
    if [ -t 2 ]; then Y=$'\e[1;33m'; G=$'\e[1;32m'; M=$'\e[1;35m'; R=$'\e[0m'; fi
    local bar="${Y}==================================================================${R}"
    echo "$bar" >&2
    echo "${Y}Job $job log files:${R}" >&2
    local recommended=""
    if [ -n "$stdout" ] && [ -f "$stdout" ]; then
        echo "  ${M}[recommend]${R} ${M}$stdout${R}" >&2
        recommended="$stdout"
    fi
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        [ "$f" = "$recommended" ] && continue
        echo "  ${Y}[also     ]${R} ${G}$f${R}" >&2
    done <<< "$extras"
    echo "$bar" >&2
    if [ -n "$recommended" ]; then
        echo "${Y}tail -f $recommended${R}" >&2
    fi
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
