#!/usr/bin/env bash
#SBATCH --job-name=phase0_ibhealth
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=results/ib_health_%j.log
#
# Layered IB-fabric health check. Run as:
#   sbatch $PLAN/acceptance/35_ib_health_check.sh
#
# Produces a single clean pass/fail summary table at the bottom suitable
# for pasting into a cluster-provider support ticket. Tests four layers:
#   1. Ethernet ping  — control (must work)
#   2. SM topology    — does the InfiniBand subnet manager see both nodes?
#   3. IB UD ping     — can a low-level IB packet transit between nodes?
#   4. IB RDMA write  — does an actual RDMA queue-pair transfer succeed?
#
# A working B200 cluster passes all four. A cluster where the IB fabric
# isn't actually routing between nodes will pass (1) and (2) but fail
# (3) and (4) — that's the smoking gun for "fabric broken" vs "cards bad".

set -uo pipefail

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"

NODES=($(scontrol show hostname "$SLURM_NODELIST"))
[ "${#NODES[@]}" -lt 2 ] && { echo "Need 2 nodes; got ${NODES[*]}" >&2; exit 1; }
NODE_A="${NODES[0]}"
NODE_B="${NODES[1]}"

HCA="${HCA:-mlx5_0}"   # which NDR HCA to test on
PORT="${PORT:-18515}"

TS_LOG="$OUT_DIR/ib_health_$(date -u +%Y%m%dT%H%M%SZ).log"

# ---------------------------------------------------------------------
# 1. Ethernet ping (control — should always pass)
# ---------------------------------------------------------------------
echo "[1/4] Ethernet ping $NODE_A -> $NODE_B"
ETH=$(srun --nodelist="$NODE_A" -N1 -n1 ping -c 3 -W 2 "$NODE_B" 2>&1 | tee -a "$TS_LOG")
if echo "$ETH" | grep -q '0% packet loss'; then ETH_RESULT="PASS"; else ETH_RESULT="FAIL"; fi
echo "  -> $ETH_RESULT"
echo

# ---------------------------------------------------------------------
# 2. SM topology — what endpoints does the InfiniBand subnet manager see?
# ---------------------------------------------------------------------
echo "[2/4] InfiniBand subnet manager topology (ibhosts)"
IBHOSTS=$(srun --nodelist="$NODE_A" -N1 -n1 ibhosts 2>&1 | tee -a "$TS_LOG")
SM_ENDPOINTS=$(echo "$IBHOSTS" | grep -c '^Ca' 2>/dev/null || echo 0)
echo "  endpoints visible from $NODE_A: $SM_ENDPOINTS"
echo "$IBHOSTS" | head -20
if [ "$SM_ENDPOINTS" -ge 2 ]; then SM_RESULT="PASS ($SM_ENDPOINTS endpoints)"; else SM_RESULT="FAIL ($SM_ENDPOINTS endpoints)"; fi
echo "  -> $SM_RESULT"
echo

# ---------------------------------------------------------------------
# 3. IB UD ping — low-level packet test (no RDMA, just unreliable datagram)
# ---------------------------------------------------------------------
echo "[3/4] InfiniBand UD ping over $HCA"
# Server in background on NODE_A
srun --nodelist="$NODE_A" -N1 -n1 ibping -d "$HCA" -P 1 -S >/tmp/ibping_srv.$$ 2>&1 &
SRV_PID=$!
sleep 3

# Grab NODE_A's base LID for this HCA
LID_HEX=$(srun --nodelist="$NODE_A" -N1 -n1 bash -c "ibstat $HCA 2>/dev/null | grep -oE 'Base lid: [0-9]+' | head -1 | awk '{print \$3}'")
echo "  $NODE_A $HCA base LID: $LID_HEX"

UDPING=$(timeout 15 srun --nodelist="$NODE_B" -N1 -n1 ibping -d "$HCA" -L "$LID_HEX" -c 5 -i 1 2>&1 | tee -a "$TS_LOG")
echo "$UDPING"
if echo "$UDPING" | grep -q '5 packets received'; then UD_RESULT="PASS"; else UD_RESULT="FAIL"; fi
kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
echo "  -> $UD_RESULT"
echo

# ---------------------------------------------------------------------
# 4. IB RDMA write — actual RDMA queue-pair transfer
# ---------------------------------------------------------------------
echo "[4/4] InfiniBand RDMA write (ib_write_bw, RC) over $HCA"
srun --nodelist="$NODE_A" -N1 -n1 ib_write_bw -d "$HCA" -p "$PORT" -D 10 -F >/tmp/ibwbw_srv.$$ 2>&1 &
SRV_PID=$!
sleep 3

CLIENT_OUT=$(timeout 30 srun --nodelist="$NODE_B" -N1 -n1 ib_write_bw -d "$HCA" -p "$PORT" -D 10 -F "$NODE_A" 2>&1 | tee -a "$TS_LOG")
echo "$CLIENT_OUT" | head -30

if echo "$CLIENT_OUT" | grep -qE 'Failed status|Couldn|error|Completion with error'; then
    RDMA_RESULT="FAIL ($(echo "$CLIENT_OUT" | grep -oE 'Failed status [0-9]+[^,]*' | head -1))"
else
    BW=$(echo "$CLIENT_OUT" | awk '/^[ ]*[0-9]+[ ]+[0-9]+[ ]+[0-9.]+[ ]+[0-9.]+/ {print $4}' | tail -1)
    if [ -n "$BW" ] && awk "BEGIN{exit !($BW > 1000)}"; then
        RDMA_RESULT="PASS ($BW MB/s)"
    else
        RDMA_RESULT="FAIL (no bandwidth measured)"
    fi
fi
kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
echo "  -> $RDMA_RESULT"
echo

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
echo "==========================================================="
echo "SUMMARY ($NODE_A <-> $NODE_B, HCA=$HCA)"
echo "==========================================================="
printf "  %-30s %s\n" "[1] Ethernet ping"            "$ETH_RESULT"
printf "  %-30s %s\n" "[2] SM endpoints visible"      "$SM_RESULT"
printf "  %-30s %s\n" "[3] IB UD ping"                "$UD_RESULT"
printf "  %-30s %s\n" "[4] IB RDMA write"             "$RDMA_RESULT"
echo "-----------------------------------------------------------"
if [ "$ETH_RESULT" = "PASS" ] && \
   { echo "$UD_RESULT" | grep -q FAIL || echo "$RDMA_RESULT" | grep -q FAIL; }; then
    echo "VERDICT: Inter-node InfiniBand fabric is NON-FUNCTIONAL."
    echo "         Nodes can reach each other on Ethernet, but InfiniBand"
    echo "         packets are not transiting between them despite both"
    echo "         HCAs reporting State=Active in ibstat."
    echo "         This requires cluster-provider intervention."
elif echo "$ETH_RESULT $UD_RESULT $RDMA_RESULT" | grep -q FAIL; then
    echo "VERDICT: One or more components failing — see per-layer results."
else
    echo "VERDICT: Inter-node IB fabric is healthy."
fi
echo "==========================================================="
echo "Full log: $TS_LOG"
