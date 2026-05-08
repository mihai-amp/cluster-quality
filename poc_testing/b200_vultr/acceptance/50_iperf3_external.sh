#!/usr/bin/env bash
# External-internet bandwidth via iperf3.
# Tests a B200 node's egress/ingress against a public iperf3 server (or your own).
# Useful for HF model pulls, dataset downloads, observability uplinks.
#
# Override IPERF_SERVER to point at a known-good endpoint.
# Public iperf3 list: https://iperf.fr/iperf-servers.php

set -uo pipefail

# Set IPERF_SERVER to AMP's GCP iperf3 endpoint (controlled, low-noise).
# Public iperf hosts are noisy and unreliable; only fall back to them if
# the GCP endpoint is unavailable.
IPERF_SERVER="${IPERF_SERVER:?set IPERF_SERVER to the GCP iperf3 endpoint}"
IPERF_PORT="${IPERF_PORT:-5201}"
DURATION="${DURATION:-30}"
PARALLEL="${PARALLEL:-8}"

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/iperf3_external_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ).log"

{
  echo "==== iperf3 external from $(hostname) -> $IPERF_SERVER:$IPERF_PORT ===="
  echo "Duration: ${DURATION}s, parallel streams: $PARALLEL"
  date -u

  echo
  echo "--- TCP downstream (server -> client) ---"
  iperf3 -c "$IPERF_SERVER" -p "$IPERF_PORT" -t "$DURATION" -P "$PARALLEL" -R

  echo
  echo "--- TCP upstream (client -> server) ---"
  iperf3 -c "$IPERF_SERVER" -p "$IPERF_PORT" -t "$DURATION" -P "$PARALLEL"

  echo
  echo "--- UDP loss/jitter (1 Gbps offered) ---"
  iperf3 -c "$IPERF_SERVER" -p "$IPERF_PORT" -t 10 -u -b 1G || true
} | tee "$OUT"

echo "Wrote $OUT"
