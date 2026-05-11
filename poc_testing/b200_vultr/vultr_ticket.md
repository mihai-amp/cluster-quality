# Vultr Support Ticket — B200 Cluster Inter-Node InfiniBand Non-Functional

> **Status:** Open
> **Filed:** 2026-05-10
> **Cluster:** 2× HGX B200 (gpu01, gpu02) + 1 controller on Vultr
> **Owner:** Mihai Tiuca (mihai@amppbc.com)
> **Impact:** Blocks all multi-node benchmarks in the AMP B200 evaluation POC. Single-node workloads continue to run normally.

## Summary

All 8 NDR ConnectX-7 rails fail to carry RDMA traffic between gpu01 and gpu02 despite IB cards reporting `State: Active`, `Rate: 400 Gb/s`, and a valid subnet manager.

## Evidence

### What works (rules out cards / driver / Ethernet / SM hardware)

- **Ethernet** between gpu01 and gpu02: 0% packet loss, 0.09 ms RTT
- **`ibstat`** on both nodes: all 8 NDR HCAs `State: Active`, `Rate: 400 Gb/s`, distinct base LIDs assigned, same `SM lid: 64` across cards
- **`ibhosts`** from gpu01 sees endpoints from both nodes including each other's ConnectX-7 HCAs
- **Single-node workloads** (NVLink internal) run cleanly — confirms compute, host stack, container runtime, NCCL intra-node fabric are all healthy

### What fails (the bug)

- **`ib_write_bw -d mlx5_0`** between gpu01 (server) and gpu02 (client):
  - QP setup completes (LIDs exchanged via Ethernet sideband)
  - First RDMA write: `Failed status 12 (IBV_WC_RETRY_EXC_ERR) syndrom 0x81`
  - `scnt=128, ccnt=0` — 128 sends issued, **zero** completions
- **Same failure** confirmed across all 8 NDR HCAs (`mlx5_0,1,2,3,4,9,12,13`) — none of them carries inter-node traffic
- **Open MPI / UCX** (4.1.7rc1) reports `no active messages transport ... Destination is unreachable` on `rc_verbs/mlx5_0:1`, `ud_verbs/mlx5_0:1`, `dc_mlx5/mlx5_0:1`
- **PMIx** subsequently fails with `UNREACHABLE in file server/pmix_server.c at line 2198`

## Interpretation

IB packets are not transiting between the two compute nodes despite all L1/L2 indicators showing healthy. Suggests one of:

1. **Subnet manager has not programmed forwarding tables** for inter-node paths (most likely)
2. The "rail-optimized" NDR subnets aren't actually carrying inter-node traffic
3. NDR switch fabric between nodes is misconfigured or absent

## Reproducer

Layered diagnostic script (in repo) reproduces all four findings in <2 minutes:

```bash
sbatch /mnt/vfs/mihai/cluster-quality/poc_testing/b200_vultr/acceptance/35_ib_health_check.sh
```

Output summary section shows `[1] Ethernet PASS`, `[2] SM PASS`, `[3] IB UD ping FAIL`, `[4] IB RDMA write FAIL`.

## Other potentially-relevant cluster quirks discovered

(Independent of this issue but worth noting for the support engineer)

- `/etc/hosts` had each compute hostname listed twice with the same IP (cloud-init template bug). Caused `hostname --ip-address` to return duplicated IP, malforming launcher scripts. Workaround applied (`awk '!seen[$0]++'`).
- Per-node local NVMe: only 1.7 TB visible at `/`, while plan expected ~38 TB (10× 3.84 TB). Other drives are visible via `lsblk` but unassembled/unmounted.

## Next steps requested

1. Verify SM state and routing tables for the inter-node NDR fabric
2. Confirm the NDR switch is reachable from both compute nodes and forwarding correctly
3. Once fixed, the reproducer above should show `[4] IB RDMA write: PASS (~46-49 GB/s)`

## Updates (append as we hear back)

_(none yet)_
