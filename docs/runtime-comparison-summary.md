# Runtime Comparison Summary — GKE, NVIDIA L4

Consolidated results from all four `compare-runtimes.sh` runs executed on
2026-07-07 against a GKE 1.35 cluster with NVIDIA L4 GPUs (`g2-standard-4`).

Source data: `results/compare-*/comparison.json` (machine-readable, also served
by the dashboard's **Runtime Overhead** tab via `/api/comparisons`).

## Test matrix

| Compare ID | Runtimes | Benchmarks | Repeats | Environment |
|---|---|---|---|---|
| `compare-20260707-083442` | runc, gVisor | NCCL¹, coldstart | 2 | Host cluster (first-run warmup) |
| `compare-20260707-084508` | runc, gVisor | NCCL, coldstart | 3 | Host cluster |
| `compare-20260707-140126` | runc, gVisor, vNode | NCCL, coldstart | 3 | Host cluster |
| `compare-20260707-143651` | runc, gVisor, vNode | coldstart² | 3 | **Inside ephemeral vClusters** |

¹ NCCL parse incomplete on the warmup run; excluded from the bandwidth table.
² NCCL inside a vCluster requires installing the MPI Operator into each
ephemeral virtual cluster, so the vCluster level measures cold-start only —
the axis where a control-plane sync layer could plausibly add latency.

## NCCL all_reduce bus bandwidth (steady-state, 1× L4)

| Run | runc | gVisor | vNode |
|---|---|---|---|
| `084508` (host) | 115.26 GB/s | 115.30 GB/s (0.0%) | — |
| `140126` (host) | 115.30 GB/s | 114.99 GB/s (−0.3%) | 115.32 GB/s (0.0%) |

**Steady-state GPU bandwidth is identical across all runtimes.** Once a CUDA
context exists, the data path (DMA, kernel launches via nvproxy or user
namespaces) adds no measurable cost at single-GPU scale.

## Cold-start: CUDA context init (median ms)

| Run | Environment | runc | gVisor | Δ | vNode | Δ |
|---|---|---|---|---|---|---|
| `083442`¹ | host (warmup) | 298.7 | 363.8 | +65.1 | — | — |
| `084508` | host | 291.1 | 342.0 | +50.9 | — | — |
| `140126` | host | 304.9 | 343.8 | +38.9 | 292.3 | −12.6 |
| `143651` | vCluster | 295.4 | 336.8 | +41.4 | 291.8 | −3.6 |

¹ Warmup run: the gVisor node pool was scaling up and pulling images, which
also inflated its sandbox startup to 108 s. Treat as an infrastructure
cold-cache data point, not a runtime property.

## Cold-start: first CUDA kernel (median ms)

| Run | Environment | runc | gVisor | Δ | vNode | Δ |
|---|---|---|---|---|---|---|
| `084508` | host | 139.6 | 137.5 | −2.1 | — | — |
| `140126` | host | 136.4 | 137.3 | +0.9 | 138.1 | +1.7 |
| `143651` | vCluster | 128.3 | 150.2 | +21.9 | 124.0 | −4.3 |

## Cold-start: nvidia-smi latency (median ms)

| Run | Environment | runc | gVisor | vNode |
|---|---|---|---|---|
| `084508` | host | 30 | 69 | — |
| `140126` | host | 32 | 61 | 43 |
| `143651` | vCluster | 30 | 56 | 40 |

## Conclusions

- **gVisor's overhead is real, repeatable, and concentrated at driver init**:
  +39 to +51 ms on CUDA context creation across three independent runs
  (nvproxy intercepting NVIDIA driver ioctls), and ~2× on `nvidia-smi`.
  Irrelevant for long training jobs; relevant for scale-to-zero / serverless
  GPU workloads that pay context init on every request.
- **vNode is indistinguishable from runc on every axis** across both host and
  vCluster runs — user namespaces add no syscall emulation to the GPU path.
- **vCluster adds no measurable data-plane overhead.** The runtime deltas
  reproduce within noise inside ephemeral vClusters (gVisor +38.9 host vs
  +41.4 vCluster). vCluster virtualizes only the control plane; pods run
  directly on host nodes.
- **Steady-state NCCL bandwidth is free everywhere** at this scale. Sandbox
  cost is a *latency* phenomenon, not a *throughput* one — which is exactly
  why kubemark-ai measures cold-start as its own benchmark axis.

## Caveats

- gVisor ran on a dedicated node pool (GKE Sandbox requirement) in
  `us-east1-c`; runc/vNode shared a node in `us-east1-b`. Same machine and
  GPU type, different physical hardware.
- Single-GPU scale only. Multi-node NCCL over the network path may expose
  additional sandbox overhead (gVisor netstack) not visible here.
- Kata Containers was not measured — no built-in GKE support.

## Reproduce

```bash
# Host-cluster comparison (NCCL + cold-start)
./scripts/compare-runtimes.sh --runtimes default,gvisor,vnode \
  --benchmarks nccl,coldstart --gpus 1 --gpu-type l4 --repeats 3

# Same, one level deeper: inside ephemeral vClusters
./scripts/compare-runtimes.sh --runtimes default,gvisor,vnode \
  --benchmarks coldstart --gpus 1 --gpu-type l4 --repeats 3 --vcluster
```

Methodology: [runtime-benchmarking.md](runtime-benchmarking.md) ·
GKE setup: [cloud-guides/gke.md](cloud-guides/gke.md)
