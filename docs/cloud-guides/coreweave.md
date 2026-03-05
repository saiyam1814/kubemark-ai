# Running kubemark-ai on CoreWeave

CoreWeave is a GPU-native cloud provider with excellent InfiniBand support and pre-configured H100/GB200 clusters.

## Prerequisites

- CoreWeave Kubernetes access (`kubectl` configured)
- GPU nodes provisioned (e.g., `gd-8xh100ib-i128`)

## Node Selectors

CoreWeave uses specific node labels for GPU types:

```bash
# H100 SXM 80GB with InfiniBand
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100
```

Or via Helm with node selector:
```bash
helm install my-bench ./k8s/helm-chart \
  --set gpu.type=h100 \
  --set gpu.totalGPUs=64 \
  --set nodeSelector."node\.coreweave\.cloud/type"=gd-8xh100ib-i128
```

## NCCL Networking

CoreWeave H100 nodes have InfiniBand. For optimal NCCL performance:

```bash
./scripts/run-benchmark.sh nccl \
  --gpus 64 \
  --gpu-type h100 \
  --nccl-ifname eth0
```

The CoreWeave NCCL test images (`ghcr.io/coreweave/nccl-tests`) are already optimized for their network stack.

## Expected Performance (Reference)

Based on CoreWeave's published benchmarks:

| Test | GPUs | Bus Bandwidth |
|------|------|---------------|
| AllReduce | 16 (2 nodes) | ~280 GB/s peak |
| AllReduce | 64 (8 nodes) | ~250 GB/s peak |
| AllReduce | 128 (16 nodes) | ~240 GB/s peak |

Your results should be within 5-10% of these numbers on similar hardware.
