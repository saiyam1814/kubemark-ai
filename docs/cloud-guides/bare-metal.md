# Running kubemark-ai on Bare Metal Kubernetes

For on-premise DGX systems or bare metal K8s clusters with NVIDIA GPUs.

## Prerequisites

- Kubernetes cluster (kubeadm, RKE2, etc.)
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) installed
- [NVIDIA Network Operator](https://github.com/Mellanox/network-operator) (for InfiniBand)
- Nodes with `nvidia.com/gpu` resources visible

## GPU Operator Setup

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true
```

## InfiniBand Configuration

For multi-node NCCL benchmarks with InfiniBand:

1. Install the Network Operator for RDMA support
2. Verify `rdma/ib` resources appear on nodes
3. Run with IB-optimized NCCL settings:

```bash
./scripts/run-benchmark.sh nccl \
  --gpus 64 \
  --gpu-type h100 \
  --nccl-ifname ib0
```

## DGX Systems

DGX H100 and DGX B200 systems come pre-configured with 8 GPUs per node:

```bash
# Full DGX node benchmark
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100

# Multi-DGX benchmark (4 DGX H100 systems)
./scripts/run-benchmark.sh nccl --gpus 32 --gpu-type h100 --tests full-suite
```

## Troubleshooting

**GPUs not visible:**
```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu'
```

**NCCL timeout errors:**
- Increase shared memory: edit the `/dev/shm` `sizeLimit` in job manifests
- Check firewall rules between nodes (NCCL uses high ports)
- Set `NCCL_DEBUG=INFO` for detailed logging: `--nccl-debug INFO`
