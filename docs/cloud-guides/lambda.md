# Running kubemark-ai on Lambda Cloud

Lambda provides on-demand GPU clusters with H100 and A100 instances.

## Prerequisites

- Lambda Cloud Kubernetes cluster or GPU instances
- `kubectl` configured for your cluster

## Quick Start

```bash
# Install prerequisites
./scripts/install-deps.sh

# Run NCCL benchmark
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100
```

## Networking Notes

Lambda instances typically use RoCE (RDMA over Converged Ethernet) rather than InfiniBand. For NCCL configuration:

```bash
# If IB is not available, disable it
./scripts/run-benchmark.sh nccl \
  --gpus 16 \
  --gpu-type h100 \
  --disable-ib
```

## Single-Node Testing

Lambda's 8x H100 instances are great for single-node benchmarks:

```bash
# NCCL on single node (intra-node NVLink bandwidth)
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --nodes 1

# Llama 3.1-8B training (fits on single node)
./scripts/run-benchmark.sh training --model llama31-8b --gpus 8 --gpu-type h100
```
