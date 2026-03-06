# kubemark-ai

**GPU Benchmarking on Kubernetes (and bare metal)** — Run NVIDIA DGX Cloud Benchmarks on any infrastructure with a single command.

kubemark-ai wraps [NVIDIA's dgxc-benchmarking](https://github.com/NVIDIA/dgxc-benchmarking) recipes (Slurm-only) and makes them run on:

- **Any Kubernetes cluster** with GPUs (DOKS, GKE, EKS, bare metal) via MPI Operator
- **Any GPU machine directly** via Docker/Podman (no K8s needed)
- **Isolated virtual clusters** via [vCluster](https://www.vcluster.com/) with zero performance overhead

**New?** See the [full Getting Started guide](docs/GETTING-STARTED.md) with step-by-step setup.

### Cloud Guides
[DigitalOcean](docs/cloud-guides/digitalocean.md) | [GKE](docs/cloud-guides/gke.md) | [EKS](docs/cloud-guides/eks.md) | [CoreWeave](docs/cloud-guides/coreweave.md) | [Lambda](docs/cloud-guides/lambda.md) | [Bare Metal](docs/cloud-guides/bare-metal.md)

## Quick Start

### Option A: On a Kubernetes Cluster

```bash
# 1. Install prerequisites (MPI Operator, check GPU nodes)
./scripts/install-deps.sh

# 2. Run NCCL AllReduce benchmark (works with any GPU count)
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type l40s        # Single GPU (testing)
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100         # Single DGX node
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100        # Multi-node

# 3. Run the full NCCL suite (5 tests)
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --tests full-suite

# 4. View results in the dashboard
```

### Option B: Directly on a GPU Machine (No K8s Needed)

```bash
# Just needs Docker + NVIDIA Container Toolkit
# Auto-detects all GPUs on the machine
./scripts/run-direct.sh nccl

# Use specific number of GPUs
./scripts/run-direct.sh nccl --gpus 4

# Full suite
./scripts/run-direct.sh nccl --tests full-suite

# View results in the dashboard
python3 dashboard/server.py
# Open http://localhost:8080
```

## What's Supported

### NCCL Communication Benchmarks

| Test | Binary | Description |
|------|--------|-------------|
| AllReduce | `all_reduce_perf` | Sum reduction across all GPUs |
| AllGather | `all_gather_perf` | Gather data from all GPUs |
| ReduceScatter | `reduce_scatter_perf` | Reduce + scatter across GPUs |
| AllToAll | `alltoall_perf` | All-to-all exchange |
| SendRecv | `sendrecv_perf` | Point-to-point send/receive |

### Training Benchmarks

| Model | Parameters | Framework | Min GPUs |
|-------|-----------|-----------|----------|
| GPT-2 | 124M (dense) | PyTorch (no credentials needed) | 1 |
| Qwen3-30B-A3B | 30B (3B active MoE) | NeMo/Megatron-Bridge | 16 |
| Llama 3.1-8B | 8B (dense) | NeMo/Megatron-Bridge | 8 |

### GPU Types

Works with any NVIDIA GPU. The `--gpu-type` flag sets GPUs-per-node automatically:

| GPU Type Flag | GPUs/Node | Examples |
|---------------|-----------|---------|
| `h100` | 8 | DGX H100, cloud 8xH100 |
| `a100` | 8 | DGX A100, cloud 8xA100 |
| `b200`, `gb200`, `gb300` | 4-8 | DGX Blackwell |
| `l40s`, `l40`, `rtx6000` | 1 | DigitalOcean, cloud single-GPU |
| `rtx4000`, `rtx4090` | 1 | DigitalOcean, consumer |
| `a10g`, `l4`, `t4` | 1 | AWS, GKE |
| `h100x4`, `a100x4` | 4 | Multi-GPU cloud instances |
| Any other | 1 (override with `--nodes`) | Custom setups |

> **Note on GPU architecture support:** The original [NVIDIA DGX Cloud Benchmarking](https://github.com/NVIDIA/dgxc-benchmarking) recipes are designed and validated for **H100, B200, GB200, and GB300** — these are the only architectures in NVIDIA's benchmark suite with published reference configurations. kubemark-ai extends this to work on any NVIDIA GPU by using the same underlying NCCL binaries and a portable GPT-2 training benchmark that runs on any GPU with sufficient VRAM. For the large model training recipes (Qwen3-30B, Llama 3.1-8B), the parallelism configurations (TP, PP, EP) are taken directly from NVIDIA's recipes and are only validated for the GPU types listed in [NVIDIA's repo](https://github.com/NVIDIA/dgxc-benchmarking#available-benchmarks).

## Architecture

```
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100
                |
                v
     ┌──────────────────────┐
     │  Create K8s Namespace │──(or vCluster with --vcluster)
     └──────────┬───────────┘
                │
     ┌──────────v───────────┐
     │  Apply MPIJob via    │
     │  envsubst templating │
     └──────────┬───────────┘
                │
     ┌──────────v───────────┐
     │  MPI Operator runs:  │
     │  - 1 Launcher Pod    │
     │  - N Worker Pods     │
     │    (each with GPUs)  │
     └──────────┬───────────┘
                │
     ┌──────────v───────────┐
     │  Collect logs, parse │
     │  results to JSON     │
     └──────────┬───────────┘
                │
     ┌──────────v───────────┐
     │  Dashboard: Chart.js │
     │  http://localhost:8080│
     └──────────────────────┘
```

## Prerequisites

- Kubernetes cluster with GPU nodes (`nvidia.com/gpu` resource)
- `kubectl`, `helm`, `jq`, `envsubst`
- [MPI Operator](https://github.com/kubeflow/mpi-operator) (installed by `install-deps.sh`)
- [vCluster CLI](https://www.vcluster.com/docs/get-started) (optional, for `--vcluster` mode)

For training benchmarks, you also need:
- NGC API key (for `nvcr.io` container images)
- HuggingFace token (for gated model weights)

## Usage

### NCCL Benchmarks

```bash
# Single test
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --tests all-reduce

# Multiple tests
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100 --tests all-reduce,all-gather,reduce-scatter

# Full suite (all 5 NCCL tests)
./scripts/run-benchmark.sh nccl --gpus 64 --gpu-type h100 --tests full-suite

# With vCluster isolation
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100 --vcluster

# Keep resources after completion (for debugging)
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --no-cleanup
```

### Training Benchmarks

```bash
# Qwen3-30B (MoE model, minimum 16 GPUs for H100)
./scripts/run-benchmark.sh training --model qwen3-30b --gpus 64 --gpu-type h100

# Llama 3.1-8B (dense model, works on a single node)
./scripts/run-benchmark.sh training --model llama31-8b --gpus 8 --gpu-type h100

# With FP8 precision
./scripts/run-benchmark.sh training --model llama31-8b --gpus 8 --gpu-type h100 --dtype fp8
```

Before running training benchmarks, create the required secrets:
```bash
# NGC registry auth (for pulling NeMo container)
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  -n <namespace>

# HuggingFace token (for downloading model weights)
kubectl create secret generic hf-token \
  --from-literal=token="$HF_TOKEN" \
  -n <namespace>
```

### Helm Chart

```bash
# NCCL benchmark
helm install my-bench ./k8s/helm-chart \
  --set gpu.type=h100 \
  --set gpu.totalGPUs=16

# Training benchmark
helm install my-bench ./k8s/helm-chart \
  --set benchmark.type=training \
  --set benchmark.model=qwen3-30b \
  --set gpu.type=h100 \
  --set gpu.totalGPUs=64

# With custom NCCL settings
helm install my-bench ./k8s/helm-chart \
  --set gpu.type=h100 \
  --set gpu.totalGPUs=64 \
  --set nccl.debug=INFO \
  --set nccl.ibDisable=0
```

### Dashboard

```bash
python3 dashboard/server.py
# Opens at http://localhost:8080
# Loads results from results/*/summary.json
```

## Results

Results are saved to `results/<run-id>/summary.json`:

```json
{
  "run_id": "20260305-143022-a1b2c3",
  "timestamp": "2026-03-05T14:30:22Z",
  "benchmark_type": "nccl",
  "gpu_type": "h100",
  "total_gpus": 16,
  "num_nodes": 2,
  "results": [{
    "test": "all_reduce",
    "avg_bus_bandwidth_gbps": 159.15,
    "data_points": [
      {"size_bytes": 1024, "busbw_gbps": 0.12, "time_us": 23.5},
      ...
    ]
  }]
}
```

## Why vCluster?

[vCluster](https://www.vcluster.com/) virtualizes the Kubernetes control plane, not the data plane. This means:

- **Zero GPU overhead** — pods run directly on host nodes with full hardware access
- **NCCL/InfiniBand works** — same network path as bare metal
- **Clean isolation** — each benchmark run gets its own virtual cluster
- **Auto-cleanup** — tear down the vCluster when done, nothing left behind

This makes it ideal for multi-tenant GPU benchmarking: cloud providers can validate infrastructure performance for each customer in isolation, with identical results to bare-metal runs.

## Project Structure

```
kubemark-ai/
├── scripts/
│   ├── run-benchmark.sh      # Main orchestrator (Kubernetes)
│   ├── run-direct.sh         # Direct GPU runner (Docker, no K8s)
│   ├── install-deps.sh       # Prerequisites installer
│   └── parse-results.sh      # Results parser (logs → JSON)
├── k8s/
│   ├── jobs/                  # MPIJob manifests (envsubst-templated)
│   │   ├── nccl-allreduce.yaml
│   │   ├── nccl-allgather.yaml
│   │   ├── nccl-reducescatter.yaml
│   │   ├── nccl-full-suite.yaml
│   │   ├── training-gpt2-bench.yaml
│   │   ├── training-qwen3-30b.yaml
│   │   └── training-llama31-8b.yaml
│   ├── base/                  # Secret templates
│   └── helm-chart/            # Helm chart
├── vcluster/
│   └── values.yaml            # vCluster config for GPU benchmarking
├── dashboard/                 # Static web dashboard (Chart.js)
├── results/                   # Benchmark results (git-ignored)
├── docs/                      # Cloud-specific guides
└── blog/                      # Blog post
```

## Sources and Acknowledgments

- [NVIDIA DGX Cloud Benchmarking](https://github.com/NVIDIA/dgxc-benchmarking) — the upstream benchmark recipes (H100, B200, GB200, GB300)
- [NVIDIA DGX Cloud Benchmarking Portal](https://developer.nvidia.com/dgx-cloud/benchmarking) — NVIDIA's official benchmarking program for Exemplar Clouds certification
- [CoreWeave nccl-tests](https://github.com/coreweave/nccl-tests) — NCCL test container images for Kubernetes
- [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) — MPIJob CRD for multi-node GPU jobs
- [vCluster](https://www.vcluster.com/) — virtual clusters for benchmark isolation

**Hardware specs used in the dashboard** are sourced from NVIDIA's published reference architectures in the [dgxc-benchmarking README](https://github.com/NVIDIA/dgxc-benchmarking#reference-infrastructure):
- H100: NVLink 4.0 (900 GB/s per GPU), 3.2 TB/s memory bandwidth
- B200: NVLink 5.0 (1.8 TB/s per GPU), 8 TB/s memory bandwidth
- GB200: NVLink 5.0 (1.8 TB/s per GPU), 8 TB/s memory bandwidth (16 TB/s total)
- GB300: NVLink 5.0 (1.8 TB/s per GPU), 12 TB/s memory bandwidth

## License

Apache License 2.0 — see [LICENSE](LICENSE)
