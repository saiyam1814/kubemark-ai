# Getting Started with kubemark-ai

Two ways to run GPU benchmarks:

| Mode | What you need | Best for |
|------|--------------|----------|
| **Direct** (`run-direct.sh`) | Docker + NVIDIA Container Toolkit + GPU machine | Quick test, no K8s needed |
| **Kubernetes** (`run-benchmark.sh`) | K8s cluster with GPU nodes | Production benchmarking, multi-node, vCluster |

---

## Path 1: Direct Mode (No Kubernetes)

The fastest way to test. SSH into any GPU machine (cloud VM, bare metal, your workstation):

```bash
# Requirements: Docker (or Podman) + NVIDIA Container Toolkit
# That's it. No K8s, no helm, no kubectl.

git clone https://github.com/saiyam1814/kubemark-ai.git && cd kubemark-ai

# Auto-detects all GPUs, runs NCCL AllReduce
./scripts/run-direct.sh nccl

# Specific GPU count
./scripts/run-direct.sh nccl --gpus 4

# Full NCCL suite (5 tests)
./scripts/run-direct.sh nccl --tests full-suite

# View results
python3 dashboard/server.py
```

Works on: any cloud VM with GPU (DigitalOcean GPU Droplet, AWS EC2 GPU, GCP GPU VM, Lambda, RunPod, etc.)

---

## Path 2: Kubernetes Mode

For multi-node benchmarks, vCluster isolation, and production-grade testing.

### What You Need

| Requirement | For NCCL benchmarks | For Training benchmarks |
|-------------|---------------------|------------------------|
| K8s cluster with GPU nodes | Yes | Yes |
| kubectl configured | Yes | Yes |
| helm v3+ | Yes | Yes |
| jq | Yes | Yes |
| NVIDIA device plugin on cluster | Yes | Yes |
| NGC API key | No | Yes |
| HuggingFace token | No | Yes |
| vcluster CLI | Only if using --vcluster | Only if using --vcluster |

## How It Works (End to End)

```
You run:  ./scripts/run-benchmark.sh nccl --gpus 2 --gpu-type l40s
                        │
   1. Script creates     │    kubectl create namespace bench-20260305-xxxx
      a K8s namespace    │
                        │
   2. Script templates   │    envsubst fills in GPU count, node count, etc.
      the MPIJob YAML    │    into k8s/jobs/nccl-allreduce.yaml
                        │
   3. Script applies     │    kubectl apply -f (the templated manifest)
      the MPIJob         │    MPI Operator creates launcher + worker pods
                        │
   4. Workers run NCCL   │    all_reduce_perf runs across all GPUs
      benchmarks         │    outputs bandwidth/latency per message size
                        │
   5. Script collects    │    kubectl logs from the launcher pod
      logs               │    saved to results/<run-id>/
                        │
   6. Parser extracts    │    scripts/parse-results.sh parses the NCCL
      structured data    │    output into results/<run-id>/summary.json
                        │
   7. Cleanup            │    kubectl delete namespace bench-20260305-xxxx
                        │
   8. Dashboard reads    │    python3 dashboard/server.py
      the JSON files     │    loads results/*/summary.json → Chart.js graphs
```

## Step 1: Get a K8s Cluster with GPUs

### Option A: GKE (Tested — recommended for getting started)

See [docs/cloud-guides/gke.md](cloud-guides/gke.md) for full walkthrough with tested results.

```bash
# Create GKE cluster
gcloud container clusters create kubemark-test \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-standard-4

# Add GPU node pool (L4 — tested and confirmed working)
gcloud container node-pools create gpu-pool \
  --cluster kubemark-test \
  --zone us-central1-a \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 \
  --num-nodes 1 \
  --spot
```

GKE automatically installs NVIDIA drivers on GPU nodes.

### Option B: Any existing cluster with GPUs

If you already have a K8s cluster with NVIDIA GPUs:

```bash
# Verify GPU nodes exist
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu'

# Should show something like:
# NAME              GPUS
# gpu-node-1        8
# gpu-node-2        8
```

If you see `<none>` for GPUs, you need:
1. NVIDIA GPU drivers on nodes
2. [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) or the device plugin
3. Container runtime configured for GPU access

## Step 2: Install Prerequisites

```bash
git clone https://github.com/saiyam1814/kubemark-ai.git && cd kubemark-ai

# This checks all tools and installs MPI Operator
./scripts/install-deps.sh
```

What it does:
- Validates: kubectl, helm, jq, envsubst
- Installs [MPI Operator](https://github.com/kubeflow/mpi-operator) via Helm
- Checks for GPU nodes in your cluster

## Step 3: Run Your First Benchmark

### Single-GPU test (cheapest, good for validation)

```bash
# If you have 1 GPU node with 1 GPU (like DO L40S or RTX 4000)
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type l40s --nodes 1
```

Note: NCCL single-GPU tests measure internal GPU bandwidth only.
Multi-GPU tests are where you see real inter-GPU/inter-node bandwidth.

### Multi-GPU single node

```bash
# 8x H100 on one node (DGX or cloud 8-GPU instance)
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100

# 4x GPUs on one node
./scripts/run-benchmark.sh nccl --gpus 4 --gpu-type h100x4
```

### Multi-node (where it gets interesting)

```bash
# 16 GPUs across 2 nodes (8 GPUs each)
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100

# 2 single-GPU nodes
./scripts/run-benchmark.sh nccl --gpus 2 --gpu-type l40s --nodes 2
```

### Full NCCL suite (all 5 tests)

```bash
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100 --tests full-suite
```

### With vCluster isolation

```bash
# Requires vcluster CLI installed
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --vcluster
```

## Step 4: View Results

### Command line

Results are saved automatically:
```bash
# After a benchmark run completes:
cat results/<run-id>/summary.json | jq '.'
```

### Dashboard

```bash
python3 dashboard/server.py
# Open http://localhost:8080
```

The dashboard:
- Loads ALL results from `results/*/summary.json`
- Shows NCCL bandwidth/latency charts
- Shows training TFLOPS charts
- Compare tab lets you overlay two runs
- Has built-in sample data for demo (if no real results exist yet)

### How dashboard gets data

```
results/
├── 20260305-143022-abc123/
│   ├── nccl-allreduce-launcher-0.log    ← raw launcher pod logs
│   └── summary.json                      ← parsed JSON (dashboard reads this)
├── 20260305-160045-def456/
│   ├── ...
│   └── summary.json
```

The `server.py` serves a `/api/results` endpoint that reads all `summary.json` files.
The `app.js` fetches from that endpoint and renders Chart.js graphs.
If `/api/results` fails (e.g., opening `index.html` directly), it falls back to built-in sample data.

## Step 5: Training Benchmarks

### GPT-2 (Easiest — no credentials needed)

GPT-2 124M is the fastest way to validate your GPU's training performance. No NGC key or HuggingFace token required:

```bash
# Single GPU — works on any NVIDIA GPU with 8+ GB VRAM
./scripts/run-benchmark.sh training --model gpt2-bench --gpus 1 --gpu-type l4

# Also works on H100, A100, etc.
./scripts/run-benchmark.sh training --model gpt2-bench --gpus 1 --gpu-type h100
```

### Large Models (Advanced — needs credentials)

For production-grade benchmarks with larger models:

```bash
# 1. Create NGC secret (for pulling NeMo container image)
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="YOUR_NGC_API_KEY"

# 2. Create HuggingFace secret (for model weights)
kubectl create secret generic hf-token \
  --from-literal=token="YOUR_HF_TOKEN"

# 3. Run training benchmark
# Llama 3.1-8B fits on a single 8-GPU node
./scripts/run-benchmark.sh training --model llama31-8b --gpus 8 --gpu-type h100

# Qwen3-30B needs at least 16 GPUs on H100
./scripts/run-benchmark.sh training --model qwen3-30b --gpus 64 --gpu-type h100
```

## Understanding Your Results

Benchmark numbers can be confusing. Here is what each metric means in plain English and how to tell if your infrastructure is healthy.

### NCCL Benchmarks: What the Numbers Mean

NCCL benchmarks measure **how fast your GPUs can talk to each other**. This is critical for any multi-GPU AI training because GPUs need to constantly exchange data (gradients, activations) during training.

**Bus Bandwidth (GB/s)** — the headline number. This tells you the effective communication speed between your GPUs. Higher is better.

| What you're testing | What determines the speed | Typical numbers |
|---|---|---|
| Single GPU | GPU memory bandwidth (no communication) | 100-300 GB/s depending on GPU |
| Multiple GPUs, same node | NVLink / NVSwitch / PCIe | 20-300 GB/s depending on interconnect |
| Multiple GPUs, across nodes | Network (InfiniBand, RoCE, Ethernet) | 1-50 GB/s depending on network |

**How to read the results:**

- **Single-GPU NCCL** is a loopback test. It measures your GPU's internal memory bandwidth, not inter-GPU communication. Use this to verify your GPU hardware is working, but do not compare it to multi-GPU numbers.
- **Multi-GPU on the same node**: This tests NVLink/NVSwitch bandwidth. On H100 DGX (NVSwitch), expect ~400 GB/s. On PCIe-only setups, expect ~20-30 GB/s.
- **Multi-GPU across nodes**: This tests your network. On InfiniBand (400Gb/s), expect ~45-50 GB/s. On 25GbE, expect ~2-3 GB/s. On 10GbE, expect ~1.2 GB/s.

**What "good" looks like:**

| Setup | Expected AllReduce Bus BW | If you see much less |
|---|---|---|
| 8x H100 (NVSwitch) | 380-420 GB/s | Check NVLink topology: `nvidia-smi topo -m` |
| 2x nodes, InfiniBand 400G | 40-50 GB/s | Check IB: `ibstat`, verify GPUDirect RDMA |
| 2x nodes, 25GbE | 2-3 GB/s | Normal for Ethernet, consider upgrading |
| 2x nodes, 10GbE | 1-1.5 GB/s | Expected — Ethernet is the bottleneck |
| 1 GPU (loopback) | 80-300 GB/s | Check GPU driver, memory clock |

**Message size matters:** Small messages (< 1 MB) have high latency overhead and low bandwidth. Large messages (1 GB+) show the true sustained bandwidth of your interconnect. When comparing results, look at the large-message bandwidth numbers.

### Training Benchmarks: What the Numbers Mean

Training benchmarks measure **how fast your GPU can do actual AI math**. We run a short model training (GPT-2 124M or larger models) and measure throughput.

**TFLOPS/GPU** — teraflops per second per GPU. This is the core measure of compute utilization.

| GPU | Peak FP16/BF16 TFLOPS | Typical training efficiency |
|---|---|---|
| H100 SXM | 989 | 40-60% (400-600 TFLOPS) for large models |
| A100 SXM | 312 | 40-55% (125-170 TFLOPS) |
| L4 | 121 (FP16) | 5-10% for small models (6-12 TFLOPS) |
| L40S | 362 (FP16) | varies by model |

Lower TFLOPS efficiency for small models (like GPT-2 124M) is **completely normal**. Small models do not saturate the GPU's compute units. This benchmark still validates that your GPU is working — you should see consistent step times with low variance.

**Step Time (ms)** — how long each training iteration takes. Lower is better. Look at the standard deviation: it should be small (< 5% of mean). Large variance means something is causing inconsistent performance (thermal throttling, noisy neighbors, storage latency).

**Tokens/sec** — how many text tokens are processed per second. Higher is better. This is directly proportional to 1/step_time.

**What to look for:**

- **Consistent step times** (low std deviation) = healthy GPU, no thermal throttling
- **TFLOPS matches expected range** for your GPU = compute is working correctly
- **If step times degrade over the run** = possible thermal throttling or memory issues

### vCluster Overhead: How to Verify Zero Impact

If you run the same benchmark both directly (namespace) and via vCluster, the results should be **within 2-3%** of each other. vCluster only virtualizes the Kubernetes control plane — your GPU pods still run on the same physical hardware with the same drivers, the same NCCL libraries, and the same network path.

If you see more than 5% difference, something else changed between runs (thermal state, competing workloads, etc.) — it is not vCluster overhead.

### Quick Checklist: Is My Infrastructure Healthy?

1. **Single-GPU NCCL**: Bus BW matches your GPU's memory bandwidth spec? GPU works.
2. **Multi-GPU same-node NCCL**: Bus BW matches your interconnect spec (NVLink/PCIe)? Interconnect works.
3. **Multi-GPU cross-node NCCL**: Bus BW matches your network spec (IB/Ethernet)? Network works.
4. **Training step time std < 5%**: No thermal throttling or noisy neighbors.
5. **Training TFLOPS in expected range**: Compute pipeline is healthy.

If any check fails, the benchmark has told you exactly which layer to investigate: GPU, interconnect, network, or software configuration.

---

## Troubleshooting

### "No GPU nodes found"

```bash
# Check if NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Check node resources
kubectl describe node <gpu-node> | grep -A5 "Allocatable"
```

### MPIJob stays pending

```bash
# Check MPI Operator is running
kubectl get pods -n mpi-operator

# Check events on the MPIJob
kubectl describe mpijob -n bench-<run-id>
```

### NCCL errors / timeout

```bash
# Run with debug logging
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --nccl-debug INFO

# On non-InfiniBand clusters, disable IB
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --disable-ib
```

### Don't clean up (for debugging)

```bash
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --no-cleanup

# Then manually inspect:
kubectl get pods -n bench-<run-id>
kubectl logs -n bench-<run-id> <launcher-pod>
```

## Supported GPU Types

| GPU Type Flag | GPUs/Node | Examples |
|---------------|-----------|---------|
| `h100` | 8 | DGX H100, cloud 8xH100 |
| `a100` | 8 | DGX A100, cloud 8xA100 |
| `b200` | 8 | DGX B200 |
| `gb200` | 4 | DGX GB200 NVL |
| `gb300` | 4 | DGX GB300 |
| `l40s` | 1 | Cloud single-GPU instances |
| `l40` | 1 | Cloud single-GPU |
| `l4` | 1 | GKE, cloud single-GPU |
| `a10g` | 1 | AWS g5 instances |
| `rtx4000` | 1 | Cloud or workstation GPUs |
| `rtx6000` | 1 | Cloud or workstation GPUs |
| `rtx4090` | 1 | Consumer/cloud |
| `t4` | 1 | GKE, AWS g4dn |
| `v100` | 1 | Legacy |
| `h100x4` | 4 | 4-GPU cloud instances |
| `a100x4` | 4 | 4-GPU cloud instances |

For unlisted GPU types, use any name — it defaults to 1 GPU/node.
Override with `--nodes` if your setup is different.
