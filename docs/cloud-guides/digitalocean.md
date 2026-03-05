# Running kubemark-ai on DigitalOcean (DOKS)

DigitalOcean Kubernetes (DOKS) supports GPU node pools natively. Drivers are auto-installed. This is one of the easiest ways to test GPU benchmarks.

## GPU Options on DigitalOcean

| GPU | VRAM | $/hr | Best For |
|-----|------|------|----------|
| RTX 4000 Ada | 20 GB | $0.76 | Testing the pipeline |
| L40S | 48 GB | $1.57 | Inference benchmarks |
| RTX 6000 Ada | 48 GB | $1.57 | Good balance |
| H100 SXM (1x) | 80 GB | $3.39 | Real NCCL benchmarks |
| H100 SXM (8x) | 640 GB | ~$27/hr | Multi-GPU training |

Billing is **per-second** with a 5-minute minimum. GPU node pools can **scale to zero**.

## Step-by-Step Setup

### 1. Install doctl

```bash
# macOS
brew install doctl

# Authenticate
doctl auth init
```

### 2. Create DOKS Cluster

```bash
# Create cluster with a small non-GPU node for the control plane workloads
doctl kubernetes cluster create kubemark-test \
  --region nyc1 \
  --node-pool "name=default;size=s-2vcpu-4gb;count=1"
```

### 3. Add GPU Node Pool

```bash
# Option A: Cheap test (1x RTX 4000 Ada, $0.76/hr)
doctl kubernetes cluster node-pool create kubemark-test \
  --name gpu-pool \
  --size gpu-4000adax1-20gb \
  --count 1

# Option B: Real benchmark (1x H100, $3.39/hr)
doctl kubernetes cluster node-pool create kubemark-test \
  --name gpu-pool \
  --size gpu-h100x1-80gb \
  --count 1

# Option C: Multi-GPU (8x H100, ~$27/hr)
doctl kubernetes cluster node-pool create kubemark-test \
  --name gpu-pool \
  --size gpu-h100x8-640gb \
  --count 1
```

### 4. Get Kubeconfig

```bash
doctl kubernetes cluster kubeconfig save kubemark-test
```

### 5. Install NVIDIA Device Plugin

DigitalOcean installs GPU drivers automatically, but you need the K8s device plugin:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --set gfd.enabled=true
```

Verify GPUs are visible:
```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu'
```

### 6. Run kubemark-ai

```bash
# Clone and install prerequisites
git clone <repo> && cd kubemark-ai
./scripts/install-deps.sh

# Run NCCL benchmark
# For 1x H100:
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type h100 --nodes 1

# For 8x H100:
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100

# For 1x L40S:
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type l40s --nodes 1

# For 1x RTX 4000 Ada:
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type rtx4000 --nodes 1
```

### 7. View Results

```bash
python3 dashboard/server.py
# Open http://localhost:8080
```

### 8. Clean Up (Stop Billing!)

```bash
# Scale GPU pool to zero
doctl kubernetes cluster node-pool update kubemark-test gpu-pool --count 0

# Or delete the whole cluster
doctl kubernetes cluster delete kubemark-test
```

## Multi-Node Benchmarks on DigitalOcean

For inter-node NCCL benchmarks (where you really see the networking):

```bash
# 2x H100 nodes (2 GPUs total, $6.78/hr)
doctl kubernetes cluster node-pool update kubemark-test gpu-pool \
  --size gpu-h100x1-80gb --count 2

./scripts/run-benchmark.sh nccl --gpus 2 --gpu-type h100 --nodes 2
```

Note: Inter-node bandwidth on DigitalOcean uses 400GbE networking (for H100 8x nodes).
Single-GPU instances use standard networking which will show lower inter-node NCCL bandwidth.

## Cost Estimate for Testing

| Test | Config | Duration | Cost |
|------|--------|----------|------|
| Quick validation | 1x RTX 4000 Ada | ~5 min | ~$0.06 |
| NCCL single node | 1x H100 | ~5 min | ~$0.28 |
| NCCL full suite | 1x H100 | ~15 min | ~$0.85 |
| Multi-node NCCL | 2x H100 | ~10 min | ~$1.13 |
| Full suite multi-node | 2x H100 8x | ~20 min | ~$18 |
