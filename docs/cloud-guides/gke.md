# Running kubemark-ai on Google Kubernetes Engine (GKE)

GKE has native GPU support with auto-provisioning of NVIDIA drivers.

## GPU Options on GKE

| GPU | Machine Type | $/hr (on-demand) | Notes |
|-----|-------------|------------------|-------|
| T4 (16GB) | `n1-standard-8` + 1x T4 | ~$0.35/GPU | Cheapest, good for testing |
| L4 (24GB) | `g2-standard-8` | ~$0.70 | Good balance |
| A100 40GB | `a2-highgpu-1g` | ~$3.67 | Serious benchmarks |
| A100 80GB | `a2-ultragpu-1g` | ~$5.07 | Large models |
| H100 80GB | `a3-highgpu-8g` | ~$8.80/GPU | 8 GPUs per node, InfiniBand |

## Step-by-Step

### 1. Create GKE Cluster

```bash
# Create cluster with a small default pool
gcloud container clusters create kubemark-test \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-standard-4

# Add GPU node pool (1x L4 — cheapest practical option)
gcloud container node-pools create gpu-pool \
  --cluster kubemark-test \
  --zone us-central1-a \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 \
  --num-nodes 1 \
  --spot  # Use spot for cheaper testing
```

GKE automatically installs NVIDIA drivers on GPU nodes.

### 2. Install device plugin (GKE does this automatically)

```bash
# Verify GPUs are visible
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu'
```

### 3. Run kubemark-ai

```bash
./scripts/install-deps.sh

# Single L4 GPU
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type l4

# For A100 8-GPU node
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type a100

# For H100 8-GPU node
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100
```

### 4. Multi-node (H100 with GPUDirect-TCPXO)

```bash
# H100 nodes use GPUDirect, need specific NCCL settings
./scripts/run-benchmark.sh nccl \
  --gpus 16 \
  --gpu-type h100 \
  --nccl-ifname eth0
```

### 5. Clean Up

```bash
# Delete GPU node pool (stop billing)
gcloud container node-pools delete gpu-pool \
  --cluster kubemark-test --zone us-central1-a

# Or delete entire cluster
gcloud container clusters delete kubemark-test --zone us-central1-a
```

## Cost Estimate

| Test | Config | Duration | Cost |
|------|--------|----------|------|
| Quick validation | 1x T4 (spot) | ~5 min | ~$0.03 |
| NCCL single node | 1x L4 | ~5 min | ~$0.06 |
| NCCL 8-GPU | 8x A100 | ~5 min | ~$2.50 |
| Full suite H100 | 8x H100 | ~15 min | ~$22 |
