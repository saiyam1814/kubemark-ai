# Running kubemark-ai on Google Kubernetes Engine (GKE)

Tested with NVIDIA L4 GPUs on GKE. GKE auto-installs NVIDIA drivers on GPU nodes.

## GPU Options on GKE

| GPU | Machine Type | $/hr (on-demand) | Notes |
|-----|-------------|------------------|-------|
| T4 (16GB) | `n1-standard-8` + 1x T4 | ~$0.35/GPU | Cheapest, good for testing |
| L4 (24GB) | `g2-standard-8` | ~$0.70 | Good balance — tested with kubemark-ai |
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

# Add GPU node pool (L4 — tested and confirmed working)
gcloud container node-pools create gpu-pool \
  --cluster kubemark-test \
  --zone us-central1-a \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 \
  --num-nodes 1 \
  --spot  # Use spot for cheaper testing

# For multi-node tests, use --num-nodes 2
gcloud container node-pools create gpu-pool \
  --cluster kubemark-test \
  --zone us-central1-a \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 \
  --num-nodes 2 \
  --spot
```

GKE automatically installs NVIDIA drivers on GPU nodes.

### 2. Verify GPUs

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu'
```

### 3. Install kubemark-ai and run benchmarks

```bash
git clone https://github.com/saiyam1814/kubemark-ai.git && cd kubemark-ai
./scripts/install-deps.sh

# Single L4 GPU — NCCL (measures internal memory bandwidth)
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type l4

# Two L4s across 2 nodes — NCCL (measures inter-node network bandwidth)
./scripts/run-benchmark.sh nccl --gpus 2 --gpu-type l4 --nodes 2

# GPT-2 training benchmark (no credentials needed)
./scripts/run-benchmark.sh training --model gpt2-bench --gpus 1 --gpu-type l4
```

### 4. View Results

```bash
python3 dashboard/server.py
# Open http://localhost:8080
```

## Tested Results (L4 on GKE)

These are actual results from running kubemark-ai on GKE with L4 GPUs:

### NCCL AllReduce — Single L4

```
#       size      time   algbw   busbw
#        (B)      (us)  (GB/s)  (GB/s)
     1048576      9.05  115.81  115.81
    16777216    137.70  121.81  121.81
   268435456   2328.4   115.29  115.29
  1073741824   9304.0   115.41  115.41
  4294967296  37230.0   115.36  115.36
```

**115.35 GB/s** — the L4 has 300 GB/s theoretical GDDR6 bandwidth. ~115 GB/s through NCCL confirms healthy GPU hardware.

### NCCL AllReduce — 2 L4s Across 2 Nodes

```
#       size      time   algbw   busbw
#        (B)      (us)  (GB/s)  (GB/s)
     1048576   1135.8     0.92    0.92
    16777216  13184.0     1.27    1.27
    67108864  41044.0     1.64    1.64
   536870912 339623.0     1.58    1.58
  1073741824 696563.0     1.54    1.54
```

**1.40 GB/s** — this is GKE standard Ethernet (~16 Gbps). The benchmark tells you exactly what your network delivers.

### GPT-2 Training — Single L4

```
Training Benchmark Summary
  Avg step time:   511.27 ms (std: 1.61)
  Avg TFLOPS/GPU:  7.82
  Avg tokens/sec:  8011
  Peak GPU memory: 9.24 GB
```

**7.82 TFLOPS/GPU** with 0.3% step time std deviation — extremely stable.

## Clean Up

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
| Quick validation | 1x L4 (spot) | ~5 min | ~$0.03 |
| NCCL + training | 1x L4 | ~10 min | ~$0.12 |
| Multi-node NCCL | 2x L4 | ~10 min | ~$0.24 |
| NCCL 8-GPU | 8x A100 | ~5 min | ~$2.50 |
| Full suite H100 | 8x H100 | ~15 min | ~$22 |
