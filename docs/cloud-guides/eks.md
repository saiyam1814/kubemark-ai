# Running kubemark-ai on Amazon EKS

EKS supports GPU nodes via EC2 GPU instances with the NVIDIA device plugin.

## GPU Options on EKS

| Instance | GPU | $/hr (on-demand) | Notes |
|----------|-----|------------------|-------|
| `g4dn.xlarge` | 1x T4 (16GB) | $0.526 | Cheapest GPU |
| `g5.xlarge` | 1x A10G (24GB) | $1.006 | Good balance |
| `g5.12xlarge` | 4x A10G | $5.672 | Multi-GPU |
| `p4d.24xlarge` | 8x A100 40GB | $32.77 | Serious benchmarks |
| `p5.48xlarge` | 8x H100 80GB | $98.32 | InfiniBand, full scale |

Spot instances can save 60-70%.

## Step-by-Step

### 1. Create EKS Cluster

```bash
# Using eksctl (easiest)
eksctl create cluster \
  --name kubemark-test \
  --region us-east-1 \
  --node-type m5.large \
  --nodes 1
```

### 2. Add GPU Node Group

```bash
# Option A: Cheap test (1x A10G, ~$1/hr)
eksctl create nodegroup \
  --cluster kubemark-test \
  --name gpu-nodes \
  --node-type g5.xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 2

# Option B: Multi-GPU (8x A100)
eksctl create nodegroup \
  --cluster kubemark-test \
  --name gpu-nodes \
  --node-type p4d.24xlarge \
  --nodes 1
```

### 3. Install NVIDIA Device Plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

Or via Helm:
```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm install nvidia-device-plugin nvdp/nvidia-device-plugin --namespace kube-system
```

### 4. Verify GPUs

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu'
```

### 5. Run kubemark-ai

```bash
./scripts/install-deps.sh

# Single A10G
./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type a10g

# 8x A100
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type a100

# Multi-node
./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type a100
```

### 6. EFA for Multi-Node (P4d/P5 instances)

P4d and P5 instances have Elastic Fabric Adapter (EFA) for high-speed networking:

```bash
# Install EFA device plugin
kubectl apply -f https://raw.githubusercontent.com/aws/eks-charts/master/stable/aws-efa-k8s-device-plugin/templates/daemonset.yaml

# Run with EFA-optimized NCCL
./scripts/run-benchmark.sh nccl \
  --gpus 16 \
  --gpu-type h100 \
  --nccl-ifname efa0
```

### 7. Clean Up

```bash
# Scale down GPU nodes
eksctl scale nodegroup --cluster kubemark-test --name gpu-nodes --nodes 0

# Or delete
eksctl delete cluster --name kubemark-test
```

## Cost Savings Tips

- Use **Spot Instances**: Add `--spot` to eksctl nodegroup creation (~60-70% cheaper)
- Use **Karpenter**: Auto-provisions GPU nodes on demand, terminates when idle
- Scale node group to 0 when not benchmarking
