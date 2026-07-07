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

## Comparing Container Runtimes on GKE (gVisor, vNode, Kata)

All commands below were validated on GKE 1.35 with NVIDIA L4 GPUs
(`g2-standard-4`). See the [runtime benchmarking guide](../runtime-benchmarking.md)
for methodology and result interpretation.

### GKE + gVisor (GKE Sandbox)

GKE ships gVisor natively — create a GPU node pool with `--sandbox type=gvisor`
and the `gvisor` RuntimeClass appears automatically:

```bash
gcloud container node-pools create gpu-gvisor \
  --cluster kubemark-test \
  --zone us-central1-a \
  --machine-type g2-standard-4 \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --num-nodes 1 \
  --sandbox type=gvisor

kubectl get runtimeclass gvisor   # registered by GKE
```

Notes:
- GKE Sandbox requires a **dedicated node pool**; sandboxed pods are pinned to
  it via an auto-injected `sandbox.gke.io/runtime=gvisor` nodeSelector + taint.
- GPU support uses gVisor's nvproxy; only specific driver versions are
  allowlisted (use `gpu-driver-version=latest`).

Minimal GPU pod under gVisor:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-gvisor-test
spec:
  runtimeClassName: gvisor        # GKE injects the sandbox nodeSelector + toleration
  restartPolicy: Never
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

### GKE + vNode (vCluster / Loft)

vNode installs onto **existing** node pools — no dedicated pool needed. It
requires a [vCluster Platform](https://www.vcluster.com/docs/platform) with a
vNode-enabled license:

```bash
helm upgrade --install vnode-runtime vnode-runtime -n vnode-runtime \
  --repo https://charts.loft.sh --create-namespace \
  --set "config.platform.host=https://YOUR-PLATFORM.example.com" \
  --set "config.platform.accessKey=$PLATFORM_ACCESS_KEY" \
  --set "daemonSet.tolerations[0].operator=Exists"

kubectl get runtimeclass vnode    # registered by the DaemonSet (~30s)
```

Notes:
- The `daemonSet.tolerations` override lets the runtime DaemonSet land on
  GPU-tainted nodes — required, or GPU nodes never get the `vnode` handler.
- GKE's COS nodes (kernel 6.6, containerd 2.x) meet vNode's requirements
  (kernel 6.1+, containerd 1.7/2.0) out of the box.
- **vCluster is not required to run workloads.** vNode is a standalone
  containerd runtime handler; the vCluster Platform connection is only for
  license validation by the DaemonSet. Any plain pod can then use
  `runtimeClassName: vnode`.
- vNode *pairs* with vCluster for multi-tenancy: the benchmarks here ran on
  the **host cluster**; running them inside a vCluster whose workloads are
  forced onto the `vnode` RuntimeClass is the combined isolation story
  (`--vcluster` mode + vNode).

Minimal GPU pod under vNode:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-vnode-test
spec:
  runtimeClassName: vnode         # no dedicated node pool, no nodeSelector needed
  restartPolicy: Never
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

### GKE + Kata Containers (untested)

GKE has no built-in Kata support. The path is the NVIDIA GPU Operator with
Kata Manager on an **Ubuntu node pool** (`--image-type UBUNTU_CONTAINERD`,
COS is not supported), which deploys the `kata-qemu-nvidia-gpu` RuntimeClass
with VFIO GPU passthrough. We have not validated this on GKE — nested
virtualization must be available on the machine type, and all GPUs on a Kata
node are bound to VFIO (the node stops serving `nvidia.com/gpu` to runc pods).
See the [Kata section of the runtime guide](../runtime-benchmarking.md#kata-containers-kata-qemu-nvidia-gpu).

### Run the comparison

```bash
./scripts/compare-runtimes.sh \
  --runtimes default,gvisor,vnode \
  --benchmarks nccl,coldstart \
  --gpus 1 --gpu-type l4 --repeats 3
```

### Measured results (L4, GKE 1.35, 3 repeats, median)

| Metric | runc (baseline) | gVisor | vNode |
|---|---|---|---|
| NCCL all_reduce busbw | 115.3 GB/s | 115.0 GB/s (−0.3%) | 115.3 GB/s (0.0%) |
| CUDA init | 304.9 ms | 343.8 ms (**+38.9 ms**) | 292.3 ms (−12.6 ms, noise) |
| First CUDA kernel | 136.4 ms | +0.9 ms | +1.7 ms |

### One level deeper: GKE + vCluster + runtime

The same comparison can run **inside an ephemeral vCluster** per benchmark,
stacking control-plane isolation (vCluster) on top of runtime isolation
(gVisor/vNode):

```bash
./scripts/compare-runtimes.sh \
  --runtimes default,gvisor,vnode \
  --benchmarks coldstart \
  --gpus 1 --gpu-type l4 --repeats 3 --vcluster
```

Each run creates a fresh vCluster (`vcluster/values.yaml` syncs host
RuntimeClasses in via `sync.fromHost.runtimeClasses`), applies the benchmark
inside it, and tears it down. The report is tagged `"vcluster": true`.

Measured (L4, GKE 1.35, cold-start medians of 3, inside vCluster):

| Metric | runc | gVisor | vNode |
|---|---|---|---|
| CUDA init | 295.4 ms | 336.8 ms (+41.4 ms) | 291.8 ms (−3.6 ms) |
| First CUDA kernel | 128.3 ms | +21.9 ms | −4.3 ms |
| Sandbox startup | 1 s | 1 s | 2 s |

Compare with the host-cluster table above: the gVisor/vNode deltas are the
same within noise — **vCluster adds no measurable data-plane overhead**,
because it virtualizes only the control plane; pods run directly on host
nodes. NCCL inside a vCluster additionally requires installing the MPI
Operator in the virtual cluster, so the vCluster comparison covers cold-start
(the axis where a sync layer could plausibly add latency — it didn't).

Takeaways:
- **Steady-state bandwidth is free** on both sandboxes — single-GPU NCCL is
  identical across all three runtimes.
- **gVisor pays at driver init** (~+40 ms CUDA context creation) from nvproxy
  ioctl interception. Negligible for training jobs; relevant for
  scale-to-zero/serverless GPU pods.
- **vNode is indistinguishable from runc** on every axis measured — expected,
  since user namespaces add no syscall emulation to the GPU path.
- Caveat: gVisor ran on its own node pool (GKE Sandbox requirement) while
  runc/vNode shared a node — same machine/GPU type, different physical
  instances.

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
