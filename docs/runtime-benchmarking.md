# Benchmarking GPU Runtime Classes (gVisor, Kata, vNode)

Kubernetes lets you run pods under different container runtimes via
[RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/).
Sandboxed runtimes add security isolation — but at a cost that shows up in
**three different places**, and measuring only one of them gives a misleading
picture:

| Axis | What it measures | Where sandboxes pay |
|------|------------------|---------------------|
| NCCL bus bandwidth | GPU-to-GPU communication | ioctl/syscall proxying (gVisor), virtio paths (Kata) |
| Training throughput | End-to-end compute | Kernel launch overhead, memory registration |
| GPU cold-start latency | Pod scheduled → CUDA ready | Sandbox/VM boot, device passthrough setup, driver init |

kubemark-ai measures all three with a single command:

```bash
# Baseline (runc) vs gVisor
./scripts/compare-runtimes.sh --runtimes default,gvisor

# Full comparison across runtimes, 8 GPUs
./scripts/compare-runtimes.sh \
  --runtimes default,gvisor,kata-qemu-nvidia-gpu,vnode \
  --benchmarks nccl,coldstart,training \
  --gpus 8 --gpu-type h100 --repeats 5
```

The first runtime listed is the **baseline**. The report
(`results/compare-<timestamp>/comparison.json`) contains per-runtime results
plus normalized overhead:

```json
{
  "baseline": "default",
  "runtimes": {
    "default": { "nccl": {"all_reduce": 280.5}, "coldstart": {"cuda_init_ms_median": 850.0} },
    "gvisor": {
      "nccl": {"all_reduce": 271.2},
      "coldstart": {"cuda_init_ms_median": 1420.0},
      "overhead_vs_baseline": {
        "nccl_all_reduce_bandwidth_pct": -3.3,
        "cuda_init_ms_delta": 570.0
      }
    }
  }
}
```

## Measured results: runc vs gVisor vs vNode on GKE

Validated end-to-end on GKE 1.35, NVIDIA L4 (`g2-standard-4`), 3 repeats,
medians reported (full setup commands in the [GKE guide](cloud-guides/gke.md#comparing-container-runtimes-on-gke-gvisor-vnode-kata)):

| Metric | runc (baseline) | gVisor (nvproxy) | vNode |
|---|---|---|---|
| NCCL all_reduce busbw | 115.3 GB/s | 115.0 GB/s (−0.3%) | 115.3 GB/s (0.0%) |
| CUDA context init | 304.9 ms | 343.8 ms (**+38.9 ms**) | 292.3 ms (−12.6 ms, noise) |
| First CUDA kernel | 136.4 ms | +0.9 ms | +1.7 ms |

- Steady-state bandwidth was identical across all three runtimes.
- gVisor's cost concentrates in driver/context init (nvproxy ioctl
  interception) — the profile predicted above, now confirmed by measurement.
- vNode measured indistinguishable from runc on every axis.
- Kata was **not** measured (no built-in GKE support; see setup notes below).

The same cold-start comparison repeated **inside ephemeral vClusters**
(`compare-runtimes.sh ... --vcluster`) reproduced the deltas within noise
(gVisor +41.4 ms CUDA init, vNode −3.6 ms) — confirming vCluster's
control-plane-only virtualization adds no data-plane overhead. See the
[GKE guide](cloud-guides/gke.md#one-level-deeper-gke--vcluster--runtime) for
the full table.

You can also run any single benchmark under a runtime class directly:

```bash
./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100 --runtime-class gvisor
./scripts/run-benchmark.sh coldstart --runtime-class kata-qemu-nvidia-gpu
./scripts/run-benchmark.sh training --model gpt2-bench --runtime-class vnode

# Helm
helm install my-bench ./k8s/helm-chart --set runtimeClassName=gvisor
```

## Runtime setup guides

### gVisor (`runsc` + nvproxy)

gVisor supports GPUs through `nvproxy`, which proxies the application's NVIDIA
driver ioctls from inside the sandbox.

- Enable with the `--nvproxy` flag on `runsc`; on GKE use the built-in
  [GKE Sandbox GPU support](https://cloud.google.com/kubernetes-engine/docs/how-to/sandbox-pods).
- **Driver version matters**: nvproxy supports a *strict allowlist* of NVIDIA
  driver versions. Check with `runsc nvproxy list-supported-drivers`. Unsupported
  drivers fail to start (override at your own risk with
  `--nvproxy-allow-unsupported-driver`).
- Only `compute` and `utility` driver capabilities are allowed by default.
- CUDA workloads (PyTorch, NCCL) are supported; expect the largest deltas in
  cold-start and ioctl-heavy phases rather than steady-state bandwidth.

Typical RuntimeClass name: `gvisor`.

On GKE: create a GPU node pool with `--sandbox type=gvisor` — the
RuntimeClass is registered automatically, and sandboxed pods are pinned to
that pool via an injected nodeSelector + taint. Because launchers/pods that
request no GPU don't get the auto-injected `nvidia.com/gpu` toleration, the
kubemark-ai manifests carry an explicit toleration so MPI launchers can
schedule onto GPU-tainted sandbox nodes.

Docs: https://gvisor.dev/docs/user_guide/gpu/

### Kata Containers (`kata-qemu-nvidia-gpu`)

Kata runs each pod in a lightweight VM. NVIDIA supports this officially through
the **GPU Operator** with Kata Manager:

- Install the GPU Operator with `kataManager.enabled=true` (containerd only).
- This deploys the `kata-qemu-nvidia-gpu` RuntimeClass (and
  `kata-qemu-nvidia-gpu-snp`/`-tdx` for confidential computing on capable
  hardware).
- GPUs are attached via **VFIO passthrough** — all GPUs on a node must be
  assigned to one Kata VM; partial node configs and vGPU are not supported.
- The VM boots an NVIDIA-optimized kernel with the driver inside the guest, so
  cold-start includes VM boot + guest driver init. Steady-state GPU compute is
  near native.

Docs: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-kata.html

### vNode (LoftLabs)

vNode isolates tenant workloads with Linux user namespaces rather than a
syscall-interception sandbox or VM, so its expected overhead profile is the
lightest of the three — confirmed by measurement (see table above). Pairs
naturally with the existing `--vcluster` mode: vCluster isolates the control
plane, vNode isolates the node runtime.

Install (requires a vCluster Platform with a vNode-enabled license):

```bash
helm upgrade --install vnode-runtime vnode-runtime -n vnode-runtime \
  --repo https://charts.loft.sh --create-namespace \
  --set "config.platform.host=https://YOUR-PLATFORM.example.com" \
  --set "config.platform.accessKey=$PLATFORM_ACCESS_KEY" \
  --set "daemonSet.tolerations[0].operator=Exists"
```

The tolerations override is required so the runtime DaemonSet also lands on
GPU-tainted nodes. Unlike gVisor on GKE, vNode needs no dedicated node pool —
it registers the `vnode` handler on existing nodes (kernel 6.1+,
containerd 1.7/2.0).

Typical RuntimeClass name: `vnode`.

Docs: https://www.vnode.com/docs/

## Methodology notes

- **Cold-start** runs a 1-GPU pod that reports `nvidia-smi` latency, CUDA
  context init, and first-kernel latency from inside the container, and the
  harness adds `sandbox_startup_s` (PodScheduled → container started) from the
  API server. The first run per node includes the image pull — use
  `--repeats 3` or more (default 3); the **median** is reported.
- **NCCL/Training** use the exact same manifests as regular runs — the only
  difference is `runtimeClassName`, so deltas are attributable to the runtime.
- Run all runtimes on the **same node pool** (same GPU, driver, topology) or
  the comparison is meaningless. Use `nodeSelector` in the Helm chart, or
  single-node clusters, to control placement.
- Kata's VFIO passthrough re-binds GPUs away from the host driver. A node
  configured for Kata GPU pods will not simultaneously expose
  `nvidia.com/gpu` to runc pods — compare across identically-shaped node
  pools in that case, or use the NVIDIA sandbox device plugin resource
  (`nvidia.com/pgpu`) as documented by the GPU Operator.

## Related: Dynamic Resource Allocation (DRA)

DRA is GA since Kubernetes 1.34 and is the successor to the device plugin
model used here (`nvidia.com/gpu` resource requests). The
[NVIDIA DRA driver](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu)
adds ResourceClaim-based GPU allocation, MIG slices, VFIO devices, and
ComputeDomains for multi-node NVLink. With the `DRAExtendedResource` feature
gate, existing `nvidia.com/gpu` requests are translated automatically, so
kubemark-ai manifests work unchanged on DRA clusters. Native ResourceClaim
manifests are on the roadmap.
