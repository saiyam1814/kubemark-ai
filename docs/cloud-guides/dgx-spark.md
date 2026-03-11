# Running kubemark-ai on NVIDIA DGX Spark

Tested with the DGX Spark (GB10 Grace Blackwell Superchip). Uses direct mode — no Kubernetes needed.

## About DGX Spark

The DGX Spark is NVIDIA's desktop AI workstation:
- **GB10 Grace Blackwell Superchip** — ARM64 (Grace) CPU + Blackwell GPU
- **128 GB unified memory** — shared between CPU and GPU (no PCIe transfer overhead)
- **200Gbps QSFP** interconnect for connecting two Sparks
- Runs Ubuntu with NVIDIA Container Toolkit pre-installed

## Prerequisites

- Docker or Podman with NVIDIA Container Toolkit (pre-installed on DGX Spark)
- That's it — no Kubernetes, no helm, no kubectl needed

## Run Benchmarks

```bash
# SSH into your DGX Spark
ssh saiyam@<your-spark-ip>

# Clone and run
git clone https://github.com/saiyam1814/kubemark-ai.git && cd kubemark-ai

# NCCL benchmark (single GPU — measures memory bandwidth)
./scripts/run-direct.sh nccl

# GPT-2 training benchmark (no credentials needed)
./scripts/run-direct.sh training

# Full NCCL suite
./scripts/run-direct.sh nccl --tests full-suite

# View results in dashboard
python3 dashboard/server.py
# Open http://<your-spark-ip>:8080
```

## Tested Results (GB10)

These are actual results from running kubemark-ai on a DGX Spark:

### NCCL AllReduce — GB10 Single GPU

```
# Using devices
#  Rank  0 Group  0 Pid  88 on 157ce8b34214 device  0 [000f:01:00] NVIDIA GB10
NCCL version 2.29.2+cuda12.9

#       size      time   algbw   busbw
#        (B)      (us)  (GB/s)  (GB/s)
     1048576     11.32   92.67   92.67
    16777216    144.90  115.80  115.80
   268435456   2313.5   116.03  116.03
  1073741824  10700.0   100.35  100.35
  4294967296  41696.0   103.01  103.01
 17179869184 164061.0   104.72  104.72
```

**104.28 GB/s average, 117.76 GB/s peak.** This is a single-GPU test measuring the GB10's internal memory bandwidth through the unified memory subsystem. The numbers confirm the memory subsystem is performing as expected.

### GPT-2 Training — GB10

```
GPU: NVIDIA GB10
VRAM: 128.5 GB
CUDA: 12.8
PyTorch: 2.7.0a0+ecf3bae40a.nv25.02
Model parameters: 162,643,968 (162.6M)
Batch size: 8, Seq length: 512
Tokens per step: 4,096

step 5:  elapsed time per iteration (ms): 652.36 | 6.13 MODEL_TFLOP/s/GPU | tokens/sec: 6279
step 10: elapsed time per iteration (ms): 654.83 | 6.10 MODEL_TFLOP/s/GPU | tokens/sec: 6255
step 15: elapsed time per iteration (ms): 628.16 | 6.36 MODEL_TFLOP/s/GPU | tokens/sec: 6521
step 20: elapsed time per iteration (ms): 611.34 | 6.54 MODEL_TFLOP/s/GPU | tokens/sec: 6700
step 24: elapsed time per iteration (ms): 621.80 | 6.43 MODEL_TFLOP/s/GPU | tokens/sec: 6587

============================================================
Training Benchmark Summary
  Avg step time:   618.07 ms (std: 28.66)
  Avg TFLOPS/GPU:  6.47
  Avg tokens/sec:  6627
  Peak GPU memory: 9.24 GB
============================================================
```

**6.47 TFLOPS/GPU, 6,627 tokens/sec.** The 4.6% step time std deviation is slightly above the ideal threshold (<5%), expected on the GB10 since NVIDIA's container itself notes early driver maturity for this architecture.

## ARM64 Note

The DGX Spark uses an ARM64 (Grace) CPU. kubemark-ai automatically detects the architecture and uses the correct container image:

- **ARM64**: `nvcr.io/nvidia/pytorch:25.02-py3` (multi-arch, works on Grace)
- **x86_64**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

This is handled transparently — no manual configuration needed.

## Multi-Spark Benchmarking

Two DGX Sparks can be connected via 200Gbps QSFP for multi-node training. NVIDIA's DGX Spark Playbooks report ~23 GB/s (189.85 Gbps) NCCL bandwidth between two Sparks ([source](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md)).

Multi-Spark support in kubemark-ai is planned — see the roadmap in the README.
