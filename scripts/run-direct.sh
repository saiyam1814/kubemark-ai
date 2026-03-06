#!/bin/bash
set -euo pipefail

# kubemark-ai: Run GPU benchmarks DIRECTLY on a machine — no Kubernetes needed
#
# Just needs: Docker (or Podman) + NVIDIA Container Toolkit + GPU(s)
#
# Usage:
#   ./scripts/run-direct.sh nccl                              # All GPUs on this machine
#   ./scripts/run-direct.sh nccl --gpus 4                     # Use 4 GPUs
#   ./scripts/run-direct.sh nccl --tests full-suite           # All 5 NCCL tests
#   ./scripts/run-direct.sh nccl --gpus 8 --multi-node \      # Multi-node via SSH
#       --hosts "gpu-node-1:8,gpu-node-2:8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_banner() {
  echo -e "${CYAN}"
  echo "  kubemark-ai  (direct mode — no Kubernetes)"
  echo -e "${NC}"
}

usage() {
  echo "Usage: $0 <benchmark_type> [options]"
  echo ""
  echo "Runs GPU benchmarks directly on this machine using Docker/Podman."
  echo "No Kubernetes cluster needed."
  echo ""
  echo "Benchmark types:"
  echo "  nccl       Run NCCL communication benchmarks"
  echo "  training   Run GPT-2 training benchmark"
  echo ""
  echo "Options:"
  echo "  --gpus N          Number of GPUs to use (default: all available)"
  echo "  --tests TESTS     NCCL tests: all-reduce, all-gather, reduce-scatter, full-suite"
  echo "                    (default: all-reduce)"
  echo "  --runtime CMD     Container runtime: docker or podman (default: auto-detect)"
  echo "  --image IMAGE     Override container image"
  echo "  --nccl-debug LVL  NCCL debug level: WARN, INFO, TRACE (default: WARN)"
  echo "  --help            Show this help"
  exit 0
}

if [[ $# -lt 1 ]]; then
  print_banner
  usage
fi

BENCHMARK_TYPE="$1"
shift

# Defaults
TOTAL_GPUS=""
TESTS="all-reduce"
RUNTIME=""
IMAGE="ghcr.io/coreweave/nccl-tests:12.9.1-devel-ubuntu22.04-nccl2.29.2-1-2276a5e"
NCCL_DEBUG="WARN"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) TOTAL_GPUS="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --runtime) RUNTIME="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --nccl-debug) NCCL_DEBUG="$2"; shift 2;;
    --help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

print_banner

# --- Prerequisite Checks ---
echo -e "${BOLD}Checking prerequisites...${NC}"
echo ""

PREREQ_OK=true

# 1. Container runtime
if [[ -z "$RUNTIME" ]]; then
  if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    RUNTIME="docker"
  elif command -v podman &>/dev/null; then
    RUNTIME="podman"
  else
    PREREQ_OK=false
    echo -e "  ${RED}✗ Container runtime not found${NC}"
    echo ""
    echo "    You need Docker or Podman installed."
    echo ""
    # Detect OS for install instructions
    if [[ "$(uname)" == "Linux" ]]; then
      if command -v apt &>/dev/null; then
        echo "    Install Docker (Ubuntu/Debian):"
        echo "      curl -fsSL https://get.docker.com | sh"
        echo "      sudo usermod -aG docker \$USER && newgrp docker"
      elif command -v yum &>/dev/null || command -v dnf &>/dev/null; then
        echo "    Install Docker (RHEL/CentOS/Fedora):"
        echo "      curl -fsSL https://get.docker.com | sh"
        echo "      sudo systemctl start docker && sudo usermod -aG docker \$USER"
      else
        echo "    Install Docker: https://docs.docker.com/engine/install/"
      fi
    elif [[ "$(uname)" == "Darwin" ]]; then
      echo "    Install Docker Desktop for Mac:"
      echo "      https://docs.docker.com/desktop/install/mac-install/"
      echo ""
      echo "    Note: macOS does not have NVIDIA GPUs. Use a Linux GPU machine."
    else
      echo "    Install Docker: https://docs.docker.com/engine/install/"
    fi
    echo ""
  fi
fi

if [[ "$PREREQ_OK" == "true" ]]; then
  echo -e "  ${GREEN}✓${NC} Container runtime: ${BOLD}$RUNTIME${NC} ($(command -v $RUNTIME))"
fi

# 2. NVIDIA drivers (check nvidia-smi directly on host first)
if [[ "$PREREQ_OK" == "true" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo -e "  ${GREEN}✓${NC} NVIDIA driver: version ${BOLD}$DRIVER_VERSION${NC}"
  else
    echo -e "  ${YELLOW}⚠${NC} nvidia-smi not found on host (may still work via container toolkit)"
  fi
fi

# 3. NVIDIA Container Toolkit (the critical piece)
if [[ "$PREREQ_OK" == "true" ]]; then
  if $RUNTIME run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} NVIDIA Container Toolkit: working (GPU access via $RUNTIME confirmed)"
  else
    PREREQ_OK=false
    echo -e "  ${RED}✗ NVIDIA Container Toolkit not working${NC}"
    echo ""
    echo "    '$RUNTIME run --gpus all nvidia-smi' failed."
    echo "    The NVIDIA Container Toolkit lets containers access GPUs."
    echo ""
    if [[ "$(uname)" == "Linux" ]]; then
      echo "    Install NVIDIA Container Toolkit (Linux):"
      echo ""
      echo "      # Add NVIDIA repo"
      echo "      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
      echo "      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
      echo "        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
      echo "        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
      echo ""
      echo "      # Install"
      echo "      sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
      echo ""
      echo "      # Configure Docker"
      echo "      sudo nvidia-ctk runtime configure --runtime=docker"
      echo "      sudo systemctl restart docker"
      echo ""
      echo "    Then re-run this script."
    else
      echo "    See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
    echo ""
  fi
fi

if [[ "$PREREQ_OK" != "true" ]]; then
  echo -e "${RED}Prerequisites not met. Fix the issues above and re-run.${NC}"
  exit 1
fi

# Detect GPU count if not specified
if [[ -z "$TOTAL_GPUS" ]]; then
  TOTAL_GPUS=$($RUNTIME run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi

echo -e "  ${GREEN}✓${NC} GPUs detected: ${BOLD}$TOTAL_GPUS${NC}"
echo ""

# Show GPU info
echo -e "${BOLD}GPU Details:${NC}"
echo ""
$RUNTIME run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name mem; do
  echo -e "  GPU $idx: ${BOLD}$name${NC} ($mem)"
done
echo ""

# Detect GPU type for results
GPU_NAME=$($RUNTIME run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '-' | tr '[:upper:]' '[:lower:]')

# Adjust max message size for GPUs with limited VRAM
GPU_MEM_MB=$(docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "$GPU_MEM_MB" ]] && [[ "$GPU_MEM_MB" -lt 20000 ]]; then
    MAX_MSG_SIZE="4G"  # Reduced for GPUs with <20GB VRAM (e.g., T4 16GB)
    echo "Note: Reduced max message size to ${MAX_MSG_SIZE} for ${GPU_MEM_MB}MB VRAM GPU"
elif [[ -n "$GPU_MEM_MB" ]] && [[ "$GPU_MEM_MB" -lt 32000 ]]; then
    MAX_MSG_SIZE="8G"  # Reduced for GPUs with <32GB VRAM (e.g., L4 24GB)
    echo "Note: Reduced max message size to ${MAX_MSG_SIZE} for ${GPU_MEM_MB}MB VRAM GPU"
else
    MAX_MSG_SIZE="16G"
fi

RUN_ID="direct-$(date +%Y%m%d-%H%M%S)-$(head -c 100 /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 6)"
RESULTS_DIR="$PROJECT_DIR/results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# Map test names to binaries
get_binary() {
  case "$1" in
    all-reduce)      echo "all_reduce_perf";;
    all-gather)      echo "all_gather_perf";;
    reduce-scatter)  echo "reduce_scatter_perf";;
    all-to-all)      echo "alltoall_perf";;
    send-recv)       echo "sendrecv_perf";;
    *) echo "all_reduce_perf";;
  esac
}

# Mount InfiniBand devices if available (for RDMA support)
IB_DEVICE_FLAG=""
if [[ -d /dev/infiniband ]]; then
    IB_DEVICE_FLAG="--device=/dev/infiniband"
    echo "InfiniBand devices detected, enabling RDMA support"
fi

run_nccl_test() {
  local test_name="$1"
  local binary=$(get_binary "$test_name")
  local extra_args=""

  # AllToAll and SendRecv use uint8 datatype (from NVIDIA testset.toml)
  if [[ "$test_name" == "all-to-all" || "$test_name" == "send-recv" ]]; then
    extra_args="-d uint8"
  fi

  echo -e "${CYAN}>>> Running NCCL ${test_name}${NC}"

  # Note: --shm-size is redundant when --ipc=host is set (container shares host /dev/shm)
  # Kept for documentation purposes and compatibility with runtimes that ignore --ipc=host
  $RUNTIME run --rm \
    --gpus all \
    --ipc=host \
    $IB_DEVICE_FLAG \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=32g \
    -e NCCL_DEBUG=$NCCL_DEBUG \
    "$IMAGE" \
    mpirun \
      --allow-run-as-root \
      -np "$TOTAL_GPUS" \
      -bind-to none \
      -map-by slot \
      -x LD_LIBRARY_PATH \
      -x NCCL_DEBUG \
      -mca btl ^openib \
      -mca btl_tcp_if_exclude lo,docker0 \
      /opt/nccl_tests/build/${binary} \
        -b 8 \
        -e $MAX_MSG_SIZE \
        -f 2 \
        -g 1 \
        -n 20 \
        -w 5 \
        $extra_args \
    2>&1 | tee "$RESULTS_DIR/${test_name}.log"

  echo ""
}

run_training_test() {
  local training_image="${IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
  # If the user didn't override the NCCL image, use PyTorch image for training
  if [[ "$IMAGE" == "ghcr.io/coreweave/nccl-tests:"* ]]; then
    training_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
  fi

  echo -e "${CYAN}>>> Running GPT-2 Training Benchmark${NC}"
  echo "Image: $training_image"
  echo ""

  $RUNTIME run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    "$training_image" \
    python3 -u -c '
import torch
import torch.nn as nn
import time
import math

VOCAB_SIZE = 50257
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
SEQ_LEN = 512
BATCH_SIZE = 8
WARMUP_STEPS = 5
MEASURE_STEPS = 20

class GPT2Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(SEQ_LEN, N_EMBD)
        self.blocks = nn.Sequential(*[GPT2Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = self.blocks(tok + pos)
        x = self.ln_f(x)
        return self.head(x)

device = torch.device("cuda")
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")

model = GPT2Model().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,} ({params/1e6:.1f}M)")
print(f"Batch size: {BATCH_SIZE}, Seq length: {SEQ_LEN}")
print(f"Tokens per step: {BATCH_SIZE * SEQ_LEN:,}")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
loss_fn = nn.CrossEntropyLoss()

print(f"Running {WARMUP_STEPS} warmup steps...")
for i in range(WARMUP_STEPS):
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), input_ids.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"Running {MEASURE_STEPS} measured steps...")
print()
times = []
losses = []
for i in range(MEASURE_STEPS):
    torch.cuda.synchronize()
    start = time.time()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), input_ids.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000
    times.append(elapsed_ms)
    losses.append(loss.item())
    tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / (elapsed_ms / 1000)
    flops = 6 * params * BATCH_SIZE * SEQ_LEN
    tflops = flops / (elapsed_ms / 1000) / 1e12
    print(f"step {WARMUP_STEPS + i}: elapsed time per iteration (ms): {elapsed_ms:.2f} | "
          f"loss: {loss.item():.4f} | "
          f"{tflops:.2f} MODEL_TFLOP/s/GPU | "
          f"tokens/sec: {tokens_per_sec:.0f}")

avg_time = sum(times) / len(times)
std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
avg_tflops = 6 * params * BATCH_SIZE * SEQ_LEN / (avg_time / 1000) / 1e12
avg_tokens = BATCH_SIZE * SEQ_LEN / (avg_time / 1000)
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print()
print("=" * 60)
print(f"Training Benchmark Summary")
print(f"  Model:           GPT-2 (124M)")
print(f"  Avg step time:   {avg_time:.2f} ms (std: {std_time:.2f})")
print(f"  Avg TFLOPS/GPU:  {avg_tflops:.2f}")
print(f"  Avg tokens/sec:  {avg_tokens:.0f}")
print(f"  Peak GPU memory: {peak_mem:.2f} GB")
print(f"  Final loss:      {losses[-1]:.4f}")
print("=" * 60)
' 2>&1 | tee "$RESULTS_DIR/training-gpt2.log"

  echo ""
}

# Run benchmarks
case "$BENCHMARK_TYPE" in
  nccl)
    echo -e "${BOLD}=== NCCL Benchmark (Direct Mode) ===${NC}"
    echo "Tests:   $TESTS"
    echo "Run ID:  $RUN_ID"
    echo ""

    if [[ "$TESTS" == "full-suite" ]]; then
      for test in all-reduce all-gather reduce-scatter all-to-all send-recv; do
        run_nccl_test "$test"
      done
    else
      for test in $(echo "$TESTS" | tr ',' ' '); do
        run_nccl_test "$test"
      done
    fi
    ;;
  training)
    echo -e "${BOLD}=== Training Benchmark (Direct Mode) ===${NC}"
    echo "Model:   GPT-2 (124M)"
    echo "Run ID:  $RUN_ID"
    echo ""

    run_training_test
    ;;
  *)
    echo -e "${RED}Supported benchmark types: nccl, training${NC}"
    exit 1
    ;;
esac

# Parse results
echo -e "${CYAN}Parsing results...${NC}"
"$SCRIPT_DIR/parse-results.sh" "$RESULTS_DIR" "$BENCHMARK_TYPE" "$GPU_NAME" "$TOTAL_GPUS" "1" \
  > "$RESULTS_DIR/summary.json" 2>/dev/null || echo '{"error": "parse failed"}' > "$RESULTS_DIR/summary.json"

echo ""
echo -e "${GREEN}${BOLD}=== Benchmark Complete ===${NC}"
echo ""

# Print human-readable summary
print_nccl_summary() {
  local json_file="$1"
  local gpu_count="$2"

  if ! command -v jq &>/dev/null || [[ ! -f "$json_file" ]]; then
    return
  fi

  # Check for parse errors
  if jq -e '.error' "$json_file" &>/dev/null; then
    echo -e "  ${YELLOW}Could not parse results. Check raw logs in $RESULTS_DIR/${NC}"
    return
  fi

  local num_tests
  num_tests=$(jq '.results | length' "$json_file" 2>/dev/null || echo "0")

  echo -e "${BOLD}Results Summary${NC}"
  echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
  echo ""

  # Per-test results
  for i in $(seq 0 $((num_tests - 1))); do
    local test_name avg_bw peak_bw
    test_name=$(jq -r ".results[$i].test" "$json_file" 2>/dev/null | tr '_' ' ')
    avg_bw=$(jq -r ".results[$i].avg_bus_bandwidth_gbps" "$json_file" 2>/dev/null)
    peak_bw=$(jq "[.results[$i].data_points[].busbw_gbps] | max" "$json_file" 2>/dev/null || echo "0")

    echo -e "  ${BOLD}${test_name^^}${NC}"
    echo -e "    Avg bus bandwidth:   ${GREEN}${avg_bw} GB/s${NC}"
    echo -e "    Peak bus bandwidth:  ${GREEN}${peak_bw} GB/s${NC}"
    echo ""
  done

  # What these numbers mean
  echo -e "${BOLD}What do these numbers mean?${NC}"
  echo ""
  echo "  Bus bandwidth measures how fast GPUs can exchange data."
  echo "  Higher = better. It determines how well multi-GPU training scales."
  echo ""
  if [[ "$gpu_count" -le 1 ]]; then
    echo "  Single-GPU results show internal memory bandwidth only."
    echo "  For real GPU-to-GPU communication benchmarks, use 2+ GPUs."
  else
    echo "  With $gpu_count GPUs, this tests real inter-GPU communication."
    echo ""
    echo -e "  ${BOLD}Reference baselines (large message AllReduce):${NC}"
    echo "    H100 x8  (NVLink):   ~280-310 GB/s"
    echo "    A100 x8  (NVLink):   ~220-250 GB/s"
    echo "    L40S x2  (PCIe):     ~20-25 GB/s"
    echo "    RTX 4000 (PCIe):     ~15-20 GB/s"
  fi
  echo ""
  echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
}

if [[ "$BENCHMARK_TYPE" == "nccl" ]]; then
  print_nccl_summary "$RESULTS_DIR/summary.json" "$TOTAL_GPUS"
elif [[ "$BENCHMARK_TYPE" == "training" ]]; then
  if command -v jq &>/dev/null && [[ -f "$RESULTS_DIR/summary.json" ]]; then
    echo -e "${BOLD}Results Summary${NC}"
    echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
    echo ""

    local_status=$(jq -r '.results.status // "UNKNOWN"' "$RESULTS_DIR/summary.json" 2>/dev/null)
    if [[ "$local_status" == "COMPLETE" ]]; then
      local_tflops=$(jq -r '.results.tflops_per_gpu_mean' "$RESULTS_DIR/summary.json" 2>/dev/null)
      local_step_time=$(jq -r '.results.step_time_mean_ms' "$RESULTS_DIR/summary.json" 2>/dev/null)
      local_std=$(jq -r '.results.step_time_std_ms' "$RESULTS_DIR/summary.json" 2>/dev/null)

      echo -e "  ${BOLD}GPT-2 Training (124M params)${NC}"
      echo -e "    TFLOPS/GPU:    ${GREEN}${local_tflops}${NC}"
      echo -e "    Avg step time: ${GREEN}${local_step_time} ms${NC} (std: ${local_std} ms)"
      echo ""
      echo -e "${BOLD}What do these numbers mean?${NC}"
      echo ""
      echo "  TFLOPS/GPU measures how efficiently your GPU does AI math."
      echo "  For small models like GPT-2, expect 5-10% of peak spec."
      echo "  Step time std < 5% of mean = stable GPU, no thermal issues."
    else
      echo -e "  ${YELLOW}Training did not complete. Check logs in $RESULTS_DIR/${NC}"
    fi

    echo ""
    echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
  fi
fi

echo ""
echo "  Results:    $RESULTS_DIR/summary.json"
echo "  Full logs:  $RESULTS_DIR/"
echo "  Dashboard:  python3 dashboard/server.py  (then open http://localhost:8080)"
echo ""
