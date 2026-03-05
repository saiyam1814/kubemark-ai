#!/bin/bash
set -euo pipefail

# kubemark-ai: Run GPU benchmarks on Kubernetes
#
# Usage:
#   ./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100
#   ./scripts/run-benchmark.sh nccl --gpus 64 --gpu-type h100 --tests full-suite --vcluster
#   ./scripts/run-benchmark.sh training --model qwen3-30b --gpus 64 --gpu-type h100

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
  echo "  _          _                            _             _ "
  echo " | | ___   _| |__   ___ _ __ ___   __ _ _ __| | __      __ _(_)"
  echo " | |/ / | | | '_ \\ / _ \\ '_ \` _ \\ / _\` | '__| |/ /___  / _\` | |"
  echo " |   <| |_| | |_) |  __/ | | | | | (_| | |  |   <|___|| (_| | |"
  echo " |_|\\_\\\\__,_|_.__/ \\___|_| |_| |_|\\__,_|_|  |_|\\_\\     \\__,_|_|"
  echo ""
  echo -e "${NC}  GPU Benchmarking on Kubernetes"
  echo ""
}

usage() {
  echo "Usage: $0 <benchmark_type> [options]"
  echo ""
  echo "Benchmark types:"
  echo "  nccl       Run NCCL communication benchmarks"
  echo "  training   Run model training benchmarks"
  echo ""
  echo "Options:"
  echo "  --gpus N          Total number of GPUs (default: 8)"
  echo "  --gpu-type TYPE   GPU type: h100, b200, gb200, gb300, a100 (default: h100)"
  echo "  --nodes N         Number of nodes (auto-calculated if not set)"
  echo "  --tests TESTS     Comma-separated tests (default: all-reduce)"
  echo "                    NCCL: all-reduce, all-gather, reduce-scatter, full-suite"
  echo "  --model MODEL     Model for training benchmarks (e.g., qwen3-30b, llama31-8b)"
  echo "  --dtype TYPE      Data type: bf16, fp8 (default: bf16)"
  echo "  --vcluster        Use vCluster for isolation"
  echo "  --no-cleanup      Don't delete namespace/vcluster after completion"
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
TOTAL_GPUS=8
GPU_TYPE="h100"
NUM_NODES=""
TESTS="all-reduce"
MODEL=""
DTYPE="bf16"
USE_VCLUSTER=false
CLEANUP=true
NCCL_DEBUG="WARN"
NCCL_SOCKET_IFNAME="eth0"
NCCL_IB_DISABLE="0"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) TOTAL_GPUS="$2"; shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --nodes) NUM_NODES="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --vcluster) USE_VCLUSTER=true; shift;;
    --no-cleanup) CLEANUP=false; shift;;
    --nccl-debug) NCCL_DEBUG="$2"; shift 2;;
    --nccl-ifname) NCCL_SOCKET_IFNAME="$2"; shift 2;;
    --disable-ib) NCCL_IB_DISABLE="1"; shift;;
    --help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

# Derive GPUs per node from GPU type
# For single-GPU cloud instances, use --gpus 1 --nodes 1
case "$GPU_TYPE" in
  # Multi-GPU DGX / HGX systems
  h100|b200|b300|a100|a100-80g) GPUS_PER_NODE=8;;
  gb200|gb300)                  GPUS_PER_NODE=4;;
  # Single-GPU cloud instances (DigitalOcean, Lambda, RunPod, etc.)
  l40s|l40|l4|a10g|a10|rtx4000|rtx6000|rtx4090|rtx3090|t4|v100)
    GPUS_PER_NODE=1;;
  # Multi-GPU but not 8 (some cloud configs)
  a100x2|h100x2) GPUS_PER_NODE=2;;
  a100x4|h100x4) GPUS_PER_NODE=4;;
  # Custom: user can override with --nodes
  *)
    echo -e "${YELLOW}Unknown GPU type: $GPU_TYPE — defaulting to 1 GPU/node${NC}"
    echo -e "${YELLOW}Override with --nodes if needed${NC}"
    GPUS_PER_NODE=1;;
esac

# Calculate nodes if not specified
if [[ -z "$NUM_NODES" ]]; then
  NUM_NODES=$(( (TOTAL_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
fi

# Generate run ID
# SHORT_ID (6 chars) is used for K8s resource names to keep DNS hostnames short.
# MPI Operator creates worker hostnames like {mpijob-name}-worker-N.{service}.{namespace}
# which must stay under DNS label limits. Full RUN_ID is kept for results and display.
SHORT_ID="$(head -c 100 /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 6)"
RUN_ID="$(date +%Y%m%d-%H%M%S)-${SHORT_ID}"
NAMESPACE="bench-${SHORT_ID}"

# Export for envsubst
export RUN_ID SHORT_ID NAMESPACE TOTAL_GPUS GPUS_PER_NODE NUM_NODES GPU_TYPE DTYPE
export NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_DISABLE

print_banner

echo -e "${BOLD}Benchmark Configuration${NC}"
echo "  Type:       $BENCHMARK_TYPE"
echo "  GPUs:       $TOTAL_GPUS ($NUM_NODES nodes x $GPUS_PER_NODE GPUs/node)"
echo "  GPU Type:   $GPU_TYPE"
if [[ "$BENCHMARK_TYPE" == "nccl" ]]; then
  echo "  Tests:      $TESTS"
else
  echo "  Model:      $MODEL"
  echo "  Dtype:      $DTYPE"
fi
echo "  vCluster:   $USE_VCLUSTER"
echo "  Namespace:  $NAMESPACE"
echo "  Run ID:     $RUN_ID"
echo ""

# Cleanup handler
cleanup() {
  if [[ "$CLEANUP" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [[ "$USE_VCLUSTER" == "true" ]]; then
      vcluster delete "bench-${SHORT_ID}" --namespace "$NAMESPACE" --driver helm 2>/dev/null || true
    fi
    kubectl delete namespace "$NAMESPACE" --wait=false 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete${NC}"
  else
    echo ""
    echo -e "${YELLOW}Skipping cleanup (--no-cleanup). Resources in namespace: $NAMESPACE${NC}"
  fi
}
trap cleanup EXIT

# Step 1: Create namespace
echo -e "${CYAN}[1/5]${NC} Creating namespace..."
kubectl create namespace "$NAMESPACE"

# Step 2: Optional vCluster
if [[ "$USE_VCLUSTER" == "true" ]]; then
  echo -e "${CYAN}[2/5]${NC} Creating vCluster for isolation..."
  if ! command -v vcluster &>/dev/null; then
    echo -e "${RED}vcluster CLI not found. Install it or remove --vcluster flag.${NC}"
    exit 1
  fi
  # Run from temp dir to avoid conflict with our vcluster/ directory
  (cd /tmp && vcluster create "bench-${SHORT_ID}" \
    --namespace "$NAMESPACE" \
    -f "$PROJECT_DIR/vcluster/values.yaml" \
    --connect=false \
    --add=false \
    --driver helm)

  # Connect and get kubeconfig
  VCLUSTER_KUBECONFIG=$(mktemp)
  vcluster connect "bench-${SHORT_ID}" \
    --namespace "$NAMESPACE" \
    --driver helm \
    --print > "$VCLUSTER_KUBECONFIG"

  export KUBECONFIG="$VCLUSTER_KUBECONFIG"
  # When using vCluster, apply to default namespace inside it
  NAMESPACE="default"
  export NAMESPACE
  echo -e "${GREEN}  vCluster ready${NC}"
else
  echo -e "${CYAN}[2/5]${NC} Skipping vCluster (use --vcluster to enable)"
fi

# Step 3: Apply benchmark manifests
echo -e "${CYAN}[3/5]${NC} Launching benchmark..."

  # Map test names to manifest filenames
map_test_to_manifest() {
  case "$1" in
    all-reduce)      echo "nccl-allreduce";;
    all-gather)      echo "nccl-allgather";;
    reduce-scatter)  echo "nccl-reducescatter";;
    full-suite)      echo "nccl-full-suite";;
    *)               echo "nccl-$1";;
  esac
}

case "$BENCHMARK_TYPE" in
  nccl)
    for test in $(echo "$TESTS" | tr ',' ' '); do
      MANIFEST_NAME=$(map_test_to_manifest "$test")
      MANIFEST="$PROJECT_DIR/k8s/jobs/${MANIFEST_NAME}.yaml"
      if [[ ! -f "$MANIFEST" ]]; then
        echo -e "${RED}  Manifest not found: $MANIFEST${NC}"
        echo "  Available: all-reduce, all-gather, reduce-scatter, full-suite"
        exit 1
      fi
      echo "  Applying: ${MANIFEST_NAME}"
      envsubst < "$MANIFEST" | kubectl apply -f -
    done
    ;;

  training)
    if [[ -z "$MODEL" ]]; then
      echo -e "${RED}  --model is required for training benchmarks${NC}"
      echo "  Available: gpt2-bench, qwen3-30b, llama31-8b"
      exit 1
    fi
    MANIFEST="$PROJECT_DIR/k8s/jobs/training-${MODEL}.yaml"
    if [[ ! -f "$MANIFEST" ]]; then
      echo -e "${RED}  Manifest not found: $MANIFEST${NC}"
      exit 1
    fi
    echo "  Applying: training-${MODEL}"
    envsubst < "$MANIFEST" | kubectl apply -f -
    ;;

  *)
    echo -e "${RED}Unknown benchmark type: $BENCHMARK_TYPE${NC}"
    usage
    ;;
esac

# Step 4: Wait for completion
echo -e "${CYAN}[4/5]${NC} Waiting for benchmark to complete..."
echo "  (this may take several minutes depending on GPU count and tests)"

# Poll for completion
TIMEOUT=3600
ELAPSED=0
INTERVAL=10
while [[ $ELAPSED -lt $TIMEOUT ]]; do
  SUCCEEDED=""
  FAILED=""

  if [[ "$BENCHMARK_TYPE" == "nccl" ]]; then
    # NCCL uses MPIJob
    SUCCEEDED=$(kubectl get mpijob -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Succeeded")].status}' 2>/dev/null || echo "")
    FAILED=$(kubectl get mpijob -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
  else
    # Training uses regular Job
    SUCCEEDED=$(kubectl get job -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "")
    FAILED=$(kubectl get job -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
  fi

  if echo "$FAILED" | grep -q "True"; then
    echo -e "${RED}  Benchmark FAILED${NC}"
    echo "  Check logs: kubectl logs -n $NAMESPACE -l app=kubemark-ai"
    exit 1
  fi

  if echo "$SUCCEEDED" | grep -q "True"; then
    echo -e "${GREEN}  Benchmark completed successfully${NC}"
    break
  fi

  # Show running pods
  RUNNING=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')
  printf "\r  Pods: %s | Elapsed: %ds / %ds" "$RUNNING" "$ELAPSED" "$TIMEOUT"

  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

if [[ $ELAPSED -ge $TIMEOUT ]]; then
  echo -e "\n${RED}  Timeout after ${TIMEOUT}s${NC}"
  exit 1
fi

# Step 5: Collect results
echo -e "${CYAN}[5/5]${NC} Collecting results..."
RESULTS_DIR="$PROJECT_DIR/results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# Get pod logs (launcher pods for MPIJob, all pods for Job)
if [[ "$BENCHMARK_TYPE" == "nccl" ]]; then
  LOG_SELECTOR="training.kubeflow.org/job-role=launcher"
else
  LOG_SELECTOR="app=kubemark-ai,benchmark=training"
fi
for pod in $(kubectl get pods -n "$NAMESPACE" -l "$LOG_SELECTOR" -o name 2>/dev/null); do
  POD_NAME=$(basename "$pod")
  echo "  Saving logs: $POD_NAME"
  kubectl logs -n "$NAMESPACE" "$pod" > "$RESULTS_DIR/${POD_NAME}.log" 2>&1
done

# Parse results
echo "  Parsing results..."
"$SCRIPT_DIR/parse-results.sh" "$RESULTS_DIR" "$BENCHMARK_TYPE" "$GPU_TYPE" "$TOTAL_GPUS" "$NUM_NODES" \
  > "$RESULTS_DIR/summary.json" 2>/dev/null || echo "{\"error\": \"parse failed\"}" > "$RESULTS_DIR/summary.json"

# Store as ConfigMap for dashboard
kubectl create configmap "results-${RUN_ID}" \
  --namespace "$NAMESPACE" \
  --from-file="$RESULTS_DIR/summary.json" \
  --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true

echo ""
echo -e "${GREEN}${BOLD}=== Benchmark Complete ===${NC}"
echo ""

# Human-readable results summary
print_readable_summary() {
  local json_file="$1"
  local bench_type="$2"
  local gpu_count="$3"
  local gpu_type="$4"

  if ! command -v jq &>/dev/null || [[ ! -f "$json_file" ]]; then
    return
  fi

  if jq -e '.error' "$json_file" &>/dev/null; then
    echo -e "  ${YELLOW}Could not parse results. Check raw logs.${NC}"
    return
  fi

  echo -e "${BOLD}Results Summary${NC}"
  echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
  echo ""

  if [[ "$bench_type" == "nccl" ]]; then
    local num_tests
    num_tests=$(jq '.results | length' "$json_file" 2>/dev/null || echo "0")

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

    echo -e "  ${BOLD}What do these numbers mean?${NC}"
    echo ""
    echo "  Bus bandwidth measures how fast GPUs exchange data via NCCL."
    echo "  Higher bandwidth = better multi-GPU scaling for AI training."
    echo ""
    if [[ "$gpu_count" -le 1 ]]; then
      echo "  Single-GPU: measures internal GPU bandwidth only."
      echo "  Use 2+ GPUs to test real inter-GPU communication."
    else
      echo -e "  ${BOLD}Reference baselines (large message AllReduce):${NC}"
      echo "    H100 x8  (NVLink):   ~280-310 GB/s"
      echo "    A100 x8  (NVLink):   ~220-250 GB/s"
      echo "    L40S x2  (PCIe):     ~20-25 GB/s"
      echo "    RTX 4000 (PCIe):     ~15-20 GB/s"
    fi

  elif [[ "$bench_type" == "training" ]]; then
    local status mean_tflops mean_step
    status=$(jq -r '.results.status' "$json_file" 2>/dev/null)
    mean_tflops=$(jq -r '.results.tflops_per_gpu_mean' "$json_file" 2>/dev/null)
    mean_step=$(jq -r '.results.step_time_mean_ms' "$json_file" 2>/dev/null)

    if [[ "$status" == "COMPLETE" ]]; then
      echo -e "  ${BOLD}Training Performance${NC}"
      echo -e "    TFLOPS/GPU:      ${GREEN}${mean_tflops}${NC}"
      echo -e "    Step time:       ${GREEN}${mean_step} ms${NC}"
      echo ""
      echo -e "  ${BOLD}What do these numbers mean?${NC}"
      echo ""
      echo "  TFLOPS/GPU = trillion floating-point operations per second per GPU."
      echo "  Higher = better GPU utilization for training."
      echo ""
      echo "  Step time = how long each training iteration takes."
      echo "  Lower = faster training."
      echo ""
      echo "  Reference (BF16, Qwen3-30B):"
      echo "    H100 x64: ~450-500 TFLOPS/GPU"
      echo "    B200 x64: ~600-650 TFLOPS/GPU"
    else
      echo -e "  ${YELLOW}Training did not complete enough iterations for measurement.${NC}"
      echo "  Check logs for errors. Parser expects 'elapsed time per iteration (ms): X.XX' output."
    fi
  fi

  echo ""
  echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
}

print_readable_summary "$RESULTS_DIR/summary.json" "$BENCHMARK_TYPE" "$TOTAL_GPUS" "$GPU_TYPE"

echo ""
echo "  Results:    $RESULTS_DIR/summary.json"
echo "  Full logs:  $RESULTS_DIR/"
echo "  Dashboard:  python3 dashboard/server.py  (then open http://localhost:8080)"
echo ""
