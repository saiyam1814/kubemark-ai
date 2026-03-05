#!/bin/bash
set -euo pipefail

# kubemark-ai: Install prerequisites for GPU benchmarking on Kubernetes
#
# What this script does:
#   1. Checks for required CLI tools (kubectl, helm, jq, envsubst)
#   2. Verifies connection to your Kubernetes cluster
#   3. Checks if GPU nodes exist (nvidia.com/gpu resource)
#   4. Installs the MPI Operator (needed for multi-GPU/multi-node benchmarks)
#
# You only need to run this ONCE per cluster.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

check_cmd() {
  if command -v "$1" &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} $1 found: $(command -v "$1")"
    return 0
  else
    echo -e "  ${RED}✗${NC} $1 not found"
    return 1
  fi
}

echo ""
echo -e "${BOLD}kubemark-ai: Cluster Setup${NC}"
echo -e "${DIM}Checks tools, cluster, GPU nodes, and installs MPI Operator${NC}"
echo ""

# --- Step 1: CLI Tools ---
echo -e "${CYAN}[1/4]${NC} ${BOLD}Checking CLI tools${NC}"
echo ""

MISSING=0
check_cmd kubectl || MISSING=1
check_cmd helm || MISSING=1
check_cmd jq || MISSING=1
check_cmd envsubst || MISSING=1

# vcluster is optional
if command -v vcluster &>/dev/null; then
  echo -e "  ${GREEN}✓${NC} vcluster CLI found (optional)"
else
  echo -e "  ${DIM}– vcluster CLI not found (optional — only needed for --vcluster flag)${NC}"
fi

echo ""

if [[ "$MISSING" -eq 1 ]]; then
  echo -e "${RED}Missing required tools above. Install them and re-run this script.${NC}"
  echo ""
  echo "  Install guides:"
  echo "    kubectl:  https://kubernetes.io/docs/tasks/tools/"
  echo "    helm:     https://helm.sh/docs/intro/install/"
  echo "    jq:       brew install jq (macOS) or apt install jq (Linux)"
  echo "    envsubst: part of gettext — brew install gettext (macOS) or apt install gettext (Linux)"
  exit 1
fi

# --- Step 2: Cluster Connection ---
echo -e "${CYAN}[2/4]${NC} ${BOLD}Checking Kubernetes cluster${NC}"
echo ""

if ! kubectl cluster-info &>/dev/null 2>&1; then
  echo -e "  ${RED}✗${NC} Cannot connect to Kubernetes cluster"
  echo ""
  echo "  Make sure:"
  echo "    1. You have a running K8s cluster (DOKS, GKE, EKS, kind, etc.)"
  echo "    2. Your kubeconfig is set: export KUBECONFIG=~/.kube/config"
  echo "    3. kubectl cluster-info works"
  exit 1
fi

CLUSTER_NAME=$(kubectl config current-context 2>/dev/null || echo "unknown")
echo -e "  ${GREEN}✓${NC} Connected to cluster: ${BOLD}$CLUSTER_NAME${NC}"
echo ""

# --- Step 3: GPU Nodes ---
echo -e "${CYAN}[3/4]${NC} ${BOLD}Checking for GPU nodes${NC}"
echo ""

GPU_NODES=$(kubectl get nodes -o json | jq '[.items[] | select(.status.capacity["nvidia.com/gpu"] != null)] | length')
if [[ "$GPU_NODES" -gt 0 ]]; then
  echo -e "  ${GREEN}✓${NC} Found ${BOLD}$GPU_NODES GPU node(s)${NC}:"
  echo ""
  kubectl get nodes -o custom-columns='    NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu,GPU_PRODUCT:.metadata.labels.nvidia\.com/gpu\.product' | head -20
else
  echo -e "  ${YELLOW}⚠${NC} No GPU nodes found"
  echo ""
  echo "  Your cluster needs nodes with nvidia.com/gpu resources."
  echo "  This requires:"
  echo "    1. Nodes with NVIDIA GPUs (physical or cloud GPU instances)"
  echo "    2. NVIDIA GPU drivers installed on the nodes"
  echo "    3. NVIDIA Device Plugin or GPU Operator deployed"
  echo ""
  echo "  Quick setup:"
  echo "    helm repo add nvdp https://nvidia.github.io/k8s-device-plugin"
  echo "    helm install nvidia-device-plugin nvdp/nvidia-device-plugin -n kube-system"
fi

echo ""

# --- Step 4: MPI Operator ---
echo -e "${CYAN}[4/4]${NC} ${BOLD}Installing MPI Operator${NC}"
echo ""
echo -e "  ${DIM}The MPI Operator (from Kubeflow) manages multi-GPU/multi-node jobs.${NC}"
echo -e "  ${DIM}It creates an MPIJob CRD that coordinates launcher + worker pods.${NC}"
echo -e "  ${DIM}This is how NCCL tests and training benchmarks run across GPUs.${NC}"
echo ""

if kubectl get crd mpijobs.kubeflow.org &>/dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} MPI Operator already installed (mpijobs.kubeflow.org CRD found)"

  # Check if the operator pod is running
  MPI_PODS=$(kubectl get pods -n mpi-operator --no-headers 2>/dev/null | grep -c "Running" || echo "0")
  if [[ "$MPI_PODS" -gt 0 ]]; then
    echo -e "  ${GREEN}✓${NC} MPI Operator pod is running"
  else
    echo -e "  ${YELLOW}⚠${NC} MPI Operator CRD exists but no running pod found in mpi-operator namespace"
    echo "    Try: kubectl get pods -A | grep mpi"
  fi
else
  echo "  Installing MPI Operator..."
  echo ""
  # Try Helm first, fall back to kubectl apply
  if helm repo add mpi-operator https://kubeflow.github.io/mpi-operator 2>/dev/null && \
     helm repo update mpi-operator 2>/dev/null; then
    helm install mpi-operator mpi-operator/mpi-operator \
      --namespace mpi-operator \
      --create-namespace \
      --wait \
      --timeout 120s 2>&1
  else
    echo "  Helm chart not available, installing via kubectl apply..."
    kubectl apply --server-side -f https://raw.githubusercontent.com/kubeflow/mpi-operator/master/deploy/v2beta1/mpi-operator.yaml 2>&1
  fi
  echo ""
  echo -e "  ${GREEN}✓${NC} MPI Operator installed successfully"
fi

echo ""
echo -e "${GREEN}${BOLD}=== Setup Complete ===${NC}"
echo ""
echo "  Everything is ready. Here's what was set up:"
echo ""
echo "    Cluster:       $CLUSTER_NAME"
echo "    GPU nodes:     $GPU_NODES"
echo "    MPI Operator:  installed (manages multi-GPU benchmark jobs)"
echo ""
echo -e "  ${BOLD}Next step — run a benchmark:${NC}"
echo ""
echo "    ./scripts/run-benchmark.sh nccl --gpus 1 --gpu-type l40s     # Single GPU test"
echo "    ./scripts/run-benchmark.sh nccl --gpus 8 --gpu-type h100     # 8-GPU single node"
echo "    ./scripts/run-benchmark.sh nccl --gpus 16 --gpu-type h100    # Multi-node"
echo ""
echo -e "  ${DIM}Tip: Use --no-cleanup to keep pods for debugging${NC}"
