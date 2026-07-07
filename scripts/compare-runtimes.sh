#!/bin/bash
set -euo pipefail

# kubemark-ai: Compare GPU benchmark performance across container runtimes
#
# Runs the same benchmarks under different Kubernetes RuntimeClasses
# (runc/default, gVisor, Kata Containers, vNode, ...) and reports the
# overhead of each sandboxed runtime relative to the baseline.
#
# Three measurement axes:
#   1. NCCL bus bandwidth      — GPU-to-GPU communication (ioctl proxy cost)
#   2. Training throughput     — end-to-end compute (GPT-2 single-GPU)
#   3. GPU cold-start latency  — sandbox creation + CUDA init cost
#
# Usage:
#   ./scripts/compare-runtimes.sh --runtimes default,gvisor
#   ./scripts/compare-runtimes.sh --runtimes default,kata-qemu-nvidia-gpu,vnode \
#       --benchmarks nccl,coldstart --gpus 8 --gpu-type h100
#
# The first runtime in the list is used as the baseline (usually "default" = runc).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --runtimes LIST    Comma-separated RuntimeClass names. Use 'default' for the"
  echo "                     cluster default runtime (runc). First entry = baseline."
  echo "                     (default: default,gvisor)"
  echo "                     Common: gvisor, kata-qemu-nvidia-gpu, vnode"
  echo "  --benchmarks LIST  Benchmarks to run: nccl, coldstart, training"
  echo "                     (default: nccl,coldstart)"
  echo "  --gpus N           GPUs for the NCCL benchmark (default: 1)"
  echo "  --gpu-type TYPE    GPU type flag passed through (default: h100)"
  echo "  --tests TESTS      NCCL tests (default: all-reduce)"
  echo "  --repeats N        Cold-start repeats per runtime; median is reported (default: 3)"
  echo "  --vcluster         Run every benchmark inside an ephemeral vCluster"
  echo "                     (control-plane isolation; pods still run on host nodes)"
  echo "  --help             Show this help"
  exit 0
}

RUNTIMES="default,gvisor"
BENCHMARKS="nccl,coldstart"
TOTAL_GPUS=1
GPU_TYPE="h100"
TESTS="all-reduce"
REPEATS=3
USE_VCLUSTER=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --runtimes) RUNTIMES="$2"; shift 2;;
    --benchmarks) BENCHMARKS="$2"; shift 2;;
    --gpus) TOTAL_GPUS="$2"; shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --vcluster) USE_VCLUSTER=true; shift;;
    --help) usage;;
    *) echo "Unknown option: $1"; usage;;
  esac
done

if ! command -v jq &>/dev/null; then
  echo -e "${RED}jq is required for runtime comparison. Install it first.${NC}"
  exit 1
fi

echo -e "${CYAN}${BOLD}"
echo "  kubemark-ai  Runtime Class Comparison"
echo -e "${NC}"
echo -e "${BOLD}Configuration${NC}"
echo "  Runtimes:    $RUNTIMES"
echo "  Benchmarks:  $BENCHMARKS"
echo "  GPUs:        $TOTAL_GPUS ($GPU_TYPE)"
echo "  NCCL tests:  $TESTS"
echo "  Repeats:     $REPEATS (coldstart)"
echo "  vCluster:    $USE_VCLUSTER"
echo ""

if [[ "$USE_VCLUSTER" == "true" ]] && ! command -v vcluster &>/dev/null; then
  echo -e "${RED}vcluster CLI not found. Install it or remove --vcluster.${NC}"
  exit 1
fi

# --- Validate runtime classes exist on the cluster ---
echo -e "${BOLD}Validating RuntimeClasses...${NC}"
for rt in $(echo "$RUNTIMES" | tr ',' ' '); do
  if [[ "$rt" == "default" || "$rt" == "runc" ]]; then
    echo -e "  ${GREEN}✓${NC} default (cluster default runtime)"
    continue
  fi
  if kubectl get runtimeclass "$rt" &>/dev/null; then
    HANDLER=$(kubectl get runtimeclass "$rt" -o jsonpath='{.handler}' 2>/dev/null || echo "?")
    echo -e "  ${GREEN}✓${NC} $rt (handler: $HANDLER)"
  else
    echo -e "  ${RED}✗ RuntimeClass '$rt' not found on this cluster${NC}"
    echo ""
    echo "  Available RuntimeClasses:"
    kubectl get runtimeclass -o custom-columns='    NAME:.metadata.name,HANDLER:.handler' 2>/dev/null || echo "    (none)"
    echo ""
    echo "  Setup guides: docs/runtime-benchmarking.md"
    exit 1
  fi
done
echo ""

COMPARE_ID="compare-$(date +%Y%m%d-%H%M%S)"
COMPARE_DIR="$PROJECT_DIR/results/${COMPARE_ID}"
mkdir -p "$COMPARE_DIR"

# Find the summary.json produced by the most recent run-benchmark.sh invocation
latest_summary() {
  ls -td "$PROJECT_DIR"/results/*/ 2>/dev/null \
    | grep -v "$COMPARE_DIR" \
    | head -1 \
    | xargs -I{} echo "{}summary.json"
}

run_one() {
  local rt="$1" bench="$2"
  shift 2
  # RuntimeClass names contain no whitespace, so word-splitting is safe here
  # (bash 3.2 on macOS treats empty-array expansion as unbound under set -u).
  local rt_flag=""
  if [[ "$rt" != "default" && "$rt" != "runc" ]]; then
    rt_flag="--runtime-class $rt"
  fi
  local vc_flag=""
  if [[ "$USE_VCLUSTER" == "true" ]]; then
    vc_flag="--vcluster"
  fi
  # shellcheck disable=SC2086
  "$SCRIPT_DIR/run-benchmark.sh" "$bench" "$@" $rt_flag $vc_flag \
    || { echo -e "${YELLOW}  Warning: $bench failed on runtime '$rt' — recording as failed${NC}"; return 1; }
}

# --- Run benchmarks per runtime ---
for rt in $(echo "$RUNTIMES" | tr ',' ' '); do
  rt_key=$(echo "$rt" | tr -c 'a-zA-Z0-9\n' '_')
  echo -e "${CYAN}${BOLD}=== Runtime: $rt ===${NC}"

  for bench in $(echo "$BENCHMARKS" | tr ',' ' '); do
    case "$bench" in
      nccl)
        echo -e "${BOLD}[$rt] NCCL ($TESTS, $TOTAL_GPUS GPUs)${NC}"
        if run_one "$rt" nccl --gpus "$TOTAL_GPUS" --gpu-type "$GPU_TYPE" --tests "$TESTS"; then
          cp "$(latest_summary)" "$COMPARE_DIR/nccl-${rt_key}.json" 2>/dev/null || true
        fi
        ;;
      training)
        echo -e "${BOLD}[$rt] Training (gpt2-bench, 1 GPU)${NC}"
        if run_one "$rt" training --model gpt2-bench --gpus 1 --gpu-type "$GPU_TYPE"; then
          cp "$(latest_summary)" "$COMPARE_DIR/training-${rt_key}.json" 2>/dev/null || true
        fi
        ;;
      coldstart)
        echo -e "${BOLD}[$rt] Cold-start x${REPEATS}${NC}"
        for i in $(seq 1 "$REPEATS"); do
          echo "  --- repeat $i/$REPEATS ---"
          if run_one "$rt" coldstart --gpus 1 --gpu-type "$GPU_TYPE"; then
            cp "$(latest_summary)" "$COMPARE_DIR/coldstart-${rt_key}-${i}.json" 2>/dev/null || true
          fi
        done
        ;;
      *)
        echo -e "${YELLOW}Unknown benchmark '$bench' — skipping${NC}"
        ;;
    esac
  done
  echo ""
done

# --- Aggregate into comparison.json ---
echo -e "${CYAN}Aggregating results...${NC}"

BASELINE=$(echo "$RUNTIMES" | cut -d, -f1)

python3 - "$COMPARE_DIR" "$RUNTIMES" "$BASELINE" "$USE_VCLUSTER" <<'PYEOF'
import json, glob, os, re, statistics, sys

compare_dir, runtimes_csv, baseline, use_vcluster = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
runtimes = runtimes_csv.split(",")

def rt_key(rt):
    return re.sub(r'[^a-zA-Z0-9]', '_', rt)

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

report = {
    "compare_id": os.path.basename(compare_dir),
    "baseline": baseline,
    "vcluster": use_vcluster == "true",
    "runtimes": {},
}

for rt in runtimes:
    k = rt_key(rt)
    entry = {}

    nccl = load(os.path.join(compare_dir, f"nccl-{k}.json"))
    if nccl and isinstance(nccl.get("results"), list):
        entry["nccl"] = {
            r.get("test", f"test{i}"): r.get("avg_bus_bandwidth_gbps")
            for i, r in enumerate(nccl["results"]) if r
        }

    training = load(os.path.join(compare_dir, f"training-{k}.json"))
    if training and isinstance(training.get("results"), dict) \
            and training["results"].get("status") == "COMPLETE":
        entry["training"] = {
            "tflops_per_gpu": training["results"].get("tflops_per_gpu_mean"),
            "step_time_ms": training["results"].get("step_time_mean_ms"),
        }

    cold_metrics = {}
    for path in sorted(glob.glob(os.path.join(compare_dir, f"coldstart-{k}-*.json"))):
        cs = load(path)
        if cs and isinstance(cs.get("results"), dict) and cs["results"].get("status") == "COMPLETE":
            for key in ("nvidia_smi_ms", "cuda_init_ms", "first_kernel_ms", "sandbox_startup_s"):
                v = cs["results"].get(key)
                if v is not None:
                    cold_metrics.setdefault(key, []).append(v)
    if cold_metrics:
        entry["coldstart"] = {
            key + "_median": round(statistics.median(vals), 1)
            for key, vals in cold_metrics.items()
        }
        entry["coldstart"]["repeats"] = max(len(v) for v in cold_metrics.values())

    report["runtimes"][rt] = entry

# Overhead vs baseline
base = report["runtimes"].get(baseline, {})
for rt in runtimes:
    if rt == baseline:
        continue
    entry = report["runtimes"].get(rt, {})
    overhead = {}
    for test, bw in (entry.get("nccl") or {}).items():
        base_bw = (base.get("nccl") or {}).get(test)
        if bw and base_bw:
            overhead[f"nccl_{test}_bandwidth_pct"] = round((bw - base_bw) / base_bw * 100, 1)
    t, bt = entry.get("training"), base.get("training")
    if t and bt and t.get("tflops_per_gpu") and bt.get("tflops_per_gpu"):
        overhead["training_tflops_pct"] = round(
            (t["tflops_per_gpu"] - bt["tflops_per_gpu"]) / bt["tflops_per_gpu"] * 100, 1)
    c, bc = entry.get("coldstart"), base.get("coldstart")
    if c and bc:
        for key in ("cuda_init_ms_median", "first_kernel_ms_median", "sandbox_startup_s_median"):
            if c.get(key) and bc.get(key):
                overhead[key.replace("_median", "_delta")] = round(c[key] - bc[key], 1)
    if overhead:
        entry["overhead_vs_baseline"] = overhead

with open(os.path.join(compare_dir, "comparison.json"), "w") as f:
    json.dump(report, f, indent=2)

# --- Human-readable table ---
print()
print("=" * 72)
print(f"  Runtime Comparison (baseline: {baseline})")
print("=" * 72)
for rt in runtimes:
    entry = report["runtimes"].get(rt, {})
    print(f"\n  {rt}")
    for test, bw in (entry.get("nccl") or {}).items():
        print(f"    NCCL {test:<16} avg busbw: {bw} GB/s")
    if entry.get("training"):
        print(f"    Training TFLOPS/GPU:     {entry['training']['tflops_per_gpu']}")
    if entry.get("coldstart"):
        cs = entry["coldstart"]
        print(f"    Cold-start (median of {cs.get('repeats', '?')}):"
              f" cuda_init={cs.get('cuda_init_ms_median', 'n/a')}ms"
              f" first_kernel={cs.get('first_kernel_ms_median', 'n/a')}ms"
              f" sandbox_startup={cs.get('sandbox_startup_s_median', 'n/a')}s")
    if entry.get("overhead_vs_baseline"):
        print(f"    vs {baseline}:")
        for key, v in entry["overhead_vs_baseline"].items():
            unit = "%" if key.endswith("_pct") else (" s" if "_s_" in key else " ms")
            sign = "+" if v > 0 else ""
            print(f"      {key:<36} {sign}{v}{unit}")
    if not entry:
        print("    (no successful results)")
print()
print("=" * 72)
PYEOF

echo ""
echo -e "${GREEN}${BOLD}=== Comparison Complete ===${NC}"
echo ""
echo "  Report:     $COMPARE_DIR/comparison.json"
echo "  Raw runs:   $COMPARE_DIR/"
echo ""
echo "  Notes:"
echo "  - Bandwidth deltas are % vs baseline (negative = slower than baseline)."
echo "  - Cold-start deltas are absolute (ms/s slower than baseline)."
echo "  - First runs include image pulls; use --repeats >= 3 so the median"
echo "    reflects warm-cache sandbox startup."
