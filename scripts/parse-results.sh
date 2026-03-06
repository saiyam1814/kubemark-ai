#!/bin/bash
set -euo pipefail

# kubemark-ai: Parse benchmark results into structured JSON
# Usage: ./scripts/parse-results.sh <results_dir> <benchmark_type> [gpu_type] [total_gpus] [num_nodes]

RESULTS_DIR="${1:?Usage: parse-results.sh <results_dir> <benchmark_type> [gpu_type] [total_gpus] [num_nodes]}"
BENCHMARK_TYPE="${2:?Specify benchmark type: nccl or training}"
GPU_TYPE="${3:-unknown}"
TOTAL_GPUS="${4:-0}"
NUM_NODES="${5:-0}"

RUN_ID=$(basename "$RESULTS_DIR")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

parse_nccl_test() {
  local log_file="$1"
  local test_name="$2"

  if [[ ! -f "$log_file" ]]; then
    echo "null"
    return
  fi

  # Try to extract test name from the log itself
  local detected_name
  detected_name=$(grep -o 'Collective test starting: [a-z_]*' "$log_file" 2>/dev/null | head -1 | sed 's/Collective test starting: //' | sed 's/_perf//')
  if [[ -n "$detected_name" ]]; then
    test_name="$detected_name"
  fi

  # Extract data rows from NCCL output
  # NCCL output format: size count type redop root | out-of-place: time algbw busbw #wrong | in-place: time algbw busbw #wrong
  # For multi-GPU: use in-place busbw (column 12). For single-GPU busbw=0, so use out-of-place algbw (column 7)
  local data_points
  data_points=$(awk '
    /^#.*[Aa]vg.*[Bb]us.*[Bb]andwidth/ {
      # Extract the last number on the line (most reliable)
      match($0, /[0-9]+\.[0-9]+[[:space:]]*$/)
      if (RSTART > 0) {
        val = substr($0, RSTART, RLENGTH)
        gsub(/[[:space:]]/, "", val)
        print "AVG_BUSBW=" val
      }
      next
    }
    /^#/ { next }
    /^=/ { next }
    # Data rows start with whitespace and a number (the size field)
    /^ *[0-9]/ {
      n = split($0, f)
      if (n >= 11) {
        size = f[1]
        if (n >= 13) {
          # AllReduce/AllGather/ReduceScatter format (13 cols: with redop + root)
          oop_time = f[6]; oop_algbw = f[7]; oop_busbw = f[8]
          ip_time = f[10]; ip_algbw = f[11]; ip_busbw = f[12]
        } else {
          # AllToAll/SendRecv format (11 cols: no redop, no root)
          oop_time = f[4]; oop_algbw = f[5]; oop_busbw = f[6]
          ip_time = f[8]; ip_algbw = f[9]; ip_busbw = f[10]
        }
        # Use in-place busbw if > 0, else fall back to out-of-place algbw
        bw = (ip_busbw + 0 > 0) ? ip_busbw : oop_algbw
        time_val = (ip_busbw + 0 > 0) ? ip_time : oop_time
        printf "{\"size_bytes\":%s,\"time_us\":%s,\"algbw_gbps\":%s,\"busbw_gbps\":%s}\n", size, time_val, oop_algbw, bw
      }
    }
  ' "$log_file")

  # Extract avg bus bandwidth (for single GPU it will be 0, use max algbw instead)
  local avg_busbw
  avg_busbw=$(echo "$data_points" | grep "^AVG_BUSBW=" | head -1 | cut -d= -f2)
  avg_busbw="${avg_busbw:-0}"

  # If avg_busbw is 0 (single GPU), compute average from algbw of large messages
  if [[ "$avg_busbw" == "0" || "$avg_busbw" == "0.00" ]]; then
    avg_busbw=$(echo "$data_points" | grep "^{" | tail -5 | sed 's/.*busbw_gbps"://' | sed 's/}//' | awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print "0"}')
  fi

  # Build JSON data points array
  local points_json
  points_json=$(echo "$data_points" | grep "^{" | paste -sd, -)
  points_json="${points_json:-}"

  cat <<EOF
{
    "test": "$test_name",
    "avg_bus_bandwidth_gbps": $avg_busbw,
    "data_points": [$points_json]
  }
EOF
}

parse_nccl_full_suite() {
  local log_file="$1"

  if [[ ! -f "$log_file" ]]; then
    echo "{}"
    return
  fi

  # Split the full suite log into individual test sections
  local tmpdir
  tmpdir=$(mktemp -d)
  # Clean up tmpdir at end of function (not via trap, to avoid clobbering outer traps)

  # Split on the ">>> [N/5]" markers, with fallback to NCCL binary output headers
  awk -v dir="$tmpdir" '
    />>> \[1\/5\] AllReduce|nccl_test:.*all_reduce_perf|# NCCL.*AllReduce/     { file=dir"/all_reduce.log"; next }
    />>> \[2\/5\] AllGather|nccl_test:.*all_gather_perf|# NCCL.*AllGather/     { file=dir"/all_gather.log"; next }
    />>> \[3\/5\] ReduceScatter|nccl_test:.*reduce_scatter_perf|# NCCL.*ReduceScatter/ { file=dir"/reduce_scatter.log"; next }
    />>> \[4\/5\] AllToAll|nccl_test:.*alltoall_perf|# NCCL.*AllToAll/        { file=dir"/all_to_all.log"; next }
    />>> \[5\/5\] SendRecv|nccl_test:.*sendrecv_perf|# NCCL.*SendRecv/        { file=dir"/send_recv.log"; next }
    /^===/ { next }
    /^---/ { next }
    file { print > file }
  ' "$log_file"

  local results=()
  for test_name in all_reduce all_gather reduce_scatter all_to_all send_recv; do
    local test_file="$tmpdir/${test_name}.log"
    if [[ -f "$test_file" ]]; then
      results+=("$(parse_nccl_test "$test_file" "$test_name")")
    fi
  done

  local IFS=','
  local output="${results[*]}"
  rm -rf "$tmpdir"
  echo "$output"
}

parse_training() {
  local log_file="$1"

  if [[ ! -f "$log_file" ]]; then
    echo "null"
    return
  fi

  # Parse training output. Supports multiple formats:
  # 1. Megatron-Bridge: "elapsed time per iteration (ms): X.XX" + "X.XX MODEL_TFLOP/s/GPU"
  # 2. NeMo: "train_step_timing in s: X.XX" + "TFLOPS_per_GPU: X.XX"
  # For large runs (50+ steps): use iterations 35-44 to skip warmup
  # For small runs (<35 steps): use the last 10 iterations (or all if fewer)
  local timings tflops measured_range
  local total_lines

  # Extract all timing lines (compatible with macOS and Linux grep)
  timings=$(grep 'elapsed time per iteration (ms):' "$log_file" 2>/dev/null | sed 's/.*elapsed time per iteration (ms): *\([0-9.]*\).*/\1/' || true)
  tflops=$(grep 'MODEL_TFLOP/s/GPU' "$log_file" 2>/dev/null | sed 's/.*| *\([0-9.]*\) *MODEL_TFLOP\/s\/GPU.*/\1/' || true)

  if [[ -z "$timings" ]]; then
    # Try NeMo format
    timings=$(grep 'train_step_timing in s:' "$log_file" 2>/dev/null | sed 's/.*train_step_timing in s: *\([0-9.]*\).*/\1/' || true)
    tflops=$(grep 'TFLOPS_per_GPU:' "$log_file" 2>/dev/null | sed 's/.*TFLOPS_per_GPU: *\([0-9.]*\).*/\1/' || true)
  fi

  if [[ -z "$timings" ]]; then
    cat <<EOF
{
    "status": "INCOMPLETE",
    "iterations_found": 0,
    "iterations_expected": 10
  }
EOF
    return
  fi

  total_lines=$(echo "$timings" | wc -l | tr -d ' ')

  if [[ "$total_lines" -ge 44 ]]; then
    # Large run: use iterations 35-44 (standard NVIDIA approach)
    timings=$(echo "$timings" | sed -n '35,44p')
    tflops=$(echo "$tflops" | sed -n '35,44p')
    measured_range="35-44"
  elif [[ "$total_lines" -gt 10 ]]; then
    # Medium run: use last 10 iterations (skip warmup)
    local start_line=$((total_lines - 9))
    timings=$(echo "$timings" | tail -10)
    tflops=$(echo "$tflops" | tail -10)
    measured_range="${start_line}-${total_lines}"
  else
    # Small run: use all iterations
    measured_range="1-${total_lines}"
  fi

  local count mean_time std_time mean_tflops std_tflops
  count=$(echo "$timings" | wc -l | tr -d ' ')

  if [[ "$count" -lt 1 ]]; then
    cat <<EOF
{
    "status": "INCOMPLETE",
    "iterations_found": 0,
    "iterations_expected": 10
  }
EOF
    return
  fi

  mean_time=$(echo "$timings" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
  std_time=$(echo "$timings" | awk -v m="$mean_time" '{sum+=($1-m)^2} END {printf "%.2f", sqrt(sum/NR)}')

  if [[ -n "$tflops" ]]; then
    mean_tflops=$(echo "$tflops" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
    std_tflops=$(echo "$tflops" | awk -v m="$mean_tflops" '{sum+=($1-m)^2} END {printf "%.2f", sqrt(sum/NR)}')
  else
    mean_tflops="0"
    std_tflops="0"
  fi

  cat <<EOF
{
    "status": "COMPLETE",
    "iterations_measured": "$measured_range",
    "iterations_found": $count,
    "step_time_mean_ms": $mean_time,
    "step_time_std_ms": $std_time,
    "tflops_per_gpu_mean": $mean_tflops,
    "tflops_per_gpu_std": $std_tflops
  }
EOF
}

# Main output
case "$BENCHMARK_TYPE" in
  nccl)
    # Check if this is a full-suite log or individual test logs
    LOGS=$(find "$RESULTS_DIR" -name "*.log" -type f)
    LOG_COUNT=$(echo "$LOGS" | wc -l | tr -d ' ')

    NCCL_RESULTS=""
    if [[ "$LOG_COUNT" -eq 1 ]]; then
      LOG_FILE=$(echo "$LOGS" | head -1)
      # Check if it's a full suite log
      if grep -q ">>> \[1/5\]" "$LOG_FILE" 2>/dev/null; then
        NCCL_RESULTS=$(parse_nccl_full_suite "$LOG_FILE")
      else
        # Try to extract test name from filename (e.g., all-reduce.log or nccl-allreduce-...-launcher-xxx.log)
        TEST_NAME="unknown"
        FNAME=$(basename "$LOG_FILE" .log)
        if echo "$FNAME" | grep -q "allreduce"; then TEST_NAME="all_reduce"
        elif echo "$FNAME" | grep -q "allgather"; then TEST_NAME="all_gather"
        elif echo "$FNAME" | grep -q "reducescatter"; then TEST_NAME="reduce_scatter"
        elif echo "$FNAME" | grep -q "alltoall\|all-to-all"; then TEST_NAME="all_to_all"
        elif echo "$FNAME" | grep -q "sendrecv\|send-recv"; then TEST_NAME="send_recv"
        fi
        NCCL_RESULTS=$(parse_nccl_test "$LOG_FILE" "$TEST_NAME")
      fi
    else
      PARTS=()
      for log in $LOGS; do
        TEST_NAME="unknown"
        FNAME=$(basename "$log" .log)
        if echo "$FNAME" | grep -q "allreduce\|all-reduce"; then TEST_NAME="all_reduce"
        elif echo "$FNAME" | grep -q "allgather\|all-gather"; then TEST_NAME="all_gather"
        elif echo "$FNAME" | grep -q "reducescatter\|reduce-scatter"; then TEST_NAME="reduce_scatter"
        elif echo "$FNAME" | grep -q "alltoall\|all-to-all"; then TEST_NAME="all_to_all"
        elif echo "$FNAME" | grep -q "sendrecv\|send-recv"; then TEST_NAME="send_recv"
        fi
        PARTS+=("$(parse_nccl_test "$log" "$TEST_NAME")")
      done
      IFS=','
      NCCL_RESULTS="${PARTS[*]}"
    fi

    cat <<EOF
{
  "run_id": "$RUN_ID",
  "timestamp": "$TIMESTAMP",
  "benchmark_type": "nccl",
  "gpu_type": "$GPU_TYPE",
  "total_gpus": $TOTAL_GPUS,
  "num_nodes": $NUM_NODES,
  "results": [
    $NCCL_RESULTS
  ]
}
EOF
    ;;

  training)
    LOG_FILE=$(find "$RESULTS_DIR" -name "*.log" -type f | head -1)
    TRAINING_RESULT=$(parse_training "$LOG_FILE")

    cat <<EOF
{
  "run_id": "$RUN_ID",
  "timestamp": "$TIMESTAMP",
  "benchmark_type": "training",
  "gpu_type": "$GPU_TYPE",
  "total_gpus": $TOTAL_GPUS,
  "num_nodes": $NUM_NODES,
  "results": $TRAINING_RESULT
}
EOF
    ;;

  *)
    echo "Unknown benchmark type: $BENCHMARK_TYPE" >&2
    exit 1
    ;;
esac
