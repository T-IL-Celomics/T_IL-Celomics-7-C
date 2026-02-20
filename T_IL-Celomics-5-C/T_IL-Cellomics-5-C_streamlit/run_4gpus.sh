#!/bin/bash
# NOTE: intentionally NO set -e — we must wait for ALL GPUs even if one fails
mkdir -p logs

POLL_INTERVAL=120  # seconds between status updates

# Helper: extract a clean one-line summary from a GPU log file.
gpu_status() {
  local f="$1"
  if [ ! -f "$f" ]; then echo "(no log yet)"; return; fi
  local n
  n=$(wc -l < "$f")
  local last
  last=$(tail -c 1000 "$f" 2>/dev/null \
    | tr '\r' '\n' \
    | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' \
    | sed 's/[^[:print:]\t]//g' \
    | sed '/^[[:space:]]*$/d' \
    | tail -1 \
    | cut -c1-100)
  printf '%s lines | %s' "$n" "${last:-(empty)}"
}

# Cleanup handler: kill all child processes on EXIT/INT/TERM
cleanup() {
  echo "Cleaning up child processes..."
  kill "$MONITOR_PID" 2>/dev/null || true
  for pid in "${GPU_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 0.5
  for pid in "${GPU_PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "=== Starting 4-GPU forecasting ==="

GPU_PIDS=()
for i in 0 1 2 3; do
  echo "[GPU $i] Launching TSA_analysis_4gpu.py (SHARD $((i+1))/4)..."
  CUDA_VISIBLE_DEVICES=$i SHARD_IDX=$i NUM_SHARDS=4 MAX_CELLS=-1 \
    python -u TSA_analysis_4gpu.py > logs/gpu${i}.txt 2>&1 &
  GPU_PIDS+=($!)
done
echo "GPU PIDs: ${GPU_PIDS[*]}"

# Background monitor: print status per GPU every POLL_INTERVAL seconds
(
  while true; do
    sleep "$POLL_INTERVAL"
    printf '\n========== Status update ==========\n'
    for i in 0 1 2 3; do
      printf '[GPU %d] %s\n' "$i" "$(gpu_status "logs/gpu${i}.txt")"
    done
    printf '====================================\n'
  done
) &
MONITOR_PID=$!

# Wait for each GPU and track exit codes
FAILED=0
for i in 0 1 2 3; do
  wait "${GPU_PIDS[$i]}" 2>/dev/null
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[GPU $i] FAILED with exit code $rc"
    FAILED=$((FAILED + 1))
  else
    echo "[GPU $i] Completed successfully"
  fi
done

kill "$MONITOR_PID" 2>/dev/null || true
# Remove the trap since we're exiting cleanly
trap - EXIT INT TERM

echo ""
echo "=== All 4 GPU processes completed ==="
for i in 0 1 2 3; do
  if [ -f "logs/gpu${i}.txt" ]; then
    LINES=$(wc -l < "logs/gpu${i}.txt")
    echo "[GPU $i] Finished — $LINES lines in logs/gpu${i}.txt"
  fi
done

if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED GPU(s) failed!"
  exit 1
fi
