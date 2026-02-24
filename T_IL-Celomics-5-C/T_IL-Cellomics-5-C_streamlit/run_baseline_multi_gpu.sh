#!/bin/bash
# ─────────────────────────────────────────────────────────────
# run_baseline_multi_gpu.sh — Launch baseline_comparison.py on 1-4 GPUs
#
# Usage:
#   NUM_GPUS=1 bash run_baseline_multi_gpu.sh   # single GPU (default)
#   NUM_GPUS=4 bash run_baseline_multi_gpu.sh   # 4 GPUs
#
# Environment variables (all optional):
#   NUM_GPUS              — number of GPUs to use (1-4, default 1)
#   PIPELINE_RAW_CSV      — path to raw_all_cells.csv
#   PIPELINE_FEATURES_FILE — path to selected_features.txt
#   PIPELINE_MAX_CELLS    — fallback MAX_CELLS for subsampling
# ─────────────────────────────────────────────────────────────
# NOTE: intentionally NO set -e — we must wait for ALL GPUs even if one fails
mkdir -p logs

NUM_GPUS=${NUM_GPUS:-1}
POLL_INTERVAL=60  # seconds between status updates

# Validate
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 4 ]; then
  echo "ERROR: NUM_GPUS must be 1-4, got $NUM_GPUS"
  exit 1
fi

echo "=== Starting Baseline Comparison on $NUM_GPUS GPU(s) ==="

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

# ========== Launch one process per GPU ==========
GPU_PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
  echo "[GPU $i] Launching baseline_comparison.py (SHARD $((i+1))/$NUM_GPUS)..."
  CUDA_VISIBLE_DEVICES=$i SHARD_IDX=$i NUM_SHARDS=$NUM_GPUS \
    python -u baseline_comparison.py > "logs/baseline_gpu${i}.txt" 2>&1 &
  GPU_PIDS+=($!)
done
echo "GPU PIDs: ${GPU_PIDS[*]}"

# Background monitor: print status per GPU every POLL_INTERVAL seconds
if [ "$NUM_GPUS" -gt 1 ]; then
  (
    while true; do
      sleep "$POLL_INTERVAL"
      printf '\n========== Baseline Status Update ==========\n'
      for (( i=0; i<NUM_GPUS; i++ )); do
        printf '[GPU %d] %s\n' "$i" "$(gpu_status "logs/baseline_gpu${i}.txt")"
      done
      printf '============================================\n'
    done
  ) &
  MONITOR_PID=$!
fi

# Wait for each GPU and track exit codes
FAILED=0
for (( i=0; i<NUM_GPUS; i++ )); do
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
echo "=== All $NUM_GPUS GPU process(es) completed ==="
for (( i=0; i<NUM_GPUS; i++ )); do
  if [ -f "logs/baseline_gpu${i}.txt" ]; then
    LINES=$(wc -l < "logs/baseline_gpu${i}.txt")
    echo "[GPU $i] Finished — $LINES lines in logs/baseline_gpu${i}.txt"
  fi
done

if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED GPU(s) failed!"
  exit 1
fi

# ========== Merge shard results ==========
if [ "$NUM_GPUS" -gt 1 ]; then
  echo ""
  echo "=== Merging baseline comparison shard results ==="

  mkdir -p baseline

  # Merge CSV shards
  HEADER_DONE=0
  > baseline/baseline_comparison.csv
  for (( i=0; i<NUM_GPUS; i++ )); do
    f="baseline/baseline_comparison_shard${i}.csv"
    if [ -f "$f" ]; then
      if [ $HEADER_DONE -eq 0 ]; then
        cat "$f" >> baseline/baseline_comparison.csv
        HEADER_DONE=1
      else
        tail -n +2 "$f" >> baseline/baseline_comparison.csv
      fi
    else
      echo "WARNING: $f not found"
    fi
  done
  echo "Merged CSV → baseline/baseline_comparison.csv ($(tail -n +2 baseline/baseline_comparison.csv | wc -l) features)"

  # Merge JSON shards
  python3 -c "
import json, glob
merged = []
for i in range($NUM_GPUS):
    shard_file = f'baseline/baseline_comparison_shard{i}.json'
    try:
        with open(shard_file) as f:
            merged.extend(json.load(f))
    except FileNotFoundError:
        print(f'WARNING: {shard_file} not found')
with open('baseline/baseline_comparison.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(merged)} entries → baseline/baseline_comparison.json')
"

  # Merge summary text files
  python3 -c "
import pandas as pd
import numpy as np

comp_df = pd.read_csv('baseline/baseline_comparison.csv')

# Discover all baseline model columns dynamically
import re
baseline_names = []
for c in comp_df.columns:
    m = re.match(r'^(.+)_mse$', c)
    if m and m.group(1) not in ('mmf_best',):
        baseline_names.append(m.group(1))

lines = []
lines.append('=' * 65)
lines.append('  MMF vs Baselines — Comparison Summary')
lines.append('=' * 65)

for name in baseline_names:
    mse_col = f'mmf_vs_{name}_%lower_mse'
    mae_col = f'mmf_vs_{name}_%lower_mae'
    rmse_col = f'mmf_vs_{name}_%lower_rmse'
    if mse_col in comp_df.columns:
        lines.append(
            f'  vs. {name}:  '
            f'{comp_df[mse_col].mean():+.1f}% MSE  |  '
            f'{comp_df[mae_col].mean():+.1f}% MAE  |  '
            f'{comp_df[rmse_col].mean():+.1f}% RMSE'
        )
        lines.append('    (positive = MMF is better, negative = baseline is better)')

lines.append('')
lines.append('Per-feature breakdown:')
lines.append(comp_df.to_string(index=False))
text = '\n'.join(lines)

with open('baseline/baseline_comparison_summary.txt', 'w') as f:
    f.write(text)
print(text)
"

  echo "=== Merge complete ==="
else
  echo ""
  echo "=== Single-GPU run — no merge needed ==="
fi

echo "=== DONE ==="
