#!/bin/bash
set -e
mkdir -p logs

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i SHARD_IDX=$i NUM_SHARDS=4 MAX_CELLS=-1 \
    python TSA_analysis_4gpu.py > logs/gpu${i}.txt 2>&1 &
done

wait
echo "done"
