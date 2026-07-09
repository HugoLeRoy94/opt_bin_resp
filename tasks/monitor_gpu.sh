#!/usr/bin/env bash
# Log GPU memory + utilisation to CSV while a simulation runs — via nvidia-smi, no
# change to the Python code. Run it in a SEPARATE terminal / tmux window for the
# whole duration of the run (start it BEFORE launching the sim to catch the ramp).
#
# Usage: ./monitor_gpu.sh [gpu_index] [interval_sec] [outfile]
#   gpu_index    which GPU to watch. On the HOST this is the real index (= MY_GPU);
#                INSIDE the docker container it is usually 0.
#   interval_sec seconds between samples (default 2)
#   outfile      CSV path (default gpu_mem_<timestamp>.csv in the cwd)
#
# The CSV has one clean header then one row per sample; align it with the sim by
# timestamp. Ctrl-C to stop.
set -e

GPU="${1:-0}"
INT="${2:-2}"
OUT="${3:-gpu_mem_$(date +%Y%m%d_%H%M%S).csv}"

echo "watching GPU $GPU every ${INT}s  ->  $OUT   (Ctrl-C to stop)"
{
    echo "timestamp,mem_used_MiB,mem_free_MiB,mem_total_MiB,gpu_util_pct,mem_util_pct"
    while true; do
        nvidia-smi -i "$GPU" --format=csv,noheader,nounits \
            --query-gpu=timestamp,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory
        sleep "$INT"
    done
} | tee "$OUT"
