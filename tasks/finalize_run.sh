#!/usr/bin/env bash
# Move a run's GPU-memory trace + stdout log INTO the sweep folder the run created,
# so they sit next to that sweep's outputs instead of loose in the task data root.
#
# The sweep folder name is chosen inside Python (timestamped), so the launcher can't
# know it up front. We recover it from the "Initiating sweep: <path>" line the
# training scripts print (captured in the run log). Runs that create no sweep
# (e.g. test_scaling.py) print no such line → the files are left where they are.
#
#   finalize_run.sh <run_log> <mem_log> <remote_data_root>
set -u
RUN_LOG="$1"; MEM_LOG="$2"; REMOTE_DATA="$3"

# Stop the GPU monitor by the PID it recorded (see monitor_gpu.sh). Done here — a real
# script with literal args — so there is no pattern match that could hit the tmux shell.
mon_pid=$(cat "$MEM_LOG.pid" 2>/dev/null || true)
[ -n "$mon_pid" ] && kill "$mon_pid" 2>/dev/null || true
rm -f "$MEM_LOG.pid"

sweep_c=$(grep -m1 'Initiating sweep:' "$RUN_LOG" 2>/dev/null | awk '{print $NF}')
[ -z "$sweep_c" ] && exit 0                       # no sweep created → leave at root

# Map the container path (/app/data/...) to the host data root.
sweep_h=${sweep_c/\/app\/data/$REMOTE_DATA}
[ -d "$sweep_h" ] || exit 0

mv -f "$MEM_LOG" "$sweep_h"/ 2>/dev/null || true
mv -f "$RUN_LOG" "$sweep_h"/ 2>/dev/null || true
echo "moved run + gpu logs into $sweep_h"
