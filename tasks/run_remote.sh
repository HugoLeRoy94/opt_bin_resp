#!/usr/bin/env bash
# Launch a task script on the cluster inside a detached tmux session, with a GPU
# memory monitor running alongside it for the whole duration.
#
# Usage:
#   ./run_remote.sh <task> <script.py> [gpu] [-- script args...]
#
# Examples:
#   ./run_remote.sh fig1 het_casc_ng3.py                 # GPU 0, no args
#   ./run_remote.sh fig1 het_casc.py 2 -- --n_genes 7    # GPU 2, pass --n_genes 7
#
# The run lives in a detached tmux session (holds live logs, survives logout).
#   attach:  ssh -t cluster tmux attach -t <session>
#   list:    ssh -t cluster tmux ls
# A GPU trace is written to data/<task>/gpu_mem_<session>.csv (so sync.sh pulls it
# back with the runs) and the monitor is stopped automatically when the run ends.
set -e

SERVER="leroy@10.187.172.7"
REMOTE_ROOT="/home/leroy/opt_bin_resp"
REMOTE_DATA="/storage/leroy/data"          # host data root (= /app/data in container)
COMPOSE="$REMOTE_ROOT/docker-compose.server.yaml"

TASK="$1"; SCRIPT="$2"; GPU="${3:-0}"
# drop task/script/gpu; anything after (incl. a leading --) is script args
shift $(( $# < 3 ? $# : 3 ))
[ "$1" = "--" ] && shift
ARGS="$*"

[ -z "$TASK" ] || [ -z "$SCRIPT" ] && { echo "usage: $0 <task> <script.py> [gpu] [-- args]"; exit 1; }

SESSION="${TASK}_${SCRIPT%.py}_$(date +%H%M%S)"
REMOTE_SCRIPT="/app/tasks/${TASK}/scripts/${SCRIPT}"
# Write the GPU trace into the data folder so sync.sh pulls it back with the runs.
MEM_LOG="$REMOTE_DATA/${TASK}/gpu_mem_${SESSION}.csv"

# 1. Pull latest code (visible — fails loudly before we launch anything).
ssh "$SERVER" "cd $REMOTE_ROOT && git pull"

# 2. Build the tmux payload: start the GPU monitor in the background (host-side,
#    watching GPU $GPU, every 5 s), run the container in the foreground, then stop
#    the monitor — matched by its unique log-file path — when the sim exits.
MON="mkdir -p $REMOTE_DATA/$TASK; $REMOTE_ROOT/tasks/monitor_gpu.sh $GPU 5 $MEM_LOG >/dev/null 2>&1 &"
SIM="MY_GPU=$GPU docker compose -f $COMPOSE run --rm gpu-runner python3 $REMOTE_SCRIPT $ARGS"
FIN="pkill -f '$MEM_LOG' 2>/dev/null; echo; echo '=== run finished | gpu log: $MEM_LOG (Ctrl-b d to detach) ==='; exec bash"
RUN="$MON $SIM; $FIN"

ssh "$SERVER" tmux new-session -d -s "$SESSION" "bash -lc \"$RUN\""

echo "launched tmux session: $SESSION"
echo "  attach:   ssh -t $SERVER tmux attach -t $SESSION"
echo "  list:     ssh -t $SERVER tmux ls"
echo "  gpu log:  $SERVER:$MEM_LOG"
