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
# Write the GPU trace + the run's stdout/stderr into the data folder so sync.sh
# pulls them back with the runs (the tmux pane is discarded when the session ends).
MEM_LOG="$REMOTE_DATA/${TASK}/gpu_mem_${SESSION}.csv"
RUN_LOG="$REMOTE_DATA/${TASK}/run_${SESSION}.log"

# 1. Pull latest code (visible — fails loudly before we launch anything).
ssh "$SERVER" "cd $REMOTE_ROOT && git pull"

# 2. Build the tmux payload: start the GPU monitor in the background (host-side,
#    watching GPU $GPU, every 5 s), run the container in the foreground, then stop
#    the monitor — matched by its unique log-file path — when the sim exits.
# -T disables the container TTY so stdout/stderr can be piped to tee (kept live in
# the tmux pane AND appended to $RUN_LOG).
MON="mkdir -p $REMOTE_DATA/$TASK; $REMOTE_ROOT/tasks/monitor_gpu.sh $GPU 5 $MEM_LOG >/dev/null 2>&1 &"
# python3 -u: without a TTY, Python block-buffers stdout, so prints never reach the
# pane/log until the buffer fills — -u forces unbuffered so output streams live.
SIM="MY_GPU=$GPU docker compose -f $COMPOSE run --rm -T gpu-runner python3 -u $REMOTE_SCRIPT $ARGS 2>&1 | tee -a $RUN_LOG"
# On exit: finalize_run.sh stops the GPU monitor (by its recorded PID) and moves the
# gpu trace + run log into the sweep folder. NO pattern-based pkill here — it would
# also match this tmux session's own shell (its command line contains $MEM_LOG) and
# kill the session, discarding any crash output. `exec bash` keeps the pane open so a
# crashed run's error stays visible for inspection.
FIN="$REMOTE_ROOT/tasks/finalize_run.sh '$RUN_LOG' '$MEM_LOG' '$REMOTE_DATA'; echo; echo '=== run finished (Ctrl-b d to detach) ==='; exec bash"
RUN="$MON $SIM; $FIN"

ssh "$SERVER" tmux new-session -d -s "$SESSION" "bash -lc \"$RUN\""

echo "launched tmux session: $SESSION"
echo "  attach:   ssh -t $SERVER tmux attach -t $SESSION"
echo "  list:     ssh -t $SERVER tmux ls"
# Kill ONE run with kill-session — NEVER 'tmux kill-server', which nukes every session
# (interactive + all sweeps) on the shared server at once.
echo "  kill:     ssh -t $SERVER tmux kill-session -t $SESSION"
echo "  run log:  $SERVER:$RUN_LOG"
echo "  gpu log:  $SERVER:$MEM_LOG"
