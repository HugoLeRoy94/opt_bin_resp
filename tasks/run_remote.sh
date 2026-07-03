#!/usr/bin/env bash
# Launch a task script on the cluster inside a detached tmux session.
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
set -e

SERVER="leroy@10.187.172.7"
REMOTE_ROOT="/home/leroy/opt_bin_resp"
COMPOSE="$REMOTE_ROOT/docker-compose.server.yaml"

TASK="$1"; SCRIPT="$2"; GPU="${3:-0}"
# drop task/script/gpu; anything after (incl. a leading --) is script args
shift $(( $# < 3 ? $# : 3 ))
[ "$1" = "--" ] && shift
ARGS="$*"

[ -z "$TASK" ] || [ -z "$SCRIPT" ] && { echo "usage: $0 <task> <script.py> [gpu] [-- args]"; exit 1; }

SESSION="${TASK}_${SCRIPT%.py}_$(date +%H%M%S)"
REMOTE_SCRIPT="/app/tasks/${TASK}/scripts/${SCRIPT}"

# 1. Pull latest code (visible — fails loudly before we launch anything).
ssh "$SERVER" "cd $REMOTE_ROOT && git pull"

# 2. Launch the container in the foreground of a detached tmux session.
ssh "$SERVER" tmux new-session -d -s "$SESSION" \
  "bash -lc \"MY_GPU=$GPU docker compose -f $COMPOSE run --rm gpu-runner python3 $REMOTE_SCRIPT $ARGS; echo; echo '=== run finished (Ctrl-b d to detach, exit to close) ==='; exec bash\""

echo "launched tmux session: $SESSION"
echo "  attach:  ssh -t $SERVER tmux attach -t $SESSION"
echo "  list:    ssh -t $SERVER tmux ls"
