#!/usr/bin/env bash
# Pull profiling artifacts (torch trace.json / torch_table.txt / cprofile.prof) back
# from the GPU server. No DB backfill — these are profiler outputs, not science runs.
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/profiling/" "${LOCAL}/profiling/"
