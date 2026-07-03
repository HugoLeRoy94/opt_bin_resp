#!/usr/bin/env bash
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/sweepD5/"  "${LOCAL}/sweepD5/"
rsync -avz --progress "${SERVER}:${REMOTE}/sweepD10/" "${LOCAL}/sweepD10/"

cd "$OPT_ROOT"
python -m src.db backfill "${LOCAL}/sweepD5/runs.db"
python -m src.db backfill "${LOCAL}/sweepD10/runs.db"
