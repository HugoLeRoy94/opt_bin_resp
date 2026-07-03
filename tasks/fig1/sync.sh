#!/usr/bin/env bash
# Sync fig1 data from GPU server and rebuild the run index.
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/fig1/"   "${LOCAL}/fig1/"
rsync -avz --progress "${SERVER}:${REMOTE}/fig1_1/" "${LOCAL}/fig1_1/"
rsync -avz --progress "${SERVER}:${REMOTE}/fig1_2/" "${LOCAL}/fig1_2/"

cd "$OPT_ROOT"
python -m src.db backfill "${LOCAL}/fig1/runs.db"
python -m src.db backfill "${LOCAL}/fig1_1/runs.db"
python -m src.db backfill "${LOCAL}/fig1_2/runs.db"
