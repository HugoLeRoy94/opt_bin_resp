#!/usr/bin/env bash
set -e

SERVER="leroy@10.187.172.7"
REMOTE="/storage/leroy/data"
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$OPT_ROOT/data"

rsync -avz --progress "${SERVER}:${REMOTE}/concentration_vs_family_spread/" "${LOCAL}/concentration_vs_family_spread/"

cd "$OPT_ROOT"
python3 -m src.db backfill "${LOCAL}/concentration_vs_family_spread/runs.db"
