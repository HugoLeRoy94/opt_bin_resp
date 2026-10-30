#!/usr/bin/env bash
# Sync large_R data through the common curation workflow.
set -e

OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "$OPT_ROOT/manage_data.py" sync large_R
