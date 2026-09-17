#!/usr/bin/env bash
# Sync Figure 1 data through the common curation workflow.
set -e

OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "$OPT_ROOT/manage_data.py" sync fig1 fig1_1 fig1_2
