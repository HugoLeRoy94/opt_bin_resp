#!/usr/bin/env bash
# Use the shared cluster/local curation and synchronization workflow.
set -e
OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "$OPT_ROOT/manage_data.py" sync gene_expression
