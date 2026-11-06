#!/usr/bin/env bash
# Sync two_families data through the common curation workflow.
set -e

OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "$OPT_ROOT/manage_data.py" sync two_families
