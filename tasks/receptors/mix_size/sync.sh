#!/usr/bin/env bash
set -e

OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 "$OPT_ROOT/manage_data.py" sync mix_size
