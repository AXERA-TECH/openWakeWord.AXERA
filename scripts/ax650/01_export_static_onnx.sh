#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python "${ROOT_DIR}/python/export_static.py"
python "${ROOT_DIR}/python/export_mel_weights.py"
