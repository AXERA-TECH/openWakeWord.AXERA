#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROOT_DIR}/python/make_pulsar2_configs.py"
bash "${ROOT_DIR}/scripts/ax650/build_check0.sh"
