#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXERA_TARGET=AX650 exec bash "${ROOT}/cpp/run_batch_openwakeword_ax.sh"
