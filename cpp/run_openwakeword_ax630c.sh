#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXERA_TARGET=AX630C exec bash "${ROOT}/cpp/run_openwakeword_ax.sh"
