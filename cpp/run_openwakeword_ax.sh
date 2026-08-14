#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${AXERA_TARGET:-AX650}"
AUDIO="${AUDIO:-${ROOT}/audio/openwakeword/alexa_test.wav}"
THRESHOLD="${THRESHOLD:-0.5}"

case "${TARGET}" in
  AX650)
    BINARY="${ROOT}/cpp/bin/openwakeword_ax650"
    MODELS_DIR="${MODELS_DIR:-${ROOT}/models/650}"
    BSP_LIB_DIR="${AXERA_BSP_LIB_DIR:-${AX650_BSP_LIB_DIR:-${ROOT}/cpp/toolchains/ax650n_bsp_sdk/msp/out/lib}}"
    BUILD_SCRIPT="cpp/build_ax650.sh"
    ;;
  AX630C)
    BINARY="${ROOT}/cpp/bin/openwakeword_ax630c"
    MODELS_DIR="${MODELS_DIR:-${ROOT}/models/630C}"
    BSP_LIB_DIR="${AXERA_BSP_LIB_DIR:-${AX630C_BSP_LIB_DIR:-${ROOT}/cpp/toolchains/ax620e_bsp_sdk/msp/out/arm64_glibc/lib}}"
    BUILD_SCRIPT="cpp/build_ax630c.sh"
    ;;
  *)
    echo "ERROR: AXERA_TARGET must be AX650 or AX630C" >&2
    exit 2
    ;;
esac

if [[ ! -x "${BINARY}" ]]; then
  echo "ERROR: ${BINARY} is missing; run bash ${BUILD_SCRIPT} first" >&2
  exit 2
fi
for name in embedding_model alexa_v0.1 hey_jarvis_v0.1 hey_mycroft_v0.1 \
    hey_rhasspy_v0.1 timer_v0.1 weather_v0.1; do
  model="${MODELS_DIR}/openwakeword__${name}.axmodel"
  if [[ ! -f "${model}" ]]; then
    echo "ERROR: ${TARGET} model is missing: ${model}" >&2
    exit 2
  fi
done

export LD_LIBRARY_PATH="${BSP_LIB_DIR}:${LD_LIBRARY_PATH:-}"
exec "${BINARY}" \
  --models-dir "${MODELS_DIR}" \
  --mel-weights "${ROOT}/config/openwakeword_mel_weights.bin" \
  --audio "${AUDIO}" \
  --threshold "${THRESHOLD}"
