#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${AXERA_TARGET:-AX650}"
AUDIO_DIR="${AUDIO_DIR:-${ROOT}/audio/openwakeword}"
THRESHOLD="${THRESHOLD:-0.5}"

case "${TARGET}" in
  AX650)
    TARGET_SLUG="ax650"
    BINARY="${ROOT}/cpp/bin/openwakeword_ax650"
    MODELS_DIR="${MODELS_DIR:-${ROOT}/models/650}"
    BSP_LIB_DIR="${AXERA_BSP_LIB_DIR:-${AX650_BSP_LIB_DIR:-${ROOT}/cpp/toolchains/ax650n_bsp_sdk/msp/out/lib}}"
    BUILD_SCRIPT="cpp/build_ax650.sh"
    ;;
  AX630C)
    TARGET_SLUG="ax630c"
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

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/cpp_${TARGET_SLUG}_batch}"
if [[ ! -x "${BINARY}" ]]; then
  echo "ERROR: ${BINARY} is missing; run bash ${BUILD_SCRIPT} first" >&2
  exit 2
fi
if [[ ! -d "${AUDIO_DIR}" ]]; then
  echo "ERROR: audio directory does not exist: ${AUDIO_DIR}" >&2
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

mapfile -d '' AUDIO_FILES < <(
  find "${AUDIO_DIR}" -maxdepth 1 -type f -iname '*.wav' -print0 | sort -z
)
if [[ "${#AUDIO_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no WAV files found under ${AUDIO_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
RESULTS="${OUTPUT_DIR}/results.tsv"
SUMMARY="${OUTPUT_DIR}/summary.txt"
printf 'file\taudio_seconds\tfeature_seconds\tnpu_seconds\tinference_seconds\trtf\tmodel_load_seconds\tdetections\n' > "${RESULTS}"

export LD_LIBRARY_PATH="${BSP_LIB_DIR}:${LD_LIBRARY_PATH:-}"
for audio in "${AUDIO_FILES[@]}"; do
  name="$(basename "${audio}")"
  stem="${name%.*}"
  log="${OUTPUT_DIR}/logs/${stem}.log"
  echo "[RUN] target=${TARGET} ${name} threshold=${THRESHOLD}"
  "${BINARY}" \
    --models-dir "${MODELS_DIR}" \
    --mel-weights "${ROOT}/config/openwakeword_mel_weights.bin" \
    --audio "${audio}" \
    --threshold "${THRESHOLD}" > "${log}" 2>&1
  cat "${log}"

  field() {
    awk -F': ' -v key="$1" '$1 == key {print $2; exit}' "${log}"
  }
  audio_seconds="$(field audio_seconds)"
  feature_seconds="$(field feature_seconds)"
  npu_seconds="$(field npu_seconds)"
  inference_seconds="$(field inference_seconds)"
  rtf="$(field rtf)"
  model_load_seconds="$(field model_load_seconds)"
  detections="$(awk '$NF == "WAKEUP" {print $1}' "${log}" | paste -sd ',' -)"
  detections="${detections:-none}"

  for value in "${audio_seconds}" "${feature_seconds}" "${npu_seconds}" \
      "${inference_seconds}" "${rtf}" "${model_load_seconds}"; do
    if [[ -z "${value}" ]]; then
      echo "ERROR: timing field missing from ${log}" >&2
      exit 1
    fi
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${audio_seconds}" "${feature_seconds}" "${npu_seconds}" \
    "${inference_seconds}" "${rtf}" "${model_load_seconds}" "${detections}" \
    >> "${RESULTS}"
done

awk -F'\t' '
  NR > 1 {
    files += 1
    audio += $2
    feature += $3
    npu += $4
    inference += $5
    model_load += $7
  }
  END {
    printf "files: %d\n", files
    printf "total_audio_seconds: %.6f\n", audio
    printf "total_feature_seconds: %.6f\n", feature
    printf "total_npu_seconds: %.6f\n", npu
    printf "total_inference_seconds: %.6f\n", inference
    printf "total_model_load_seconds: %.6f\n", model_load
    printf "rtf: %.6f\n", inference / audio
  }
' "${RESULTS}" > "${SUMMARY}"

echo "[SUMMARY]"
cat "${SUMMARY}"
echo "results: ${RESULTS}"
echo "logs: ${OUTPUT_DIR}/logs"
