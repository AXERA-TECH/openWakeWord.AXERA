#!/usr/bin/env bash
set -euo pipefail

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAINS_DIR="${CPP_DIR}/toolchains"
PLATFORM="${1:-all}"

download_sdk() {
  local directory="$1"
  local archive_name="$2"
  local extracted_name="$3"
  local url="$4"

  if [[ -d "${TOOLCHAINS_DIR}/${directory}" ]]; then
    echo "Found ${TOOLCHAINS_DIR}/${directory}"
    return
  fi
  mkdir -p "${TOOLCHAINS_DIR}"
  wget -c "${url}" -O "${TOOLCHAINS_DIR}/${archive_name}"
  unzip -q "${TOOLCHAINS_DIR}/${archive_name}" -d "${TOOLCHAINS_DIR}"
  mv "${TOOLCHAINS_DIR}/${extracted_name}" "${TOOLCHAINS_DIR}/${directory}"
  rm -f "${TOOLCHAINS_DIR}/${archive_name}"
  echo "Downloaded ${TOOLCHAINS_DIR}/${directory}"
}

case "${PLATFORM}" in
  650|AX650|ax650)
    download_sdk \
      ax650n_bsp_sdk msp_50_3.10.2.zip msp_50_3.10.2 \
      https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_50_3.10.2.zip
    ;;
  630C|AX630C|ax630c)
    download_sdk \
      ax620e_bsp_sdk msp_20e_3.0.0.zip msp_20e_3.0.0 \
      https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_20e_3.0.0.zip
    ;;
  all)
    bash "${BASH_SOURCE[0]}" 650
    bash "${BASH_SOURCE[0]}" 630C
    ;;
  *)
    echo "Usage: bash cpp/download_bsp.sh [650|630C|all]" >&2
    exit 2
    ;;
esac
