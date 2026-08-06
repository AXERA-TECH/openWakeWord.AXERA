#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PULSAR_ROOT="${ROOT_DIR}/model_convert/pulsar2/check0"
MANIFEST="${PULSAR_ROOT}/manifest.json"
RESULTS="${PULSAR_ROOT}/check0_results.json"

export WORK_TMP="${ROOT_DIR}/.work_tmp"
export TMPDIR="${WORK_TMP}/tmp"
export MPLCONFIGDIR="${WORK_TMP}/matplotlib"
export XDG_CACHE_HOME="${WORK_TMP}/xdg_cache"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if ! command -v pulsar2 >/dev/null 2>&1; then
    echo "pulsar2 is unavailable; source the target machine's npu_dev first" >&2
    exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing ${MANIFEST}; run make_pulsar2_configs.py first" >&2
    exit 2
fi

mkdir -p "${PULSAR_ROOT}/axmodels" "${PULSAR_ROOT}/logs"
while IFS=$'\t' read -r key config shapes build_dir output_name axmodel log; do
    if [[ -f "${axmodel}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
        echo "=== skip ${key}: ${axmodel} already exists ==="
        continue
    fi
    echo "=== pulsar2 check0: ${key} ==="
    echo "input_shapes=${shapes}"
    mkdir -p "${build_dir}" "$(dirname "${log}")" "$(dirname "${axmodel}")"
    pulsar2 build --config "${config}" --input_shapes "${shapes}" 2>&1 | tee "${log}"
    test -s "${build_dir}/${output_name}"
    cp "${build_dir}/${output_name}" "${axmodel}"
done < <(
    python - "${MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
selected = {
    value.strip()
    for value in os.environ.get("BUILD_KEYS", "").split(",")
    if value.strip()
}
for row in manifest["models"]:
    if selected and row["key"] not in selected:
        continue
    print("\t".join(
        str(row[name])
        for name in ("key", "config", "input_shapes", "build_dir", "output_name", "axmodel", "log")
    ))
PY
)

python - "${MANIFEST}" "${PULSAR_ROOT}/axmodels" <<'PY'
import json
import shutil
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
output_dir = Path(sys.argv[2])
for alias, source in manifest["aliases_after_build"].items():
    source_path = output_dir / source
    alias_path = output_dir / alias
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    shutil.copy2(source_path, alias_path)
PY

python - "${MANIFEST}" "${RESULTS}" <<'PY'
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
results_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
aliases = manifest["aliases_after_build"]
error_pattern = re.compile(
    r"traceback|codeexception|\berror:|\bfailed\b|\bexception\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


records = []
for row in manifest["models"]:
    axmodel = Path(row["axmodel"])
    log = Path(row["log"])
    log_text = log.read_text(errors="replace") if log.exists() else ""
    error_markers = sorted({match.group(0) for match in error_pattern.finditer(log_text)})
    passed = axmodel.is_file() and axmodel.stat().st_size > 0 and log.is_file() and not error_markers
    records.append(
        {
            "name": axmodel.name,
            "kind": "build",
            "path": str(axmodel),
            "size_bytes": axmodel.stat().st_size if axmodel.exists() else 0,
            "sha256": sha256(axmodel) if axmodel.is_file() else None,
            "log": str(log),
            "log_error_markers": error_markers,
            "check0_passed": passed,
        }
    )

by_name = {record["name"]: record for record in records}
for alias_name, source_name in aliases.items():
    alias = Path(next(iter(by_name.values()))["path"]).parent / alias_name
    source = by_name[source_name]
    alias_hash = sha256(alias) if alias.is_file() else None
    passed = (
        source["check0_passed"]
        and alias.is_file()
        and alias.stat().st_size > 0
        and alias_hash == source["sha256"]
    )
    records.append(
        {
            "name": alias.name,
            "kind": "alias",
            "source": source_name,
            "path": str(alias),
            "size_bytes": alias.stat().st_size if alias.exists() else 0,
            "sha256": alias_hash,
            "log": source["log"],
            "log_error_markers": source["log_error_markers"],
            "check0_passed": passed,
        }
    )

records.sort(key=lambda record: record["name"])
results = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "manifest": str(manifest_path),
    "target_hardware": manifest["target_hardware"],
    "npu_mode": manifest["npu_mode"],
    "compiler_check": manifest["compiler_check"],
    "build_count": len(manifest["models"]),
    "axmodel_count": len(records),
    "all_passed": all(record["check0_passed"] for record in records),
    "models": records,
}
results_path.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n")
if not results["all_passed"]:
    raise SystemExit(f"check0 result audit failed; inspect {results_path}")
print(f"wrote {results_path} with {len(records)} passing axmodels")
PY

find "${PULSAR_ROOT}/axmodels" -maxdepth 1 -type f -name '*.axmodel' -print | sort
