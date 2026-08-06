#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tarfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[1]
OPEN_SOURCE_DIR = ROOT / "weights/openwakeword"
SHERPA_SOURCE_DIR = ROOT / "weights/sherpa-unused"
STATIC_ROOT = ROOT / "model_convert/static_onnx"
CALIB_ROOT = ROOT / "model_convert/calib_data"
PULSAR_ROOT = ROOT / "model_convert/pulsar2/check0"

OPEN_CORE_MODELS = {
    "alexa_v0.1.onnx",
    "embedding_model.onnx",
    "hey_jarvis_v0.1.onnx",
    "hey_mycroft_v0.1.onnx",
    "hey_rhasspy_v0.1.onnx",
    "melspectrogram.onnx",
    "timer_v0.1.onnx",
    "weather_v0.1.onnx",
}

SHERPA_QUANT_MODELS: set[str] = set()


def model_key(group: str, path: Path) -> str:
    return f"{group}__{path.stem}"


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elem_type_to_numpy(elem_type: int) -> np.dtype:
    mapping = {
        onnx.TensorProto.FLOAT: np.dtype(np.float32),
        onnx.TensorProto.DOUBLE: np.dtype(np.float64),
        onnx.TensorProto.FLOAT16: np.dtype(np.float16),
        onnx.TensorProto.INT64: np.dtype(np.int64),
        onnx.TensorProto.INT32: np.dtype(np.int32),
        onnx.TensorProto.INT16: np.dtype(np.int16),
        onnx.TensorProto.INT8: np.dtype(np.int8),
        onnx.TensorProto.UINT64: np.dtype(np.uint64),
        onnx.TensorProto.UINT32: np.dtype(np.uint32),
        onnx.TensorProto.UINT16: np.dtype(np.uint16),
        onnx.TensorProto.UINT8: np.dtype(np.uint8),
        onnx.TensorProto.BOOL: np.dtype(np.bool_),
    }
    if elem_type not in mapping:
        raise ValueError(f"unsupported ONNX tensor type: {elem_type}")
    return mapping[elem_type]


def value_info_shape(value_info: Any) -> list[int]:
    result = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value <= 0:
            raise ValueError(f"dynamic dimension remains in {value_info.name}: {dim}")
        result.append(int(dim.dim_value))
    return result


def fixed_input_shapes(path: Path, model: onnx.ModelProto) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for value_info in model.graph.input:
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            shape.append(int(dim.dim_value) if dim.dim_value > 0 else 1)
        result[value_info.name] = shape

    if path.name == "melspectrogram.onnx":
        return {"input": [1, 1760]}
    if path.name == "silero_vad.onnx":
        return {"input": [1, 480], "sr": [], "h": [2, 1, 64], "c": [2, 1, 64]}
    return result


def output_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    if ref.shape != got.shape:
        return {
            "shape_match": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(got.shape),
        }

    if not np.issubdtype(ref.dtype, np.floating):
        return {
            "shape_match": True,
            "exact": bool(np.array_equal(ref, got)),
            "reference_shape": list(ref.shape),
        }

    a = ref.astype(np.float64, copy=False).reshape(-1)
    b = got.astype(np.float64, copy=False).reshape(-1)
    diff = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        cosine = 1.0 if np.array_equal(a, b) else 0.0
    else:
        cosine = float(np.dot(a, b) / denom)
    return {
        "shape_match": True,
        "reference_shape": list(ref.shape),
        "cosine": cosine,
        "mse": float(np.mean(diff * diff)) if diff.size else 0.0,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def openwakeword_mel_host_postprocess(value: np.ndarray) -> np.ndarray:
    """Apply the original log10 and 80 dB clipping after the no-Log mel graph."""
    clipped = np.asarray(value, dtype=np.float32)
    decibels = np.log(clipped).astype(np.float32)
    decibels = decibels * np.float32(10.0)
    decibels = decibels / np.float32(2.3025851249694824)
    minimum = np.max(decibels).astype(np.float32) - np.float32(80.0)
    return np.maximum(decibels, minimum).astype(np.float32)


def metrics_pass(metrics: dict[str, Any], cosine_threshold: float = 0.99) -> bool:
    if not metrics.get("shape_match", False):
        return False
    if "exact" in metrics:
        return bool(metrics["exact"])
    return bool(metrics.get("cosine", 0.0) >= cosine_threshold)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"expected mono 16-bit PCM WAV: {path}")
        sample_rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).copy()
    return samples, sample_rate


def select_evenly(values: list[Any], count: int) -> list[Any]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return [values[int(index)] for index in indices]


def write_feed_archives(
    output_dir: Path,
    feeds: dict[str, list[dict[str, np.ndarray]]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for key, records in sorted(feeds.items()):
        if not records:
            raise ValueError(f"no calibration feeds recorded for {key}")
        input_names = list(records[0])
        for record in records:
            if list(record) != input_names:
                raise ValueError(f"inconsistent input order for {key}")
        for input_name in input_names:
            safe = safe_name(input_name)
            input_dir = output_dir / key / safe
            input_dir.mkdir(parents=True, exist_ok=True)
            tar_path = output_dir / key / f"{safe}.tar.gz"
            entries = []
            with tarfile.open(tar_path, "w:gz") as archive:
                for index, record in enumerate(records):
                    value = np.asarray(record[input_name])
                    npy_path = input_dir / f"{index:05d}.npy"
                    np.save(npy_path, value)
                    archive.add(npy_path, arcname=f"{safe}/{npy_path.name}")
                    entries.append(
                        {
                            "file": str(npy_path.relative_to(output_dir)),
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                        }
                    )
            manifest.setdefault(key, {})[input_name] = {
                "tar": str(tar_path.relative_to(output_dir)),
                "count": len(records),
                "entries": entries,
            }
    return manifest


def model_input_description(path: Path) -> dict[str, dict[str, Any]]:
    model = onnx.load(path)
    result = {}
    initializers = {value.name for value in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializers:
            continue
        result[value_info.name] = {
            "shape": value_info_shape(value_info),
            "dtype": onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type),
        }
    return result


def iter_model_entries() -> Iterable[tuple[str, Path, Path]]:
    sources = sorted(OPEN_SOURCE_DIR.glob("*.onnx"))
    if not sources:
        raise FileNotFoundError(f"no ONNX models found under {OPEN_SOURCE_DIR}")
    for source in sources:
        yield "openwakeword", source, STATIC_ROOT / "openwakeword" / source.name


def source_for_key(key: str) -> Path:
    group, stem = key.split("__", 1)
    root = OPEN_SOURCE_DIR if group == "openwakeword" else SHERPA_SOURCE_DIR
    return root / f"{stem}.onnx"


def static_for_key(key: str) -> Path:
    group, stem = key.split("__", 1)
    return STATIC_ROOT / group / f"{stem}.onnx"
