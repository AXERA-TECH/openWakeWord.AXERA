#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from common import (
    OPEN_SOURCE_DIR,
    ROOT,
    SHERPA_SOURCE_DIR,
    STATIC_ROOT,
    openwakeword_mel_host_postprocess,
    output_metrics,
    read_wav,
    write_json,
)


OPEN_CLASSIFIERS = (
    "alexa_v0.1.onnx",
    "hey_jarvis_v0.1.onnx",
    "hey_mycroft_v0.1.onnx",
    "hey_rhasspy_v0.1.onnx",
    "timer_v0.1.onnx",
    "weather_v0.1.onnx",
)
OPEN_TEST_AUDIO = (
    ROOT / "tests/audio/alexa_test.wav",
    ROOT / "tests/audio/hey_mycroft_test.wav",
    ROOT / "tests/audio/hey_jane.wav",
)
OPEN_EXPECTED_POSITIVES = {
    "alexa_test.wav": "alexa_v0.1.onnx",
    "hey_mycroft_test.wav": "hey_mycroft_v0.1.onnx",
}


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def update_metric_summary(summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    if not metrics.get("shape_match", False):
        summary["shape_match"] = False
        return
    summary["comparisons"] += 1
    summary["minimum_cosine"] = min(
        summary["minimum_cosine"], float(metrics.get("cosine", 1.0))
    )
    summary["maximum_mse"] = max(summary["maximum_mse"], float(metrics.get("mse", 0.0)))
    summary["maximum_abs"] = max(
        summary["maximum_abs"], float(metrics.get("max_abs", 0.0))
    )


def new_metric_summary() -> dict[str, Any]:
    return {
        "shape_match": True,
        "comparisons": 0,
        "minimum_cosine": 1.0,
        "maximum_mse": 0.0,
        "maximum_abs": 0.0,
    }


def pad_chunks(samples: np.ndarray, chunk_size: int) -> np.ndarray:
    remainder = samples.size % chunk_size
    if remainder == 0:
        return samples
    return np.pad(samples, (0, chunk_size - remainder))


def run_openwakeword_clip(
    audio_path: Path,
    source_sessions: dict[str, ort.InferenceSession],
    static_sessions: dict[str, ort.InferenceSession],
) -> dict[str, Any]:
    samples, sample_rate = read_wav(audio_path)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio: {audio_path} has {sample_rate}")

    samples = pad_chunks(samples, 1280)
    history = np.zeros(480, dtype=np.int16)
    source_mel_buffer = np.ones((76, 32), dtype=np.float32)
    static_mel_buffer = np.ones((76, 32), dtype=np.float32)
    source_feature_buffer = np.zeros((34, 96), dtype=np.float32)
    static_feature_buffer = np.zeros((34, 96), dtype=np.float32)

    stage_metrics = {
        "melspectrogram": new_metric_summary(),
        "embedding": new_metric_summary(),
        "classifiers": new_metric_summary(),
    }
    source_scores = {name: [] for name in OPEN_CLASSIFIERS}
    static_scores = {name: [] for name in OPEN_CLASSIFIERS}

    source_mel = source_sessions["melspectrogram.onnx"]
    static_mel = static_sessions["melspectrogram.onnx"]
    source_embedding = source_sessions["embedding_model.onnx"]
    static_embedding = static_sessions["embedding_model.onnx"]

    for start in range(0, samples.size, 1280):
        chunk = samples[start : start + 1280]
        mel_input = np.concatenate((history, chunk)).astype(np.float32)[None, :]
        history = np.concatenate((history, chunk))[-480:].astype(np.int16, copy=False)

        source_spec_raw = source_mel.run(None, {source_mel.get_inputs()[0].name: mel_input})[0]
        static_spec_raw = static_mel.run(None, {static_mel.get_inputs()[0].name: mel_input})[0]
        static_spec_raw = openwakeword_mel_host_postprocess(static_spec_raw)
        update_metric_summary(
            stage_metrics["melspectrogram"], output_metrics(source_spec_raw, static_spec_raw)
        )

        source_spec = np.squeeze(source_spec_raw).astype(np.float32) / 10.0 + 2.0
        static_spec = np.squeeze(static_spec_raw).astype(np.float32) / 10.0 + 2.0
        source_mel_buffer = np.vstack((source_mel_buffer, source_spec))[-970:]
        static_mel_buffer = np.vstack((static_mel_buffer, static_spec))[-970:]

        source_embedding_input = source_mel_buffer[-76:][None, :, :, None].astype(np.float32)
        static_embedding_input = static_mel_buffer[-76:][None, :, :, None].astype(np.float32)
        source_embedding_raw = source_embedding.run(
            None, {source_embedding.get_inputs()[0].name: source_embedding_input}
        )[0]
        static_embedding_raw = static_embedding.run(
            None, {static_embedding.get_inputs()[0].name: static_embedding_input}
        )[0]
        update_metric_summary(
            stage_metrics["embedding"],
            output_metrics(source_embedding_raw, static_embedding_raw),
        )

        source_feature = np.asarray(source_embedding_raw).reshape(-1, 96)[-1]
        static_feature = np.asarray(static_embedding_raw).reshape(-1, 96)[-1]
        source_feature_buffer = np.vstack((source_feature_buffer, source_feature))[-120:]
        static_feature_buffer = np.vstack((static_feature_buffer, static_feature))[-120:]

        for model_name in OPEN_CLASSIFIERS:
            source_classifier = source_sessions[model_name]
            static_classifier = static_sessions[model_name]
            frame_count = int(static_classifier.get_inputs()[0].shape[1])
            source_input = source_feature_buffer[-frame_count:][None, :, :].astype(np.float32)
            static_input = static_feature_buffer[-frame_count:][None, :, :].astype(np.float32)
            source_output = source_classifier.run(
                None, {source_classifier.get_inputs()[0].name: source_input}
            )[0]
            static_output = static_classifier.run(
                None, {static_classifier.get_inputs()[0].name: static_input}
            )[0]
            update_metric_summary(
                stage_metrics["classifiers"], output_metrics(source_output, static_output)
            )
            source_scores[model_name].append(np.asarray(source_output).reshape(-1).tolist())
            static_scores[model_name].append(np.asarray(static_output).reshape(-1).tolist())

    source_max_scores = {
        name: np.max(np.asarray(values, dtype=np.float64), axis=0).tolist()
        for name, values in source_scores.items()
    }
    static_max_scores = {
        name: np.max(np.asarray(values, dtype=np.float64), axis=0).tolist()
        for name, values in static_scores.items()
    }
    expected_model = OPEN_EXPECTED_POSITIVES.get(audio_path.name)
    expected_score = None
    expected_detected = None
    if expected_model is not None:
        expected_score = float(np.max(static_max_scores[expected_model]))
        expected_detected = expected_score >= 0.5

    numerical_match = all(
        value["shape_match"]
        and value["maximum_abs"] <= 1e-4
        and (
            value["minimum_cosine"] >= 0.99999
            or value["maximum_abs"] <= 1e-5
        )
        for value in stage_metrics.values()
    )
    return {
        "audio": str(audio_path.relative_to(ROOT)),
        "sample_rate": sample_rate,
        "sample_count": int(samples.size),
        "frame_count": int(samples.size // 1280),
        "initial_context": "480 zero PCM samples for fixed [1,1760] mel input",
        "stage_metrics": stage_metrics,
        "source_max_scores": source_max_scores,
        "static_max_scores": static_max_scores,
        "expected_model": expected_model,
        "expected_score": expected_score,
        "expected_detected": expected_detected,
        "numerical_match": numerical_match,
        "passed": numerical_match and expected_detected is not False,
    }


def validate_openwakeword() -> dict[str, Any]:
    model_names = ("melspectrogram.onnx", "embedding_model.onnx") + OPEN_CLASSIFIERS
    source_sessions = {
        name: create_session(OPEN_SOURCE_DIR / name) for name in model_names
    }
    static_sessions = {
        name: create_session(STATIC_ROOT / "openwakeword" / name) for name in model_names
    }
    clips = [
        run_openwakeword_clip(path, source_sessions, static_sessions)
        for path in OPEN_TEST_AUDIO
    ]
    return {
        "pipeline": "fixed 80 ms streaming chunks with 30 ms PCM history",
        "clips": clips,
        "passed": all(clip["passed"] for clip in clips),
    }


def read_float_wave(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = read_wav(path)
    return samples.astype(np.float32) / 32768.0, sample_rate


def run_sherpa_audio(kws: Any, audio_path: Path) -> dict[str, Any]:
    samples, sample_rate = read_float_wave(audio_path)
    stream = kws.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.accept_waveform(
        sample_rate, np.zeros(int(0.8 * sample_rate), dtype=np.float32)
    )
    stream.input_finished()

    detections = []
    decode_calls = 0
    while kws.is_ready(stream):
        kws.decode_stream(stream)
        decode_calls += 1
        if decode_calls > 10000:
            raise RuntimeError(f"decode loop did not terminate for {audio_path}")
        result = kws.get_result(stream)
        if result:
            detections.append(str(result))
            kws.reset_stream(stream)

    return {
        "audio": str(audio_path.relative_to(ROOT)),
        "sample_rate": sample_rate,
        "sample_count": int(samples.size),
        "decode_calls": decode_calls,
        "detections": detections,
    }


def create_sherpa_spotter(model_root: Path, chunk_size: int) -> Any:
    import sherpa_onnx

    suffix = f"epoch-13-avg-2-chunk-{chunk_size}-left-64.onnx"
    return sherpa_onnx.KeywordSpotter(
        tokens=str(SHERPA_SOURCE_DIR / "tokens.txt"),
        encoder=str(model_root / f"encoder-{suffix}"),
        decoder=str(model_root / f"decoder-{suffix}"),
        joiner=str(model_root / f"joiner-{suffix}"),
        keywords_file=str(SHERPA_SOURCE_DIR / "test_wavs/keywords.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        max_active_paths=1,
        provider="cpu",
    )


def validate_sherpa_variant(chunk_size: int) -> dict[str, Any]:
    audio_files = sorted((SHERPA_SOURCE_DIR / "test_wavs").glob("*.wav"))
    source_spotter = create_sherpa_spotter(SHERPA_SOURCE_DIR, chunk_size)
    static_spotter = create_sherpa_spotter(STATIC_ROOT / "sherpa", chunk_size)

    source_results = [run_sherpa_audio(source_spotter, path) for path in audio_files]
    static_results = [run_sherpa_audio(static_spotter, path) for path in audio_files]
    comparisons = []
    for source, static in zip(source_results, static_results):
        same = source["detections"] == static["detections"]
        comparisons.append(
            {
                "audio": source["audio"],
                "source_detections": source["detections"],
                "static_detections": static["detections"],
                "source_decode_calls": source["decode_calls"],
                "static_decode_calls": static["decode_calls"],
                "match": same,
            }
        )

    return {
        "chunk_size": chunk_size,
        "max_active_paths": 1,
        "audio_count": len(audio_files),
        "comparisons": comparisons,
        "detected_audio_count": sum(bool(row["static_detections"]) for row in comparisons),
        "passed": all(row["match"] for row in comparisons),
    }


def validate_sherpa() -> dict[str, Any]:
    import sherpa_onnx

    variants = [validate_sherpa_variant(chunk_size) for chunk_size in (8, 16)]
    return {
        "sherpa_onnx_version": sherpa_onnx.__version__,
        "variants": variants,
        "passed": all(variant["passed"] for variant in variants),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model_convert/local_inference/results.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    openwakeword = validate_openwakeword()
    result = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
        "openwakeword": openwakeword,
        "elapsed_seconds": time.perf_counter() - started,
        "all_passed": openwakeword["passed"],
    }
    write_json(args.output, result)
    print(f"wrote {args.output}")
    print(f"openwakeword_passed={openwakeword['passed']}")
    print(f"all_passed={result['all_passed']}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
