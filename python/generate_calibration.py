#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import (
    CALIB_ROOT,
    OPEN_CORE_MODELS,
    OPEN_SOURCE_DIR,
    ROOT,
    SHERPA_QUANT_MODELS,
    SHERPA_SOURCE_DIR,
    STATIC_ROOT,
    metrics_pass,
    model_key,
    openwakeword_mel_host_postprocess,
    output_metrics,
    read_wav,
    select_evenly,
    source_for_key,
    static_for_key,
    write_feed_archives,
    write_json,
)


PROVIDERS = ["CPUExecutionProvider"]


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=PROVIDERS)


def openwakeword_audio_chunks(samples: np.ndarray) -> list[np.ndarray]:
    chunks = []
    overlap = 480
    step = 1280
    for position in range(0, len(samples), step):
        chunk = np.zeros(overlap + step, dtype=np.float32)
        source_start = max(0, position - overlap)
        source_end = min(len(samples), position + step)
        target_start = source_start - (position - overlap)
        chunk[target_start : target_start + source_end - source_start] = samples[
            source_start:source_end
        ]
        chunks.append(chunk[None, :])
    return chunks


def select_with_high_responses(
    values: list[np.ndarray], responses: list[float], count: int
) -> list[np.ndarray]:
    if len(values) <= count:
        return values

    priority_count = min(len(values), max(1, count // 4))
    priority_indices = np.argsort(np.asarray(responses))[-priority_count:][::-1]
    even_indices = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    selected = []
    seen = set()
    for index in np.concatenate((priority_indices, even_indices)):
        integer_index = int(index)
        if integer_index in seen:
            continue
        selected.append(values[integer_index])
        seen.add(integer_index)
        if len(selected) == count:
            break
    return selected


def generate_openwakeword(
    feeds: dict[str, list[dict[str, np.ndarray]]],
    audio_paths: list[Path],
    max_samples: int,
) -> None:
    melspec_path = STATIC_ROOT / "openwakeword/melspectrogram.onnx"
    embedding_path = STATIC_ROOT / "openwakeword/embedding_model.onnx"
    melspec_session = session(melspec_path)
    embedding_session = session(embedding_path)
    classifier_sessions = {}
    for name in sorted(OPEN_CORE_MODELS):
        if name in {"melspectrogram.onnx", "embedding_model.onnx"}:
            continue
        classifier_sessions[name] = session(STATIC_ROOT / "openwakeword" / name)

    all_chunks: list[np.ndarray] = []
    embedding_inputs: list[np.ndarray] = []
    classifier_inputs: dict[str, list[np.ndarray]] = defaultdict(list)
    classifier_responses: dict[str, list[float]] = defaultdict(list)
    for path in audio_paths:
        samples, sample_rate = read_wav(path)
        if sample_rate != 16000:
            raise ValueError(f"openWakeWord calibration expects 16 kHz audio: {path}")

        mel_buffer = np.ones((76, 32), dtype=np.float32)
        feature_buffer = np.zeros((34, 96), dtype=np.float32)
        for mel_input in openwakeword_audio_chunks(samples):
            all_chunks.append(mel_input)
            mel_output = np.squeeze(
                openwakeword_mel_host_postprocess(
                    melspec_session.run(None, {"input": mel_input})[0]
                )
            )
            if mel_output.ndim != 2 or mel_output.shape[1] != 32:
                raise RuntimeError(f"unexpected openWakeWord mel shape: {mel_output.shape}")
            mel_buffer = np.vstack((mel_buffer, mel_output / 10.0 + 2.0))[-970:]

            embedding_input = mel_buffer[-76:][None, :, :, None].astype(np.float32)
            embedding_inputs.append(embedding_input)
            embedding_output = embedding_session.run(
                None, {"input_1": embedding_input}
            )[0]
            feature = np.asarray(embedding_output).reshape(-1, 96)[-1]
            feature_buffer = np.vstack((feature_buffer, feature))[-120:]

            for name, classifier in classifier_sessions.items():
                input_meta = classifier.get_inputs()[0]
                sequence_length = int(input_meta.shape[1])
                classifier_input = feature_buffer[-sequence_length:][None, :, :].astype(
                    np.float32
                )
                classifier_output = classifier.run(
                    None, {input_meta.name: classifier_input}
                )[0]
                classifier_inputs[name].append(classifier_input)
                classifier_responses[name].append(
                    float(np.max(np.asarray(classifier_output)))
                )

    recorded_chunks = select_evenly(all_chunks, max_samples)
    melspec_key = model_key("openwakeword", melspec_path)
    feeds[melspec_key] = [{"input": value} for value in recorded_chunks]

    embedding_key = model_key("openwakeword", embedding_path)
    feeds[embedding_key] = [
        {"input_1": value} for value in select_evenly(embedding_inputs, max_samples)
    ]

    for name, classifier in classifier_sessions.items():
        path = STATIC_ROOT / "openwakeword" / name
        input_meta = classifier.get_inputs()[0]
        candidates = select_with_high_responses(
            classifier_inputs[name], classifier_responses[name], max_samples
        )
        key = model_key("openwakeword", path)
        feeds[key] = [
            {input_meta.name: value.astype(np.float32)} for value in candidates
        ]


def fbank(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError(f"Sherpa calibration expects 16 kHz audio, got {sample_rate}")
    waveform = torch.from_numpy(samples.astype(np.float32) / 32768.0).unsqueeze(0)
    features = kaldi.fbank(
        waveform,
        dither=0.0,
        energy_floor=0.0,
        frame_length=25.0,
        frame_shift=10.0,
        high_freq=-400.0,
        low_freq=20.0,
        num_mel_bins=80,
        sample_frequency=16000.0,
        snip_edges=False,
        window_type="povey",
    )
    return features.cpu().numpy().astype(np.float32)


def zero_encoder_states(encoder: ort.InferenceSession) -> dict[str, np.ndarray]:
    result = {}
    for meta in encoder.get_inputs()[1:]:
        shape = tuple(int(value) for value in meta.shape)
        dtype = np.int64 if meta.type == "tensor(int64)" else np.float32
        result[meta.name] = np.zeros(shape, dtype=dtype)
    return result


def next_encoder_states(
    encoder: ort.InferenceSession,
    outputs: list[np.ndarray],
) -> dict[str, np.ndarray]:
    by_name = {meta.name: value for meta, value in zip(encoder.get_outputs(), outputs)}
    result = {}
    for meta in encoder.get_inputs()[1:]:
        output_name = f"new_{meta.name}"
        if output_name not in by_name:
            raise KeyError(f"missing recurrent encoder output: {output_name}")
        result[meta.name] = by_name[output_name]
    return result


def keyword_token_contexts() -> list[np.ndarray]:
    token_map = {}
    for line in (SHERPA_SOURCE_DIR / "tokens.txt").read_text().splitlines():
        token, value = line.rsplit(" ", 1)
        token_map[token] = int(value)
    contexts = []
    for line in (SHERPA_SOURCE_DIR / "test_wavs/keywords.txt").read_text().splitlines():
        tokens = [value for value in line.split() if not value.startswith("@")]
        history = [0, 0]
        contexts.append(np.asarray(history, dtype=np.int64)[None, :])
        for token in tokens:
            history = [history[-1], token_map[token]]
            contexts.append(np.asarray(history, dtype=np.int64)[None, :])
    return contexts


def generate_sherpa(
    feeds: dict[str, list[dict[str, np.ndarray]]],
    audio_paths: list[Path],
    max_samples: int,
) -> None:
    feature_sets = []
    for path in audio_paths:
        samples, sample_rate = read_wav(path)
        feature_sets.append(fbank(samples, sample_rate))

    encoder_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for chunk in (8, 16):
        name = f"encoder-epoch-13-avg-2-chunk-{chunk}-left-64.onnx"
        path = STATIC_ROOT / "sherpa" / name
        encoder = session(path)
        key = model_key("sherpa", path)
        records = []
        input_frames = int(encoder.get_inputs()[0].shape[1])
        shift = 16 if chunk == 8 else 32
        for features in feature_sets:
            states = zero_encoder_states(encoder)
            for start in range(0, len(features), shift):
                x = np.zeros((1, input_frames, 80), dtype=np.float32)
                current = features[start : start + input_frames]
                x[0, : len(current)] = current
                feed = {"x": x, **states}
                if len(records) < max_samples:
                    records.append({name: np.asarray(value).copy() for name, value in feed.items()})
                outputs = encoder.run(None, feed)
                encoder_out = np.asarray(outputs[0]).reshape(-1, 320)
                encoder_vectors[key].extend(encoder_out)
                states = next_encoder_states(encoder, outputs)
                if len(records) >= max_samples and len(encoder_vectors[key]) >= max_samples * 2:
                    break
            if len(records) >= max_samples and len(encoder_vectors[key]) >= max_samples * 2:
                break
        feeds[key] = records

    decoder_name = "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
    decoder_path = STATIC_ROOT / "sherpa" / decoder_name
    decoder = session(decoder_path)
    decoder_key = model_key("sherpa", decoder_path)
    contexts = select_evenly(keyword_token_contexts(), max_samples)
    feeds[decoder_key] = [{"y": value} for value in contexts]
    decoder_vectors = [decoder.run(None, {"y": value})[0] for value in contexts]

    joiner_name = "joiner-epoch-13-avg-2-chunk-16-left-64.onnx"
    joiner_path = STATIC_ROOT / "sherpa" / joiner_name
    joiner_key = model_key("sherpa", joiner_path)
    all_encoder_vectors = []
    for values in encoder_vectors.values():
        all_encoder_vectors.extend(values)
    all_encoder_vectors = select_evenly(all_encoder_vectors, max_samples)
    records = []
    for index, encoder_out in enumerate(all_encoder_vectors):
        decoder_out = decoder_vectors[index % len(decoder_vectors)]
        records.append(
            {
                "encoder_out": np.asarray(encoder_out, dtype=np.float32).reshape(1, 320),
                "decoder_out": np.asarray(decoder_out, dtype=np.float32).reshape(1, 320),
            }
        )
    feeds[joiner_key] = records


def validate_recorded_feeds(
    feeds: dict[str, list[dict[str, np.ndarray]]],
    count: int = 3,
) -> dict[str, object]:
    report = {}
    for key, records in sorted(feeds.items()):
        reference = session(source_for_key(key))
        static = session(static_for_key(key))
        output_names = [value.name for value in reference.get_outputs()]
        comparisons = []
        for record in select_evenly(records, count):
            ref_values = reference.run(None, record)
            got_values = static.run(None, record)
            if key == "openwakeword__melspectrogram":
                got_values[0] = openwakeword_mel_host_postprocess(got_values[0])
            metrics = {
                name: output_metrics(ref, got)
                for name, ref, got in zip(output_names, ref_values, got_values)
            }
            if any(not metrics_pass(value) for value in metrics.values()):
                raise RuntimeError(f"calibration-path validation failed for {key}")
            comparisons.append(metrics)
        cosines = [
            value["cosine"]
            for comparison in comparisons
            for value in comparison.values()
            if "cosine" in value
        ]
        report[key] = {
            "samples_checked": len(comparisons),
            "minimum_output_cosine": min(cosines) if cosines else None,
            "comparisons": comparisons,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    open_audio = sorted((ROOT / "tests/audio").glob("*.wav"))
    feeds: dict[str, list[dict[str, np.ndarray]]] = {}

    print("generating openWakeWord calibration feeds", flush=True)
    generate_openwakeword(feeds, open_audio, args.max_samples)

    expected = {
        *(model_key("openwakeword", Path(name)) for name in OPEN_CORE_MODELS),
    }
    if set(feeds) != expected:
        raise RuntimeError(
            f"calibration model mismatch, missing={sorted(expected - set(feeds))}, "
            f"extra={sorted(set(feeds) - expected)}"
        )

    print("validating static ONNX with recorded calibration feeds", flush=True)
    validation = validate_recorded_feeds(feeds)
    archive_manifest = write_feed_archives(CALIB_ROOT, feeds)
    manifest = {
        "environment": "kws-quant",
        "max_samples_per_model": args.max_samples,
        "openwakeword_audio": [str(path.relative_to(ROOT)) for path in open_audio],
        "validation": validation,
        "archives": archive_manifest,
    }
    write_json(CALIB_ROOT / "manifest.json", manifest)
    print(f"wrote {CALIB_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
