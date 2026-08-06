#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx

from common import (
    CALIB_ROOT,
    OPEN_CORE_MODELS,
    PULSAR_ROOT,
    SHERPA_QUANT_MODELS,
    STATIC_ROOT,
    model_input_description,
    model_key,
    safe_name,
    write_json,
)


def shape_argument(inputs: dict[str, dict[str, object]]) -> str:
    values = []
    for name, description in inputs.items():
        shape = description["shape"]
        encoded = "x".join(str(value) for value in shape) if shape else "1"
        values.append(f"{name}:{encoded}")
    return ";".join(values)


def model_suffix(target_hardware: str, npu_mode: str) -> str:
    # Pulsar2 identifies AX630C as AX620E/NPU2, but deployment names use AX630C.
    if target_hardware == "AX620E" and npu_mode == "NPU2":
        return "AX630C"
    return target_hardware


def quant_models() -> list[tuple[str, Path]]:
    result = [
        ("openwakeword", STATIC_ROOT / "openwakeword" / name)
        for name in sorted(OPEN_CORE_MODELS)
    ]
    result.extend(
        ("sherpa", STATIC_ROOT / "sherpa" / name)
        for name in sorted(SHERPA_QUANT_MODELS)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-hardware", default="AX650")
    parser.add_argument("--npu-mode", default="NPU3")
    parser.add_argument("--default-data-type", default="U16")
    args = parser.parse_args()
    suffix = model_suffix(args.target_hardware, args.npu_mode)

    config_dir = PULSAR_ROOT / "configs"
    build_root = PULSAR_ROOT / "build"
    axmodel_dir = PULSAR_ROOT / "axmodels"
    log_dir = PULSAR_ROOT / "logs"
    for path in (config_dir, build_root, axmodel_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    records = []
    for group, path in quant_models():
        if not path.exists():
            raise FileNotFoundError(f"missing static ONNX: {path}")
        onnx.checker.check_model(onnx.load(path))
        key = model_key(group, path)
        inputs = model_input_description(path)
        input_configs = []
        for name in inputs:
            archive = CALIB_ROOT / key / f"{safe_name(name)}.tar.gz"
            if not archive.exists():
                raise FileNotFoundError(f"missing calibration archive: {archive}")
            input_configs.append(
                {
                    "tensor_name": name,
                    "calibration_dataset": str(archive.resolve()),
                    "calibration_format": "Numpy",
                    "calibration_size": -1,
                }
            )

        build_dir = build_root / key
        output_name = f"{key}__{suffix}.axmodel"
        if key == "openwakeword__melspectrogram":
            layer_configs = [
                {
                    "start_tensor_names": ["DEFAULT"],
                    "end_tensor_names": ["DEFAULT"],
                    "data_type": "FP32",
                }
            ]
        else:
            layer_configs = [
                {"op_type": "Pow", "data_type": "U8"},
                {"op_types": ["Softmax", "ReduceMean"], "data_type": "FP32"},
                {
                    "start_tensor_names": ["DEFAULT"],
                    "end_tensor_names": ["DEFAULT"],
                    "data_type": args.default_data_type,
                },
            ]
        config = {
            "model_type": "ONNX",
            "npu_mode": args.npu_mode,
            "input": str(path.resolve()),
            "output_name": output_name,
            "output_dir": str(build_dir.resolve()),
            "target_hardware": args.target_hardware,
            "onnx_opt": {
                "disable_onnx_optimization": False,
                "enable_onnxsim": True,
            },
            "quant": {
                "input_configs": input_configs,
                "layer_configs": layer_configs,
                "calibration_method": "MinMax",
                "enable_smooth_quant": False,
                "conv_bias_data_type": "FP32",
                "precision_analysis": False,
                "precision_analysis_method": "EndToEnd",
                "disable_auto_refine_scale": True,
                "transformer_opt_level": 0,
            },
            "input_processors": [{"tensor_name": "DEFAULT"}],
            "compiler": {"check": 0, "enable_slice_mode": False},
        }
        config_path = config_dir / f"config_{key}.json"
        write_json(config_path, config)
        records.append(
            {
                "key": key,
                "group": group,
                "onnx": str(path.resolve()),
                "config": str(config_path.resolve()),
                "input_shapes": shape_argument(inputs),
                "build_dir": str(build_dir.resolve()),
                "output_name": output_name,
                "axmodel": str((axmodel_dir / output_name).resolve()),
                "log": str((log_dir / f"{key}.log").resolve()),
            }
        )

    manifest = {
        "target_hardware": args.target_hardware,
        "npu_mode": args.npu_mode,
        "model_suffix": suffix,
        "compiler_check": 0,
        "default_data_type": args.default_data_type,
        "models": records,
        "aliases_after_build": {},
    }
    write_json(PULSAR_ROOT / "manifest.json", manifest)
    print(f"wrote {PULSAR_ROOT / 'manifest.json'} with {len(records)} builds")


if __name__ == "__main__":
    main()
