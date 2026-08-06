#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_path = ROOT / "model_convert/static_onnx/openwakeword/melspectrogram.onnx"
    output_path = ROOT / "model_convert/board_assets/openwakeword_mel_weights.npz"
    model = onnx.load(model_path)
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        real=np.asarray(
            initializers["0.stft.conv_real.weight"][:, 0, :], dtype=np.float32
        ),
        imag=np.asarray(
            initializers["0.stft.conv_imag.weight"][:, 0, :], dtype=np.float32
        ),
        mel=np.asarray(initializers["1.melW"], dtype=np.float32),
        floor=np.asarray(initializers["onnx::Clip_39"], dtype=np.float32),
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
