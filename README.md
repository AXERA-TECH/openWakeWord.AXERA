# openWakeWord.AXERA

openWakeWord 的静态 ONNX 导出、验证和 AX650/AX630C 量化。

## 目录结构

```text
openWakeWord.AXERA/
├── weights/       # 原始 ONNX 模型
├── tests/audio/   # 验证音频
├── model_convert/ # 静态 ONNX、校准数据和 Pulsar2 产物
├── python/        # 导出、验证和量化配置脚本
└── scripts/ax650/
```

## 环境

```bash
cd openwakeword-axera-export
conda create -n kws-quant python=3.12
conda activate kws-quant
pip install -r requirements.txt
```

将 openWakeWord ONNX 模型放入：

```text
weights/openwakeword/
```

模型可从 [openWakeWord](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1) 获取。

## 导出与量化

```bash
# 导出并简化静态 ONNX
bash scripts/ax650/01_export_static_onnx.sh

# 原始 ONNX 与静态 ONNX 推理验证
bash scripts/ax650/02_validate_onnx.sh

# 生成校准数据
# 可选：先合成 6 个官方唤醒词的 TTS 正/负样本（显著提升分类器校准质量，
# 见下方"验证与已知问题"）
bash scripts/ax650/05_generate_calibration_audio.sh
bash scripts/ax650/03_generate_calib.sh

# 激活 Pulsar2 环境后量化
bash scripts/ax650/04_quant.sh
```

### AX630C 量化

```bash
python python/make_pulsar2_configs.py \
  --target-hardware AX620E \
  --npu-mode NPU2 \
  --default-data-type U16

激活 Pulsar2 环境执行：
bash scripts/ax650/build_check0.sh
```

配置和量化产物位于：

```text
model_convert/pulsar2/check0/
├── configs/       # AX620E/NPU2 Pulsar2 配置
├── logs/          # 每个模型的 check0 日志
└── axmodels/     # 生成的 AX630C .axmodel
```


## 参考

- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/pulsar2/introduction.html)

## 验证与已知问题

- **分类器验证应使用整段音频的峰值窗口**：openwakeword 是流式模型，分类器输入是
  最近 N 帧 embedding（N=16/22/34）。只看末尾窗口会误判模型精度；应取整段逐帧
  得分的最大值（`scripts/openwakeword_ax.py` 已按 `max_scores` 汇总）。
- **hey_rhasspy / weather 的 NPU 分类器精度对校准数据敏感**：仅用少量通用语音
  校准会让这两个模型在真实触发窗口坍缩（峰值窗口得分 0.005 / 0.79）；用
  `05_generate_calibration_audio.sh` 补充 TTS 唤醒词正样本后可恢复到 0.97 / 1.00。
- **timer_v0.1 在 AX620E/NPU2 上需 U16 Softmax**：其输出为 2D（[1,7]）Softmax，
  Pulsar2 7.0 的 AX620E/NPU2 后端对 FP32 2D Softmax 编译报
  `TileFailException`；`make_pulsar2_configs.py` 已对 timer + AX620E 自动改用 U16
  （编译通过且精度与 ONNX 一致）。
- **melspectrogram.axmodel 在板上输出全零**：NPU 定点无法表达 mel 前端的动态范围，
  默认 `mel_backend=numpy`（见 `config.json`），C++/Python 运行时均使用 CPU mel 权重。
