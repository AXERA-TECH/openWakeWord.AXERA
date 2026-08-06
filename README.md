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
