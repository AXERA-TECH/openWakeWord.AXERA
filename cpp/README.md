# openWakeWord C++ 推理

本目录提供 AX650 和 AX630C C++ 推理程序。WAV、STFT、mel 和缓存更新在 CPU
端完成，embedding 与 classifier 使用对应平台的 `axmodel`。

## 编译环境

AX650 与 AX630C 共用 Arm GNU 9.2 AArch64 交叉编译器：

```bash
mkdir -p cpp/toolchains
wget -P cpp/toolchains \
  https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf cpp/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz \
  -C cpp/toolchains
```

下载 BSP：

```bash
bash cpp/download_bsp.sh all
```

| 平台 | BSP 压缩包 | 编译目录 |
| --- | --- | --- |
| AX650 | `msp_50_3.10.2.zip` | `cpp/toolchains/ax650n_bsp_sdk/msp/out` |
| AX630C | `msp_20e_3.0.0.zip` | `cpp/toolchains/ax620e_bsp_sdk/msp/out/arm64_glibc` |

## 交叉编译

```bash
export TOOLCHAIN_ROOT="cpp/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

bash cpp/build_ax650.sh
bash cpp/build_ax630c.sh
```

生成：

```text
cpp/bin/openwakeword_ax650
cpp/bin/openwakeword_ax630c
```

如 BSP 放在其他位置，可设置 `BSP_MSP_DIR`。AX650 应指向包含 `include/` 和
`lib/` 的 `msp/out`；AX630C 应指向 `msp/out/arm64_glibc`。

## 板端运行

AX650 单条推理：

```bash
AUDIO=audio/openwakeword/alexa_test.wav \
bash cpp/run_openwakeword_ax650.sh
```

AX630C 单条推理：

```bash
AUDIO=audio/openwakeword/alexa_test.wav \
bash cpp/run_openwakeword_ax630c.sh
```

模型目录必须与平台对应：

```text
models/650/
models/630C/
```

运行脚本会检查 embedding 和 6 个 classifier 模型，避免误用其他平台的 axmodel。
输入音频必须为 16 kHz、单声道、16-bit PCM WAV。

单条推理会输出 classifier 分数和 `WAKEUP`，并只输出一个 `rtf`。`rtf` 包含
CPU mel 和 NPU 推理，不包含一次性模型加载与初始化。

阈值可通过环境变量调整：

```bash
THRESHOLD=0.6 AUDIO=audio/openwakeword/hey_mycroft_test.wav \
bash cpp/run_openwakeword_ax630c.sh
```

## 二次开发（共享库）

编译产物除可执行文件外，还包含共享库：

```text
cpp/lib/ax650/libopenwakeword_ax.so      # AX650
cpp/lib/ax630c/libopenwakeword_ax.so     # AX630C
cpp/include/openwakeword_ax.hpp          # 公共 API 头文件
```

在自己的 CMake 工程中链接：

```cmake
add_library(openwakeword_ax SHARED IMPORTED)
set_target_properties(openwakeword_ax PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/ax650/libopenwakeword_ax.so")
target_include_directories(your_app PRIVATE cpp/include)
target_link_libraries(your_app PRIVATE openwakeword_ax)
```

最小使用示例：

```cpp
#include "openwakeword_ax.hpp"

openwakeword::WakeWordDetector detector(
    "models/650",                    // axmodel 目录
    "config/openwakeword_mel_weights.bin",  // CPU mel 权重
    0.5f);                            // 阈值

// 每 80ms 喂一帧 1280 个 int16 样本（16 kHz）
openwakeword::FrameResult result = detector.ProcessFrame(pcm_frame);
for (std::size_t i = 0; i < result.names.size(); ++i) {
    printf("%s max_score=%.4f\n", result.names[i].c_str(),
           result.max_scores[i]);
}
if (result.any_triggered) { /* 唤醒 */ }
detector.Reset();  // 新会话时清空流式状态
```

板端运行时需提供 BSP 动态库路径（AX650 通常为 `/soc/lib`，AX630C 为
`/opt/lib`）：

```bash
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH}
```
