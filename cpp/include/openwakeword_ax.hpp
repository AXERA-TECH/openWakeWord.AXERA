#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openwakeword {

// 单帧（80ms）推理结果：6 个官方唤醒词的逐类 logits 与汇总。
struct FrameResult {
    std::vector<std::string> names;            // 分类器名（alexa_v0.1 ...）
    std::vector<std::vector<float>> logits;    // 每个分类器的逐类 logit
    std::vector<float> max_scores;             // 每个分类器的最大 logit
    bool any_triggered = false;                // 任一 max_score > threshold
};

// 流式唤醒词检测器：mel 在 CPU 计算，embedding + 6 个分类器在 NPU 执行。
class WakeWordDetector {
public:
    // models_dir：包含 openwakeword__*.axmodel 的目录（models/650 或 models/630C）
    // mel_weights：openwakeword_mel_weights.bin（CPU mel 权重）
    WakeWordDetector(const std::string& models_dir,
                     const std::string& mel_weights,
                     float threshold = 0.5f);
    ~WakeWordDetector();

    WakeWordDetector(const WakeWordDetector&) = delete;
    WakeWordDetector& operator=(const WakeWordDetector&) = delete;

    // 清空流式状态（新会话/断句后调用）
    void Reset();

    // 处理一帧 1280 个 int16 样本（80ms @16kHz），返回 6 个分类器的得分。
    // 前 5 帧为预热窗口（返回 0），与 openwakeword 官方行为一致。
    FrameResult ProcessFrame(const int16_t* samples);

    float threshold() const { return threshold_; }

private:
    struct Impl;
    Impl* impl_;
    float threshold_;
};

}  // namespace openwakeword
