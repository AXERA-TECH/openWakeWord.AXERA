#include "openwakeword_ax.hpp"

#include "engine_wrapper.hpp"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 1280;
constexpr int kHistorySamples = 480;
constexpr int kFftSize = 512;
constexpr int kSpectrumBins = 257;
constexpr int kMelBins = 32;
constexpr int kMelFrames = 8;
constexpr int kEmbeddingFrames = 76;
constexpr int kEmbeddingSize = 96;
constexpr int kFeatureFrames = 34;

std::string Join(const std::string& left, const std::string& right) {
    return left.empty() || left.back() == '/' ? left + right
                                              : left + "/" + right;
}

class AxRuntime {
public:
    AxRuntime() {
        if (AX_SYS_Init() != 0) throw std::runtime_error("AX_SYS_Init failed");
        sys_initialized_ = true;
        AX_ENGINE_NPU_ATTR_T attr{};
        if (AX_ENGINE_Init(&attr) != 0) {
            AX_SYS_Deinit();
            sys_initialized_ = false;
            throw std::runtime_error("AX_ENGINE_Init failed");
        }
        engine_initialized_ = true;
    }
    ~AxRuntime() {
        if (engine_initialized_) AX_ENGINE_Deinit();
        if (sys_initialized_) AX_SYS_Deinit();
    }

private:
    bool sys_initialized_ = false;
    bool engine_initialized_ = false;
};

template <typename T>
T ReadScalar(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("Truncated mel weight file");
    return value;
}

struct MelWeights {
    std::vector<float> real;
    std::vector<float> imag;
    std::vector<float> mel;
    float floor = 0.0f;

    static MelWeights Load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open mel weights: " + path);
        char magic[8]{};
        input.read(magic, 8);
        if (std::memcmp(magic, "OWWMEL1", 7) != 0 ||
            ReadScalar<uint32_t>(input) != 1) {
            throw std::runtime_error("Invalid openWakeWord mel weight file");
        }
        const uint32_t real_rows = ReadScalar<uint32_t>(input);
        const uint32_t real_cols = ReadScalar<uint32_t>(input);
        const uint32_t imag_rows = ReadScalar<uint32_t>(input);
        const uint32_t imag_cols = ReadScalar<uint32_t>(input);
        const uint32_t mel_rows = ReadScalar<uint32_t>(input);
        const uint32_t mel_cols = ReadScalar<uint32_t>(input);
        MelWeights result;
        result.floor = ReadScalar<float>(input);
        if (real_rows != kSpectrumBins || real_cols != kFftSize ||
            imag_rows != kSpectrumBins || imag_cols != kFftSize ||
            mel_rows != kSpectrumBins || mel_cols != kMelBins) {
            throw std::runtime_error("Unexpected openWakeWord mel weight shapes");
        }
        result.real.resize(static_cast<std::size_t>(real_rows) * real_cols);
        result.imag.resize(static_cast<std::size_t>(imag_rows) * imag_cols);
        result.mel.resize(static_cast<std::size_t>(mel_rows) * mel_cols);
        input.read(reinterpret_cast<char*>(result.real.data()),
                   result.real.size() * sizeof(float));
        input.read(reinterpret_cast<char*>(result.imag.data()),
                   result.imag.size() * sizeof(float));
        input.read(reinterpret_cast<char*>(result.mel.data()),
                   result.mel.size() * sizeof(float));
        if (!input) throw std::runtime_error("Truncated openWakeWord mel weights");
        return result;
    }
};

// 与官方 numpy/ONNX 实现逐位对齐的 CPU mel（含 max-80 下限与 /10+2）。
std::array<float, kMelFrames * kMelBins> ComputeMel(
    const std::array<float, kHistorySamples + kChunkSamples>& samples,
    const MelWeights& weights) {
    std::array<float, kMelFrames * kMelBins> result{};
    std::array<float, kSpectrumBins> power{};
    float max_db = -std::numeric_limits<float>::infinity();
    for (int frame = 0; frame < kMelFrames; ++frame) {
        const float* frame_samples = samples.data() + frame * 160;
        for (int frequency = 0; frequency < kSpectrumBins; ++frequency) {
            const float* real = weights.real.data() + frequency * kFftSize;
            const float* imag = weights.imag.data() + frequency * kFftSize;
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            for (int n = 0; n < kFftSize; ++n) {
                real_sum += frame_samples[n] * real[n];
                imag_sum += frame_samples[n] * imag[n];
            }
            power[frequency] = real_sum * real_sum + imag_sum * imag_sum;
        }
        for (int bin = 0; bin < kMelBins; ++bin) {
            float value = 0.0f;
            for (int frequency = 0; frequency < kSpectrumBins; ++frequency) {
                value += power[frequency] *
                         weights.mel[frequency * kMelBins + bin];
            }
            value = std::max(value, weights.floor);
            const float db = std::log(value) * 10.0f / 2.3025851249694824f;
            result[frame * kMelBins + bin] = db;
            max_db = std::max(max_db, db);
        }
    }
    const float minimum = max_db - 80.0f;
    for (float& value : result) {
        value = std::max(value, minimum) / 10.0f + 2.0f;
    }
    return result;
}

struct Classifier {
    std::string name;
    int frames = 0;
    EngineWrapper engine;
};

}  // namespace

namespace openwakeword {

struct WakeWordDetector::Impl {
    AxRuntime runtime;
    MelWeights weights;
    EngineWrapper embedding;
    std::vector<Classifier> classifiers;
    std::array<int16_t, kHistorySamples> history{};
    std::array<float, kEmbeddingFrames * kMelBins> mel_buffer{};
    std::array<float, kFeatureFrames * kEmbeddingSize> feature_buffer{};
    int frame_count = 0;

    Impl(const std::string& models_dir, const std::string& mel_weights)
        : weights(MelWeights::Load(mel_weights)) {
        if (embedding.Init(Join(models_dir,
                                "openwakeword__embedding_model.axmodel")) != 0) {
            throw std::runtime_error("Failed to load embedding model");
        }
        const std::array<std::pair<const char*, int>, 6> definitions{{
            {"alexa_v0.1", 16},
            {"hey_jarvis_v0.1", 16},
            {"hey_mycroft_v0.1", 16},
            {"hey_rhasspy_v0.1", 16},
            {"timer_v0.1", 34},
            {"weather_v0.1", 22},
        }};
        for (const auto& definition : definitions) {
            Classifier classifier;
            classifier.name = definition.first;
            classifier.frames = definition.second;
            const std::string model =
                Join(models_dir, "openwakeword__" + classifier.name + ".axmodel");
            if (classifier.engine.Init(model) != 0) {
                throw std::runtime_error("Failed to load classifier: " +
                                         classifier.name);
            }
            classifiers.push_back(std::move(classifier));
        }
        Reset();
    }

    void Reset() {
        history.fill(0);
        mel_buffer.fill(1.0f);
        feature_buffer.fill(0.0f);
        frame_count = 0;
    }

    FrameResult ProcessFrame(const int16_t* samples) {
        std::array<float, kHistorySamples + kChunkSamples> mel_input{};
        for (int i = 0; i < kHistorySamples; ++i) mel_input[i] = history[i];
        for (int i = 0; i < kChunkSamples; ++i) {
            mel_input[kHistorySamples + i] = samples[i];
        }
        for (int i = 0; i < kHistorySamples; ++i) {
            history[i] = samples[kChunkSamples - kHistorySamples + i];
        }

        const auto mel = ComputeMel(mel_input, weights);
        std::memmove(mel_buffer.data(), mel_buffer.data() + kMelFrames * kMelBins,
                     (kEmbeddingFrames - kMelFrames) * kMelBins * sizeof(float));
        std::memcpy(mel_buffer.data() + (kEmbeddingFrames - kMelFrames) * kMelBins,
                    mel.data(), mel.size() * sizeof(float));

        const std::string& embedding_input = embedding.InputName(0);
        if (embedding.SetInputByName(embedding_input, mel_buffer.data(),
                                     mel_buffer.size() * sizeof(float)) != 0) {
            throw std::runtime_error("Failed to set embedding input");
        }
        if (embedding.RunSync() != 0) {
            throw std::runtime_error("Embedding inference failed");
        }
        std::array<float, kEmbeddingSize> feature{};
        if (embedding.GetOutputByName(embedding.OutputName(0), feature.data(),
                                      feature.size() * sizeof(float)) != 0) {
            throw std::runtime_error("Failed to read embedding output");
        }
        std::memmove(feature_buffer.data(), feature_buffer.data() + kEmbeddingSize,
                     (kFeatureFrames - 1) * kEmbeddingSize * sizeof(float));
        std::memcpy(feature_buffer.data() + (kFeatureFrames - 1) * kEmbeddingSize,
                    feature.data(), feature.size() * sizeof(float));

        FrameResult result;
        for (Classifier& classifier : classifiers) {
            const float* input =
                feature_buffer.data() +
                (kFeatureFrames - classifier.frames) * kEmbeddingSize;
            const std::string& input_name = classifier.engine.InputName(0);
            if (classifier.engine.SetInputByName(
                    input_name, input,
                    classifier.frames * kEmbeddingSize * sizeof(float)) != 0) {
                throw std::runtime_error("Failed to set classifier input: " +
                                         classifier.name);
            }
            if (classifier.engine.RunSync() != 0) {
                throw std::runtime_error("Classifier inference failed: " +
                                         classifier.name);
            }
            const int output_bytes = classifier.engine.GetOutputSizeByName(
                classifier.engine.OutputName(0));
            std::vector<float> output(output_bytes / sizeof(float));
            if (classifier.engine.GetOutputByName(
                    classifier.engine.OutputName(0), output.data(),
                    output.size() * sizeof(float)) != 0) {
                throw std::runtime_error("Failed to read classifier output");
            }
            result.names.push_back(classifier.name);
            result.logits.push_back(output);
            const float max_score =
                output.empty() ? 0.0f
                               : *std::max_element(output.begin(), output.end());
            result.max_scores.push_back(max_score);
        }

        ++frame_count;
        if (frame_count <= 5) {
            for (auto& logits : result.logits) {
                std::fill(logits.begin(), logits.end(), 0.0f);
            }
            result.max_scores.assign(result.max_scores.size(), 0.0f);
        }
        return result;
    }
};

WakeWordDetector::WakeWordDetector(const std::string& models_dir,
                                   const std::string& mel_weights,
                                   float threshold)
    : impl_(new Impl(models_dir, mel_weights)), threshold_(threshold) {}

WakeWordDetector::~WakeWordDetector() { delete impl_; }

void WakeWordDetector::Reset() { impl_->Reset(); }

FrameResult WakeWordDetector::ProcessFrame(const int16_t* samples) {
    FrameResult result = impl_->ProcessFrame(samples);
    for (float score : result.max_scores) {
        if (score > threshold_) {
            result.any_triggered = true;
            break;
        }
    }
    return result;
}

}  // namespace openwakeword
