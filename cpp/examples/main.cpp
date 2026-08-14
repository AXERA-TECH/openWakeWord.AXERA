#include "openwakeword_ax.hpp"
#include "wav_reader.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string models_dir = "models";
    std::string weights = "config/openwakeword_mel_weights.bin";
    std::string audio = "audio/openwakeword/alexa_test.wav";
    float threshold = 0.5f;
};

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Missing value for " + key);
            return argv[i];
        };
        if (key == "--models-dir") {
            args.models_dir = value();
        } else if (key == "--mel-weights") {
            args.weights = value();
        } else if (key == "--audio") {
            args.audio = value();
        } else if (key == "--threshold") {
            args.threshold = std::stof(value());
        } else if (key == "-h" || key == "--help") {
            std::printf(
                "Usage: %s [--models-dir DIR] [--mel-weights FILE] "
                "[--audio WAV] [--threshold VALUE]\n",
                argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const PcmWav wav = ReadPcmWav(args.audio);
        if (wav.sample_rate != 16000) {
            throw std::runtime_error("Input WAV must use 16 kHz sample rate");
        }

        openwakeword::WakeWordDetector detector(args.models_dir, args.weights,
                                                args.threshold);
        std::vector<int16_t> padded = wav.samples;
        const std::size_t remainder = padded.size() % 1280;
        if (remainder != 0) padded.resize(padded.size() + 1280 - remainder);

        std::vector<float> maximum;
        std::vector<std::string> names;
        bool any_detected = false;
        for (std::size_t start = 0; start < padded.size(); start += 1280) {
            const openwakeword::FrameResult result =
                detector.ProcessFrame(padded.data() + start);
            if (maximum.empty()) {
                maximum.assign(result.max_scores.size(), -1.0f);
                names = result.names;
            }
            for (std::size_t i = 0; i < result.max_scores.size(); ++i) {
                maximum[i] = std::max(maximum[i], result.max_scores[i]);
                if (result.any_triggered) any_detected = true;
            }
        }

        std::printf("openWakeWord C++ inference complete\n");
        std::printf("audio: %s\n", args.audio.c_str());
        std::printf("threshold: %.3f\n", args.threshold);
        for (std::size_t i = 0; i < maximum.size(); ++i) {
            std::printf("%-20s %.6f%s\n",
                        i < names.size() ? names[i].c_str() : "?",
                        maximum[i],
                        maximum[i] >= args.threshold ? "  WAKEUP" : "");
        }
        std::printf("detected: %s\n", any_detected ? "true" : "false");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
