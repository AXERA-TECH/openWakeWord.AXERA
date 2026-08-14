#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct PcmWav {
  int sample_rate = 0;
  std::vector<int16_t> samples;
};

inline uint16_t ReadLe16(std::istream &input) {
  uint8_t bytes[2]{};
  input.read(reinterpret_cast<char *>(bytes), 2);
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

inline uint32_t ReadLe32(std::istream &input) {
  uint8_t bytes[4]{};
  input.read(reinterpret_cast<char *>(bytes), 4);
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

inline PcmWav ReadPcmWav(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open WAV: " + path);
  char riff[4]{};
  char wave[4]{};
  input.read(riff, 4);
  ReadLe32(input);
  input.read(wave, 4);
  if (std::memcmp(riff, "RIFF", 4) != 0 ||
      std::memcmp(wave, "WAVE", 4) != 0) {
    throw std::runtime_error("Invalid RIFF/WAVE file: " + path);
  }

  uint16_t format = 0;
  uint16_t channels = 0;
  uint16_t bits = 0;
  uint32_t sample_rate = 0;
  std::vector<uint8_t> data;
  while (input && (!format || data.empty())) {
    char id[4]{};
    input.read(id, 4);
    if (!input) break;
    const uint32_t size = ReadLe32(input);
    if (std::memcmp(id, "fmt ", 4) == 0) {
      if (size < 16) throw std::runtime_error("Invalid WAV fmt chunk");
      format = ReadLe16(input);
      channels = ReadLe16(input);
      sample_rate = ReadLe32(input);
      ReadLe32(input);
      ReadLe16(input);
      bits = ReadLe16(input);
      input.seekg(size - 16, std::ios::cur);
    } else if (std::memcmp(id, "data", 4) == 0) {
      data.resize(size);
      input.read(reinterpret_cast<char *>(data.data()), size);
    } else {
      input.seekg(size, std::ios::cur);
    }
    if (size & 1U) input.seekg(1, std::ios::cur);
  }
  if (format != 1 || channels != 1 || bits != 16 || sample_rate == 0 ||
      data.empty() || data.size() % sizeof(int16_t) != 0) {
    throw std::runtime_error("Expected mono 16-bit PCM WAV: " + path);
  }
  PcmWav result;
  result.sample_rate = static_cast<int>(sample_rate);
  result.samples.resize(data.size() / sizeof(int16_t));
  std::memcpy(result.samples.data(), data.data(), data.size());
  return result;
}
