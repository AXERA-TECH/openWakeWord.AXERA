#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "ax_engine_api.h"

class EngineWrapper {
 public:
  EngineWrapper() = default;
  ~EngineWrapper();

  EngineWrapper(const EngineWrapper &) = delete;
  EngineWrapper &operator=(const EngineWrapper &) = delete;
  EngineWrapper(EngineWrapper &&other) noexcept;
  EngineWrapper &operator=(EngineWrapper &&other) noexcept;

  int Init(const std::string &model_path);
  void Release();
  int SetInputByName(const std::string &name, const void *data,
                     std::size_t size = 0);
  int ZeroInputByName(const std::string &name);
  int RunSync();
  int GetOutputByName(const std::string &name, void *data,
                      std::size_t size = 0) const;
  int GetInputSizeByName(const std::string &name) const;
  int GetOutputSizeByName(const std::string &name) const;
  const std::string &InputName(std::size_t index) const;
  const std::string &OutputName(std::size_t index) const;

 private:
  int InputIndex(const std::string &name) const;
  int OutputIndex(const std::string &name) const;

  bool initialized_ = false;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *info_ = nullptr;
  AX_ENGINE_IO_T io_{};
  std::unordered_map<std::string, int> inputs_;
  std::unordered_map<std::string, int> outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
