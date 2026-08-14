#include "engine_wrapper.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "ax_sys_api.h"

namespace {
constexpr int kAlignment = 128;

void FreeIo(AX_ENGINE_IO_T *io) {
  for (AX_U32 i = 0; io->pInputs && i < io->nInputSize; ++i) {
    if (io->pInputs[i].pVirAddr) {
      AX_SYS_MemFree(io->pInputs[i].phyAddr, io->pInputs[i].pVirAddr);
    }
  }
  for (AX_U32 i = 0; io->pOutputs && i < io->nOutputSize; ++i) {
    if (io->pOutputs[i].pVirAddr) {
      AX_SYS_MemFree(io->pOutputs[i].phyAddr, io->pOutputs[i].pVirAddr);
    }
  }
  delete[] io->pInputs;
  delete[] io->pOutputs;
  std::memset(io, 0, sizeof(*io));
}
}  // namespace

EngineWrapper::~EngineWrapper() { Release(); }

EngineWrapper::EngineWrapper(EngineWrapper &&other) noexcept
    : initialized_(other.initialized_),
      handle_(other.handle_),
      info_(other.info_),
      io_(other.io_),
      inputs_(std::move(other.inputs_)),
      outputs_(std::move(other.outputs_)),
      input_names_(std::move(other.input_names_)),
      output_names_(std::move(other.output_names_)) {
    other.initialized_ = false;
    other.handle_ = nullptr;
    other.info_ = nullptr;
    std::memset(&other.io_, 0, sizeof(other.io_));
}

EngineWrapper &EngineWrapper::operator=(EngineWrapper &&other) noexcept {
    if (this != &other) {
        Release();
        initialized_ = other.initialized_;
        handle_ = other.handle_;
        info_ = other.info_;
        io_ = other.io_;
        inputs_ = std::move(other.inputs_);
        outputs_ = std::move(other.outputs_);
        input_names_ = std::move(other.input_names_);
        output_names_ = std::move(other.output_names_);
        other.initialized_ = false;
        other.handle_ = nullptr;
        other.info_ = nullptr;
        std::memset(&other.io_, 0, sizeof(other.io_));
    }
    return *this;
}

int EngineWrapper::Init(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "Failed to read model: %s\n", path.c_str());
    return -1;
  }
  input.seekg(0, std::ios::end);
  const std::streamoff model_size = input.tellg();
  if (model_size <= 0 || model_size > static_cast<std::streamoff>(UINT32_MAX)) {
    std::fprintf(stderr, "Invalid model size: %s\n", path.c_str());
    return -1;
  }
  input.seekg(0, std::ios::beg);
  std::vector<char> model(static_cast<std::size_t>(model_size));
  input.read(model.data(), model_size);
  if (!input) {
    std::fprintf(stderr, "Failed to read model: %s\n", path.c_str());
    return -1;
  }
  const AX_S32 create_ret = AX_ENGINE_CreateHandle(
      &handle_, model.data(), static_cast<AX_U32>(model.size()));
  if (create_ret != 0 || !handle_) {
    std::fprintf(stderr, "AX_ENGINE_CreateHandle failed: %s ret=0x%x\n",
                 path.c_str(), static_cast<unsigned>(create_ret));
    return -1;
  }
  if (AX_ENGINE_CreateContext(handle_) != 0 ||
      AX_ENGINE_GetIOInfo(handle_, &info_) != 0 || !info_) {
    Release();
    return -1;
  }

  io_.nInputSize = info_->nInputSize;
  io_.nOutputSize = info_->nOutputSize;
  io_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_.nInputSize]{};
  io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_.nOutputSize]{};
  AX_S8 input_tag[] = "KWS-INPUT";
  AX_S8 output_tag[] = "KWS-OUTPUT";
  for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
    io_.pInputs[i].nSize = info_->pInputs[i].nSize;
    if (AX_SYS_MemAlloc(&io_.pInputs[i].phyAddr,
                        &io_.pInputs[i].pVirAddr, io_.pInputs[i].nSize,
                        kAlignment, input_tag) != 0) {
      Release();
      return -1;
    }
    const std::string name = info_->pInputs[i].pName;
    inputs_[name] = static_cast<int>(i);
    input_names_.push_back(name);
  }
  for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
    io_.pOutputs[i].nSize = info_->pOutputs[i].nSize;
    if (AX_SYS_MemAlloc(&io_.pOutputs[i].phyAddr,
                        &io_.pOutputs[i].pVirAddr, io_.pOutputs[i].nSize,
                        kAlignment, output_tag) != 0) {
      Release();
      return -1;
    }
    const std::string name = info_->pOutputs[i].pName;
    outputs_[name] = static_cast<int>(i);
    output_names_.push_back(name);
  }
  initialized_ = true;
  std::printf("Loaded %s (inputs=%u outputs=%u)\n", path.c_str(),
              io_.nInputSize, io_.nOutputSize);
  return 0;
}

void EngineWrapper::Release() {
  FreeIo(&io_);
  if (handle_) AX_ENGINE_DestroyHandle(handle_);
  initialized_ = false;
  handle_ = nullptr;
  info_ = nullptr;
  inputs_.clear();
  outputs_.clear();
  input_names_.clear();
  output_names_.clear();
}

int EngineWrapper::InputIndex(const std::string &name) const {
  const auto it = inputs_.find(name);
  return it == inputs_.end() ? -1 : it->second;
}

int EngineWrapper::OutputIndex(const std::string &name) const {
  const auto it = outputs_.find(name);
  return it == outputs_.end() ? -1 : it->second;
}

int EngineWrapper::SetInputByName(const std::string &name, const void *data,
                                  std::size_t size) {
  const int index = InputIndex(name);
  if (!initialized_ || index < 0 || !data) return -1;
  const std::size_t expected = io_.pInputs[index].nSize;
  if (size != 0 && size != expected) return -1;
  std::memcpy(io_.pInputs[index].pVirAddr, data, expected);
  return 0;
}

int EngineWrapper::ZeroInputByName(const std::string &name) {
  const int index = InputIndex(name);
  if (!initialized_ || index < 0) return -1;
  std::memset(io_.pInputs[index].pVirAddr, 0, io_.pInputs[index].nSize);
  return 0;
}

int EngineWrapper::RunSync() {
  if (!initialized_) return -1;
  for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
    const AX_S32 flush_ret = AX_SYS_MflushCache(
        io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr,
        io_.pInputs[i].nSize);
    if (flush_ret != 0) {
      std::fprintf(stderr, "AX_SYS_MflushCache failed input=%u ret=0x%x\n", i,
                   static_cast<unsigned>(flush_ret));
      return flush_ret;
    }
  }
  const AX_S32 ret = AX_ENGINE_RunSync(handle_, &io_);
  if (ret != 0) {
    std::fprintf(stderr, "AX_ENGINE_RunSync failed ret=0x%x\n",
                 static_cast<unsigned>(ret));
    return ret;
  }
  for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
    const AX_S32 invalidate_ret = AX_SYS_MinvalidateCache(
        io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr,
        io_.pOutputs[i].nSize);
    if (invalidate_ret != 0) {
      std::fprintf(stderr,
                   "AX_SYS_MinvalidateCache failed output=%u ret=0x%x\n", i,
                   static_cast<unsigned>(invalidate_ret));
      return invalidate_ret;
    }
  }
  return 0;
}

int EngineWrapper::GetOutputByName(const std::string &name, void *data,
                                   std::size_t size) const {
  const int index = OutputIndex(name);
  if (!initialized_ || index < 0 || !data) return -1;
  const std::size_t expected = io_.pOutputs[index].nSize;
  if (size != 0 && size != expected) return -1;
  std::memcpy(data, io_.pOutputs[index].pVirAddr, expected);
  return 0;
}

int EngineWrapper::GetInputSizeByName(const std::string &name) const {
  const int index = InputIndex(name);
  return index < 0 ? -1 : static_cast<int>(io_.pInputs[index].nSize);
}

int EngineWrapper::GetOutputSizeByName(const std::string &name) const {
  const int index = OutputIndex(name);
  return index < 0 ? -1 : static_cast<int>(io_.pOutputs[index].nSize);
}

const std::string &EngineWrapper::InputName(std::size_t index) const {
  return input_names_.at(index);
}

const std::string &EngineWrapper::OutputName(std::size_t index) const {
  return output_names_.at(index);
}
