#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>

using LaunchFn = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                                unsigned int, unsigned int, unsigned int,
                                unsigned int, unsigned int, hipStream_t,
                                void **, void **);

extern "C" hipError_t hipModuleLaunchKernel(
    hipFunction_t function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_mem, hipStream_t stream,
    void **kernel_params, void **extra) {
  static auto real = reinterpret_cast<LaunchFn>(dlsym(RTLD_NEXT, "hipModuleLaunchKernel"));
  const char *capture = std::getenv("QWEN_TRITON_KERNARG_CAPTURE");
  if (capture && extra) {
    void *buffer = nullptr;
    size_t size = 0;
    for (size_t index = 0; extra[index] != HIP_LAUNCH_PARAM_END; index += 2) {
      if (extra[index] == HIP_LAUNCH_PARAM_BUFFER_POINTER)
        buffer = *reinterpret_cast<void **>(extra[index + 1]);
      else if (extra[index] == HIP_LAUNCH_PARAM_BUFFER_SIZE)
        size = *reinterpret_cast<size_t *>(extra[index + 1]);
    }
    if (buffer && size == 88) {
      std::ofstream output(capture, std::ios::trunc);
      const auto *bytes = static_cast<const uint8_t *>(buffer);
      output << "{\"grid\":[" << grid_x << ',' << grid_y << ',' << grid_z
             << "],\"block\":[" << block_x << ',' << block_y << ',' << block_z
             << "],\"shared\":" << shared_mem << ",\"size\":" << size
             << ",\"u64\":[";
      for (size_t offset = 0; offset < size; offset += 8) {
        uint64_t value = 0;
        std::memcpy(&value, bytes + offset, sizeof(value));
        output << (offset ? "," : "") << "\"0x" << std::hex << value << std::dec << "\"";
      }
      output << "]}\n";
    } else if (kernel_params) {
      std::ofstream output(capture, std::ios::trunc);
      output << "{\"grid\":[" << grid_x << ',' << grid_y << ',' << grid_z
             << "],\"block\":[" << block_x << ',' << block_y << ',' << block_z
             << "],\"shared\":" << shared_mem << ",\"kernel_params\":[";
      for (size_t index = 0; index < 12; ++index) {
        uint64_t value = 0;
        if (kernel_params[index])
          std::memcpy(&value, kernel_params[index], sizeof(value));
        output << (index ? "," : "") << "\"0x" << std::hex << value << std::dec << "\"";
      }
      output << "]}\n";
    }
  }
  return real(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
              shared_mem, stream, kernel_params, extra);
}
