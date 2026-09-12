#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      std::cerr << #expr << ": " << hipGetErrorString(status) << '\n';        \
      return 1;                                                                \
    }                                                                          \
  } while (false)

struct SmokeArgs {
  void *input;
  void *output;
  uint32_t addend;
  uint32_t reserved;
};
static_assert(sizeof(SmokeArgs) == 24, "assembly kernarg ABI changed");

int main(int argc, char **argv) {
  const char *hsaco = argc > 1 ? argv[1] : "kernel.hsaco";
  constexpr uint32_t kBlock = 128;
  constexpr uint32_t kGrid = 4;
  constexpr uint32_t kCount = kBlock * kGrid;
  constexpr uint32_t kAddend = 0x13579bdfU;

  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
  HIP_CHECK(hipInit(0));
  HIP_CHECK(hipModuleLoad(&module, hsaco));
  HIP_CHECK(hipModuleGetFunction(&function, module, "qwen_gfx942_asm_smoke"));

  std::vector<uint32_t> input(kCount), output(kCount, 0);
  for (uint32_t i = 0; i < kCount; ++i)
    input[i] = i * 17U + 3U;

  void *d_input = nullptr;
  void *d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_input, kCount * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&d_output, kCount * sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(uint32_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_output, 0, output.size() * sizeof(uint32_t)));

  SmokeArgs args{d_input, d_output, kAddend, 0};
  size_t arg_size = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(function, kGrid, 1, 1, kBlock, 1, 1, 0,
                                  nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.data(), d_output,
                      output.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));

  uint32_t bad = kCount;
  for (uint32_t i = 0; i < kCount; ++i) {
    if (output[i] != input[i] + kAddend) {
      bad = i;
      break;
    }
  }
  std::cout << "{\"hsaco\":\"" << hsaco << "\",\"symbol\":\"qwen_gfx942_asm_smoke\""
            << ",\"grid\":" << kGrid << ",\"block\":" << kBlock
            << ",\"count\":" << kCount << ",\"correct\":"
            << (bad == kCount ? "true" : "false");
  if (bad != kCount)
    std::cout << ",\"first_bad\":" << bad << ",\"got\":" << output[bad]
              << ",\"expected\":" << input[bad] + kAddend;
  std::cout << "}\n";

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipModuleUnload(module));
  return bad == kCount ? 0 : 2;
}
