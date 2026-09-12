#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kHg = 4, kH = 8, kK = 128, kV = 128, kBT = 64;
constexpr size_t kStateElements = kH * kV * kK;
constexpr unsigned kDynamicSharedBytes = 57344;

void Check(hipError_t status, const char *where) {
  if (status != hipSuccess) throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(status));
}

template <typename T> std::vector<T> Read(const std::string &path, size_t count) {
  std::vector<T> values(count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read " + path);
  stream.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (stream.gcount() != static_cast<std::streamsize>(count * sizeof(T))) throw std::runtime_error("unexpected size for " + path);
  return values;
}

struct DeviceBuffers {
  void *k = nullptr, *v = nullptr, *w = nullptr, *v_new = nullptr, *g = nullptr, *h = nullptr, *h0 = nullptr, *ht = nullptr;
  ~DeviceBuffers() { for (void *ptr : {k, v, w, v_new, g, h, h0, ht}) if (ptr) (void)hipFree(ptr); }
};

void CopyToDevice(void **dst, const void *src, size_t bytes) {
  Check(hipMalloc(dst, bytes), "hipMalloc");
  Check(hipMemcpy(*dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpy");
}

struct Module {
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
  std::string name;
  ~Module() { if (module) (void)hipModuleUnload(module); }
};

std::array<float, 3> Stats(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return {values[values.size() / 2], values[(values.size() - 1) / 10], values[(values.size() - 1) * 9 / 10]};
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: round_robin_harness <original.hsaco> <rebuilt.hsaco> <asm_v0.hsaco> <T> <input-dir> <warmup> <repeat>\n";
    return 64;
  }
  try {
    int32_t t = std::stoi(argv[4]);
    const std::string input_dir = argv[5];
    const int warmup = std::stoi(argv[6]), repeat = std::stoi(argv[7]);
    if (t < static_cast<int32_t>(kBT) || t % kBT || repeat <= 0) throw std::runtime_error("T must be a positive multiple of 64 and repeat must be positive");
    const size_t tokens = static_cast<size_t>(t), chunks = tokens / kBT;
    const size_t k_elements = tokens * kHg * kK, tv_elements = tokens * kH * kV, g_elements = tokens * kH, h_elements = chunks * kStateElements;
    const auto k = Read<uint16_t>(input_dir + "/k_bf16.bin", k_elements);
    const auto v = Read<float>(input_dir + "/v_fp32.bin", tv_elements);
    const auto w = Read<float>(input_dir + "/w_fp32.bin", tv_elements);
    const auto g = Read<float>(input_dir + "/g_fp32.bin", g_elements);
    const auto h0 = Read<float>(input_dir + "/h0_fp32.bin", kStateElements);

    Check(hipInit(0), "hipInit");
    std::array<Module, 3> modules;
    const std::array<std::string, 3> paths = {argv[1], argv[2], argv[3]};
    const std::array<std::string, 3> names = {
        "golden_triton_original_hsaco", "golden_triton_rebuilt_hsaco", "avelang_asm_v0"};
    const std::array<const char *, 3> symbols = {
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "qwen_gdn_bt64_gfx942_asm_v0"};
    for (size_t index = 0; index < modules.size(); ++index) {
      modules[index].name = names[index];
      Check(hipModuleLoad(&modules[index].module, paths[index].c_str()), "hipModuleLoad");
      Check(hipModuleGetFunction(&modules[index].function, modules[index].module, symbols[index]), "hipModuleGetFunction");
    }

    DeviceBuffers device;
    CopyToDevice(&device.k, k.data(), k.size() * sizeof(uint16_t));
    CopyToDevice(&device.v, v.data(), v.size() * sizeof(float));
    CopyToDevice(&device.w, w.data(), w.size() * sizeof(float));
    CopyToDevice(&device.g, g.data(), g.size() * sizeof(float));
    CopyToDevice(&device.h0, h0.data(), h0.size() * sizeof(float));
    Check(hipMalloc(&device.v_new, tv_elements * sizeof(float)), "hipMalloc(v_new)");
    Check(hipMalloc(&device.h, h_elements * sizeof(uint16_t)), "hipMalloc(h)");
    Check(hipMalloc(&device.ht, kStateElements * sizeof(float)), "hipMalloc(ht)");
    hipDeviceptr_t global_scratch = 0, profile_scratch = 0;
    void *params[] = {&device.k, &device.v, &device.w, &device.v_new, &device.g, &device.h, &device.h0, &device.ht, &t, &global_scratch, &profile_scratch};
    const auto launch = [&](size_t index) {
      Check(hipModuleLaunchKernel(modules[index].function, 4, 8, 1, 256, 1, 1,
                                  kDynamicSharedBytes, nullptr, params, nullptr), "hipModuleLaunchKernel");
    };
    for (int iteration = 0; iteration < warmup; ++iteration) {
      for (size_t offset = 0; offset < modules.size(); ++offset) launch((iteration + offset) % modules.size());
    }
    Check(hipDeviceSynchronize(), "warmup synchronize");
    hipEvent_t start = nullptr, end = nullptr;
    Check(hipEventCreate(&start), "event start");
    Check(hipEventCreate(&end), "event end");
    std::array<std::vector<float>, 3> times;
    for (int iteration = 0; iteration < repeat; ++iteration) {
      for (size_t offset = 0; offset < modules.size(); ++offset) {
        const size_t index = (iteration + offset) % modules.size();
        float elapsed = 0.0f;
        Check(hipEventRecord(start), "record start");
        launch(index);
        Check(hipEventRecord(end), "record end");
        Check(hipEventSynchronize(end), "event synchronize");
        Check(hipEventElapsedTime(&elapsed, start, end), "elapsed");
        times[index].push_back(elapsed);
      }
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(end);
    std::cout << std::fixed << std::setprecision(9) << "{\"T\":" << t << ",\"rows\":[";
    for (size_t index = 0; index < modules.size(); ++index) {
      const auto stats = Stats(times[index]);
      if (index) std::cout << ',';
      std::cout << "{\"implementation\":\"" << modules[index].name << "\",\"median_ms\":" << stats[0]
                << ",\"p10_ms\":" << stats[1] << ",\"p90_ms\":" << stats[2] << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
