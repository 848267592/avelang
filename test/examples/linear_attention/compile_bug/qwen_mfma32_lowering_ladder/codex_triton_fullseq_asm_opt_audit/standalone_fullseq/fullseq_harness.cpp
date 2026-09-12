#include <hip/hip_runtime.h>

#include <algorithm>
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

template <typename T> void Write(const std::string &path, const std::vector<T> &values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot write " + path);
  stream.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

struct DeviceBuffers {
  void *k = nullptr, *v = nullptr, *w = nullptr, *v_new = nullptr, *g = nullptr, *h = nullptr, *h0 = nullptr, *ht = nullptr;
  ~DeviceBuffers() { for (void *ptr : {k, v, w, v_new, g, h, h0, ht}) if (ptr) (void)hipFree(ptr); }
};

void CopyToDevice(void **dst, const void *src, size_t bytes) {
  Check(hipMalloc(dst, bytes), "hipMalloc");
  Check(hipMemcpy(*dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpy");
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: fullseq_harness <hsaco> <T> <input-dir> <output-dir> <warmup> <repeat>\n";
    return 64;
  }
  try {
    const std::string hsaco = argv[1], input_dir = argv[3], output_dir = argv[4];
    int32_t t = std::stoi(argv[2]);
    const int warmup = std::stoi(argv[5]), repeat = std::stoi(argv[6]);
    if (t < static_cast<int32_t>(kBT) || t % kBT) throw std::runtime_error("T must be a positive multiple of 64");
    const size_t tokens = static_cast<size_t>(t), chunks = tokens / kBT;
    const size_t k_elements = tokens * kHg * kK, tv_elements = tokens * kH * kV, g_elements = tokens * kH;
    const size_t h_elements = chunks * kStateElements;
    auto k = Read<uint16_t>(input_dir + "/k_bf16.bin", k_elements);
    auto v = Read<float>(input_dir + "/v_fp32.bin", tv_elements);
    auto w = Read<float>(input_dir + "/w_fp32.bin", tv_elements);
    auto g = Read<float>(input_dir + "/g_fp32.bin", g_elements);
    auto h0 = Read<float>(input_dir + "/h0_fp32.bin", kStateElements);

    Check(hipInit(0), "hipInit");
    hipModule_t module = nullptr; hipFunction_t function = nullptr;
    Check(hipModuleLoad(&module, hsaco.c_str()), "hipModuleLoad");
    Check(hipModuleGetFunction(&function, module, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"), "hipModuleGetFunction");
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
    const auto launch = [&] { Check(hipModuleLaunchKernel(function, 4, 8, 1, 256, 1, 1, kDynamicSharedBytes, nullptr, params, nullptr), "hipModuleLaunchKernel"); };
    launch(); Check(hipDeviceSynchronize(), "initial synchronize");
    std::vector<uint16_t> h(h_elements); std::vector<float> v_new(tv_elements), ht(kStateElements);
    Check(hipMemcpy(h.data(), device.h, h.size() * sizeof(uint16_t), hipMemcpyDeviceToHost), "copy h");
    Check(hipMemcpy(v_new.data(), device.v_new, v_new.size() * sizeof(float), hipMemcpyDeviceToHost), "copy v_new");
    Check(hipMemcpy(ht.data(), device.ht, ht.size() * sizeof(float), hipMemcpyDeviceToHost), "copy ht");
    Write(output_dir + "/h_bf16.bin", h); Write(output_dir + "/v_new_fp32.bin", v_new); Write(output_dir + "/ht_fp32.bin", ht);
    std::vector<float> elapsed;
    if (repeat > 0) {
      for (int i = 0; i < warmup; ++i) launch();
      Check(hipDeviceSynchronize(), "warmup synchronize");
      hipEvent_t start = nullptr, end = nullptr; Check(hipEventCreate(&start), "event start"); Check(hipEventCreate(&end), "event end");
      for (int i = 0; i < repeat; ++i) { float value = 0; Check(hipEventRecord(start), "record start"); launch(); Check(hipEventRecord(end), "record end"); Check(hipEventSynchronize(end), "event synchronize"); Check(hipEventElapsedTime(&value, start, end), "elapsed"); elapsed.push_back(value); }
      (void)hipEventDestroy(start); (void)hipEventDestroy(end); std::sort(elapsed.begin(), elapsed.end());
    }
    Check(hipModuleUnload(module), "hipModuleUnload");
    const auto percentile = [&](size_t index) { return elapsed.empty() ? 0.0f : elapsed[index]; };
    std::cout << std::fixed << std::setprecision(9)
              << "{\"symbol\":\"chunk_gated_delta_rule_fwd_kernel_h_blockdim64\",\"T\":" << t
              << ",\"kernarg_size\":88,\"grid\":[4,8,1],\"block\":[256,1,1],\"shared_memory\":" << kDynamicSharedBytes
              << ",\"median_ms\":" << percentile(elapsed.size() / 2)
              << ",\"p10_ms\":" << percentile(elapsed.empty() ? 0 : (elapsed.size() - 1) / 10)
              << ",\"p90_ms\":" << percentile(elapsed.empty() ? 0 : (elapsed.size() - 1) * 9 / 10) << "}\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
