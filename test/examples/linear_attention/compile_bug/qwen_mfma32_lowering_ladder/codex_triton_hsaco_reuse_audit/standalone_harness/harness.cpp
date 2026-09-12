#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kT = 64;
constexpr size_t kHg = 4;
constexpr size_t kH = 8;
constexpr size_t kK = 128;
constexpr size_t kV = 128;
constexpr size_t kKElements = kT * kHg * kK;
constexpr size_t kTVElements = kT * kH * kV;
constexpr size_t kGElements = kT * kH;
constexpr size_t kStateElements = kH * kV * kK;
constexpr unsigned int kDynamicSharedBytes = 57344;

void Check(hipError_t error, const char *expr) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(expr) + ": " + hipGetErrorString(error));
}

template <typename T> std::vector<T> Read(const std::string &path, size_t count) {
  std::vector<T> data(count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot read " + path);
  stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (stream.gcount() != static_cast<std::streamsize>(count * sizeof(T)))
    throw std::runtime_error("unexpected size for " + path);
  return data;
}

template <typename T> void Write(const std::string &path, const std::vector<T> &data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error("cannot write " + path);
  stream.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(T)));
}

// Exact selected Triton code-object ABI. The two final null pointers are the
// non-varlen cu_seqlens/chunk_offsets slots retained in the metadata.
struct KernelArgs {
  void *k;
  void *v;
  void *w;
  void *v_new;
  void *g;
  void *h;
  void *h0;
  void *ht;
  int32_t t;
  int32_t padding;
  void *cu_seqlens;
  void *chunk_offsets;
};
static_assert(sizeof(KernelArgs) == 88, "Triton metadata says kernarg is 88 bytes");

struct DeviceBuffers {
  void *k = nullptr, *v = nullptr, *w = nullptr, *v_new = nullptr;
  void *g = nullptr, *h = nullptr, *h0 = nullptr, *ht = nullptr;
  ~DeviceBuffers() {
    for (void *p : {k, v, w, v_new, g, h, h0, ht})
      if (p) hipFree(p);
  }
};

void CopyToDevice(void **dst, const void *src, size_t bytes) {
  Check(hipMalloc(dst, bytes), "hipMalloc");
  Check(hipMemcpy(*dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpyHostToDevice");
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: harness <hsaco> <input-dir> <output-dir> <warmup> <repeat>\n";
    return 64;
  }
  try {
    const std::string hsaco = argv[1];
    const std::string in_dir = argv[2];
    const std::string out_dir = argv[3];
    const int warmup = std::stoi(argv[4]);
    const int repeat = std::stoi(argv[5]);

    auto k = Read<uint16_t>(in_dir + "/k_bf16.bin", kKElements);
    auto v = Read<float>(in_dir + "/v_fp32.bin", kTVElements);
    auto w = Read<float>(in_dir + "/w_fp32.bin", kTVElements);
    auto g = Read<float>(in_dir + "/g_fp32.bin", kGElements);
    auto h0 = Read<float>(in_dir + "/h0_fp32.bin", kStateElements);

    Check(hipInit(0), "hipInit");
    hipModule_t module = nullptr;
    hipFunction_t kernel = nullptr;
    Check(hipModuleLoad(&module, hsaco.c_str()), "hipModuleLoad");
    Check(hipModuleGetFunction(&kernel, module,
                               "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
          "hipModuleGetFunction");

    DeviceBuffers d;
    CopyToDevice(&d.k, k.data(), k.size() * sizeof(uint16_t));
    CopyToDevice(&d.v, v.data(), v.size() * sizeof(float));
    CopyToDevice(&d.w, w.data(), w.size() * sizeof(float));
    CopyToDevice(&d.g, g.data(), g.size() * sizeof(float));
    CopyToDevice(&d.h0, h0.data(), h0.size() * sizeof(float));
    Check(hipMalloc(&d.v_new, kTVElements * sizeof(float)), "hipMalloc(v_new)");
    Check(hipMalloc(&d.h, kStateElements * sizeof(uint16_t)), "hipMalloc(h)");
    Check(hipMalloc(&d.ht, kStateElements * sizeof(float)), "hipMalloc(ht)");

    // Triton's generated AMD launcher uses the kernelParams form, not the
    // HIP_LAUNCH_PARAM_BUFFER form. It appends global/profile scratch slots
    // even when both are null; those slots occupy metadata offsets 72 and 80.
    int32_t t_arg = static_cast<int32_t>(kT);
    hipDeviceptr_t global_scratch = 0;
    hipDeviceptr_t profile_scratch = 0;
    void *params[] = {&d.k, &d.v, &d.w, &d.v_new, &d.g, &d.h, &d.h0, &d.ht,
                      &t_arg, &global_scratch, &profile_scratch};
    const auto launch = [&] {
      Check(hipModuleLaunchKernel(kernel, 4, 8, 1, 256, 1, 1, kDynamicSharedBytes, nullptr,
                                  params, nullptr),
            "hipModuleLaunchKernel");
    };

    launch();
    Check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    std::vector<uint16_t> h(kStateElements);
    std::vector<float> v_new(kTVElements), ht(kStateElements);
    Check(hipMemcpy(h.data(), d.h, h.size() * sizeof(uint16_t), hipMemcpyDeviceToHost), "copy h");
    Check(hipMemcpy(v_new.data(), d.v_new, v_new.size() * sizeof(float), hipMemcpyDeviceToHost), "copy v_new");
    Check(hipMemcpy(ht.data(), d.ht, ht.size() * sizeof(float), hipMemcpyDeviceToHost), "copy ht");
    Write(out_dir + "/h_bf16.bin", h);
    Write(out_dir + "/v_new_fp32.bin", v_new);
    Write(out_dir + "/ht_fp32.bin", ht);

    double median_ms = 0.0, p10_ms = 0.0, p90_ms = 0.0;
    if (repeat > 0) {
      for (int i = 0; i < warmup; ++i) launch();
      Check(hipDeviceSynchronize(), "warmup synchronize");
      hipEvent_t start = nullptr, end = nullptr;
      Check(hipEventCreate(&start), "hipEventCreate(start)");
      Check(hipEventCreate(&end), "hipEventCreate(end)");
      std::vector<float> times;
      times.reserve(repeat);
      for (int i = 0; i < repeat; ++i) {
        Check(hipEventRecord(start, nullptr), "hipEventRecord(start)");
        launch();
        Check(hipEventRecord(end, nullptr), "hipEventRecord(end)");
        Check(hipEventSynchronize(end), "hipEventSynchronize(end)");
        float elapsed = 0;
        Check(hipEventElapsedTime(&elapsed, start, end), "hipEventElapsedTime");
        times.push_back(elapsed);
      }
      std::sort(times.begin(), times.end());
      median_ms = times[times.size() / 2];
      p10_ms = times[(times.size() - 1) / 10];
      p90_ms = times[(times.size() - 1) * 9 / 10];
      hipEventDestroy(start);
      hipEventDestroy(end);
    }
    Check(hipModuleUnload(module), "hipModuleUnload");
    std::cout << std::fixed << std::setprecision(9)
              << "{\"symbol\":\"chunk_gated_delta_rule_fwd_kernel_h_blockdim64\""
              << ",\"kernarg_size\":88,\"grid\":[4,8,1],\"block\":[256,1,1]"
              << ",\"shared_memory\":" << kDynamicSharedBytes << ",\"median_ms\":" << median_ms
              << ",\"p10_ms\":" << p10_ms << ",\"p90_ms\":" << p90_ms << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
