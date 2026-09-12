#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <mutex>
#include <string>

namespace {

constexpr unsigned kGridX = 4;
constexpr unsigned kGridY = 8;
constexpr unsigned kBlockX = 256;
constexpr unsigned kDynamicLdsBytes = 57344;
constexpr const char *kKernelName = "qwen_gdn_bt64_gfx942_asm_v0";

std::mutex g_mutex;
hipModule_t g_module = nullptr;
hipFunction_t g_function = nullptr;
std::string g_path;
unsigned g_loads = 0;
thread_local std::string g_error;

bool Check(hipError_t status, const char *where) {
  if (status == hipSuccess) return true;
  g_error = std::string(where) + ": " + hipGetErrorString(status);
  return false;
}

bool Load(const char *path) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_function != nullptr && g_path == path) return true;
  if (g_module != nullptr) (void)hipModuleUnload(g_module);
  g_module = nullptr;
  g_function = nullptr;
  if (!Check(hipInit(0), "hipInit")) return false;
  if (!Check(hipModuleLoad(&g_module, path), "hipModuleLoad")) return false;
  ++g_loads;
  if (!Check(hipModuleGetFunction(&g_function, g_module, kKernelName), "hipModuleGetFunction")) {
    (void)hipModuleUnload(g_module);
    g_module = nullptr;
    return false;
  }
  g_path = path;
  return true;
}

}  // namespace

extern "C" const char *qwen_gdn_bt64_gfx942_asm_v0_last_error() {
  return g_error.c_str();
}

extern "C" unsigned qwen_gdn_bt64_gfx942_asm_v0_module_load_count() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_loads;
}

extern "C" int qwen_gdn_bt64_gfx942_asm_v0_launch(
    const char *hsaco_path,
    uint64_t stream_value,
    int32_t t,
    void *k,
    void *v,
    void *w,
    void *v_new,
    void *g,
    void *h,
    void *h0,
    void *ht) {
  g_error.clear();
  if (hsaco_path == nullptr || k == nullptr || v == nullptr || w == nullptr ||
      v_new == nullptr || g == nullptr || h == nullptr || h0 == nullptr ||
      ht == nullptr || t < 64 || t % 64 != 0) {
    g_error = "qwen asm v0 requires non-null buffers and T divisible by 64";
    return 1;
  }
  if (!Load(hsaco_path)) return 2;

  hipDeviceptr_t global_scratch = 0;
  hipDeviceptr_t profile_scratch = 0;
  void *params[] = {&k, &v, &w, &v_new, &g, &h, &h0, &ht, &t,
                    &global_scratch, &profile_scratch};
  if (!Check(hipModuleLaunchKernel(g_function, kGridX, kGridY, 1, kBlockX, 1,
                                   1, kDynamicLdsBytes,
                                   reinterpret_cast<hipStream_t>(stream_value),
                                   params, nullptr),
             "hipModuleLaunchKernel")) {
    return 3;
  }
  return 0;
}
