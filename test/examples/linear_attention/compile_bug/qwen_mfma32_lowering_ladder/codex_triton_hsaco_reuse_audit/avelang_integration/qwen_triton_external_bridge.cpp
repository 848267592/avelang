#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <mutex>
#include <string>

namespace {
std::mutex g_mutex;
hipModule_t g_module = nullptr;
hipFunction_t g_function = nullptr;
std::string g_path;
unsigned g_module_load_count = 0;
thread_local std::string g_error;
constexpr unsigned kGridX = 4, kGridY = 8, kBlockX = 256, kShared = 57344;

bool Check(hipError_t status, const char *where) {
  if (status == hipSuccess) return true;
  g_error = std::string(where) + ": " + hipGetErrorString(status);
  return false;
}

bool Load(const char *path) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_function && g_path == path) return true;
  if (g_module) (void)hipModuleUnload(g_module);
  g_module = nullptr;
  g_function = nullptr;
  if (!Check(hipInit(0), "hipInit")) return false;
  if (!Check(hipModuleLoad(&g_module, path), "hipModuleLoad")) return false;
  ++g_module_load_count;
  if (!Check(hipModuleGetFunction(&g_function, g_module,
                                  "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
             "hipModuleGetFunction")) {
    (void)hipModuleUnload(g_module); g_module = nullptr; return false;
  }
  g_path = path;
  return true;
}
} // namespace

extern "C" const char *qwen_triton_external_last_error() { return g_error.c_str(); }
extern "C" unsigned qwen_triton_external_module_load_count() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_module_load_count;
}

extern "C" int qwen_triton_external_launch(
    const char *hsaco_path, uint64_t stream_value, void *k, void *v, void *w,
    void *v_new, void *g, void *h, void *h0, void *ht) {
  g_error.clear();
  if (!hsaco_path || !k || !v || !w || !v_new || !g || !h || !h0 || !ht) {
    g_error = "external BT64 kernel requires non-null k/v/w/v_new/g/h/h0/ht";
    return 1;
  }
  if (!Load(hsaco_path)) return 2;
  int32_t t = 64;
  hipDeviceptr_t global_scratch = 0, profile_scratch = 0;
  void *params[] = {&k, &v, &w, &v_new, &g, &h, &h0, &ht, &t,
                    &global_scratch, &profile_scratch};
  if (!Check(hipModuleLaunchKernel(g_function, kGridX, kGridY, 1, kBlockX, 1,
                                   1, kShared, reinterpret_cast<hipStream_t>(stream_value),
                                   params, nullptr), "hipModuleLaunchKernel"))
    return 3;
  return 0;
}
