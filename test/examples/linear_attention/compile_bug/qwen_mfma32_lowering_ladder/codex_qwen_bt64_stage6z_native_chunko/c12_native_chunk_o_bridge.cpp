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

bool Check(hipError_t status, const char *where) {
  if (status == hipSuccess)
    return true;
  g_error = std::string(where) + ": " + hipGetErrorString(status);
  return false;
}

bool Load(const char *path) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_function && g_path == path)
    return true;
  if (g_module)
    (void)hipModuleUnload(g_module);
  g_module = nullptr;
  g_function = nullptr;
  if (!Check(hipInit(0), "hipInit"))
    return false;
  if (!Check(hipModuleLoad(&g_module, path), "hipModuleLoad"))
    return false;
  ++g_module_load_count;
  if (!Check(hipModuleGetFunction(&g_function, g_module, "chunk_fwd_kernel_o"),
             "hipModuleGetFunction(chunk_fwd_kernel_o)")) {
    (void)hipModuleUnload(g_module);
    g_module = nullptr;
    return false;
  }
  g_path = path;
  return true;
}
} // namespace

extern "C" const char *c12_native_chunk_o_last_error() {
  return g_error.c_str();
}

extern "C" unsigned c12_native_chunk_o_module_load_count() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_module_load_count;
}

extern "C" int c12_native_chunk_o_launch(
    const char *hsaco_path, uint64_t stream_value, void *q, void *k, void *v,
    void *h, void *g, void *o, float scale, int32_t num_tokens, int32_t grid_x,
    int32_t grid_y, int32_t grid_z, int32_t block_x, uint32_t shared_bytes) {
  g_error.clear();
  if (!hsaco_path || !q || !k || !v || !h || !g || !o) {
    g_error = "chunk-o external control received a null pointer";
    return 1;
  }
  if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0 || block_x <= 0) {
    g_error = "chunk-o external control received an invalid launch shape";
    return 1;
  }
  if (!Load(hsaco_path))
    return 2;

  // The captured Triton ABI is q, k, v, h, g, o, scale, T, followed by two
  // readnone optional pointer arguments.  The last two pointers are null for
  // this fixed contract.
  void *cu_seqlens = nullptr;
  void *chunk_indices = nullptr;
  void *params[] = {&q, &k, &v, &h, &g, &o, &scale, &num_tokens,
                    &cu_seqlens, &chunk_indices};
  if (!Check(hipModuleLaunchKernel(
                 g_function, static_cast<unsigned>(grid_x),
                 static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z),
                 static_cast<unsigned>(block_x), 1, 1, shared_bytes,
                 reinterpret_cast<hipStream_t>(stream_value), params, nullptr),
             "hipModuleLaunchKernel"))
    return 3;
  return 0;
}
