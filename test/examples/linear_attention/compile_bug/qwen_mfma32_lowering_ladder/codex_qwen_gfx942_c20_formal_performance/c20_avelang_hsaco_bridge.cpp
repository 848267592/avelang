#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
struct Entry {
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
};

std::mutex g_mutex;
std::unordered_map<std::string, Entry> g_entries;
thread_local std::string g_error;

bool Check(hipError_t status, const char *where) {
  if (status == hipSuccess)
    return true;
  g_error = std::string(where) + ": " + hipGetErrorString(status);
  return false;
}

hipFunction_t Lookup(const char *path, const char *symbol) {
  const std::string key = std::string(path) + "\n" + symbol;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_entries.find(key);
  if (it != g_entries.end())
    return it->second.function;

  Entry entry;
  if (!Check(hipInit(0), "hipInit") ||
      !Check(hipModuleLoad(&entry.module, path), "hipModuleLoad") ||
      !Check(hipModuleGetFunction(&entry.function, entry.module, symbol),
             "hipModuleGetFunction")) {
    if (entry.module != nullptr)
      (void)hipModuleUnload(entry.module);
    return nullptr;
  }
  g_entries.emplace(key, entry);
  return entry.function;
}
} // namespace

extern "C" const char *c20_avelang_last_error() { return g_error.c_str(); }

// The frozen C18/P2/C19 Avelang code objects have six pointer arguments.  T,
// num_chunks and scale are constexpr-specialized in each T=2048 object, so
// there are no hidden runtime scalar arguments in this bridge.
extern "C" int c20_avelang_launch(
    const char *hsaco_path, const char *symbol, uint64_t stream_value,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x,
    void *q, void *k, void *v_new, void *h, void *g, void *output) {
  g_error.clear();
  if (hsaco_path == nullptr || symbol == nullptr || q == nullptr || k == nullptr ||
      v_new == nullptr || h == nullptr || g == nullptr || output == nullptr ||
      grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0) {
    g_error = "invalid C20 frozen Avelang launch arguments";
    return 1;
  }

  hipFunction_t function = Lookup(hsaco_path, symbol);
  if (function == nullptr)
    return 2;

  void *params[] = {&q, &k, &v_new, &h, &g, &output};
  if (!Check(hipModuleLaunchKernel(
                 function, grid_x, grid_y, grid_z, block_x, 1, 1, 0,
                 reinterpret_cast<hipStream_t>(stream_value), params, nullptr),
             "hipModuleLaunchKernel"))
    return 3;
  return 0;
}
