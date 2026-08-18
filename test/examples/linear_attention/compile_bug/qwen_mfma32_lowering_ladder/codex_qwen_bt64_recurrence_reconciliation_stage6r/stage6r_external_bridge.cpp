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
  if (status == hipSuccess) return true;
  g_error = std::string(where) + ": " + hipGetErrorString(status);
  return false;
}

hipFunction_t Lookup(const char *path, const char *symbol) {
  const std::string key = std::string(path) + "\n" + symbol;
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto existing = g_entries.find(key);
  if (existing != g_entries.end()) return existing->second.function;
  Entry entry;
  if (!Check(hipInit(0), "hipInit") || !Check(hipModuleLoad(&entry.module, path), "hipModuleLoad") ||
      !Check(hipModuleGetFunction(&entry.function, entry.module, symbol), "hipModuleGetFunction")) {
    if (entry.module != nullptr) (void)hipModuleUnload(entry.module);
    return nullptr;
  }
  g_entries.emplace(key, entry);
  return entry.function;
}

}  // namespace

extern "C" const char *stage6r_external_last_error() { return g_error.c_str(); }

// The fixed Qwen recurrence ABI has eight runtime tensor pointers and T. The
// remaining source-level nullable arguments are constexpr-specialized away;
// ROCm appends the two standard scratch pointers used by the compiled code.
extern "C" int stage6r_external_recurrence_launch(
    const char *hsaco_path, const char *symbol, uint64_t stream_value,
    uint32_t grid_x, uint32_t grid_y, uint32_t block_x, uint32_t dynamic_lds,
    int32_t t, void *k, void *v, void *w, void *v_new, void *g, void *h,
    void *h0, void *ht) {
  g_error.clear();
  if (hsaco_path == nullptr || symbol == nullptr || k == nullptr || v == nullptr || w == nullptr ||
      v_new == nullptr || g == nullptr || h == nullptr || h0 == nullptr || ht == nullptr || t < 64 ||
      t % 64 != 0 || grid_x == 0 || grid_y == 0 || block_x == 0) {
    g_error = "stage6r bridge received an invalid fixed-recurrence launch";
    return 1;
  }
  hipFunction_t function = Lookup(hsaco_path, symbol);
  if (function == nullptr) return 2;
  hipDeviceptr_t global_scratch = 0;
  hipDeviceptr_t profile_scratch = 0;
  void *params[] = {&k, &v, &w, &v_new, &g, &h, &h0, &ht, &t,
                    &global_scratch, &profile_scratch};
  if (!Check(hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x, 1, 1,
                                   dynamic_lds,
                                   reinterpret_cast<hipStream_t>(stream_value),
                                   params, nullptr),
             "hipModuleLaunchKernel")) {
    return 3;
  }
  return 0;
}
