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
  if (!Check(hipInit(0), "hipInit") ||
      !Check(hipModuleLoad(&entry.module, path), "hipModuleLoad") ||
      !Check(hipModuleGetFunction(&entry.function, entry.module, symbol),
             "hipModuleGetFunction")) {
    if (entry.module != nullptr) (void)hipModuleUnload(entry.module);
    return nullptr;
  }
  g_entries.emplace(key, entry);
  return entry.function;
}

}  // namespace

extern "C" const char *stage6s_external_solve_last_error() {
  return g_error.c_str();
}

// The captured Stage 5B code object has exactly two runtime kernargs, a and
// out. T and chunk count were constexpr-specialized when the immutable HSACO
// was compiled, so they must not be appended to this ABI.
extern "C" int stage6s_external_hierarchical_solve_launch(
    const char *hsaco_path, const char *symbol, uint64_t stream_value,
    uint32_t grid_x, uint32_t block_x, void *a, void *out) {
  g_error.clear();
  if (hsaco_path == nullptr || symbol == nullptr || a == nullptr ||
      out == nullptr || grid_x == 0 || block_x != 256) {
    g_error = "stage6s solve bridge received an invalid fixed solve launch";
    return 1;
  }
  hipFunction_t function = Lookup(hsaco_path, symbol);
  if (function == nullptr) return 2;
  void *params[] = {&a, &out};
  if (!Check(hipModuleLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1,
                                   /*sharedMemBytes=*/0,
                                   reinterpret_cast<hipStream_t>(stream_value),
                                   params, nullptr),
             "hipModuleLaunchKernel")) {
    return 3;
  }
  return 0;
}
