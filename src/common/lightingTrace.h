#ifndef KYTY_COMMON_LIGHTING_TRACE_H_
#define KYTY_COMMON_LIGHTING_TRACE_H_

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace Common {
template <typename... Args>
inline void LightingApiTrace(const char* format, Args... args) {
    static const bool enabled = [] {
        const char* value = std::getenv("KYTY_DBG_LIGHT_APIS");
        return value != nullptr && value[0] == '1';
    }();
    if (!enabled) return;
    static std::atomic<unsigned> calls {0};
    if (calls.fetch_add(1, std::memory_order_relaxed) >= 128) return;
    std::printf("NHL27LIGHTAPI: ");
    std::printf(format, args...);
    std::printf("\n");
}
} // namespace Common
#endif
