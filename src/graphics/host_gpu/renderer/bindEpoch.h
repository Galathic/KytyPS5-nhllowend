#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDEPOCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDEPOCH_H_

#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

// Incremented by every recording path outside RenderExecutor::CommitBindings that pushes
// descriptors or push constants (or binds descriptor sets), so the redundant-bind cache in
// CommitBindings never assumes stale bound state.
inline std::atomic<uint64_t> g_bind_epoch {0};

inline void InvalidateRedundantBinds() noexcept {
	g_bind_epoch.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t BindEpoch() noexcept {
	return g_bind_epoch.load(std::memory_order_relaxed);
}

// Incremented for every recorded draw and dispatch. Buffer copies/fills and image transitions
// carry their own barriers, so two guest global barriers with no draw or dispatch between them
// in one command buffer are redundant.
inline std::atomic<uint64_t> g_work_epoch {0};

inline void NoteShaderWork() noexcept {
	g_work_epoch.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t WorkEpoch() noexcept {
	return g_work_epoch.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDEPOCH_H_
