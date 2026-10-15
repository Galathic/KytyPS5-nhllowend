#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_STAGESTATS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_STAGESTATS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

// KYTY_DBG_STAGE_STATS=1: where the GPU pipeline's execution and resolve threads wait, measured
// only when they actually wait (an uncontended lock or a ready operation costs one branch).
// Printed every 5 s as NHL27STALLS by the execution thread. With the frame rate capped (menus,
// a paused match), execution-thread busy time per frame still shows what a change saves.
namespace Libs::Graphics::StageStats {

enum Counter : uint32_t {
	ExecIdle,        // execution thread waiting for operations
	ExecProgramLock, // execution thread waiting for the program lock
	ResolveProgramLock,
	ResolveQueueFull, // resolve thread waiting for room in the execution queue
	CounterCount,
};

enum class Role : uint8_t { Other, Execution, Resolve };

inline std::atomic<uint64_t> g_ns[CounterCount] {};
inline std::atomic<uint64_t> g_ops {0}; // operations the execution thread ran (written by it only)
// Execution-thread time and count by operation kind (GpuOperationKind; execution thread only).
inline std::atomic<uint64_t> g_kind_ns[8] {};
inline std::atomic<uint64_t> g_kind_count[8] {};
inline thread_local Role     t_role = Role::Other;

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_STAGE_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline void Add(Counter counter, std::chrono::steady_clock::time_point start) {
	g_ns[counter].fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                                  std::chrono::steady_clock::now() - start)
	                                                  .count()),
	                        std::memory_order_relaxed);
}

// The program-lock counter of the calling stage thread, or CounterCount for other threads.
[[nodiscard]] inline Counter ProgramLockCounter() {
	switch (t_role) {
		case Role::Execution: return ExecProgramLock;
		case Role::Resolve: return ResolveProgramLock;
		default: return CounterCount;
	}
}

} // namespace Libs::Graphics::StageStats

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_STAGESTATS_H_ */
