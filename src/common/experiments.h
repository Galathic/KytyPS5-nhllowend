#ifndef EMULATOR_SRC_COMMON_EXPERIMENTS_H_
#define EMULATOR_SRC_COMMON_EXPERIMENTS_H_

#include <atomic>
#include <cstdint>

namespace Common::Experiments {

// Startup switches that can also be changed while the emulator runs, so A/B comparisons can
// alternate inside one match instead of comparing noisy separate runs.
// Each bit starts from its KYTY_* environment variable: "1" = on, "0" = off, unset = the default
// noted below (the NHL 27 recommended set is on by default). When KYTY_EXPERIMENT_FILE names a file,
// a background thread reads a decimal mask from it every 500 ms; while the file exists its mask
// replaces the environment defaults.
//
// Features that also have a live runtime flag (common/runtimeFlags.h, flags.ctl) are NOT here:
// KYTY_GUEST_SYNC, KYTY_LONG_BUFFER_GC_AGE, KYTY_RENDER_TARGET_MEMO (flag KYTY_RT_DECODE_MEMO),
// KYTY_FAULT_COALESCE, KYTY_SPARSE_BUFFER_GC, KYTY_DIRECT_EOP_INTERRUPTS, KYTY_READBACK_QUEUE.
// Their bit numbers (0, 13, 14, 16, 17, 19, 20) and the unused ones (2, 8, 9) are retired.
// KYTY_RESOURCE_MULTI_CACHE (16-state resource memo) is read only by the resource memo itself.
enum Bit : uint32_t {
	SkipRedundantBinds = 1u << 1u, // KYTY_SKIP_REDUNDANT_BINDS (default on)
	CpuCopyForward     = 1u << 3u, // KYTY_CPU_COPY_FORWARD (default off)
	FalseShareReads    = 1u << 6u, // KYTY_FALSE_SHARE_READS (default off)
	SrtBatchReads      = 1u << 7u, // KYTY_SRT_BATCH_READS (default on; also read by the program cache)
	DirectSrtReads     = 1u << 10u, // KYTY_DIRECT_SRT_READS (default on; SRT reads without the IR walk)
	TextureDescCache   = 1u << 11u, // KYTY_TEXTURE_DESC_CACHE (default on; superseded by the draw-prep delta, no consumer)
	ImageLookupHint    = 1u << 12u, // KYTY_IMAGE_LOOKUP_HINT (default on; skip the region scan for known images)
	AdaptiveResourceMemo = 1u << 15u, // KYTY_ADAPTIVE_RESOURCE_MEMO (default on; superseded by KYTY_RESOURCE_MEMO_BYPASS, no consumer)
	CleanFromDirect      = 1u << 18u, // KYTY_CLEAN_FROM_DIRECT (default on; strict evaluator reuses direct SRT words)
	AsyncSubmit          = 1u << 21u, // KYTY_ASYNC_SUBMIT (default off; vkQueueSubmit on a worker thread, in tick order)
	LazySharedWrites     = 1u << 22u, // KYTY_LAZY_SHARED_WRITES (default off; CPU stores next to few GPU-written bytes do not wait)
	NativeFp16           = 1u << 23u, // KYTY_NATIVE_FP16 (default off; f16vec2 math for packed f16 ops, needs shaderFloat16)
	FaultPageReads       = 1u << 24u, // KYTY_FAULT_PAGE_READS (default off; SRT reads of pages that faulted before use the backing)
	GpuColorClear        = 1u << 26u, // KYTY_GPU_COLOR_CLEAR (default off; fast-clear checks of GPU-written metadata on the GPU)
	ExecSelects          = 1u << 25u, // KYTY_EXEC_SELECTS (default off; drop EXEC-mask selects nobody observes, part of shader keys)
	PooledBdaBuffers     = 1u << 27u, // KYTY_POOLED_BDA_BUFFERS (default off; small device-address buffers share VMA blocks)
	AsyncGuestReads      = 1u << 28u, // KYTY_ASYNC_GUEST_READS (default off; a guest thread waits for its own readback)
	BdaDirtyRegions      = 1u << 29u, // KYTY_BDA_DIRTY_REGIONS (default off; BDA preparation syncs only CPU-written regions)
	// A/B only (no environment variable): selects the code a pending change replaces.
	AbLegacy = 1u << 31u,
};

namespace Detail {
// The live mask; bit 32 stays set until the first use reads the environment. Enabled() runs
// millions of times per second on the GPU thread, so its fast path is inline.
inline constexpr uint64_t Uninitialized = uint64_t {1} << 32u;
extern std::atomic<uint64_t> g_mask;
[[nodiscard]] uint64_t InitMask() noexcept;
} // namespace Detail

[[nodiscard]] inline uint32_t Mask() noexcept {
	uint64_t mask = Detail::g_mask.load(std::memory_order_relaxed);
	if ((mask & Detail::Uninitialized) != 0) [[unlikely]] {
		mask = Detail::InitMask();
	}
	return static_cast<uint32_t>(mask);
}

[[nodiscard]] inline bool Enabled(Bit bit) noexcept {
	return (Mask() & bit) != 0;
}

} // namespace Common::Experiments

namespace Common::Timeline {

// Debugging: KYTY_DBG_TIMELINE=<seconds> records marks from every thread for 1.5 s after that
// delay, then prints them once in time order (NHL27TL lines): who waits on whom in a frame.
[[nodiscard]] bool Enabled() noexcept;
void Mark(const char* what, uint64_t value = 0) noexcept;

} // namespace Common::Timeline

#endif // EMULATOR_SRC_COMMON_EXPERIMENTS_H_
