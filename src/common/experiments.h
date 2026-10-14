#ifndef EMULATOR_SRC_COMMON_EXPERIMENTS_H_
#define EMULATOR_SRC_COMMON_EXPERIMENTS_H_

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
	SrtJit             = 1u << 4u, // KYTY_SRT_JIT (default off; upstream x64 flat walker for plans without control flow)
	FalseShareReads    = 1u << 6u, // KYTY_FALSE_SHARE_READS (default off)
	SrtBatchReads      = 1u << 7u, // KYTY_SRT_BATCH_READS (default on; also read by the program cache)
	DirectSrtReads     = 1u << 10u, // KYTY_DIRECT_SRT_READS (default on; SRT reads without the IR walk)
	TextureDescCache   = 1u << 11u, // KYTY_TEXTURE_DESC_CACHE (default on; superseded by the draw-prep delta, no consumer)
	ImageLookupHint    = 1u << 12u, // KYTY_IMAGE_LOOKUP_HINT (default on; skip the region scan for known images)
	AdaptiveResourceMemo = 1u << 15u, // KYTY_ADAPTIVE_RESOURCE_MEMO (default on; superseded by KYTY_RESOURCE_MEMO_BYPASS, no consumer)
	CleanFromDirect      = 1u << 18u, // KYTY_CLEAN_FROM_DIRECT (default on; strict evaluator reuses direct SRT words)
	LazySharedWrites     = 1u << 22u, // KYTY_LAZY_SHARED_WRITES (default off; CPU stores next to few GPU-written bytes do not wait)
	ProtectSharedLock    = 1u << 23u, // KYTY_PROTECT_SHARED_LOCK (default off; page-tracking protections of different pages run in parallel)
	ClockWaitPredict     = 1u << 24u, // KYTY_CLOCK_WAIT_PREDICT (default off; WAIT_REG_MEM on a pending clock write need not wait for it)
	FalseShareAnyThread  = 1u << 29u, // KYTY_FALSE_SHARE_ANY_THREAD (default off; CP and guest reads of clean bytes on GPU-dirty pages use the backing)
	CpCleanRegReads      = 1u << 25u, // KYTY_CP_CLEAN_REG_READS (default off; the command processor reads clean indirect register tables from the backing)
	CpSkipIdleSpin       = 1u << 26u, // KYTY_CP_SKIP_IDLE_SPIN (default off; work queued while the command processor waited for the execution thread starts at once, not after a 2 ms spin)
	SpeculateDirtyReads  = 1u << 30u, // KYTY_SPECULATE_DIRTY_READS (default off; shader-table reads of GPU-written bytes the GPU keeps rewriting unchanged use the backing, checked later)
	// A/B only (no environment variable): selects the code a pending change replaces.
	AbLegacy = 1u << 31u,
};

[[nodiscard]] bool Enabled(Bit bit) noexcept;
[[nodiscard]] uint32_t Mask() noexcept;

} // namespace Common::Experiments

namespace Common::Timeline {

// Debugging: KYTY_DBG_TIMELINE=<seconds> records marks from every thread for 1.5 s after that
// delay, then prints them once in time order (NHL27TL lines): who waits on whom in a frame.
[[nodiscard]] bool Enabled() noexcept;
void Mark(const char* what, uint64_t value = 0) noexcept;

} // namespace Common::Timeline

#endif // EMULATOR_SRC_COMMON_EXPERIMENTS_H_
