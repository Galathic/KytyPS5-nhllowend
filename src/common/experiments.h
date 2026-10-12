#ifndef EMULATOR_SRC_COMMON_EXPERIMENTS_H_
#define EMULATOR_SRC_COMMON_EXPERIMENTS_H_

#include <cstdint>

namespace Common::Experiments {

// Opt-in performance experiments that can be switched while the emulator runs, so A/B
// comparisons can alternate inside one match instead of comparing noisy separate runs.
// Each bit starts from its KYTY_* environment variable ("1" = on). When KYTY_EXPERIMENT_FILE
// names a file, a background thread reads a decimal mask from it every 500 ms; while the file
// exists its mask replaces the environment defaults.
enum Bit : uint32_t {
	GuestSync          = 1u << 0u, // KYTY_GUEST_SYNC
	SkipRedundantBinds = 1u << 1u, // KYTY_SKIP_REDUNDANT_BINDS
	DedupBarriers      = 1u << 2u, // KYTY_DEDUP_BARRIERS
	CpuCopyForward     = 1u << 3u, // KYTY_CPU_COPY_FORWARD
	FalseShareReads    = 1u << 6u, // KYTY_FALSE_SHARE_READS
	SrtBatchReads      = 1u << 7u, // KYTY_SRT_BATCH_READS (also read directly by the program cache)
	ResourceMultiCache = 1u << 8u, // KYTY_RESOURCE_MULTI_CACHE (16-state resource memo)
	BackingReadCache   = 1u << 9u, // KYTY_BACKING_READ_CACHE (lock-free per-thread backing reads)
	DirectSrtReads     = 1u << 10u, // KYTY_DIRECT_SRT_READS (SRT reads without the IR walk)
	TextureDescCache   = 1u << 11u, // KYTY_TEXTURE_DESC_CACHE (memoized T# decoding per draw)
	ImageLookupHint    = 1u << 12u, // KYTY_IMAGE_LOOKUP_HINT (skip the region scan for known images)
	LongBufferGcAge    = 1u << 13u, // KYTY_LONG_BUFFER_GC_AGE (keep unused buffers 10x longer)
	RenderTargetMemo   = 1u << 14u, // KYTY_RENDER_TARGET_MEMO (memoized color target decoding)
	AdaptiveResourceMemo = 1u << 15u, // KYTY_ADAPTIVE_RESOURCE_MEMO (skip read capture for volatile inputs)
	FaultCoalesce        = 1u << 16u, // KYTY_FAULT_COALESCE (a write fault releases its clean 64 KiB window)
	SparseBufferGc       = 1u << 17u, // KYTY_SPARSE_BUFFER_GC (scan for idle buffers every 16th submission)
	CleanFromDirect      = 1u << 18u, // KYTY_CLEAN_FROM_DIRECT (strict evaluator reuses direct SRT words)
	DirectEopInterrupts  = 1u << 19u, // KYTY_DIRECT_EOP_INTERRUPTS (EOP interrupts at CP time, like labels)
	ReadbackQueue        = 1u << 20u, // KYTY_READBACK_QUEUE (downloads wait only for the buffer's last writer)
	AsyncSubmit          = 1u << 21u, // KYTY_ASYNC_SUBMIT (vkQueueSubmit on a worker thread, in tick order)
	LazySharedWrites     = 1u << 22u, // KYTY_LAZY_SHARED_WRITES (CPU writes next to few GPU-written bytes do not wait)
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
