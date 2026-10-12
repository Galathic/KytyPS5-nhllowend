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
};

[[nodiscard]] bool Enabled(Bit bit) noexcept;
[[nodiscard]] uint32_t Mask() noexcept;

} // namespace Common::Experiments

#endif // EMULATOR_SRC_COMMON_EXPERIMENTS_H_
