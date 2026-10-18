#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

// KYTY_PERF_STATS diagnostics: page-protection changes and GPU-tracking faults.
struct ProtectStats {
	static void Count(uint64_t bytes) noexcept;
	static void CountFault(bool write) noexcept;
	// KYTY_DBG_CP_READS: write faults per page (PrintFaultPages, every 2 s with NHL27PROTECT).
	static void NoteWriteFaultPage(uint64_t vaddr) noexcept;
	static void PrintFaultPages() noexcept;
	// Totals since start: protection calls, bytes, calls made on the GPU thread, read/write faults.
	static void Read(uint64_t& calls, uint64_t& bytes, uint64_t& gpu_calls, uint64_t& read_faults,
	                 uint64_t& write_faults) noexcept;
};

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;
	// Whether cache tracking currently restricts access to the page holding vaddr.
	[[nodiscard]] bool IsWatched(uint64_t vaddr) const;
	// Whether reads of the page holding vaddr fault (it holds GPU-written bytes not downloaded).
	[[nodiscard]] bool IsReadWatched(uint64_t vaddr) const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
