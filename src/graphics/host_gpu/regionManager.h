#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			std::atomic_signal_fence(std::memory_order_seq_cst);
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			// Summary first (over-report), then the page bits; see cpu_summary.
			if constexpr (source == DirtySource::Cpu) {
				cpu_summary.fetch_or(PageWordMask(start, end), std::memory_order_release);
			}
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				RefreshCpuSummary();
			}
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				RefreshCpuSummary();
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	TrackingSpinLock lock;

	// KYTY_DIRTY_WALK_FAST: one bit per 64-page word of m_cpu_dirty (16 bits for 1024 pages).
	// Invariant: bit w is set whenever any CPU-dirty page bit in word w is set (it may over-report
	// transiently, never under-report). All m_cpu_dirty mutations hold `lock`. Setters fetch_or the
	// summary (release) BEFORE setting page bits; clearers recompute it exactly AFTER clearing
	// (under the lock). A lock-free reader that acquires a zero summary for its word span therefore
	// observed a state in which every CPU-dirty marking whose ChangeState completed before the
	// reader's walk started is either absent from the span or already visible (the marking thread
	// published through the same lock / release chain the walk already relies on). A marking still
	// inside ChangeState concurrently with the walk is racing with the walk today as well and is
	// also queued (QueueCpuDirty precedes it) for the next drain.
	std::atomic_uint32_t cpu_summary {(1u << (TRACKER_REGION_PAGES / 64)) - 1u};
	static_assert(TRACKER_REGION_PAGES / 64 <= 32);

	// Mask of summary words covering the byte span [offset, offset + bytes), bytes > 0.
	[[nodiscard]] static uint32_t SpanWordMask(uint64_t offset, uint64_t bytes) {
		return PageWordMask(static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		                    static_cast<size_t>((offset + bytes + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE));
	}
	// Caller holds `lock`: exact summary computed from the page bits.
	[[nodiscard]] uint32_t ExactCpuSummary() const {
		uint32_t summary = 0;
		for (size_t w = 0; w < RegionBits::Words(); ++w) {
			summary |= (m_cpu_dirty.WordAt(w) != 0 ? 1u : 0u) << w;
		}
		return summary;
	}
	// Intrusive pending-upload list: fault handlers never allocate or take a queue mutex.
	std::atomic_bool cpu_dirty_queued {false};
	RegionManager* cpu_dirty_next = nullptr;

private:
	[[nodiscard]] static uint32_t PageWordMask(size_t start, size_t end) {
		const size_t first = start / 64, last = (end - 1) / 64;
		return ((2u << last) - 1u) & ~((1u << first) - 1u);
	}
	void RefreshCpuSummary() { cpu_summary.store(ExactCpuSummary(), std::memory_order_release); }

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_writable;
	RegionBits   m_readable;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
