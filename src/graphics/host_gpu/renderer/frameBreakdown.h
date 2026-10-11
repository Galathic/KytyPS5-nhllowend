#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "common/runtimeFlags.h"

// KYTY_PERF_STATS=1 frame time breakdown. Prints one line about once per second:
//   NHL27FRAME fps=... frame=... gpu_thread_busy=... wait_gpu=... readback=... faults=...
//              compile=... draw_prep=... draws=... dispatches=...
// Every value except fps and frames is a per-frame average (a frame is one presented flip
// interval). Header-only so probes in several translation units share one set of counters
// without touching the build files. Every probe is a no-op unless the gate is set.
namespace Libs::Graphics::FrameBreakdown {

using Clock = std::chrono::steady_clock;

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Fine-grained CPU preparation timing is intentionally opt-in; it can perturb hot paths.
inline bool DetailEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_DRAW_DETAIL");
		return Enabled() && value != nullptr && value[0] == '1';
	}();
	return enabled;
}

struct Counters {
	std::atomic<uint64_t> flip_ns {0};          // presented flip -> presented flip
	std::atomic<uint64_t> flip_intervals {0};   // number of those intervals (= frames)
	// Flip intervals rounded to 60 Hz vblanks: [0] = 1 vblank ... [4] = 5 or more.
	std::array<std::atomic<uint64_t>, 5> flip_vblanks {};
	std::atomic<uint64_t> gpu_process_ns {0};   // GPU thread: GuestGpu::Process wall time
	std::atomic<uint64_t> gpu_wait_ns {0};      // GPU thread: blocked in MasterSemaphore::Wait
	std::atomic<uint64_t> gpu_wait_count {0};   // GPU thread: waits that actually blocked
	std::atomic<uint64_t> readback_ns {0};      // synchronous GPU->CPU readbacks, all threads
	std::atomic<uint64_t> readback_count {0};
	std::atomic<uint64_t> fault_ns {0};         // guest page-fault handling on GPU-tracked memory
	std::atomic<uint64_t> fault_count {0};
	std::atomic<uint64_t> compile_ns {0};       // shader permutation compile (SPIR-V + module)
	std::atomic<uint64_t> compile_count {0};
	std::atomic<uint64_t> draw_prep_ns {0};     // PrepareDrawRenderState entry -> vkCmd draw
	std::atomic<uint64_t> draws {0};            // draw calls emitted (indirect = 1 call)
	std::atomic<uint64_t> dispatches {0};       // guest compute dispatches emitted
	// These are inclusive aggregates: graphics program lookup and color-target resolution
	// contain shared graphics work, while descriptor preparation also includes compute paths.
	std::atomic<uint64_t> prep_graphics_programs_ns {0};
	std::atomic<uint64_t> prep_compute_program_ns {0};
	std::atomic<uint64_t> prep_color_target_ns {0};
	std::atomic<uint64_t> prep_find_buffers_ns {0};
	std::atomic<uint64_t> prep_rebind_images_ns {0};
	std::atomic<uint64_t> prep_rebind_buffers_ns {0};
	std::atomic<uint64_t> prep_commit_graphics_ns {0};
	std::atomic<uint64_t> prep_commit_compute_ns {0};
};
inline Counters g_counters;

// Adds the elapsed time to sink when the scope ends, or earlier through Stop().
class Timer {
public:
	explicit Timer(std::atomic<uint64_t>& sink, bool condition = true) noexcept
	    : m_sink(&sink), m_active(condition && Enabled()) {
		if (m_active) {
			m_start = Clock::now();
		}
	}
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;
	~Timer() {
		Stop();
	}
	void Stop() noexcept {
		if (!m_active) {
			return;
		}
		m_active = false;
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_start);
		m_sink->fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
	}

private:
	std::atomic<uint64_t>* m_sink;
	Clock::time_point      m_start {};
	bool                   m_active;
};

inline void Add(std::atomic<uint64_t>& counter, uint64_t amount = 1) noexcept {
	if (Enabled()) {
		counter.fetch_add(amount, std::memory_order_relaxed);
	}
}

inline void AddReadback(uint64_t ns) noexcept {
	if (Enabled()) {
		g_counters.readback_ns.fetch_add(ns, std::memory_order_relaxed);
		g_counters.readback_count.fetch_add(1, std::memory_order_relaxed);
	}
}

// Draw prep starts at PrepareDrawRenderState entry and ends at the vkCmd draw emit. The start is
// thread-local, so a draw skipped before its emit is simply overwritten by the next begin.
inline thread_local Clock::time_point g_draw_prep_start {};

inline void DrawPrepBegin() noexcept {
	if (Enabled()) {
		g_draw_prep_start = Clock::now();
	}
}

inline void DrawEmitted() noexcept {
	if (!Enabled()) {
		return;
	}
	g_counters.draws.fetch_add(1, std::memory_order_relaxed);
	if (g_draw_prep_start != Clock::time_point {}) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_draw_prep_start);
		g_counters.draw_prep_ns.fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
		g_draw_prep_start = Clock::time_point {};
	}
}

inline std::mutex         g_window_mutex;
inline Clock::time_point  g_window_start {};
inline Clock::time_point  g_last_flip {};

inline uint64_t Take(std::atomic<uint64_t>& value) noexcept {
	return value.exchange(0, std::memory_order_relaxed);
}

// Caller holds g_window_mutex. Prints and resets the window once it spans at least one second.
inline void PrintIfDueLocked(Clock::time_point now) {
	if (g_window_start == Clock::time_point {}) {
		g_window_start = now;
	}
	const double seconds = std::chrono::duration<double>(now - g_window_start).count();
	if (seconds < 1.0) {
		return;
	}
	auto& c = g_counters;
	const uint64_t frames        = Take(c.flip_intervals);
	const uint64_t frame_ns      = Take(c.flip_ns);
	const uint64_t process_ns    = Take(c.gpu_process_ns);
	const uint64_t wait_ns       = Take(c.gpu_wait_ns);
	const uint64_t wait_count    = Take(c.gpu_wait_count);
	const uint64_t readback_ns   = Take(c.readback_ns);
	const uint64_t readback_n    = Take(c.readback_count);
	const uint64_t fault_ns      = Take(c.fault_ns);
	const uint64_t fault_n       = Take(c.fault_count);
	const uint64_t compile_ns    = Take(c.compile_ns);
	const uint64_t compile_n     = Take(c.compile_count);
	const uint64_t draw_prep_ns  = Take(c.draw_prep_ns);
	const uint64_t draws         = Take(c.draws);
	const uint64_t dispatches    = Take(c.dispatches);
	const uint64_t prep_graphics_programs_ns = Take(c.prep_graphics_programs_ns);
	const uint64_t prep_compute_program_ns = Take(c.prep_compute_program_ns);
	const uint64_t prep_color_target_ns = Take(c.prep_color_target_ns);
	const uint64_t prep_find_buffers_ns = Take(c.prep_find_buffers_ns);
	const uint64_t prep_rebind_images_ns = Take(c.prep_rebind_images_ns);
	const uint64_t prep_rebind_buffers_ns = Take(c.prep_rebind_buffers_ns);
	const uint64_t prep_commit_graphics_ns = Take(c.prep_commit_graphics_ns);
	const uint64_t prep_commit_compute_ns = Take(c.prep_commit_compute_ns);
	g_window_start = now;
	if (frames == 0) {
		return;
	}
	const double n       = static_cast<double>(frames);
	const auto   ms      = [n](uint64_t ns) { return static_cast<double>(ns) / 1e6 / n; };
	const auto   per     = [n](uint64_t v) { return static_cast<double>(v) / n; };
	const uint64_t busy_ns = process_ns > wait_ns ? process_ns - wait_ns : 0;
	const double draw_us = draws != 0 ? static_cast<double>(draw_prep_ns) / 1e3 / static_cast<double>(draws) : 0.0;
	std::printf("NHL27FRAME fps=%.1f frame=%.2fms frames=%" PRIu64
	            " gpu_thread_busy=%.2fms wait_gpu=%.2fms(%.2f) readback=%.2fms(%.2f) "
	            "faults=%.2fms(%.2f) compile=%.2fms(%.3f) draw_prep=%.2fms(%.1fus/draw) "
	            "draws=%.1f dispatches=%.1f vblanks=%llu,%llu,%llu,%llu,%llu\n",
	            static_cast<double>(frames) / seconds, ms(frame_ns), frames, ms(busy_ns),
	            ms(wait_ns), per(wait_count), ms(readback_ns), per(readback_n), ms(fault_ns),
	            per(fault_n), ms(compile_ns), per(compile_n), ms(draw_prep_ns), draw_us,
	            per(draws), per(dispatches), static_cast<unsigned long long>(Take(c.flip_vblanks[0])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[1])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[2])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[3])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[4])));
	// Flush every line: when stdout is a pipe (the launcher pipes into
	// Tee-Object) libc uses a block buffer, so without this the window lines
	// arrive minutes late and an automated collector cannot attribute them.
	std::fflush(stdout);
	if (DetailEnabled()) {
		std::printf("NHL27PREP frames=%" PRIu64
		            " graphics_programs=%.3fms compute_program=%.3fms color_target=%.3fms"
		            " find_buffers=%.3fms rebind_images=%.3fms rebind_buffers=%.3fms"
		            " commit_graphics=%.3fms commit_compute=%.3fms\n",
		            frames, ms(prep_graphics_programs_ns), ms(prep_compute_program_ns),
		            ms(prep_color_target_ns), ms(prep_find_buffers_ns), ms(prep_rebind_images_ns),
		            ms(prep_rebind_buffers_ns), ms(prep_commit_graphics_ns), ms(prep_commit_compute_ns));
		std::fflush(stdout);
	}
}

// Called on every presented flip (guest PM4 EOP flip and CPU flip both end here).
inline void OnPresentedFlip() {
	// Refresh runtime experiment flags at the frame boundary. This is a safe point,
	// and it keeps the published snapshot live even when no probe site on this
	// frame's path runs, so a collector always sees the NHL27CFG tag it waits for.
	// Throttled internally: a timestamp compare per flip.
	Common::RuntimeFlags::Refresh();

	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_window_mutex);
	const auto now = Clock::now();
	if (g_last_flip != Clock::time_point {}) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_last_flip);
		g_counters.flip_ns.fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
		g_counters.flip_intervals.fetch_add(1, std::memory_order_relaxed);
		const auto vblanks = (static_cast<uint64_t>(ns.count()) + 8'333'333u) / 16'666'667u;
		g_counters.flip_vblanks[std::clamp<uint64_t>(vblanks, 1u, 5u) - 1u].fetch_add(
		    1, std::memory_order_relaxed);
	}
	g_last_flip = now;
	PrintIfDueLocked(now);
}

// Called from the GPU thread after each Process slice so the line still prints if flips pause.
inline void MaybePrint() {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_window_mutex);
	PrintIfDueLocked(Clock::now());
}

} // namespace Libs::Graphics::FrameBreakdown
