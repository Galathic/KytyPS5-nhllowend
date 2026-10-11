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
//   NHL27DRAWPREP total=... materialize=... textures=... buffers=... descriptors=... pipeline=... other=...
//   NHL27BUFFERS vertex_index=... indirect=... find=... bda=... rebind=... rebind_lookup=... rebind_upload=...
//   NHL27PROTECT calls=... ms=...
//   NHL27FAULTCOST n=... gpu_r/gpu_w/guest_r/guest_w=... pages=... refault=... total=... veh=... lock=...
//   NHL27FAULTPARSE passes=... zero=... faults=... ring_wait=...ms
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
	// GPU thread waiting for guest work: total and gap count by length (<1, 1-4, 4-10, 10+ ms).
	std::atomic<uint64_t> idle_ns {0};
	std::array<std::atomic<uint64_t>, 4> idle_gaps {};
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
	// NHL27DRAWPREP sub-buckets. Each is a plain Timer (gated only by KYTY_PERF_STATS). They are
	// sequential, non-overlapping scopes inside the draw prep window, so none nests in another.
	// Draws that return before their vkCmd draw still add to these buckets but not to draw_prep_ns,
	// so "other" is clamped at zero.
	std::atomic<uint64_t> dp_materialize_ns {0};  // program lookup + resource materialization
	std::atomic<uint64_t> dp_textures_ns {0};     // image / render-target / sampled-image resolution
	std::atomic<uint64_t> dp_buffers_ns {0};      // buffer lookup, BDA, storage/vertex/index/indirect buffers
	// Sub-buckets of dp_buffers_ns, each timed in addition to it. vertex_index + indirect + find_buffers +
	// bda + rebind equals dp_buffers_ns. rebind_lookup and rebind_upload are subsets of rebind.
	std::atomic<uint64_t> dp_vertex_index_ns {0};  // mesh index, AcquireVertexBuffers, PrepareIndexBuffer
	std::atomic<uint64_t> dp_indirect_ns {0};      // indirect args/count FindBuffer, ObtainBuffer and copies
	std::atomic<uint64_t> dp_find_buffers_ns {0};  // FindBuffers (descriptor -> buffer id)
	std::atomic<uint64_t> dp_bda_ns {0};           // PrepareBda (BDA dirty-range synchronization)
	std::atomic<uint64_t> dp_rebind_ns {0};        // RebindBuffers as a whole
	std::atomic<uint64_t> dp_rebind_lookup_ns {0}; // RebindBuffers: NativeStorageBuffer / ObtainBuffer loop
	std::atomic<uint64_t> dp_rebind_upload_ns {0}; // RebindBuffers: NativeUpload of SRT and shader data
	std::atomic<uint64_t> dp_descriptors_ns {0};  // CommitBindings (descriptor writes)
	std::atomic<uint64_t> dp_pipeline_ns {0};     // pipeline lookup, render targets, dynamic state, draw emit
	// NHL27FAULTPARSE: BDA fault-parser pass counters (FaultManager::ProcessFaultBuffer).
	std::atomic<uint64_t> fault_parse_passes {0};
	std::atomic<uint64_t> fault_parse_zero {0};   // passes whose count word was 0
	std::atomic<uint64_t> fault_parse_faults {0}; // total fault entries returned by all passes
	std::atomic<uint64_t> fault_ring_wait_ns {0}; // CPU time waiting on a reused download-ring slot
	// PageManager::Impl::Protect (host page-protection changes), all threads.
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> protect_ns {0};
	// NHL27BDA: RenderContext::PrepareBda, GPU command thread. Counts are per call or per walk.
	// Time buckets are sequential inside bda_total_ns; other = total minus lock and sync.
	std::atomic<uint64_t> bda_calls {0};
	std::atomic<uint64_t> bda_skips {0};     // early-outs through the unchanged-generation cache
	std::atomic<uint64_t> bda_scans {0};     // full walks over the mapped ranges
	std::atomic<uint64_t> bda_regions {0};   // dirty regions drained from the memory tracker queue
	std::atomic<uint64_t> bda_ranges {0};    // full walk: mapped ranges; dirty path: drained regions
	std::atomic<uint64_t> bda_pages {0};     // TRACKER_PAGE_SIZE pages covered by those ranges
	std::atomic<uint64_t> bda_total_ns {0};  // whole PrepareBda body, excluding the one-time log
	std::atomic<uint64_t> bda_lock_ns {0};   // m_mapped_ranges_mutex shared-lock acquisition
	std::atomic<uint64_t> bda_sync_ns {0};   // BufferCache sync; the dirty path includes the queue drain
	// NHL27SYNC: BufferCache::SynchronizeBuffer upload events (each ends rendering + barriers).
	std::atomic<uint64_t> sync_uploads {0};
	std::atomic<uint64_t> sync_bytes {0};
};
inline Counters g_counters;

// NHL27FAULTCOST: breakdown of guest page-fault handling (RenderContext::HandleFault), all threads.
// Time buckets are summed over every faulting thread (guest CPU threads run in parallel with the GPU
// thread, so the sum can exceed wall time). Nested buckets: buf_inv includes tracker lock, protect and
// (for GPU-dirty pages) readmem; tex_inv includes tex_lock; readmem includes the GPU queue wait.
struct FaultCost {
	std::atomic<uint64_t> total_ns {0};      // HandleFault body wall time (same span as faults=)
	std::atomic<uint64_t> veh_ns {0};        // VEH entry -> HandleFault entry
	std::atomic<uint64_t> lock_ns {0};       // tracker region spinlock wait + texture-cache m_lock wait
	std::atomic<uint64_t> buf_inv_ns {0};    // BufferCache::InvalidateMemory (write faults)
	std::atomic<uint64_t> tex_inv_ns {0};    // TextureCache::InvalidateMemory (write faults)
	std::atomic<uint64_t> readmem_ns {0};    // BufferCache::ReadMemory, any fault path (incl. GPU wait)
	std::atomic<uint64_t> rd_readmem_ns {0}; // ReadMemory from read faults only
	std::atomic<uint64_t> protect_ns {0};    // PageManager Protect (VirtualProtect) during a fault
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> total_gpu_ns {0};  // total_ns spent on the GPU thread (inside gpu_thread_busy)
	std::atomic<uint64_t> total_guest_ns {0};
	std::atomic<uint64_t> n[4] {};           // gpu_r, gpu_w, guest_r, guest_w
	std::atomic<uint64_t> distinct {0};      // distinct 4 KiB pages faulted, counted once per frame
	std::atomic<uint64_t> epoch {1};         // frame counter, bumped on every presented flip
	static constexpr size_t kSlots = 16384;
	std::atomic<uint64_t> slots[kSlots] {};  // epoch<<36 | page index
};
inline FaultCost g_fault_cost;
// True while this thread is inside RenderContext::HandleFault.
inline thread_local bool g_in_fault = false;

inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

// Counts the page once per frame epoch; open addressing with stale-epoch slots reusable.
inline void NoteFaultPage(uint64_t vaddr) noexcept {
	auto&          fc    = g_fault_cost;
	const uint64_t page  = (vaddr >> 12) & ((1ull << 36) - 1);
	const uint64_t epoch = fc.epoch.load(std::memory_order_relaxed);
	const uint64_t want  = (epoch << 36) | page;
	size_t         idx   = static_cast<size_t>((page * 0x9E3779B97F4A7C15ull) >> 50) % FaultCost::kSlots;
	for (int probe = 0; probe < 32; ++probe, idx = (idx + 1) % FaultCost::kSlots) {
		uint64_t cur = fc.slots[idx].load(std::memory_order_relaxed);
		if (cur == want) {
			return;
		}
		if ((cur >> 36) != epoch) {
			if (fc.slots[idx].compare_exchange_strong(cur, want, std::memory_order_relaxed)) {
				fc.distinct.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			if (cur == want) {
				return;
			}
		}
	}
	fc.distinct.fetch_add(1, std::memory_order_relaxed); // table full: count conservatively
}

// Times a nested fault bucket, only on a thread that is currently inside HandleFault.
class FaultTimer {
public:
	explicit FaultTimer(std::atomic<uint64_t>& sink) noexcept: m_sink(&sink), m_active(g_in_fault) {
		if (m_active) {
			m_start = NowNs();
		}
	}
	FaultTimer(const FaultTimer&)            = delete;
	FaultTimer& operator=(const FaultTimer&) = delete;
	~FaultTimer() {
		if (m_active) {
			m_sink->fetch_add(NowNs() - m_start, std::memory_order_relaxed);
		}
	}

private:
	std::atomic<uint64_t>* m_sink;
	uint64_t               m_start = 0;
	bool                   m_active;
};

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

inline void AddIdleGap(uint64_t ns) noexcept {
	if (Enabled()) {
		g_counters.idle_ns.fetch_add(ns, std::memory_order_relaxed);
		const size_t bucket = ns < 1'000'000u ? 0 : ns < 4'000'000u ? 1 : ns < 10'000'000u ? 2 : 3;
		g_counters.idle_gaps[bucket].fetch_add(1, std::memory_order_relaxed);
	}
}

inline void AddReadback(uint64_t ns) noexcept {
	if (Enabled()) {
		g_counters.readback_ns.fetch_add(ns, std::memory_order_relaxed);
		g_counters.readback_count.fetch_add(1, std::memory_order_relaxed);
	}
}

// NHL27GPUBUSY: exclusive attribution of GPU-thread busy time (Process wall time, KYTY_PERF_STATS).
// Between GpuBusyBegin and GpuBusyEnd (the same window as gpu_process_ns) every nanosecond is charged
// to exactly one bucket: the innermost active GpuBusyScope, or pm4_loop when none is active. Scopes
// nest; a nested scope's time is removed from its parent, so the buckets add up to the Process wall
// time. wait is the master-semaphore block time (gpu_thread_busy excludes it). Scopes are no-ops on
// any thread/time outside that window. draw_prep is the DrawPrepBegin -> DrawEmitted window minus
// nested scopes (its NHL27DRAWPREP total is the inclusive figure).
enum GpuBucket : uint8_t {
	GB_PM4_LOOP,         // Process/CommandProcessor loop outside packet handlers (fetch, parse, split checks)
	GB_HANDLER,          // other packet handlers (register sets, NOP/flip, indirect buffer, ...)
	GB_DRAW_HANDLER,     // draw packet handlers excluding draw_prep and nested scopes
	GB_DISPATCH_HANDLER, // dispatch packet handlers excluding nested scopes (compute prep + record)
	GB_SYNC_PKT,         // EOP/RELEASE_MEM/EVENT_WRITE/WRITE_DATA/WAIT_REG_MEM/semaphore/sync packets
	GB_DMA_PKT,          // DMA_DATA/CP_DMA/COPY_DATA packets
	GB_DRAW_PREP,        // DrawPrepBegin -> DrawEmitted, exclusive
	GB_GPU_CMDS,         // GuestGpu::ProcessCommands (commands posted from other threads)
	GB_FAULT,            // RenderContext::HandleFault on the GPU thread
	GB_GC,               // RenderContext::RunGarbageCollector
	GB_DEFERRED,         // CommandScheduler::PopPendingOperations callbacks
	GB_SUBMIT,           // CommandScheduler::Submit excluding vkQueueSubmit (End, flags refresh, trace)
	GB_QUEUE_SUBMIT,     // vkQueueSubmit incl. queue mutex
	GB_BEGIN_CMD,        // CommandScheduler::BeginCommand (pool Commit + vkBeginCommandBuffer)
	GB_RENDER_PASS,      // CommandBuffer::BeginRendering/EndRendering (actual begin/end only)
	GB_BARRIER,          // Image::Transit and shader access/hazard barriers
	GB_TEX_UPLOAD,       // Image::Upload
	GB_WAIT,             // MasterSemaphore::Wait blocked (not part of gpu_thread_busy)
	GB_COUNT
};

struct GpuBusyState {
	bool             active = false;
	uint8_t          cur    = GB_PM4_LOOP;
	uint8_t          prep_prev = GB_PM4_LOOP;
	Clock::time_point last {};
	uint64_t         acc[GB_COUNT] {};
};
inline thread_local GpuBusyState g_gpu_busy;
inline std::array<std::atomic<uint64_t>, GB_COUNT> g_gpu_busy_ns {};

inline void GpuBusySwitch(uint8_t bucket) noexcept {
	auto&      s   = g_gpu_busy;
	const auto now = Clock::now();
	s.acc[s.cur] += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - s.last).count());
	s.last = now;
	s.cur  = bucket;
}

inline void GpuBusyBegin() noexcept {
	if (!Enabled()) {
		return;
	}
	auto& s = g_gpu_busy;
	s.active = true;
	s.cur    = GB_PM4_LOOP;
	s.last   = Clock::now();
}

inline void GpuBusyEnd() noexcept {
	auto& s = g_gpu_busy;
	if (!s.active) {
		return;
	}
	GpuBusySwitch(GB_PM4_LOOP);
	for (size_t i = 0; i < GB_COUNT; ++i) {
		g_gpu_busy_ns[i].fetch_add(s.acc[i], std::memory_order_relaxed);
		s.acc[i] = 0;
	}
	s.active = false;
}

class GpuBusyScope {
public:
	explicit GpuBusyScope(GpuBucket bucket) noexcept: m_on(g_gpu_busy.active) {
		if (m_on) {
			m_prev = g_gpu_busy.cur;
			GpuBusySwitch(bucket);
		}
	}
	GpuBusyScope(const GpuBusyScope&)            = delete;
	GpuBusyScope& operator=(const GpuBusyScope&) = delete;
	~GpuBusyScope() {
		if (m_on && g_gpu_busy.active) {
			GpuBusySwitch(m_prev);
		}
	}

private:
	uint8_t m_prev = GB_PM4_LOOP;
	bool    m_on;
};

// Draw prep starts at PrepareDrawRenderState entry and ends at the vkCmd draw emit. The start is
// thread-local, so a draw skipped before its emit is simply overwritten by the next begin.
inline thread_local Clock::time_point g_draw_prep_start {};

inline void DrawPrepBegin() noexcept {
	if (Enabled()) {
		g_draw_prep_start = Clock::now();
		if (g_gpu_busy.active && g_gpu_busy.cur != GB_DRAW_PREP) {
			g_gpu_busy.prep_prev = g_gpu_busy.cur;
			GpuBusySwitch(GB_DRAW_PREP);
		}
	}
}

inline void DrawEmitted() noexcept {
	if (!Enabled()) {
		return;
	}
	// A draw skipped before its emit leaves draw_prep selected until the enclosing handler scope ends.
	if (g_gpu_busy.active && g_gpu_busy.cur == GB_DRAW_PREP) {
		GpuBusySwitch(g_gpu_busy.prep_prev);
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
	const uint64_t dp_materialize_ns = Take(c.dp_materialize_ns);
	const uint64_t dp_textures_ns    = Take(c.dp_textures_ns);
	const uint64_t dp_buffers_ns     = Take(c.dp_buffers_ns);
	const uint64_t dp_descriptors_ns = Take(c.dp_descriptors_ns);
	const uint64_t dp_pipeline_ns    = Take(c.dp_pipeline_ns);
	const uint64_t dp_vertex_index_ns  = Take(c.dp_vertex_index_ns);
	const uint64_t dp_indirect_ns      = Take(c.dp_indirect_ns);
	const uint64_t dp_find_buffers_ns  = Take(c.dp_find_buffers_ns);
	const uint64_t dp_bda_ns           = Take(c.dp_bda_ns);
	const uint64_t dp_rebind_ns        = Take(c.dp_rebind_ns);
	const uint64_t dp_rebind_lookup_ns = Take(c.dp_rebind_lookup_ns);
	const uint64_t dp_rebind_upload_ns = Take(c.dp_rebind_upload_ns);
	const uint64_t fp_passes         = Take(c.fault_parse_passes);
	const uint64_t fp_zero           = Take(c.fault_parse_zero);
	const uint64_t fp_faults         = Take(c.fault_parse_faults);
	const uint64_t fp_ring_wait_ns   = Take(c.fault_ring_wait_ns);
	const uint64_t protect_calls     = Take(c.protect_calls);
	const uint64_t protect_ns        = Take(c.protect_ns);
	const uint64_t bda_calls         = Take(c.bda_calls);
	const uint64_t bda_skips         = Take(c.bda_skips);
	const uint64_t bda_scans         = Take(c.bda_scans);
	const uint64_t bda_regions       = Take(c.bda_regions);
	const uint64_t bda_ranges        = Take(c.bda_ranges);
	const uint64_t bda_pages         = Take(c.bda_pages);
	const uint64_t bda_total_ns      = Take(c.bda_total_ns);
	const uint64_t bda_lock_ns       = Take(c.bda_lock_ns);
	const uint64_t bda_sync_ns       = Take(c.bda_sync_ns);
	const uint64_t sync_uploads      = Take(c.sync_uploads);
	const uint64_t sync_bytes        = Take(c.sync_bytes);
	uint64_t gb[GB_COUNT];
	for (size_t i = 0; i < GB_COUNT; ++i) {
		gb[i] = Take(g_gpu_busy_ns[i]);
	}
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
	            "draws=%.1f dispatches=%.1f vblanks=%llu,%llu,%llu,%llu,%llu idle=%.2fms gaps=%.1f,%.1f,%.1f,%.1f\n",
	            static_cast<double>(frames) / seconds, ms(frame_ns), frames, ms(busy_ns),
	            ms(wait_ns), per(wait_count), ms(readback_ns), per(readback_n), ms(fault_ns),
	            per(fault_n), ms(compile_ns), per(compile_n), ms(draw_prep_ns), draw_us,
	            per(draws), per(dispatches), static_cast<unsigned long long>(Take(c.flip_vblanks[0])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[1])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[2])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[3])),
	            static_cast<unsigned long long>(Take(c.flip_vblanks[4])), ms(Take(c.idle_ns)),
	            per(Take(c.idle_gaps[0])), per(Take(c.idle_gaps[1])), per(Take(c.idle_gaps[2])),
	            per(Take(c.idle_gaps[3])));
	// Flush every line: when stdout is a pipe (the launcher pipes into
	// Tee-Object) libc uses a block buffer, so without this the window lines
	// arrive minutes late and an automated collector cannot attribute them.
	std::fflush(stdout);
	// Exclusive buckets (see GpuBucket): they sum to Process wall time, so busy = sum - wait.
	// unattributed = busy - sum(non-wait buckets); it is only timer skew / uncovered gaps.
	{
		uint64_t attributed = 0;
		for (size_t i = 0; i < GB_COUNT; ++i) {
			if (i != GB_WAIT) {
				attributed += gb[i];
			}
		}
		const double unattr_ms = ms(busy_ns) - ms(attributed);
		std::printf("NHL27GPUBUSY busy=%.2fms pm4_loop=%.2fms handler=%.2fms draw_handler=%.2fms "
		            "dispatch_handler=%.2fms sync_pkt=%.2fms dma_pkt=%.2fms draw_prep_excl=%.2fms "
		            "gpu_cmds=%.2fms fault=%.2fms gc=%.2fms deferred_ops=%.2fms submit=%.2fms "
		            "queue_submit=%.2fms begin_cmd=%.2fms render_pass=%.2fms barrier=%.2fms "
		            "tex_upload=%.2fms wait=%.2fms unattributed=%.2fms "
		            "(draw_prep_incl=%.2fms)\n",
		            ms(busy_ns), ms(gb[GB_PM4_LOOP]), ms(gb[GB_HANDLER]), ms(gb[GB_DRAW_HANDLER]),
		            ms(gb[GB_DISPATCH_HANDLER]), ms(gb[GB_SYNC_PKT]), ms(gb[GB_DMA_PKT]),
		            ms(gb[GB_DRAW_PREP]), ms(gb[GB_GPU_CMDS]), ms(gb[GB_FAULT]), ms(gb[GB_GC]),
		            ms(gb[GB_DEFERRED]), ms(gb[GB_SUBMIT]), ms(gb[GB_QUEUE_SUBMIT]),
		            ms(gb[GB_BEGIN_CMD]), ms(gb[GB_RENDER_PASS]), ms(gb[GB_BARRIER]),
		            ms(gb[GB_TEX_UPLOAD]), ms(gb[GB_WAIT]), unattr_ms, ms(draw_prep_ns));
		std::fflush(stdout);
	}
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
	// Same window as NHL27FRAME. other = draw_prep minus the named buckets (clamped at zero).
	{
		const double named_ms = ms(dp_materialize_ns) + ms(dp_textures_ns) + ms(dp_buffers_ns) +
		                        ms(dp_descriptors_ns) + ms(dp_pipeline_ns);
		const double other_ms = std::max(0.0, ms(draw_prep_ns) - named_ms);
		std::printf("NHL27DRAWPREP total=%.2fms materialize=%.2fms textures=%.2fms buffers=%.2fms "
		            "descriptors=%.2fms pipeline=%.2fms other=%.2fms\n",
		            ms(draw_prep_ns), ms(dp_materialize_ns), ms(dp_textures_ns), ms(dp_buffers_ns),
		            ms(dp_descriptors_ns), ms(dp_pipeline_ns), other_ms);
		std::printf("NHL27BUFFERS vertex_index=%.2fms indirect=%.2fms find=%.2fms bda=%.2fms "
		            "rebind=%.2fms rebind_lookup=%.2fms rebind_upload=%.2fms\n",
		            ms(dp_vertex_index_ns), ms(dp_indirect_ns), ms(dp_find_buffers_ns), ms(dp_bda_ns),
		            ms(dp_rebind_ns), ms(dp_rebind_lookup_ns), ms(dp_rebind_upload_ns));
			// drain_ms is always 0: the queue drain runs inside BufferCache::SynchronizeBuffersInRanges,
			// so its time is counted in sync_ms. pagetable_ms is always 0: PrepareBda never writes the
			// BDA page table (BufferCache::ChangeRegister does, at buffer registration).
			const double bda_lock_ms  = ms(bda_lock_ns);
			const double bda_sync_ms  = ms(bda_sync_ns);
			const double bda_other_ms = std::max(0.0, ms(bda_total_ns) - bda_lock_ms - bda_sync_ms);
			std::printf("NHL27BDA calls=%.2f skips=%.2f scans=%.2f regions=%.2f ranges=%.2f pages=%.2f "
			            "lock_ms=%.3f drain_ms=%.3f sync_ms=%.3f pagetable_ms=%.3f other_ms=%.3f\n",
			            per(bda_calls), per(bda_skips), per(bda_scans), per(bda_regions), per(bda_ranges),
			            per(bda_pages), bda_lock_ms, 0.0, bda_sync_ms, 0.0, bda_other_ms);
			std::printf("NHL27SYNC uploads=%.2f kb=%.2f\n", per(sync_uploads), per(sync_bytes) / 1024.0);
		std::printf("NHL27PROTECT calls=%.2f ms=%.3f\n", per(protect_calls), ms(protect_ns));
		{
			auto&          fc  = g_fault_cost;
			const uint64_t gr  = Take(fc.n[0]);
			const uint64_t gw  = Take(fc.n[1]);
			const uint64_t cr  = Take(fc.n[2]);
			const uint64_t cw  = Take(fc.n[3]);
			const uint64_t all = gr + gw + cr + cw;
			const uint64_t dis = Take(fc.distinct);
			const uint64_t tot = Take(fc.total_ns);
			const uint64_t buf = Take(fc.buf_inv_ns);
			const uint64_t tex = Take(fc.tex_inv_ns);
			const uint64_t rdm = Take(fc.rd_readmem_ns);
			const uint64_t rem = tot > buf + tex + rdm ? tot - buf - tex - rdm : 0;
			const uint64_t pn  = Take(fc.protect_calls);
			std::printf("NHL27FAULTCOST n=%.1f gpu_r=%.1f gpu_w=%.1f guest_r=%.1f guest_w=%.1f "
			            "pages=%.1f refault=%.2f total=%.2fms(gpu=%.2f guest=%.2f) us/fault=%.1f "
			            "veh=%.2fms lock=%.2fms buf_inv=%.2fms tex_inv=%.2fms readmem=%.2fms "
			            "rd_readmem=%.2fms protect=%.2fms(%.1f) other=%.2fms\n",
			            per(all), per(gr), per(gw), per(cr), per(cw), per(dis),
			            dis != 0 ? static_cast<double>(all) / static_cast<double>(dis) : 0.0, ms(tot),
			            ms(Take(fc.total_gpu_ns)), ms(Take(fc.total_guest_ns)),
			            all != 0 ? static_cast<double>(tot) / 1e3 / static_cast<double>(all) : 0.0,
			            ms(Take(fc.veh_ns)), ms(Take(fc.lock_ns)), ms(buf), ms(tex),
			            ms(Take(fc.readmem_ns)), ms(rdm), ms(Take(fc.protect_ns)), per(pn), ms(rem));
		}
		std::printf("NHL27FAULTPARSE passes=%.2f zero=%.2f faults=%.2f ring_wait=%.2fms\n",
		            per(fp_passes), per(fp_zero), per(fp_faults), ms(fp_ring_wait_ns));
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
	g_fault_cost.epoch.fetch_add(1, std::memory_order_relaxed);
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
