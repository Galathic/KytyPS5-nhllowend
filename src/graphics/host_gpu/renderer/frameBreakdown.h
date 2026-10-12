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
#include <utility>
#include <vector>

#include "common/runtimeFlags.h"

// KYTY_PERF_STATS=1 frame time breakdown. Prints one line about once per second:
//   NHL27FRAME fps=... frame=... gpu_thread_busy=... wait_gpu=... readback=... faults=...
//              compile=... draw_prep=... draws=... dispatches=...
//   NHL27DRAWPREP total=... materialize=... textures=... buffers=... descriptors=... pipeline=... other=...
//   NHL27MATERIALIZE calls=... total=... flat=... eval=...(n) indirect=... spec=... rest=... reuse_hit=... reuse_miss=... reuse_fallback=...
//   NHL27BUFFERS vertex_index=... indirect=... find=... bda=... rebind=... rebind_lookup=... rebind_upload=...
//   NHL27PROTECT calls=... ms=...
//   NHL27FAULTCOST n=... gpu_r/gpu_w/guest_r/guest_w=... pages=... refault=... total=... veh=... lock=...
//   NHL27FAULTPARSE passes=... zero=... faults=... ring_wait=...ms
// Every value except fps and frames is a per-frame average (a frame is one presented flip
// interval). Header-only so probes in several translation units share one set of counters
// without touching the build files. Every probe is a no-op unless the gate is set.
namespace Libs::Graphics::FrameBreakdown {

using Clock = std::chrono::steady_clock;

// True on the KYTY_DRAWPREP_PREFETCH worker thread. Every probe below is gated on Enabled(), which is
// false there, so worker materializations never add to (or race with) the GPU-thread statistics.
inline thread_local bool g_prefetch_worker = false;

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled && !g_prefetch_worker;
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
	std::atomic<uint64_t> obtain_cache_hits {0};   // KYTY_BUFFER_FAST: ObtainBuffer result cache hits
	std::atomic<uint64_t> obtain_cache_misses {0}; // KYTY_BUFFER_FAST: lookups that went to ObtainBuffer
	std::atomic<uint64_t> dp_descriptors_ns {0};  // CommitBindings (descriptor writes)
	std::atomic<uint64_t> dp_pipeline_ns {0};     // pipeline lookup, render targets, dynamic state, draw emit
	// NHL27MATERIALIZE: MaterializeResourcesImpl split. total is the whole call; flat, eval, indirect and
	// spec are sequential sub-scopes (eval includes the clean walker); rest = total - the four.
	std::atomic<uint64_t> mat_calls {0};
	std::atomic<uint64_t> mat_total_ns {0};
	std::atomic<uint64_t> mat_flat_ns {0};      // RefreshFlatBuffer
	std::atomic<uint64_t> mat_eval_ns {0};      // EvaluateDescriptor (direct sources)
	std::atomic<uint64_t> mat_indirect_ns {0};  // MaterializeIndirectDescriptor
	std::atomic<uint64_t> mat_spec_ns {0};      // BuildResourceSpecialization
	std::atomic<uint64_t> mat_evals {0};        // EvaluateDescriptor calls
	std::atomic<uint64_t> mat_flat_reads {0};     // RefreshFlatBuffer: guest-memory reader callbacks (words)
	std::atomic<uint64_t> mat_flat_read_ns {0};   // time inside those callbacks (KYTY_PERF_DRAW_DETAIL=1 only)
	std::atomic<uint64_t> mat_flat_ops {0};       // tape ops executed by RefreshFlatBuffer
	std::atomic<uint64_t> mat_flat_alloc_ns {0};  // flat.assign + trace setup (KYTY_PERF_DRAW_DETAIL=1 only)
	std::atomic<uint64_t> mat_flat_trace_hits {0};   // refreshes served by ReplayCfgTrace
	std::atomic<uint64_t> mat_flat_hit_ns {0};     // time in RefreshFlatBuffer calls served by trace replay
	std::atomic<uint64_t> mat_flat_nocfg_ns {0};   // time in RefreshFlatBuffer calls on non-CFG plans
	std::atomic<uint64_t> mat_hit_steps {0};  // trace steps replayed on hits
	std::atomic<uint64_t> mat_hit_reads {0};  // reader callbacks (RawRead execs) on hits
	std::atomic<uint64_t> mat_hit_memo {0};   // memo saves (published) + restores (Memo steps) on hits
	std::atomic<uint64_t> mat_flat_nocfg_ops {0};  // tape ops executed on non-CFG plans
	std::atomic<uint64_t> mat_flat_trace_other {0};  // refreshes that were not (miss/ineligible/non-CFG)
	std::atomic<uint64_t> mat_tr_why[16] {};  // ReplayCfgTrace outcome reasons (see NHL27TRACE print)
	std::atomic<uint64_t> mat_reuse_hits {0};   // KYTY_SRT_SOURCE_REUSE: gather output equals the previous one
	std::atomic<uint64_t> mat_reuse_misses {0}; // gather output differs / first time
	std::atomic<uint64_t> mat_reuse_fallback {0}; // sources not on the gather fast path
	// NHL27DISPATCHPREP buckets: RenderExecutor::DispatchDirect / DispatchIndirect, GPU thread.
	// Sequential, non-overlapping scopes inside dc_total_ns; other = total minus the named buckets.
	std::atomic<uint64_t> dc_total_ns {0};        // whole handler body (after the context lock)
	std::atomic<uint64_t> dc_materialize_ns {0};  // GetComputeProgram (program lookup + materialization)
	std::atomic<uint64_t> dc_textures_ns {0};     // RebindImages
	std::atomic<uint64_t> dc_buffers_ns {0};      // FindBuffers + PrepareBda + shared memory + RebindBuffers
	std::atomic<uint64_t> dc_bda_ns {0};          // PrepareBda (subset of buffers)
	std::atomic<uint64_t> dc_descriptors_ns {0};  // CommitBindings
	std::atomic<uint64_t> dc_pipeline_ns {0};     // GetComputePipeline + PrepareBindings + bindPipeline
	std::atomic<uint64_t> dc_barrier_ns {0};      // write-hazard and access barriers
	std::atomic<uint64_t> dc_record_ns {0};       // breadcrumb + vkCmdDispatch
	std::atomic<uint64_t> dc_handled {0};         // handler invocations
	std::atomic<uint64_t> dc_bind_skipped {0};    // vkCmdBindPipeline calls elided (KYTY_DISPATCH_FAST)
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

// NHL27SPECULATE (Stage 0 of parallel draw prep): how much materialization could have run ahead of time.
// Seq counts materializations (draws + dispatches). Page tables are direct-mapped by hashed 4K page;
// aliasing only over-blocks. Everything is gated on Enabled() at the call sites.
struct SpecStats {
	std::atomic<uint64_t> mats {0}, spec8 {0}, spec48 {0}, blk_gpu {0}, blk_cpu8 {0}, blk_cpu48 {0};
	std::atomic<uint64_t> blk_wait8 {0}, blk_wait48 {0}, ranges {0}, noreads {0};
	std::atomic<uint64_t> blk_pw8 {0}, blk_pw48 {0};
};
inline SpecStats g_spec;

// NHL27PREFETCH (KYTY_DRAWPREP_PREFETCH): content-validated compute materialization prefetch.
// Counted on the GPU thread (scans, requests, adopt outcomes) and on the worker (done, aborts, worker_ns).
struct PrefetchStats {
	std::atomic<uint64_t> scans {0};            // look-aheads started (one per DISPATCH_DIRECT)
	std::atomic<uint64_t> candidates {0};       // look-ahead dispatches decoded
	std::atomic<uint64_t> requests {0};         // requests enqueued for the worker
	std::atomic<uint64_t> skip_no_plan {0};     // program not seen yet (no plan to materialize against)
	std::atomic<uint64_t> skip_likely_hit {0};  // the plan's memo already holds these register inputs
	std::atomic<uint64_t> skip_dup {0};         // same request already pending or done
	std::atomic<uint64_t> skip_full {0};        // queue or result store full
	std::atomic<uint64_t> worker_done {0};      // results produced
	std::atomic<uint64_t> abort_read {0};       // a read was not plainly readable guest backing
	std::atomic<uint64_t> abort_fail {0};       // materialization reported failure on speculative data
	std::atomic<uint64_t> abort_exc {0};        // exception in the worker
	std::atomic<uint64_t> used_direct {0};      // results adopted by the GPU thread (memo-bypass direct path)
	std::atomic<uint64_t> fb_dirty {0};         // a recorded range was GPU-dirty at use time
	std::atomic<uint64_t> fb_mismatch {0};      // a recorded range differed (or was unavailable) at use time
	std::atomic<uint64_t> fb_regs {0};          // program matched but the user-data registers differed
	std::atomic<uint64_t> fb_notfound {0};      // no result for this (plan, base, workgroups)
	std::atomic<uint64_t> peek_fail {0};        // look-ahead clean peek failed; scan skipped
	std::atomic<uint64_t> verify_checked {0};   // KYTY_DRAWPREP_VERIFY: adoptions re-materialized and compared
	std::atomic<uint64_t> verify_mismatch {0};  // ... of which the adopted state differed (must be 0)
	std::atomic<uint64_t> unused {0};           // results replaced or dropped without being used
	std::atomic<uint64_t> validate_ns {0};      // GPU thread: TryDirect attempts that found a candidate (excl. verify)
	std::atomic<uint64_t> lookup_ns {0};        // ... of which: result lookup under the state mutex
	std::atomic<uint64_t> ranges_ns {0};        // ... of which: dirty + content validation of the recorded ranges
	std::atomic<uint64_t> lock_wait_ns {0};     // ... of which: waiting for the plan lock stripe (worker materializing)
	std::atomic<uint64_t> install_ns {0};       // ... of which: moving the outputs in (after the lock)
	std::atomic<uint64_t> verify_ns {0};        // KYTY_DRAWPREP_VERIFY re-materialization (NOT in validate_ns)
	std::atomic<uint64_t> miss_ns {0};          // GPU thread: TryDirect calls that found nothing
	std::atomic<uint64_t> skip_scanned {0};     // look-ahead dispatch already requested by an earlier scan
	std::atomic<uint64_t> nf_pending {0};       // no result: request queued or running (not ready yet)
	std::atomic<uint64_t> nf_reuse {0};         // no result: identical inputs were adopted earlier (result consumed)
	std::atomic<uint64_t> nf_gone {0};          // no result: requested earlier, result aborted/evicted/stale
	std::atomic<uint64_t> nf_never {0};         // no result: never requested (not scanned, no plan, or queue full)
	std::atomic<uint64_t> pred_checked {0};     // dispatches whose look-ahead prediction was compared at execution
	std::atomic<uint64_t> pred_ok {0};          // ... prediction matched exactly
	std::atomic<uint64_t> pred_none {0};        // ... no prediction was made for this dispatch
	std::atomic<uint64_t> pred_base {0};        // ... shader base / user-data count differed
	std::atomic<uint64_t> pred_regs {0};        // ... a user-data register differed
	std::atomic<uint64_t> pred_groups {0};      // ... workgroup counts differed
	std::atomic<uint64_t> pred_regmask {0};     // bitmask of user-data slots that differed (diagnostic)
	std::atomic<uint64_t> worker_ns {0};        // worker: time spent materializing
	std::atomic<uint64_t> used_worker_ns {0};   // worker time of the results that were used
};
inline PrefetchStats g_prefetch;
inline constexpr uint64_t kSpecTableBits = 18;
inline constexpr uint64_t kSpecTableSize = 1ull << kSpecTableBits;
inline std::atomic<uint64_t> g_spec_fault_seq[kSpecTableSize];  // page -> seq of last CPU write fault
inline std::atomic<uint64_t> g_spec_gpu_epoch[kSpecTableSize];  // page -> batch epoch of last GPU write
inline std::atomic<uint64_t> g_spec_pkt_seq[kSpecTableSize];    // page -> seq of last PM4 packet guest-memory write
// Measurement-only read collector for the materialization in progress on this thread (fed from the
// low-level shader memory readers; null outside NHL27SPECULATE materialization).
inline thread_local std::vector<std::pair<uint64_t, uint64_t>>* g_spec_collect = nullptr;
inline void SpecCollectRead(uint64_t addr, uint64_t size) {
	auto* c = g_spec_collect;
	if (c == nullptr) return;
	if (!c->empty() && c->back().first + c->back().second == addr) {
		c->back().second += size;
		return;
	}
	if (c->size() < 4096) c->emplace_back(addr, size);
}
inline std::atomic<uint64_t> g_spec_seq {0};
inline std::atomic<uint64_t> g_spec_wait_seq {0};
inline std::atomic<uint64_t> g_spec_epoch {1};
inline uint64_t SpecIndex(uint64_t page) { return (page ^ (page >> kSpecTableBits)) & (kSpecTableSize - 1); }
inline void SpecNoteFault(uint64_t vaddr) {
	g_spec_fault_seq[SpecIndex(vaddr >> 12)].store(g_spec_seq.load(std::memory_order_relaxed) + 1,
	                                               std::memory_order_relaxed);
}
inline void SpecNoteGpuWrite(uint64_t vaddr, uint64_t size) {
	const uint64_t first = vaddr >> 12;
	const uint64_t last  = (vaddr + (size != 0 ? size - 1 : 0)) >> 12;
	const uint64_t epoch = g_spec_epoch.load(std::memory_order_relaxed);
	for (uint64_t p = first; p <= last && p - first < 4096; ++p) {
		g_spec_gpu_epoch[SpecIndex(p)].store(epoch, std::memory_order_relaxed);
	}
}
inline void SpecNotePktWrite(uint64_t vaddr, uint64_t size) {
	const uint64_t first = vaddr >> 12;
	const uint64_t last  = (vaddr + (size != 0 ? size - 1 : 0)) >> 12;
	const uint64_t at    = g_spec_seq.load(std::memory_order_relaxed) + 1;
	for (uint64_t p = first; p <= last && p - first < 4096; ++p) {
		g_spec_pkt_seq[SpecIndex(p)].store(at, std::memory_order_relaxed);
	}
}
inline void SpecNoteWait() { g_spec_wait_seq.store(g_spec_seq.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed); }
inline void SpecNoteSubmit() { g_spec_epoch.fetch_add(1, std::memory_order_relaxed); }
// Called once per materialization on the GPU thread with the walker's captured read ranges.
inline void SpecObserve(const std::vector<std::pair<uint64_t, uint64_t>>& reads) {
	const uint64_t seq   = g_spec_seq.fetch_add(1, std::memory_order_relaxed) + 1;
	const uint64_t epoch = g_spec_epoch.load(std::memory_order_relaxed);
	uint64_t       newest_fault = 0, newest_pw = 0;
	bool           gpu = false;
	for (const auto& [addr, size] : reads) {
		const uint64_t first = addr >> 12;
		const uint64_t last  = (addr + (size != 0 ? size - 1 : 0)) >> 12;
		for (uint64_t p = first; p <= last && p - first < 256; ++p) {
			const uint64_t i = SpecIndex(p);
			newest_fault     = std::max(newest_fault, g_spec_fault_seq[i].load(std::memory_order_relaxed));
			newest_pw        = std::max(newest_pw, g_spec_pkt_seq[i].load(std::memory_order_relaxed));
			gpu |= g_spec_gpu_epoch[i].load(std::memory_order_relaxed) == epoch;
		}
	}
	const uint64_t wait   = g_spec_wait_seq.load(std::memory_order_relaxed);
	const auto     within = [seq](uint64_t at, uint64_t n) { return at != 0 && seq - at < n; };
	const bool cpu8 = within(newest_fault, 8), cpu48 = within(newest_fault, 48);
	const bool w8 = within(wait, 8), w48 = within(wait, 48);
	const bool pw8 = within(newest_pw, 8), pw48 = within(newest_pw, 48);
	auto& st = g_spec;
	st.mats.fetch_add(1, std::memory_order_relaxed);
	st.ranges.fetch_add(reads.size(), std::memory_order_relaxed);
	if (reads.empty()) st.noreads.fetch_add(1, std::memory_order_relaxed);
	if (gpu) st.blk_gpu.fetch_add(1, std::memory_order_relaxed);
	if (cpu8) st.blk_cpu8.fetch_add(1, std::memory_order_relaxed);
	if (cpu48) st.blk_cpu48.fetch_add(1, std::memory_order_relaxed);
	if (pw8) st.blk_pw8.fetch_add(1, std::memory_order_relaxed);
	if (pw48) st.blk_pw48.fetch_add(1, std::memory_order_relaxed);
	if (w8) st.blk_wait8.fetch_add(1, std::memory_order_relaxed);
	if (w48) st.blk_wait48.fetch_add(1, std::memory_order_relaxed);
	if (!gpu && !cpu8 && !w8 && !pw8) st.spec8.fetch_add(1, std::memory_order_relaxed);
	if (!gpu && !cpu48 && !w48 && !pw48) st.spec48.fetch_add(1, std::memory_order_relaxed);
}

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
	const uint64_t mat_calls         = Take(c.mat_calls);
	const uint64_t mat_total_ns      = Take(c.mat_total_ns);
	const uint64_t mat_flat_ns       = Take(c.mat_flat_ns);
	const uint64_t mat_eval_ns       = Take(c.mat_eval_ns);
	const uint64_t mat_indirect_ns   = Take(c.mat_indirect_ns);
	const uint64_t mat_spec_ns       = Take(c.mat_spec_ns);
	const uint64_t mat_evals         = Take(c.mat_evals);
	const uint64_t mat_flat_reads    = Take(c.mat_flat_reads);
	const uint64_t mat_flat_read_ns  = Take(c.mat_flat_read_ns);
	const uint64_t mat_flat_ops      = Take(c.mat_flat_ops);
	const uint64_t mat_flat_alloc_ns = Take(c.mat_flat_alloc_ns);
	const uint64_t mat_flat_trace_hits  = Take(c.mat_flat_trace_hits);
	const uint64_t mat_flat_trace_other = Take(c.mat_flat_trace_other);
	const uint64_t mat_flat_hit_ns      = Take(c.mat_flat_hit_ns);
	const uint64_t mat_flat_nocfg_ns    = Take(c.mat_flat_nocfg_ns);
	const uint64_t mat_flat_nocfg_ops   = Take(c.mat_flat_nocfg_ops);
	const uint64_t mat_hit_steps        = Take(c.mat_hit_steps);
	const uint64_t mat_hit_reads        = Take(c.mat_hit_reads);
	const uint64_t mat_hit_memo         = Take(c.mat_hit_memo);
	uint64_t mat_tr_why[16];
	for (int i = 0; i < 16; ++i) mat_tr_why[i] = Take(c.mat_tr_why[i]);
	const uint64_t mat_reuse_hits    = Take(c.mat_reuse_hits);
	const uint64_t mat_reuse_misses  = Take(c.mat_reuse_misses);
	const uint64_t mat_reuse_fallback = Take(c.mat_reuse_fallback);
	const uint64_t dp_vertex_index_ns  = Take(c.dp_vertex_index_ns);
	const uint64_t dp_indirect_ns      = Take(c.dp_indirect_ns);
	const uint64_t dp_find_buffers_ns  = Take(c.dp_find_buffers_ns);
	const uint64_t dp_bda_ns           = Take(c.dp_bda_ns);
	const uint64_t dp_rebind_ns        = Take(c.dp_rebind_ns);
	const uint64_t dp_rebind_lookup_ns = Take(c.dp_rebind_lookup_ns);
	const uint64_t dp_rebind_upload_ns = Take(c.dp_rebind_upload_ns);
	const uint64_t obtain_cache_hits   = Take(c.obtain_cache_hits);
	const uint64_t obtain_cache_misses = Take(c.obtain_cache_misses);
	const uint64_t dc_total_ns       = Take(c.dc_total_ns);
	const uint64_t dc_materialize_ns = Take(c.dc_materialize_ns);
	const uint64_t dc_textures_ns    = Take(c.dc_textures_ns);
	const uint64_t dc_buffers_ns     = Take(c.dc_buffers_ns);
	const uint64_t dc_bda_ns         = Take(c.dc_bda_ns);
	const uint64_t dc_descriptors_ns = Take(c.dc_descriptors_ns);
	const uint64_t dc_pipeline_ns    = Take(c.dc_pipeline_ns);
	const uint64_t dc_barrier_ns     = Take(c.dc_barrier_ns);
	const uint64_t dc_record_ns      = Take(c.dc_record_ns);
	const uint64_t dc_handled        = Take(c.dc_handled);
	const uint64_t dc_bind_skipped   = Take(c.dc_bind_skipped);
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
		{
			const uint64_t mat_named = mat_flat_ns + mat_eval_ns + mat_indirect_ns + mat_spec_ns;
			const uint64_t mat_rest  = mat_total_ns > mat_named ? mat_total_ns - mat_named : 0;
			std::printf("NHL27MATERIALIZE calls=%.1f total=%.2fms flat=%.2fms eval=%.2fms(%.1f) "
			            "indirect=%.2fms spec=%.2fms rest=%.2fms reuse_hit=%.1f reuse_miss=%.1f "
			            "reuse_fallback=%.1f\n",
			            per(mat_calls), ms(mat_total_ns), ms(mat_flat_ns), ms(mat_eval_ns), per(mat_evals),
			            ms(mat_indirect_ns), ms(mat_spec_ns), ms(mat_rest), per(mat_reuse_hits),
			            per(mat_reuse_misses), per(mat_reuse_fallback));
			std::printf("NHL27FLAT words=%.1f read=%.2fms(detail) ops=%.1f alloc=%.2fms(detail) "
			            "trace_hit=%.1f trace_other=%.1f hit_t=%.2fms(%.2fus/call) nocfg_t=%.2fms(%.2fus/call) "
			            "nocfg_ops_per_call=%.1f\n",
			            per(mat_flat_reads), ms(mat_flat_read_ns), per(mat_flat_ops), ms(mat_flat_alloc_ns),
			            per(mat_flat_trace_hits), per(mat_flat_trace_other), ms(mat_flat_hit_ns),
			            mat_flat_trace_hits > 0 ? ms(mat_flat_hit_ns) * 1000.0 / per(mat_flat_trace_hits) : 0.0,
			            ms(mat_flat_nocfg_ns),
			            mat_tr_why[0] > 0 ? ms(mat_flat_nocfg_ns) * 1000.0 / per(mat_tr_why[0]) : 0.0,
			            mat_tr_why[0] > 0 ? per(mat_flat_nocfg_ops) / per(mat_tr_why[0]) : 0.0);
			std::printf("NHL27HIT steps_per_call=%.1f reads_per_call=%.1f memo_per_call=%.1f\n",
			            mat_flat_trace_hits > 0 ? per(mat_hit_steps) / per(mat_flat_trace_hits) : 0.0,
			            mat_flat_trace_hits > 0 ? per(mat_hit_reads) / per(mat_flat_trace_hits) : 0.0,
			            mat_flat_trace_hits > 0 ? per(mat_hit_memo) / per(mat_flat_trace_hits) : 0.0);
			{
				auto&          sp  = g_spec;
				const uint64_t sm  = Take(sp.mats);
				const uint64_t srg = Take(sp.ranges);
				std::printf("NHL27SPECULATE mats=%.1f spec8=%.1f spec48=%.1f blocked_pkt_write8=%.1f "
				            "blocked_pkt_write48=%.1f blocked_wait8=%.1f blocked_wait48=%.1f "
				            "blocked_cpu_fault8=%.1f blocked_cpu_fault48=%.1f blocked_gpu_write=%.1f "
				            "no_reads=%.1f avg_ranges=%.2f\n",
				            per(sm), per(Take(sp.spec8)), per(Take(sp.spec48)), per(Take(sp.blk_pw8)),
				            per(Take(sp.blk_pw48)), per(Take(sp.blk_wait8)), per(Take(sp.blk_wait48)),
				            per(Take(sp.blk_cpu8)), per(Take(sp.blk_cpu48)), per(Take(sp.blk_gpu)),
				            per(Take(sp.noreads)),
				            sm != 0 ? static_cast<double>(srg) / static_cast<double>(sm) : 0.0);
			}
			{
				// Take() every counter into a local first: printf argument evaluation order is unspecified.
				auto&          pf      = g_prefetch;
				const uint64_t used_ns = Take(pf.used_worker_ns), val_ns = Take(pf.validate_ns);
				const uint64_t work_ns = Take(pf.worker_ns);
				const uint64_t ab_read = Take(pf.abort_read), ab_fail = Take(pf.abort_fail);
				const uint64_t ab_exc  = Take(pf.abort_exc);
				const uint64_t scans = Take(pf.scans), cands = Take(pf.candidates), reqs = Take(pf.requests);
				const uint64_t sk_plan = Take(pf.skip_no_plan), sk_hit = Take(pf.skip_likely_hit);
				const uint64_t sk_dup = Take(pf.skip_dup), sk_full = Take(pf.skip_full);
				const uint64_t done = Take(pf.worker_done), used = Take(pf.used_direct);
				const uint64_t fb_dirty = Take(pf.fb_dirty), fb_mism = Take(pf.fb_mismatch);
				const uint64_t fb_regs = Take(pf.fb_regs), nfound = Take(pf.fb_notfound);
				const uint64_t peek_fail = Take(pf.peek_fail), v_chk = Take(pf.verify_checked);
				const uint64_t v_bad = Take(pf.verify_mismatch);
				const uint64_t unused = Take(pf.unused);
				const uint64_t lk_ns = Take(pf.lookup_ns), rg_ns = Take(pf.ranges_ns), lw_ns = Take(pf.lock_wait_ns);
				const uint64_t in_ns = Take(pf.install_ns), vf_ns = Take(pf.verify_ns), ms_ns = Take(pf.miss_ns);
				const uint64_t sk_scan = Take(pf.skip_scanned), nf_pend = Take(pf.nf_pending);
				const uint64_t nf_reuse = Take(pf.nf_reuse), nf_gone = Take(pf.nf_gone), nf_never = Take(pf.nf_never);
				const uint64_t pr_chk = Take(pf.pred_checked), pr_ok = Take(pf.pred_ok), pr_none = Take(pf.pred_none);
				const uint64_t pr_base = Take(pf.pred_base), pr_regs = Take(pf.pred_regs);
				const uint64_t pr_grp = Take(pf.pred_groups);
				const uint64_t pr_mask = pf.pred_regmask.exchange(0, std::memory_order_relaxed);
				std::printf("NHL27PREFETCH scans=%.1f candidates=%.1f requests=%.1f skip_no_plan=%.1f "
				            "skip_likely_hit=%.1f skip_dup=%.1f skip_full=%.1f worker_done=%.1f "
				            "worker_aborts=%.1f(read=%.1f fail=%.1f exc=%.1f) used_direct=%.1f fb_dirty=%.1f "
				            "fb_mismatch=%.1f fb_regs=%.1f fb_notfound=%.1f peek_fail=%.1f verify_checked=%.1f "
				            "verify_mismatch=%.1f unused=%.1f validate_ns=%.3fms worker=%.3fms saved_est=%.3fms\n",
				            per(scans), per(cands), per(reqs), per(sk_plan), per(sk_hit), per(sk_dup),
				            per(sk_full), per(done), per(ab_read + ab_fail + ab_exc), per(ab_read),
				            per(ab_fail), per(ab_exc), per(used), per(fb_dirty), per(fb_mism), per(fb_regs),
				            per(nfound), per(peek_fail), per(v_chk), per(v_bad), per(unused),
				            ms(val_ns), ms(work_ns), ms(used_ns > val_ns ? used_ns - val_ns : 0));
				std::printf("NHL27PREFETCH2 validate=%.3fms(lookup=%.3f ranges=%.3f lockwait=%.3f install=%.3f) "
				            "verify=%.3fms miss=%.3fms skip_scanned=%.1f not_found(pending=%.1f reuse=%.1f "
				            "gone=%.1f never=%.1f) pred(checked=%.1f ok=%.1f none=%.1f base=%.1f regs=%.1f "
				            "groups=%.1f regmask=0x%llx)\n",
				            ms(val_ns), ms(lk_ns), ms(rg_ns), ms(lw_ns), ms(in_ns), ms(vf_ns), ms(ms_ns),
				            per(sk_scan), per(nf_pend), per(nf_reuse), per(nf_gone), per(nf_never), per(pr_chk),
				            per(pr_ok), per(pr_none), per(pr_base), per(pr_regs), per(pr_grp),
				            static_cast<unsigned long long>(pr_mask));
			}
			std::printf("NHL27TRACE nocfg=%.1f flag_off=%.1f inelig_mode=%.1f empty=%.1f disabled=%.1f tape_nr=%.1f "
			            "strict=%.1f m_memo=%.1f m_exec=%.1f m_memoslot=%.1f m_cond=%.1f m_eval=%.1f m_condeval=%.1f "
			            "rec_abort=%.1f merge_refused=%.1f merge_new=%.1f\n",
			            per(mat_tr_why[0]), per(mat_tr_why[1]), per(mat_tr_why[2]), per(mat_tr_why[3]),
			            per(mat_tr_why[4]), per(mat_tr_why[5]), per(mat_tr_why[6]), per(mat_tr_why[7]),
			            per(mat_tr_why[8]), per(mat_tr_why[9]), per(mat_tr_why[10]), per(mat_tr_why[11]),
			            per(mat_tr_why[12]), per(mat_tr_why[13]), per(mat_tr_why[14]), per(mat_tr_why[15]));
			const double dc_named = ms(dc_materialize_ns) + ms(dc_textures_ns) + ms(dc_buffers_ns) +
			                        ms(dc_descriptors_ns) + ms(dc_pipeline_ns) + ms(dc_barrier_ns) +
			                        ms(dc_record_ns);
			std::printf("NHL27DISPATCHPREP total=%.2fms materialize=%.2fms textures=%.2fms buffers=%.2fms "
			            "bda=%.2fms descriptors=%.2fms pipeline=%.2fms barrier=%.2fms record=%.2fms "
			            "other=%.2fms count=%.1f bind_skipped=%.1f\n",
			            ms(dc_total_ns), ms(dc_materialize_ns), ms(dc_textures_ns), ms(dc_buffers_ns),
			            ms(dc_bda_ns), ms(dc_descriptors_ns), ms(dc_pipeline_ns), ms(dc_barrier_ns),
			            ms(dc_record_ns), std::max(0.0, ms(dc_total_ns) - dc_named), per(dc_handled),
			            per(dc_bind_skipped));
		}
		std::printf("NHL27BUFFERS vertex_index=%.2fms indirect=%.2fms find=%.2fms bda=%.2fms "
		            "rebind=%.2fms rebind_lookup=%.2fms rebind_upload=%.2fms\n",
		            ms(dp_vertex_index_ns), ms(dp_indirect_ns), ms(dp_find_buffers_ns), ms(dp_bda_ns),
		            ms(dp_rebind_ns), ms(dp_rebind_lookup_ns), ms(dp_rebind_upload_ns),
		            static_cast<unsigned long long>(obtain_cache_hits),
		            static_cast<unsigned long long>(obtain_cache_misses));
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
