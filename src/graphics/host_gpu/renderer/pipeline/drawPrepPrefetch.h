#ifndef KYTY_GRAPHICS_DRAWPREP_PREFETCH_H_
#define KYTY_GRAPHICS_DRAWPREP_PREFETCH_H_

// KYTY_DRAWPREP_PREFETCH (parallel draw prep, redesign 2026-10-12): content-validated prefetch of
// compute-dispatch resource materialization, for the memo-bypass configuration (KYTY_RESOURCE_MEMO_BYPASS=1).
//
// Scope: DISPATCH_DIRECT only. When the GPU thread executes a dispatch it bulk-copies the next PM4 dwords
// through the clean-backing peek (never dereferencing the stream), scans the copy (up to 8 draws/dispatches),
// replays the compute SH-register writes it finds on a copy of the compute shadow, and for every later
// DISPATCH_DIRECT whose program is already known it asks ONE worker thread to run MaterializeResources
// against the guest memory as it is now.
//
// The worker's result is {key (plan, shader base, workgroups, ALL user-data registers), snapshot,
// specialization, plan.active_sources, merged read ranges {address, bytes}}. The ranges are the union of
// every guest read the materialization performed (sorted, overlapping/adjacent reads merged; a result
// whose overlapping reads disagree is dropped). When the GPU thread reaches that dispatch with the memo
// disabled, TryDirect() looks the result up by (plan, shader base, workgroups), compares every user-data
// register, and validates every range with Memory::TryValidateCleanBacking: the range must not be
// GPU-dirty and its backing bytes must equal the recorded bytes. Only then is the result installed.
//
// Exactness: the materialization output is a pure function of (plan, shader base, workgroup counts,
// user-data registers, the bytes it reads). Registers and workgroups are compared at use time, and every
// byte the walk read is compared at use time, so a fresh walk would take the identical path, read the
// identical bytes and produce the identical output. A GPU-dirty range is never trusted (the normal path
// would synchronize it), and any failed check falls through to the normal materialization.
// KYTY_DRAWPREP_VERIFY=1 additionally re-materializes every 8th adoption and compares.
//
// Thread-safety (audit): see PlanAccessGuard in ResourceMaterialization.h. A ResourcePlan's mutable
// scratch (evaluation memo, SRT tapes, CFG traces, active_sources, Value::evaluation_index) is only
// touched inside MaterializeResources or under the plan's lock stripe. The worker never touches a
// SourceEntry, its memo, its snapshot or its specialization cache, only the immutable plan graph (under
// the guard) and its own private result objects. FrameBreakdown probes are gated on Enabled(), which is
// false on the worker. Everything else reachable from materialization is thread_local, atomic, or
// mutex-protected (checked: SrtWalker/ResourceMaterialization statics).

#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/pipeline/resourceMemo.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::DrawPrepPrefetch {

using Plan           = ShaderRecompiler::IR::ResourcePlan;
using Runtime        = ShaderRecompiler::IR::SrtRuntime;
using Snapshot       = ShaderRecompiler::IR::ResourceSnapshot;
using Specialization = ShaderRecompiler::IR::ResourceSpecialization;

inline constexpr size_t   kMaxUserData      = 32;   // HW::UserSgprInfo::SGPRS_MAX
inline constexpr size_t   kMaxPending       = 16;
inline constexpr size_t   kMaxResults       = 24;
inline constexpr size_t   kMaxRecordedWords = 4096; // retention limit for one result's recorded reads
inline constexpr uint32_t kLookaheadWork    = 8;
inline constexpr uint32_t kVerifyEvery      = 8;

// One materialization request: the exact inputs of SrtRuntime for a compute stage.
struct Key {
	const Plan* plan        = nullptr;
	uint64_t    shader_base = 0;
	uint32_t    user_count  = 0;
	uint32_t    workgroups[3] = {0, 0, 0};
	uint32_t    user_data[kMaxUserData] = {};
};

inline bool SameProgram(const Key& a, const Key& b) {
	return a.plan == b.plan && a.shader_base == b.shader_base && a.workgroups[0] == b.workgroups[0] &&
	       a.workgroups[1] == b.workgroups[1] && a.workgroups[2] == b.workgroups[2];
}
inline bool SameInputs(const Key& a, const Key& b) {
	return SameProgram(a, b) && a.user_count == b.user_count &&
	       std::equal(a.user_data, a.user_data + a.user_count, b.user_data);
}

// A contiguous guest range the materialization read, with the bytes it saw (offset into Result::bytes).
struct Range {
	uint64_t address = 0;
	uint32_t offset  = 0;
	uint32_t size    = 0;
};

struct Result {
	Key                  key {};
	Snapshot             snapshot;
	Specialization       specialization;
	std::vector<uint8_t> active_sources;
	std::vector<Range>   ranges;
	std::vector<uint8_t> bytes;
	uint64_t             worker_ns = 0;
};

struct State {
	std::mutex                           mutex;
	std::condition_variable              wake;
	std::condition_variable              stopped;
	std::deque<Key>                      queue;
	Key                                  in_flight {};
	bool                                 has_in_flight = false;
	std::vector<std::unique_ptr<Result>> results;
	std::atomic<uint32_t>                result_count {0};
	bool                                 stop  = false;
	bool                                 alive = false;
	// GPU thread only: (shader base, user-data count) -> plan seen for it by ProgramCache::Get.
	std::unordered_map<uint64_t, const Plan*> registry;
};

// Intentionally leaked: the detached worker may outlive static destruction.
inline State& GetState() {
	static State* const state = new State;
	return *state;
}

// Kill switch: set to stop all prefetching for the rest of the process (nothing sets it today).
inline std::atomic<bool> g_disabled {false};

inline bool Active() {
	return !g_disabled.load(std::memory_order_relaxed) && Common::RuntimeFlags::Current().drawprep_prefetch;
}

// KYTY_DRAWPREP_VERIFY=1 (read once): re-materialize every kVerifyEvery-th adoption and compare.
inline bool VerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAWPREP_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline uint64_t RegistryKey(uint64_t shader_base, size_t user_count) {
	return (shader_base << 6u) ^ static_cast<uint64_t>(user_count);
}

// GPU thread: ProgramCache::Get resolved a compute program.
inline void NotePlan(uint64_t shader_base, size_t user_count, const Plan* plan) {
	GetState().registry[RegistryKey(shader_base, user_count)] = plan;
}

// ---- worker side -------------------------------------------------------------------------------------

// Side-effect-free readers: plain reads of the guest backing. No GPU-dirty download, no capture into shared
// structures. Anything not plainly readable aborts the request (the data could be stale or absent). Every
// successful read is appended to a blob (address, size, bytes) for the merge below.
struct RawRead {
	uint64_t address = 0;
	uint32_t size    = 0;
	uint32_t offset  = 0;
};
struct WorkerReadState {
	bool                 failed   = false;
	bool                 overflow = false;
	std::vector<RawRead> reads;
	std::vector<uint8_t> blob;
};
inline bool ReadBacking(WorkerReadState& state, uint64_t address, std::span<uint32_t> values) {
	if (values.empty() ||
	    !Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	if (state.blob.size() + values.size_bytes() > kMaxRecordedWords * sizeof(uint32_t)) {
		state.overflow = true;
		return true; // the value is still correct; the result is dropped after materialization
	}
	state.reads.push_back({address, static_cast<uint32_t>(values.size_bytes()),
	                       static_cast<uint32_t>(state.blob.size())});
	const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
	state.blob.insert(state.blob.end(), bytes, bytes + values.size_bytes());
	return true;
}
inline bool WorkerReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& state = *static_cast<WorkerReadState*>(userdata);
	if (ReadBacking(state, address, values)) return true;
	state.failed = true;
	return false;
}
inline bool WorkerReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) return true;
	auto& state = *static_cast<WorkerReadState*>(userdata);
	if (ReadBacking(state, address, values)) return true;
	state.failed = true;
	return false;
}
// Optional span capability: declining performs no read and is not an error; the walker falls back to the
// ordered scalar reads above.
inline bool WorkerSpan(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return ReadBacking(*static_cast<WorkerReadState*>(userdata), address, values);
}

// Sorts the recorded reads and merges overlapping/adjacent ones into contiguous ranges. Returns false when
// two reads of the same address disagree (the guest changed memory under the walk): the result is dropped.
inline bool MergeReads(const WorkerReadState& state, std::vector<Range>& ranges, std::vector<uint8_t>& bytes) {
	std::vector<uint32_t> order(state.reads.size());
	for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
		const RawRead& left = state.reads[a];
		const RawRead& right = state.reads[b];
		return left.address != right.address ? left.address < right.address : left.size > right.size;
	});
	ranges.clear();
	bytes.clear();
	bytes.reserve(state.blob.size());
	for (const uint32_t index: order) {
		const RawRead&  read = state.reads[index];
		const uint8_t*  src  = state.blob.data() + read.offset;
		const uint64_t  end  = read.address + read.size;
		if (end < read.address) return false;
		if (!ranges.empty()) {
			Range&         last     = ranges.back();
			const uint64_t last_end = last.address + last.size;
			if (read.address <= last_end) {
				const uint64_t overlap_end = std::min(end, last_end);
				if (overlap_end > read.address &&
				    std::memcmp(bytes.data() + last.offset + (read.address - last.address), src,
				                static_cast<size_t>(overlap_end - read.address)) != 0) {
					return false;
				}
				if (end > last_end) {
					bytes.insert(bytes.end(), src + (last_end - read.address), src + read.size);
					last.size += static_cast<uint32_t>(end - last_end);
				}
				continue;
			}
		}
		ranges.push_back({read.address, static_cast<uint32_t>(bytes.size()), read.size});
		bytes.insert(bytes.end(), src, src + read.size);
	}
	return true;
}

inline std::unique_ptr<Result> Produce(const Key& key) {
	auto& pf     = FrameBreakdown::g_prefetch;
	auto  result = std::make_unique<Result>();
	result->key  = key;

	WorkerReadState reader;
	reader.reads.reserve(64);
	reader.blob.reserve(1024);
	Runtime         runtime;
	runtime.user_data                  = std::span<const uint32_t>(result->key.user_data, key.user_count);
	runtime.shader_base                = key.shader_base;
	runtime.workgroup_counts           = std::span<const uint32_t>(result->key.workgroups, 3);
	runtime.userdata                   = &reader;
	runtime.read_memory                = WorkerReadOrdinary;
	runtime.read_specialization_memory = WorkerReadStrict;
	if (Common::RuntimeFlags::Current().srt_scalar_batch) {
		runtime.try_read_memory_span          = WorkerSpan;
		runtime.try_read_specialization_span = WorkerSpan;
	}
	// specialization_cache stays null: it belongs to the GPU thread's SourceEntry and is derived data only.

	// The emulator is built without exceptions, so there is no catch: an allocation failure terminates here
	// exactly as it would on the GPU thread. (abort_exc stays in the counters for a future guarded variant.)
	const auto start = std::chrono::steady_clock::now();
	bool       ok    = false;
	{
		// One guard scope covers the materialization and the read of plan.active_sources.
		const ShaderRecompiler::IR::PlanAccessGuard plan_access(*key.plan);
		ok = ShaderRecompiler::IR::MaterializeResources(*key.plan, runtime, result->snapshot, result->specialization);
		if (ok) result->active_sources = key.plan->active_sources;
	}
	result->worker_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                              std::chrono::steady_clock::now() - start).count());
	pf.worker_ns.fetch_add(result->worker_ns, std::memory_order_relaxed);
	if (reader.failed) {
		pf.abort_read.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	if (!ok || reader.overflow || !MergeReads(reader, result->ranges, result->bytes)) {
		pf.abort_fail.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	pf.worker_done.fetch_add(1, std::memory_order_relaxed);
	return result;
}

inline void WorkerMain() {
	FrameBreakdown::g_prefetch_worker = true;
	auto& st = GetState();
	auto& pf = FrameBreakdown::g_prefetch;
	for (;;) {
		Key key;
		{
			std::unique_lock lock(st.mutex);
			st.wake.wait(lock, [&] { return st.stop || !st.queue.empty(); });
			if (st.stop) break;
			key               = st.queue.front();
			st.queue.pop_front();
			st.in_flight      = key;
			st.has_in_flight  = true;
		}
		auto result = Produce(key);
		{
			std::lock_guard lock(st.mutex);
			st.has_in_flight = false;
			if (result != nullptr && !st.stop) {
				if (st.results.size() >= kMaxResults) {
					st.results.erase(st.results.begin());
					pf.unused.fetch_add(1, std::memory_order_relaxed);
				}
				st.results.push_back(std::move(result));
				st.result_count.store(static_cast<uint32_t>(st.results.size()), std::memory_order_relaxed);
			}
		}
	}
	std::lock_guard lock(st.mutex);
	st.alive = false;
	st.stopped.notify_all();
}

// ---- GPU thread side ---------------------------------------------------------------------------------

// GPU thread, from ResourceMemo::Refresh's memo-disabled branch (see PrefetchDirectFn). On success the
// outputs and plan.active_sources are exactly what a fresh materialization would have produced; on failure
// nothing has been modified.
inline bool TryDirect(const Plan& plan, const Runtime& runtime, Snapshot& snapshot, Specialization& specialization) {
	auto& st = GetState();
	auto& pf = FrameBreakdown::g_prefetch;
	if (runtime.workgroup_counts.size() != 3) return false; // not a compute stage
	if (st.result_count.load(std::memory_order_relaxed) == 0) {
		pf.fb_notfound.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const auto start = std::chrono::steady_clock::now();
	const auto elapsed = [&] {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now() - start).count());
	};
	std::unique_ptr<Result> candidate;
	bool                    program_seen = false;
	{
		std::lock_guard lock(st.mutex);
		for (auto it = st.results.begin(); it != st.results.end(); ++it) {
			const Key& key = (*it)->key;
			if (key.plan != &plan || key.shader_base != runtime.shader_base ||
			    key.workgroups[0] != runtime.workgroup_counts[0] ||
			    key.workgroups[1] != runtime.workgroup_counts[1] ||
			    key.workgroups[2] != runtime.workgroup_counts[2]) {
				continue;
			}
			program_seen = true;
			// ALL user-data registers, count included: they are inputs of the walk (and of push data).
			if (key.user_count == runtime.user_data.size() &&
			    std::equal(runtime.user_data.begin(), runtime.user_data.end(), key.user_data)) {
				candidate = std::move(*it);
				st.results.erase(it);
				break;
			}
		}
		st.result_count.store(static_cast<uint32_t>(st.results.size()), std::memory_order_relaxed);
	}
	if (candidate == nullptr) {
		(program_seen ? pf.fb_regs : pf.fb_notfound).fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	// Every byte the walk read must still be there, clean (not GPU-dirty) and equal.
	for (const Range& range: candidate->ranges) {
		const auto verdict = Libs::LibKernel::Memory::TryValidateCleanBacking(
		    range.address, candidate->bytes.data() + range.offset, range.size);
		if (verdict == Libs::LibKernel::Memory::CleanBackingCompare::Equal) continue;
		(verdict == Libs::LibKernel::Memory::CleanBackingCompare::Dirty ? pf.fb_dirty : pf.fb_mismatch)
		    .fetch_add(1, std::memory_order_relaxed);
		pf.unused.fetch_add(1, std::memory_order_relaxed);
		pf.validate_ns.fetch_add(elapsed(), std::memory_order_relaxed);
		return false;
	}

	candidate->snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	static uint32_t adoptions = 0; // GPU thread only
	if (VerifyEnabled() && (adoptions++ % kVerifyEvery) == 0) {
		// Reference: the normal materialization with the caller's real readers, into scratch.
		Snapshot       fresh_snapshot;
		Specialization fresh_specialization;
		bool           same         = false;
		bool           materialized = false;
		{
			const ShaderRecompiler::IR::PlanAccessGuard plan_access(plan);
			materialized = ShaderRecompiler::IR::MaterializeResources(plan, runtime, fresh_snapshot,
			                                                          fresh_specialization);
			same = materialized && plan.active_sources == candidate->active_sources &&
			       ResourceMemoEntry::SameResourceSnapshot(candidate->snapshot, fresh_snapshot) &&
			       candidate->specialization == fresh_specialization;
		}
		if (!materialized) return false; // let the caller's own materialization report it
		pf.verify_checked.fetch_add(1, std::memory_order_relaxed);
		if (!same) pf.verify_mismatch.fetch_add(1, std::memory_order_relaxed);
		// plan.active_sources already holds the fresh run's value; install the reference outputs.
		snapshot       = std::move(fresh_snapshot);
		specialization = std::move(fresh_specialization);
	} else {
		{
			const ShaderRecompiler::IR::PlanAccessGuard plan_access(plan);
			plan.active_sources = std::move(candidate->active_sources);
		}
		snapshot       = std::move(candidate->snapshot);
		specialization = std::move(candidate->specialization);
	}
	pf.used_direct.fetch_add(1, std::memory_order_relaxed);
	pf.used_worker_ns.fetch_add(candidate->worker_ns, std::memory_order_relaxed);
	pf.validate_ns.fetch_add(elapsed(), std::memory_order_relaxed);
	return true;
}

// Process exit (EXIT paths run static destructors but not ~ProgramCache): stop the detached worker before
// the memory subsystem and the plans go away. Bounded wait, so a wedged worker can never hang shutdown.
inline void StopForExit() {
	auto& st = GetState();
	g_disabled.store(true, std::memory_order_relaxed);
	std::unique_lock lock(st.mutex);
	g_prefetch_direct.store(nullptr, std::memory_order_release);
	if (!st.alive) return;
	st.stop = true;
	st.wake.notify_all();
	st.stopped.wait_for(lock, std::chrono::milliseconds(500), [&] { return !st.alive; });
}

// Starts the worker on first use. Order matters: plan locking must be on before the worker (and the adopt
// hook) can exist, and this runs on the GPU thread outside any materialization.
inline void EnsureStarted(State& st) {
	std::lock_guard lock(st.mutex);
	if (st.alive) return;
	ShaderRecompiler::IR::g_plan_access_enabled.store(true, std::memory_order_release);
	g_prefetch_direct.store(&TryDirect, std::memory_order_release);
	static const bool registered = [] {
		std::atexit(StopForExit);
		return true;
	}();
	(void)registered;
	st.stop  = false;
	st.alive = true;
	std::thread(WorkerMain).detach();
}

// GPU thread: a look-ahead found a DISPATCH_DIRECT that will run with these compute inputs.
inline void RequestCompute(uint64_t shader_base, std::span<const uint32_t> user_data,
                           const std::array<uint32_t, 3>& workgroups) {
	auto& st = GetState();
	auto& pf = FrameBreakdown::g_prefetch;
	pf.candidates.fetch_add(1, std::memory_order_relaxed);
	if (user_data.size() > kMaxUserData) return;
	const auto found = st.registry.find(RegistryKey(shader_base, user_data.size()));
	if (found == st.registry.end()) {
		pf.skip_no_plan.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const Plan& plan = *found->second;
	Key key {};
	key.plan        = &plan;
	key.shader_base = shader_base;
	key.user_count  = static_cast<uint32_t>(user_data.size());
	std::copy(workgroups.begin(), workgroups.end(), key.workgroups);
	std::copy(user_data.begin(), user_data.end(), key.user_data);

	EnsureStarted(st);
	std::lock_guard lock(st.mutex);
	const auto dup = [&] {
		if (st.has_in_flight && SameInputs(st.in_flight, key)) return true;
		if (std::ranges::any_of(st.queue, [&](const Key& queued) { return SameInputs(queued, key); })) return true;
		return std::ranges::any_of(st.results, [&](const auto& result) { return SameInputs(result->key, key); });
	}();
	if (dup) {
		pf.skip_dup.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (st.queue.size() >= kMaxPending) {
		pf.skip_full.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	st.queue.push_back(key);
	pf.requests.fetch_add(1, std::memory_order_relaxed);
	st.wake.notify_one();
}

// GPU thread, before the plans the worker references are destroyed (ProgramCache destructor).
inline void Shutdown() {
	auto& st = GetState();
	std::unique_lock lock(st.mutex);
	g_prefetch_direct.store(nullptr, std::memory_order_release);
	if (st.alive) {
		st.stop = true;
		st.wake.notify_all();
		st.stopped.wait(lock, [&] { return !st.alive; });
	}
	st.stop = false;
	st.queue.clear();
	st.results.clear();
	st.result_count.store(0, std::memory_order_relaxed);
	st.registry.clear();
}

} // namespace Libs::Graphics::DrawPrepPrefetch

#endif
