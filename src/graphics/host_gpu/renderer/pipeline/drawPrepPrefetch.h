#ifndef KYTY_GRAPHICS_DRAWPREP_PREFETCH_H_
#define KYTY_GRAPHICS_DRAWPREP_PREFETCH_H_

// KYTY_DRAWPREP_PREFETCH (Stage 1 of parallel draw prep): content-validated prefetch of compute-dispatch
// resource materialization.
//
// Scope: DISPATCH_DIRECT only. When the GPU thread executes a dispatch it scans ahead in the same PM4
// indirect buffer (up to 8 draws/dispatches), replays the compute SH-register writes it finds on a copy of
// the compute shadow, and for every later DISPATCH_DIRECT whose program is already known it asks ONE worker
// thread to run the normal single-entry-memo materialization (ResourceMemoEntry::Refresh on a private,
// cold entry) against the guest memory as it is now.
//
// The worker's result is a complete ResourceMemoEntry: the plan, shader base, workgroup counts, the
// register inputs the materialization depended on, and every guest read it performed as (address, width,
// strict/ordinary reader, bytes). When the GPU thread later reaches that dispatch and its own memo misses,
// ResourceMemo calls Adopt(): the worker's entry is re-validated EXACTLY like a memo hit (register inputs
// compared, every recorded read re-issued through the real GPU-thread readers and compared byte for
// byte, which also synchronizes GPU-dirty ranges as the normal path would). Only if all of it matches is
// the entry installed as the memo state; otherwise the normal materialization runs. So the installed
// state is, by the memo's own contract, identical to what a fresh materialization would produce: the
// output is a pure function of (plan, shader base, workgroup counts, dependency registers, bytes read),
// and all of those were re-checked at use time.
//
// Thread-safety (audit): see PlanAccessGuard in ResourceMaterialization.h. A ResourcePlan's mutable
// scratch (evaluation memo, SRT tapes, CFG traces, active_sources, Value::evaluation_index) is only
// touched inside MaterializeResources or ResourceMemoEntry::Refresh, both of which hold the plan's lock
// stripe. The worker never touches a SourceEntry, its memo, its snapshot or its specialization cache, only
// the immutable plan graph (under the guard) and its own private result objects. FrameBreakdown probes are
// gated on Enabled(), which is false on the worker. Everything else reachable from materialization is
// thread_local, atomic, or mutex-protected (checked: SrtWalker/ResourceMaterialization statics).

#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/pipeline/resourceMemo.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
inline constexpr size_t   kMaxRecordedWords = 4096; // same retention limit as the single-entry memo
inline constexpr uint32_t kLookaheadWork    = 8;

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

struct Result {
	Key               key {};
	ResourceMemoEntry memo;
	Snapshot          snapshot;
	Specialization    specialization;
	uint64_t          worker_ns = 0;
};

struct PlanInfo {
	const Plan*         plan = nullptr;
	const ResourceMemo* memo = nullptr;
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
	std::unordered_map<uint64_t, PlanInfo> registry;
};

// Intentionally leaked: the detached worker may outlive static destruction.
inline State& GetState() {
	static State* const state = new State;
	return *state;
}

// Kill switch: set to stop all prefetching for the rest of the process (nothing sets it today).
inline std::atomic<bool> g_disabled {false};

inline bool Active() {
	static const bool env_ok = [] {
		// The look-ahead reads PM4 dwords ahead of the cursor with plain loads; the clean-backing modes
		// exist precisely because such loads can fault on GPU-owned pages.
		const char* backing  = std::getenv("KYTY_PM4_CLEAN_BACKING");
		const char* metadata = std::getenv("KYTY_PM4_CLEAN_METADATA");
		return !(backing != nullptr && backing[0] == '1') && !(metadata != nullptr && metadata[0] == '1');
	}();
	return env_ok && !g_disabled.load(std::memory_order_relaxed) &&
	       Common::RuntimeFlags::Current().drawprep_prefetch;
}

inline uint64_t RegistryKey(uint64_t shader_base, size_t user_count) {
	return (shader_base << 6u) ^ static_cast<uint64_t>(user_count);
}

// GPU thread: ProgramCache::Get resolved a compute program (and refreshed its memo).
inline void NotePlan(uint64_t shader_base, size_t user_count, const Plan* plan, const ResourceMemo* memo) {
	GetState().registry[RegistryKey(shader_base, user_count)] = PlanInfo {plan, memo};
}

// ---- worker side -------------------------------------------------------------------------------------

// Side-effect-free readers: plain reads of the guest backing. No GPU-dirty download, no capture into shared
// structures. Anything not plainly readable aborts the request (the data could be stale or absent).
struct WorkerReadState {
	bool failed = false;
};
inline bool ReadBacking(uint64_t address, std::span<uint32_t> values) {
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes());
}
inline bool WorkerReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (ReadBacking(address, values)) return true;
	static_cast<WorkerReadState*>(userdata)->failed = true;
	return false;
}
inline bool WorkerReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) return true;
	if (ReadBacking(address, values)) return true;
	static_cast<WorkerReadState*>(userdata)->failed = true;
	return false;
}
// Optional span capability: declining performs no read and is not an error; the walker falls back to the
// ordered scalar reads above.
inline bool WorkerSpan(void*, uint64_t address, std::span<uint32_t> values) {
	return ReadBacking(address, values);
}

inline std::unique_ptr<Result> Produce(const Key& key) {
	auto& pf     = FrameBreakdown::g_prefetch;
	auto  result = std::make_unique<Result>();
	result->key  = key;

	WorkerReadState reader;
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
	bool       reused = false;
	const auto start  = std::chrono::steady_clock::now();
	const bool ok = result->memo.Refresh(*key.plan, runtime, result->snapshot, result->specialization, true, reused);
	result->worker_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                              std::chrono::steady_clock::now() - start).count());
	pf.worker_ns.fetch_add(result->worker_ns, std::memory_order_relaxed);
	if (reader.failed) {
		pf.abort_read.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	if (!ok || result->memo.RecordedWords() > kMaxRecordedWords) {
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

// Called by ResourceMemo with the single-entry memo already known to miss. See PrefetchAdoptFn.
inline bool Adopt(ResourceMemoEntry& memo, const Plan& plan, const Runtime& runtime, Snapshot& snapshot,
                  Specialization& specialization) {
	auto& st = GetState();
	auto& pf = FrameBreakdown::g_prefetch;
	if (st.result_count.load(std::memory_order_relaxed) == 0 || runtime.workgroup_counts.size() != 3) {
		pf.not_found.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const auto start = std::chrono::steady_clock::now();
	const auto elapsed = [&] {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now() - start).count());
	};
	bool found_any = false;
	for (size_t attempt = 0; attempt < kMaxResults; ++attempt) {
		std::unique_ptr<Result> candidate;
		{
			std::lock_guard lock(st.mutex);
			for (auto it = st.results.begin(); it != st.results.end(); ++it) {
				const Key& key = (*it)->key;
				if (key.plan == &plan && key.shader_base == runtime.shader_base &&
				    key.workgroups[0] == runtime.workgroup_counts[0] &&
				    key.workgroups[1] == runtime.workgroup_counts[1] &&
				    key.workgroups[2] == runtime.workgroup_counts[2]) {
					candidate = std::move(*it);
					st.results.erase(it);
					break;
				}
			}
			st.result_count.store(static_cast<uint32_t>(st.results.size()), std::memory_order_relaxed);
		}
		if (candidate == nullptr) break;
		found_any = true;
		// Probe-only refresh = the memo hit path: register inputs, then every recorded read through the
		// real readers (which synchronize GPU-dirty ranges), then the hit-side effects (active_sources
		// restore, dependency notifications, push-data copy). It never materializes.
		bool reused = false;
		if (candidate->memo.Refresh(plan, runtime, candidate->snapshot, candidate->specialization, true,
		                            reused, true) &&
		    reused) {
			memo           = std::move(candidate->memo);
			snapshot       = std::move(candidate->snapshot);
			specialization = std::move(candidate->specialization);
			pf.used.fetch_add(1, std::memory_order_relaxed);
			pf.used_worker_ns.fetch_add(candidate->worker_ns, std::memory_order_relaxed);
			pf.validate_ns.fetch_add(elapsed(), std::memory_order_relaxed);
			return true;
		}
		pf.unused.fetch_add(1, std::memory_order_relaxed);
	}
	if (found_any) {
		pf.validate_fail.fetch_add(1, std::memory_order_relaxed);
		pf.validate_ns.fetch_add(elapsed(), std::memory_order_relaxed);
	} else {
		pf.not_found.fetch_add(1, std::memory_order_relaxed);
	}
	return false;
}

// Process exit (EXIT paths run static destructors but not ~ProgramCache): stop the detached worker before
// the memory subsystem and the plans go away. Bounded wait, so a wedged worker can never hang shutdown.
inline void StopForExit() {
	auto& st = GetState();
	g_disabled.store(true, std::memory_order_relaxed);
	std::unique_lock lock(st.mutex);
	g_prefetch_adopt.store(nullptr, std::memory_order_release);
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
	g_prefetch_adopt.store(&Adopt, std::memory_order_release);
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
	if (found == st.registry.end() || found->second.memo == nullptr || !found->second.memo->SingleEntry()) {
		pf.skip_no_plan.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const Plan& plan = *found->second.plan;
	if (found->second.memo->InputsMatchLast(plan, shader_base, user_data, workgroups)) {
		pf.skip_likely_hit.fetch_add(1, std::memory_order_relaxed);
		return;
	}
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
	g_prefetch_adopt.store(nullptr, std::memory_order_release);
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
