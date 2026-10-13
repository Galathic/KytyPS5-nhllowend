#include "graphics/host_gpu/renderer/pipeline/computeLookahead.h"

#include "common/experiments.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ComputeLookahead {

namespace {

using Plan           = ShaderRecompiler::IR::ResourcePlan;
using Runtime        = ShaderRecompiler::IR::SrtRuntime;
using Snapshot       = ShaderRecompiler::IR::ResourceSnapshot;
using Specialization = ShaderRecompiler::IR::ResourceSpecialization;

constexpr uint32_t kWindowDwords = 512;
constexpr uint32_t kMaxAhead     = 12;
constexpr size_t   kSlots        = 256;
constexpr size_t   kProbe        = 4;
constexpr size_t   kMaxBytes     = 64 * 1024;
constexpr size_t   kPlanStripes  = 64;

struct ProgramInfo {
	const void* entry       = nullptr;
	const Plan* plan        = nullptr;
	bool        batch_reads = false;
};

struct Read {
	uint64_t address = 0;
	uint32_t size    = 0;
	uint32_t offset  = 0;
	bool     strict  = false;
};

struct Result {
	// 0 = free, 1 = owned (the worker writes it or the GPU thread validates it), 2 = ready.
	std::atomic<uint32_t>    state {0};
	std::atomic<uint64_t>    hash {0};
	const void*              entry      = nullptr;
	uint64_t                 base       = 0;
	std::array<uint32_t, 3>  groups     {};
	uint32_t                 user_count = 0;
	std::array<uint32_t, 32> user_data  {};
	uint64_t                 made_seq   = 0; // publish sequence the worker prepared it under
	Snapshot                 snapshot;
	Specialization           specialization;
	std::vector<Read>        reads;
	std::vector<uint8_t>     bytes;
};

struct Recorder {
	Result* result   = nullptr;
	bool    overflow = false;
};

struct State {
	std::mutex            publish_mutex;
	HW::ComputeShaderInfo published_cs;
	const uint32_t*       published_next      = nullptr;
	uint32_t              published_remaining = 0;
	std::atomic<uint64_t> publish_seq {0};

	std::mutex                                programs_mutex;
	std::unordered_map<uint64_t, ProgramInfo> programs;

	std::array<Result, kSlots>                  results;
	std::array<std::atomic_flag, kPlanStripes> plan_locks {};
	std::once_flag                              started;

	// KYTY_DBG_LOOKAHEAD=1: NHL27LOOKAHEAD counters every 5 s.
	std::atomic<uint64_t> prepared {0}, adopted {0}, rejected {0}, missing {0}, busy {0}, failed {0};
	std::atomic<uint64_t> waited {0}, prepare_ns {0}, evicted {0}, lookups {0};
};

// Never destroyed: the worker may still run while the process exits.
State& GetState() {
	static auto* state = new State;
	return *state;
}

constexpr uint64_t kSeed = 0xcbf29ce484222325ull;

uint64_t Mix(uint64_t hash, uint64_t value) {
	hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
	return hash;
}

// The compute registers BuildStageStaticKey depends on: the same program address with other
// values is another program-cache entry.
uint64_t Fingerprint(const HW::CsStageRegisters& r) {
	uint64_t hash = kSeed;
	hash          = Mix(hash, r.data_addr);
	hash          = Mix(hash, (uint64_t {r.num_thread_x} << 32u) | r.num_thread_y);
	hash          = Mix(hash, (uint64_t {r.num_thread_z} << 32u) | r.lds_size);
	hash          = Mix(hash, (uint64_t {r.float_mode} << 0u) | (uint64_t {r.wave_size} << 8u) |
	                              (uint64_t {r.scratch_en} << 16u) | (uint64_t {r.user_sgpr} << 24u) |
	                              (uint64_t {r.tgid_x_en} << 32u) | (uint64_t {r.tgid_y_en} << 33u) |
	                              (uint64_t {r.tgid_z_en} << 34u) | (uint64_t {r.tg_size_en} << 35u) |
	                              (uint64_t {r.tidig_comp_cnt} << 40u));
	return hash;
}

uint64_t InputHash(const void* entry, uint64_t base, std::span<const uint32_t> groups,
                   std::span<const uint32_t> user) {
	uint64_t hash = Mix(kSeed, reinterpret_cast<uintptr_t>(entry));
	hash          = Mix(hash, base);
	for (const auto value: groups) hash = Mix(hash, value);
	hash = Mix(hash, user.size());
	for (const auto value: user) hash = Mix(hash, value);
	return hash;
}

size_t Stripe(const Plan* plan) {
	return (reinterpret_cast<uintptr_t>(plan) >> 6u) % kPlanStripes;
}

bool Diagnostics() {
	static const bool enabled = std::getenv("KYTY_DBG_LOOKAHEAD") != nullptr;
	return enabled;
}

void MaybePrint() {
	if (!Diagnostics()) return;
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	last     = now;
	auto& st = GetState();
	const auto prepared = st.prepared.exchange(0);
	std::printf("NHL27LOOKAHEAD lookups=%llu adopted=%llu waited=%llu rejected=%llu missing=%llu prepared=%llu "
	            "(%.2f us each) evicted=%llu plan_busy=%llu failed=%llu\n",
	            static_cast<unsigned long long>(st.lookups.exchange(0)),
	            static_cast<unsigned long long>(st.adopted.exchange(0)),
	            static_cast<unsigned long long>(st.waited.exchange(0)),
	            static_cast<unsigned long long>(st.rejected.exchange(0)),
	            static_cast<unsigned long long>(st.missing.exchange(0)),
	            static_cast<unsigned long long>(prepared),
	            prepared != 0 ? static_cast<double>(st.prepare_ns.exchange(0)) / 1e3 / static_cast<double>(prepared) : 0.0,
	            static_cast<unsigned long long>(st.evicted.exchange(0)),
	            static_cast<unsigned long long>(st.busy.exchange(0)),
	            static_cast<unsigned long long>(st.failed.exchange(0)));
	std::fflush(stdout);
}

thread_local const HW::ComputeShaderInfo* t_current        = nullptr;
thread_local const uint32_t*              t_stream         = nullptr;
thread_local uint32_t                     t_stream_dwords  = 0;

// Worker readers: every value comes from the backing (never a fault on a GPU-owned page) and is
// recorded with its address so the GPU thread can check it is still what the walk saw.
bool RecordRead(void* userdata, uint64_t address, std::span<uint32_t> values, bool strict) {
	auto&      recorder = *static_cast<Recorder*>(userdata);
	const auto size     = values.size_bytes();
	if (size == 0) return true;
	if (!LibKernel::Memory::TryReadBacking(address, values.data(), size)) return false;
	auto& result = *recorder.result;
	if (result.bytes.size() + size > kMaxBytes) {
		recorder.overflow = true;
		return false;
	}
	const auto offset = result.bytes.size();
	result.bytes.resize(offset + size);
	std::memcpy(result.bytes.data() + offset, values.data(), size);
	result.reads.push_back({address, static_cast<uint32_t>(size), static_cast<uint32_t>(offset), strict});
	return true;
}

bool WorkerOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return RecordRead(userdata, address, values, false);
}

bool WorkerStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return RecordRead(userdata, address, values, true);
}

void Prepare(State& st, const Dispatch& dispatch, uint64_t seq) {
	const auto& regs = dispatch.cs.cs_regs;
	if (regs.data_addr == 0 || regs.user_sgpr > 16u) return;
	ProgramInfo info;
	{
		std::scoped_lock lock {st.programs_mutex};
		const auto found = st.programs.find(Fingerprint(regs));
		if (found == st.programs.end()) return;
		info = found->second;
	}
	const auto     user = std::span<const uint32_t>(dispatch.cs.cs_user_sgpr.value, regs.user_sgpr);
	const uint64_t hash = InputHash(info.entry, regs.data_addr, dispatch.groups, user);
	Result*        result = nullptr;
	for (size_t k = 0; k < kProbe; k++) {
		auto& slot = st.results[(hash + k) % kSlots];
		if (slot.state.load(std::memory_order_acquire) == 2u && slot.hash.load(std::memory_order_relaxed) == hash) {
			return; // already prepared
		}
	}
	// A free slot first, else a result prepared long ago that nobody took (a path not taken).
	for (int pass = 0; pass < 2 && result == nullptr; pass++) {
		for (size_t k = 0; k < kProbe && result == nullptr; k++) {
			auto&    slot     = st.results[(hash + k) % kSlots];
			uint32_t expected = slot.state.load(std::memory_order_relaxed);
			if (expected == 1u || (pass == 0 && expected != 0u) || (pass == 1 && slot.made_seq + 64u > seq)) {
				continue;
			}
			if (slot.state.compare_exchange_strong(expected, 1u, std::memory_order_acquire)) {
				result = &slot;
				if (expected == 2u) st.evicted.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
	if (result == nullptr) return;
	// The GPU thread holds a plan only while it materializes one dispatch itself: wait for it.
	auto&      plan_lock = st.plan_locks[Stripe(info.plan)];
	const auto deadline  = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
	while (plan_lock.test_and_set(std::memory_order_acquire)) {
		if (std::chrono::steady_clock::now() > deadline) {
			result->state.store(0u, std::memory_order_release);
			st.busy.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		std::this_thread::yield();
	}
	const auto started = std::chrono::steady_clock::now();
	// The hash is visible while the slot is owned: the GPU thread waits for a dispatch in progress.
	result->hash.store(hash, std::memory_order_relaxed);
	result->entry      = info.entry;
	result->base       = regs.data_addr;
	result->groups     = dispatch.groups;
	result->made_seq   = seq;
	result->user_count = regs.user_sgpr;
	std::copy(user.begin(), user.end(), result->user_data.begin());
	result->reads.clear();
	result->bytes.clear();
	Recorder recorder {result};
	Runtime  runtime {
	     .user_data                  = std::span<const uint32_t>(result->user_data.data(), result->user_count),
	     .shader_base                = regs.data_addr,
	     .read_memory                = WorkerOrdinaryRead,
	     .userdata                   = &recorder,
	     .read_specialization_memory = WorkerStrictRead,
	     .workgroup_counts           = std::span<const uint32_t>(result->groups),
	     .batch_reads                = info.batch_reads,
	};
	const bool ok = ShaderRecompiler::IR::MaterializeResources(*info.plan, runtime, result->snapshot,
	                                                           result->specialization);
	plan_lock.clear(std::memory_order_release);
	if (!ok || recorder.overflow) {
		result->hash.store(0, std::memory_order_relaxed);
		result->state.store(0u, std::memory_order_release);
		st.failed.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	result->state.store(2u, std::memory_order_release);
	st.prepared.fetch_add(1, std::memory_order_relaxed);
	st.prepare_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                            std::chrono::steady_clock::now() - started).count()),
	                        std::memory_order_relaxed);
}

void WorkerLoop() {
	auto&                              st   = GetState();
	uint64_t                           seen = 0;
	std::vector<uint32_t>              window(kWindowDwords);
	std::array<Dispatch, kMaxAhead>    ahead {};
	for (;;) {
		// Dispatches come in bursts: poll for the next publish for a while before sleeping, since
		// waking a sleeping thread takes longer than preparing one dispatch.
		const auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
		while (st.publish_seq.load(std::memory_order_acquire) == seen &&
		       std::chrono::steady_clock::now() < spin_until) {
			std::this_thread::yield();
		}
		st.publish_seq.wait(seen, std::memory_order_acquire);
		HW::ComputeShaderInfo cs;
		const uint32_t*       next      = nullptr;
		uint32_t              remaining = 0;
		{
			std::scoped_lock lock {st.publish_mutex};
			cs        = st.published_cs;
			next      = st.published_next;
			remaining = st.published_remaining;
			seen      = st.publish_seq.load(std::memory_order_relaxed);
		}
		const uint32_t dwords = std::min(remaining, kWindowDwords);
		if (dwords < 2u || next == nullptr ||
		    !LibKernel::Memory::TryReadBacking(reinterpret_cast<uint64_t>(next), window.data(),
		                                       uint64_t {dwords} * sizeof(uint32_t))) {
			continue;
		}
		const uint32_t count = ScanCompute(cs, std::span<const uint32_t>(window.data(), dwords), ahead);
		for (uint32_t i = 0; i < count; i++) {
			// A newer position from the GPU thread: what it is about to run comes first.
			if (i >= 2u && st.publish_seq.load(std::memory_order_relaxed) != seen) break;
			Prepare(st, ahead[i], seen);
		}
	}
}

} // namespace

bool Active() noexcept {
	return Common::Experiments::Enabled(Common::Experiments::ComputeLookahead);
}

void SetCurrent(const HW::ComputeShaderInfo* cs) noexcept {
	t_current = cs;
}

void SetStream(const uint32_t* next, uint32_t remaining_dw) noexcept {
	t_stream        = next;
	t_stream_dwords = remaining_dw;
}

void PublishCurrent() {
	if (t_current == nullptr || t_stream == nullptr) return;
	const auto* next         = t_stream;
	const auto  remaining_dw = t_stream_dwords;
	const auto& cs           = *t_current;
	t_stream                 = nullptr;
	auto& st                 = GetState();
	std::call_once(st.started, [] { std::thread(WorkerLoop).detach(); });
	{
		std::scoped_lock lock {st.publish_mutex};
		st.published_cs        = cs;
		st.published_next      = next;
		st.published_remaining = remaining_dw;
		st.publish_seq.fetch_add(1, std::memory_order_release);
	}
	st.publish_seq.notify_one();
	MaybePrint();
}

void NoteProgram(uint64_t shader_base, const void* entry, const Plan* plan, bool batch_reads) {
	if (t_current == nullptr || t_current->cs_regs.data_addr != shader_base) return;
	// The GPU thread notes the same few programs over and over: skip the map when unchanged.
	static thread_local uint64_t    last_key   = 0;
	static thread_local const void* last_entry = nullptr;
	const uint64_t                  key        = Fingerprint(t_current->cs_regs);
	if (key == last_key && entry == last_entry) return;
	last_key   = key;
	last_entry = entry;
	auto&            st = GetState();
	std::scoped_lock lock {st.programs_mutex};
	st.programs[key] = {entry, plan, batch_reads};
}

bool TryAdopt(const void* entry, const Runtime& runtime, Snapshot& snapshot, Specialization& specialization) {
	auto&          st   = GetState();
	const uint64_t hash = InputHash(entry, runtime.shader_base, runtime.workgroup_counts, runtime.user_data);
	st.lookups.fetch_add(1, std::memory_order_relaxed);
	for (size_t k = 0; k < kProbe; k++) {
		auto& slot = st.results[(hash + k) % kSlots];
		if (slot.hash.load(std::memory_order_relaxed) != hash) continue;
		uint32_t expected = 2u;
		if (!slot.state.compare_exchange_strong(expected, 1u, std::memory_order_acquire)) {
			if (expected != 1u) continue;
			// The worker is preparing this dispatch right now: waiting is cheaper than doing it.
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(40);
			while (slot.state.load(std::memory_order_acquire) == 1u && std::chrono::steady_clock::now() < deadline) {
				std::this_thread::yield();
			}
			expected = 2u;
			if (slot.hash.load(std::memory_order_relaxed) != hash ||
			    !slot.state.compare_exchange_strong(expected, 1u, std::memory_order_acquire)) {
				continue;
			}
			st.waited.fetch_add(1, std::memory_order_relaxed);
		}
		if (slot.hash.load(std::memory_order_relaxed) != hash || slot.entry != entry ||
		    slot.base != runtime.shader_base || runtime.workgroup_counts.size() != 3u ||
		    !std::equal(slot.groups.begin(), slot.groups.end(), runtime.workgroup_counts.begin()) ||
		    slot.user_count != runtime.user_data.size() ||
		    !std::equal(runtime.user_data.begin(), runtime.user_data.end(), slot.user_data.begin())) {
			slot.state.store(2u, std::memory_order_release);
			continue;
		}
		// Every value the walk read must still be there, read the way the GPU thread reads it.
		thread_local std::vector<uint32_t> current;
		bool                               valid = true;
		for (const auto& read: slot.reads) {
			current.resize(read.size / sizeof(uint32_t));
			const auto values = std::span<uint32_t>(current.data(), current.size());
			bool       ok     = true;
			if (read.strict) {
				ok = runtime.read_specialization_memory != nullptr &&
				     runtime.read_specialization_memory(runtime.userdata, read.address, values);
			} else if (runtime.read_memory != nullptr) {
				ok = runtime.read_memory(runtime.userdata, read.address, values);
			} else {
				// Like CaptureOrdinaryRead: pages that faulted before are read through the backing.
				ShaderRecompiler::IR::ReadOrdinaryMemory(read.address, current.data(), read.size);
			}
			if (!ok || std::memcmp(current.data(), slot.bytes.data() + read.offset, read.size) != 0) {
				valid = false;
				break;
			}
		}
		if (!valid) {
			slot.state.store(0u, std::memory_order_release);
			st.rejected.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		std::swap(snapshot, slot.snapshot);
		std::swap(specialization, slot.specialization);
		slot.state.store(0u, std::memory_order_release);
		st.adopted.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	st.missing.fetch_add(1, std::memory_order_relaxed);
	return false;
}

void LockPlan(const Plan* plan) noexcept {
	// The worker holds a stripe for one materialization at most (microseconds): spin.
	auto& flag = GetState().plan_locks[Stripe(plan)];
	while (flag.test_and_set(std::memory_order_acquire)) {
		std::this_thread::yield();
	}
}

void UnlockPlan(const Plan* plan) noexcept {
	GetState().plan_locks[Stripe(plan)].clear(std::memory_order_release);
}

} // namespace Libs::Graphics::ComputeLookahead
