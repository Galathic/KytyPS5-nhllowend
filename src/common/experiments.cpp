#include "common/experiments.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <array>
#include <cinttypes>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace Common::Experiments {

namespace {

// "1" enables, "0" disables, anything else (including unset) keeps the default.
bool EnvSwitch(const char* name, bool on_by_default) {
	const char* value = std::getenv(name);
	if (value != nullptr && (value[0] == '0' || value[0] == '1') && value[1] == '\0') {
		return value[0] == '1';
	}
	return on_by_default;
}

uint64_t EnvironmentMask() {
	uint64_t mask = 0;
	const auto set = [&mask](const char* name, Bit bit, bool on_by_default) {
		if (EnvSwitch(name, on_by_default)) mask |= bit;
	};
	// The NHL 27 recommended set is on unless its variable is set to 0.
	set("KYTY_SKIP_REDUNDANT_BINDS", SkipRedundantBinds, true);
	set("KYTY_SRT_BATCH_READS", SrtBatchReads, true);
	set("KYTY_DIRECT_SRT_READS", DirectSrtReads, true);
	set("KYTY_NATIVE_MINMAX", NativeMinMax, true);
	set("KYTY_IMAGE_LOOKUP_HINT", ImageLookupHint, true);
	set("KYTY_WIDE_FAULT_WINDOWS", WideFaultWindows, true);
	set("KYTY_CLEAN_FROM_DIRECT", CleanFromDirect, true);
	set("KYTY_LDS_LANE_SLOTS", LdsLaneSlots, true);
	// Not part of the recommended set: opt in with 1.
	set("KYTY_CPU_COPY_FORWARD", CpuCopyForward, false);
	set("KYTY_WIDE_WRITE_DOWNLOADS", WideWriteDownloads, true);
	set("KYTY_ASYNC_SUBMIT", AsyncSubmit, false);
	set("KYTY_LAZY_SHARED_WRITES", LazySharedWrites, false);
	set("KYTY_NATIVE_FP16", NativeFp16, false);
	set("KYTY_FAULT_PAGE_READS", FaultPageReads, false);
	set("KYTY_EXEC_SELECTS", ExecSelects, false);
	set("KYTY_GPU_COLOR_CLEAR", GpuColorClear, false);
	set("KYTY_POOLED_BDA_BUFFERS", PooledBdaBuffers, false);
	set("KYTY_ASYNC_GUEST_READS", AsyncGuestReads, false);
	set("KYTY_BDA_DIRTY_REGIONS", BdaDirtyRegions, false);
	set("KYTY_CP_BACKING_READS", CpBackingReads, true);
	set("KYTY_EARLY_SUBMIT_WRITERS", EarlySubmitWriters, false);
	set("KYTY_READBACK_COMPUTE", ReadbackCompute, true);
	set("KYTY_ROBUST_LOADS", RobustLoads, true);
	set("KYTY_FUSED_MAD", FusedMad, false);
	set("KYTY_FAULT_PAGE_DIRECT", FaultPageDirect, true);
	set("KYTY_BACKING_READ_MRU", BackingReadMru, true);
	set("KYTY_DISPATCH_AHEAD", DispatchAhead, true);
	set("KYTY_GPU_CONDITIONS_BOTH_PATHS", GpuConditionsBothPaths, true);
	set("KYTY_COMPUTE_UNLOCKED", ComputeUnlocked, true);
	set("KYTY_AHEAD_UNLOCKED", AheadUnlocked, true);
	set("KYTY_BACKING_LOOKUP_CACHE", BackingLookupCache, true);
	set("KYTY_PROTECT_SHARED_LOCK", ProtectSharedLock, false);
	set("KYTY_CLOCK_WAIT_PREDICT", ClockWaitPredict, false);
	set("KYTY_FALSE_SHARE_ANY_THREAD", FalseShareAnyThread, false);
	set("KYTY_SPECULATE_DIRTY_READS", SpeculateDirtyReads, false);
	set("KYTY_CP_CLEAN_REG_READS", CpCleanRegReads, false);
	set("KYTY_CP_SKIP_IDLE_SPIN", CpSkipIdleSpin, false);
	set("KYTY_SPECULATE_STRICT", SpeculateStrict, false);
	set("KYTY_ASYNC_READBACK", AsyncReadback, false);
	set("KYTY_DIRECT_GPU_READS", DirectGpuReads, false);
	set("KYTY_PROGRAM_LOCK_SPLIT", ProgramLockSplit, false);
	set("KYTY_UNMAP_BUFFER_WAIT", UnmapBufferWait, false);
	set("KYTY_FILL_PATTERNS", FillPatterns, false);
	return mask;
}

struct State {
	std::atomic<uint64_t>& mask = Detail::g_mask;

	State() {
		const uint64_t defaults = EnvironmentMask();
		mask.store(defaults, std::memory_order_relaxed);
		const char* path_env = std::getenv("KYTY_EXPERIMENT_FILE");
		if (path_env == nullptr || path_env[0] == '\0') {
			return;
		}
		std::thread([this, path = std::string(path_env), defaults] {
			uint64_t last = defaults;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				uint64_t value = defaults;
				if (std::FILE* f = std::fopen(path.c_str(), "rb"); f != nullptr) {
					uint64_t parsed = 0;
					if (std::fscanf(f, "%" SCNu64, &parsed) == 1) {
						value = parsed;
					}
					std::fclose(f);
				}
				if (value != last) {
					mask.store(value, std::memory_order_relaxed);
					std::printf("Experiments: mask %" PRIu64 " -> %" PRIu64 "\n", last, value);
					std::fflush(stdout);
					last = value;
				}
			}
		}).detach();
	}
};

State& GetState() {
	static State state;
	return state;
}

} // namespace

bool FalseShareReads() noexcept {
	static const bool enabled = EnvSwitch("KYTY_FALSE_SHARE_READS", false);
	return enabled;
}

bool DirectGpuReadsMemory() noexcept {
	static const bool enabled = EnvSwitch("KYTY_DIRECT_GPU_READS", false);
	return enabled;
}

bool SrtUnionDirty() noexcept {
	static const bool enabled = EnvSwitch("KYTY_SRT_UNION_DIRTY", false);
	return enabled;
}

namespace Detail {
constinit std::atomic<uint64_t> g_mask {Uninitialized};

uint64_t InitMask() noexcept {
	(void)GetState();
	return g_mask.load(std::memory_order_relaxed);
}
} // namespace Detail

} // namespace Common::Experiments

namespace Common::Timeline {

namespace {

struct Event {
	int64_t     ns    = 0;
	uint32_t    tid   = 0;
	const char* what  = nullptr;
	uint64_t    value = 0;
};

struct Recorder {
	static constexpr size_t Capacity = 1u << 19u;
	std::unique_ptr<Event[]> events = std::make_unique<Event[]>(Capacity);
	std::atomic<size_t>      next {0};
	std::atomic<bool>        recording {false};
	bool                     enabled = false;
	std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();

	Recorder() {
		const char* value = std::getenv("KYTY_DBG_TIMELINE");
		if (value == nullptr || value[0] == '\0') return;
		enabled = true;
		// A number is a delay in seconds; anything else names a file whose appearance starts it.
		const bool        by_file = value[0] < '0' || value[0] > '9';
		const auto        delay   = by_file ? 0 : std::atoi(value);
		const std::string trigger = by_file ? value : "";
		std::thread([this, delay, trigger] {
			if (trigger.empty()) {
				std::this_thread::sleep_for(std::chrono::seconds(delay));
			} else {
				for (;;) {
					if (std::FILE* f = std::fopen(trigger.c_str(), "rb"); f != nullptr) {
						std::fclose(f);
						break;
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
				}
			}
			origin = std::chrono::steady_clock::now();
			recording.store(true, std::memory_order_release);
			// KYTY_DBG_TIMELINE_MS: capture length (default 1500 ms).
			const char* length = std::getenv("KYTY_DBG_TIMELINE_MS");
			std::this_thread::sleep_for(std::chrono::milliseconds(length != nullptr ? std::atoi(length) : 1500));
			recording.store(false, std::memory_order_release);
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			const auto count = std::min(next.load(), Capacity);
#if defined(_WIN32)
			// Return addresses in marks are absolute; this base turns them into image offsets.
			std::printf("NHL27TL: image base 0x%llx\n",
			            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr))));
#endif
			std::vector<Event> sorted(events.get(), events.get() + count);
			std::sort(sorted.begin(), sorted.end(),
			          [](const Event& a, const Event& b) { return a.ns < b.ns; });
			for (const auto& e: sorted) {
				std::printf("NHL27TL: %9.3f ms tid=%u %s %llu\n", static_cast<double>(e.ns) / 1e6, e.tid,
				            e.what, static_cast<unsigned long long>(e.value));
			}
			std::fflush(stdout);
		}).detach();
	}
};

Recorder& GetRecorder() {
	static Recorder recorder;
	return recorder;
}

uint32_t ThreadTag() {
	static std::atomic<uint32_t> counter {0};
	thread_local const uint32_t tag = counter.fetch_add(1) + 1;
	return tag;
}

} // namespace

bool Enabled() noexcept {
	return GetRecorder().enabled;
}

void MarkStack() noexcept {
	auto& recorder = GetRecorder();
	if (!recorder.enabled || !recorder.recording.load(std::memory_order_acquire)) return;
#if defined(_WIN32)
	void*      frames[10] {};
	const auto count = RtlCaptureStackBackTrace(1, 10, frames, nullptr);
	for (USHORT i = 0; i < count; i++) {
		Mark("ra", reinterpret_cast<uint64_t>(frames[i]));
	}
#endif
}

void Mark(const char* what, uint64_t value) noexcept {
	auto& recorder = GetRecorder();
	if (!recorder.enabled || !recorder.recording.load(std::memory_order_acquire)) return;
	const auto index = recorder.next.fetch_add(1, std::memory_order_relaxed);
	if (index >= Recorder::Capacity) return;
	recorder.events[index] = {
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
	                                                         recorder.origin).count(),
	    ThreadTag(), what, value};
}

} // namespace Common::Timeline

namespace Common::Sampler {

#if defined(_WIN32) && defined(_M_X64) || defined(_WIN32) && defined(__x86_64__)
namespace {

constexpr int MaxThreads = 8;
constexpr int MaxFrames  = 24;

struct Target {
	HANDLE      handle = nullptr;
	const char* name   = nullptr;
};

struct SamplerState {
	std::mutex                  lock;
	std::array<Target, MaxThreads> targets {};
	int                         count   = 0;
	bool                        started = false;
};

SamplerState& GetSampler() {
	static SamplerState state;
	return state;
}

bool SamplingEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_SAMPLE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Walks the suspended thread's stack with the x64 unwind tables. Nothing here may allocate or
// take a lock the target could hold (heap, CRT): the target is suspended meanwhile.
int Unwind(CONTEXT context, uint64_t* functions) {
	int depth = 0;
	for (int i = 0; i < MaxFrames && context.Rip != 0; i++) {
		DWORD64 image_base = 0;
		auto*   entry      = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
		if (entry == nullptr) {
			// Leaf function or code without unwind data (guest or generated code).
			functions[depth++] = context.Rip | 1u;  // odd = raw address, not a function start
			if (i != 0) break;
			context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
			context.Rsp += 8;
			continue;
		}
		functions[depth++] = image_base + entry->BeginAddress;
		void*   handler_data = nullptr;
		DWORD64 establisher  = 0;
		RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, entry, &context, &handler_data,
		                 &establisher, nullptr);
	}
	return depth;
}

void Run() {
	struct Counts {
		std::unordered_map<uint64_t, uint64_t> self;
		std::unordered_map<uint64_t, uint64_t> incl;
		// Caller -> callee pairs on the sampled stacks (each pair once per sample), so a function's
		// time can be split by who called it.
		struct Edge {
			uint64_t from  = 0;
			uint64_t to    = 0;
			uint64_t count = 0;
		};
		std::unordered_map<uint64_t, Edge>     edges;
		uint64_t                               samples = 0;
	};
	std::array<Counts, MaxThreads> counts {};
	auto last = std::chrono::steady_clock::now();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		std::array<Target, MaxThreads> targets {};
		int                            n = 0;
		{
			auto&            s = GetSampler();
			std::scoped_lock guard(s.lock);
			targets = s.targets;
			n       = s.count;
		}
		for (int t = 0; t < n; t++) {
			uint64_t functions[MaxFrames] {};
			int      depth = 0;
			if (SuspendThread(targets[t].handle) == static_cast<DWORD>(-1)) continue;
			CONTEXT context {};
			context.ContextFlags = CONTEXT_FULL;
			if (GetThreadContext(targets[t].handle, &context) != 0) {
				depth = Unwind(context, functions);
			}
			ResumeThread(targets[t].handle);
			if (depth == 0) continue;
			auto& c = counts[t];
			c.samples++;
			c.self[functions[0]]++;
			std::unordered_set<uint64_t> seen;
			for (int i = 0; i < depth; i++) {
				if (seen.insert(functions[i]).second) c.incl[functions[i]]++;
			}
			std::unordered_set<uint64_t> seen_edges;
			for (int i = 0; i + 1 < depth; i++) {
				const uint64_t key = functions[i + 1] ^ (functions[i] * 0x9E3779B97F4A7C15ull);
				if (!seen_edges.insert(key).second) continue;
				auto& e = c.edges[key];
				e.from  = functions[i + 1];
				e.to    = functions[i];
				e.count++;
			}
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - last < std::chrono::seconds(10)) continue;
		last = now;
		for (int t = 0; t < n; t++) {
			auto& c = counts[t];
			if (c.samples == 0) continue;
			const auto print = [&](const char* kind, const std::unordered_map<uint64_t, uint64_t>& map, size_t top) {
				std::vector<std::pair<uint64_t, uint64_t>> sorted(map.begin(), map.end());
				std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				for (size_t i = 0; i < sorted.size() && i < top; i++) {
					std::printf("NHL27SAMPLE: thread=%s %s %.1f%% addr=0x%" PRIx64 "\n", targets[t].name, kind,
					            100.0 * static_cast<double>(sorted[i].second) / static_cast<double>(c.samples),
					            sorted[i].first);
				}
			};
			std::printf("NHL27SAMPLE: thread=%s samples=%" PRIu64 "\n", targets[t].name, c.samples);
			print("self", c.self, 30);
			print("incl", c.incl, 80);
			{
				std::vector<const Counts::Edge*> sorted;
				sorted.reserve(c.edges.size());
				for (const auto& [key, e]: c.edges) sorted.push_back(&e);
				std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) { return a->count > b->count; });
				for (size_t i = 0; i < sorted.size() && i < 150; i++) {
					std::printf("NHL27SAMPLE: thread=%s edge %.1f%% from=0x%" PRIx64 " to=0x%" PRIx64 "\n", targets[t].name,
					            100.0 * static_cast<double>(sorted[i]->count) / static_cast<double>(c.samples),
					            sorted[i]->from, sorted[i]->to);
				}
			}
			c = Counts {};
		}
		std::fflush(stdout);
	}
}

} // namespace

void RegisterCurrentThread(const char* name) noexcept {
	if (!SamplingEnabled()) return;
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0) == 0) {
		return;
	}
	auto&            s = GetSampler();
	std::scoped_lock guard(s.lock);
	if (s.count == MaxThreads) return;
	s.targets[static_cast<size_t>(s.count++)] = {handle, name};
	if (!s.started) {
		s.started = true;
		std::thread(Run).detach();
	}
}
#else
void RegisterCurrentThread(const char*) noexcept {}
#endif

} // namespace Common::Sampler
