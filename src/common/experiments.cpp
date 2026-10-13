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

uint32_t EnvironmentMask() {
	uint32_t mask = 0;
	const auto set = [&mask](const char* name, Bit bit, bool on_by_default) {
		if (EnvSwitch(name, on_by_default)) mask |= bit;
	};
	// The NHL 27 recommended set is on unless its variable is set to 0.
	set("KYTY_SKIP_REDUNDANT_BINDS", SkipRedundantBinds, true);
	set("KYTY_SRT_BATCH_READS", SrtBatchReads, true);
	set("KYTY_DIRECT_SRT_READS", DirectSrtReads, true);
	set("KYTY_TEXTURE_DESC_CACHE", TextureDescCache, true);
	set("KYTY_IMAGE_LOOKUP_HINT", ImageLookupHint, true);
	set("KYTY_ADAPTIVE_RESOURCE_MEMO", AdaptiveResourceMemo, true);
	set("KYTY_CLEAN_FROM_DIRECT", CleanFromDirect, true);
	set("KYTY_LDS_LANE_SLOTS", LdsLaneSlots, true);
	set("KYTY_IMAGE_RECYCLE", ImageRecycle, true);
	// Not part of the recommended set: opt in with 1.
	set("KYTY_CPU_COPY_FORWARD", CpuCopyForward, false);
	set("KYTY_FALSE_SHARE_READS", FalseShareReads, false);
	set("KYTY_ASYNC_SUBMIT", AsyncSubmit, false);
	set("KYTY_LAZY_SHARED_WRITES", LazySharedWrites, false);
	set("KYTY_NATIVE_FP16", NativeFp16, false);
	set("KYTY_FAULT_PAGE_READS", FaultPageReads, false);
	set("KYTY_EXEC_SELECTS", ExecSelects, false);
	set("KYTY_GPU_COLOR_CLEAR", GpuColorClear, false);
	set("KYTY_POOLED_BDA_BUFFERS", PooledBdaBuffers, false);
	set("KYTY_ASYNC_GUEST_READS", AsyncGuestReads, false);
	set("KYTY_BDA_DIRTY_REGIONS", BdaDirtyRegions, false);
	set("KYTY_GPU_THREAD_PRIORITY", GpuThreadPriority, false);
	return mask;
}

struct State {
	std::atomic<uint64_t>& mask = Detail::g_mask;

	State() {
		const uint32_t defaults = EnvironmentMask();
		mask.store(defaults, std::memory_order_relaxed);
		const char* path_env = std::getenv("KYTY_EXPERIMENT_FILE");
		if (path_env == nullptr || path_env[0] == '\0') {
			return;
		}
		std::thread([this, path = std::string(path_env), defaults] {
			uint32_t last = defaults;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				uint32_t value = defaults;
				if (std::FILE* f = std::fopen(path.c_str(), "rb"); f != nullptr) {
					unsigned parsed = 0;
					if (std::fscanf(f, "%u", &parsed) == 1) {
						value = parsed;
					}
					std::fclose(f);
				}
				if (value != last) {
					mask.store(value, std::memory_order_relaxed);
					std::printf("Experiments: mask %u -> %u\n", last, value);
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
	static constexpr size_t Capacity = 1u << 16u;
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
			std::this_thread::sleep_for(std::chrono::milliseconds(1500));
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
