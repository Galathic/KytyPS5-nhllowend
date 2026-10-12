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

bool EnvOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] == '1';
}

uint32_t EnvironmentMask() {
	uint32_t mask = 0;
	if (EnvOn("KYTY_GUEST_SYNC")) mask |= GuestSync;
	if (EnvOn("KYTY_SKIP_REDUNDANT_BINDS")) mask |= SkipRedundantBinds;
	if (EnvOn("KYTY_DEDUP_BARRIERS")) mask |= DedupBarriers;
	if (EnvOn("KYTY_CPU_COPY_FORWARD")) mask |= CpuCopyForward;
	if (EnvOn("KYTY_FALSE_SHARE_READS")) mask |= FalseShareReads;
	if (EnvOn("KYTY_SRT_BATCH_READS")) mask |= SrtBatchReads;
	if (EnvOn("KYTY_RESOURCE_MULTI_CACHE")) mask |= ResourceMultiCache;
	if (EnvOn("KYTY_BACKING_READ_CACHE")) mask |= BackingReadCache;
	if (EnvOn("KYTY_DIRECT_SRT_READS")) mask |= DirectSrtReads;
	if (EnvOn("KYTY_TEXTURE_DESC_CACHE")) mask |= TextureDescCache;
	if (EnvOn("KYTY_IMAGE_LOOKUP_HINT")) mask |= ImageLookupHint;
	if (EnvOn("KYTY_LONG_BUFFER_GC_AGE")) mask |= LongBufferGcAge;
	if (EnvOn("KYTY_RENDER_TARGET_MEMO")) mask |= RenderTargetMemo;
	if (EnvOn("KYTY_ADAPTIVE_RESOURCE_MEMO")) mask |= AdaptiveResourceMemo;
	if (EnvOn("KYTY_FAULT_COALESCE")) mask |= FaultCoalesce;
	if (EnvOn("KYTY_SPARSE_BUFFER_GC")) mask |= SparseBufferGc;
	if (EnvOn("KYTY_CLEAN_FROM_DIRECT")) mask |= CleanFromDirect;
	if (EnvOn("KYTY_DIRECT_EOP_INTERRUPTS")) mask |= DirectEopInterrupts;
	if (EnvOn("KYTY_READBACK_QUEUE")) mask |= ReadbackQueue;
	if (EnvOn("KYTY_ASYNC_SUBMIT")) mask |= AsyncSubmit;
	return mask;
}

struct State {
	std::atomic<uint32_t> mask {0};

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

bool Enabled(Bit bit) noexcept {
	return (GetState().mask.load(std::memory_order_relaxed) & bit) != 0;
}

uint32_t Mask() noexcept {
	return GetState().mask.load(std::memory_order_relaxed);
}

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
