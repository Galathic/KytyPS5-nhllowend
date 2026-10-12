#pragma once

// Diagnostic joined event timeline (KYTY_IDLE_TRACE). Header-only, lock-free.
// Record() is a single relaxed load + branch when not capturing.
// Capture starts on the flag's 0->1 transition (Poll, called from a submit safe point),
// lasts kDurationNs or until the buffer fills, then Poll() writes idle-trace-<unix>.csv to the cwd.
#include "common/runtimeFlags.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <thread>

namespace Common::EventTrace {

#define KYTY_EVENT_TRACE_TYPES(X) \
	X(GPU_WAIT_BEGIN) X(GPU_WAIT_END) X(GPU_CMD_BEGIN) X(GPU_CMD_END) X(GUEST_SUBMIT) \
	X(GUEST_FLIP_REQ) X(GUEST_EQ_WAIT_BEGIN) X(GUEST_EQ_WAIT_END) X(GUEST_SEMA_WAIT_BEGIN) \
	X(GUEST_SEMA_WAIT_END) X(EOP_TRIGGER) X(LABEL_WRITE) X(FLIP_EVENT) X(VBLANK_EVENT) \
	X(NATIVE_SUBMIT) X(NATIVE_DONE) X(SUSPEND_ACQ_BEGIN) X(SUSPEND_ACQ_END) X(SUSPEND_RELEASE) \
	X(FLIP_READY) X(PRESENT) X(FAULT_PARSE) X(GPU_BEGIN) X(GPU_END) \
	X(GPU_DRAW) X(GPU_DRAW_END) X(GPU_DISPATCH) X(GPU_DISPATCH_END)

enum Type : uint32_t {
#define X(n) n,
	KYTY_EVENT_TRACE_TYPES(X)
#undef X
	TYPE_COUNT
};

inline constexpr const char* kTypeNames[] = {
#define X(n) #n,
    KYTY_EVENT_TRACE_TYPES(X)
#undef X
};

struct Event {
	uint64_t t_ns;
	uint32_t type;
	uint32_t thread;
	uint64_t a;
	uint64_t b;
};

inline constexpr uint32_t kCapacity   = 2u * 1024u * 1024u;
inline constexpr int64_t  kDurationNs = 3'000'000'000;

inline std::atomic<bool>     g_capturing {false};
inline std::atomic<uint32_t> g_index {0};
inline std::atomic<bool>     g_prev_flag {false};
inline Event*                g_events   = nullptr;
inline int64_t               g_start_ns = 0;

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline uint32_t ThreadTag() {
	static std::atomic<uint32_t> next {1};
	thread_local const uint32_t  tag = next.fetch_add(1, std::memory_order_relaxed);
	return tag;
}

inline void RecordSlow(uint32_t type, uint64_t a, uint64_t b) {
	const auto i = g_index.fetch_add(1, std::memory_order_relaxed);
	if (i >= kCapacity) {
		return;
	}
	auto& e = g_events[i];
	e.t_ns   = static_cast<uint64_t>(NowNs());
	e.type   = type;
	e.thread = ThreadTag();
	e.a      = a;
	e.b      = b;
}

// Records an event with an explicit timestamp in NowNs() (steady_clock) units, e.g. a GPU time
// converted to the CPU clock. Such events may be out of order in the buffer. Pre-start times are dropped.
inline void RecordAt(uint32_t type, int64_t t_ns, uint64_t a = 0, uint64_t b = 0) {
	if (g_capturing.load(std::memory_order_relaxed)) [[unlikely]] {
		if (t_ns < g_start_ns) {
			return;
		}
		const auto i = g_index.fetch_add(1, std::memory_order_relaxed);
		if (i >= kCapacity) {
			return;
		}
		auto& e  = g_events[i];
		e.t_ns   = static_cast<uint64_t>(t_ns);
		e.type   = type;
		e.thread = ThreadTag();
		e.a      = a;
		e.b      = b;
	}
}

inline void Record(uint32_t type, uint64_t a = 0, uint64_t b = 0) {
	if (g_capturing.load(std::memory_order_relaxed)) [[unlikely]] {
		RecordSlow(type, a, b);
	}
}

// Called after the CSV was written with the unix time used in its file name (side files reuse it).
inline void (*g_csv_written_hook)(long long unix_s) = nullptr;

inline void WriteCsv(uint32_t count) {
	char name[64];
	const auto unix_s = static_cast<long long>(std::time(nullptr));
	std::snprintf(name, sizeof(name), "idle-trace-%lld.csv", unix_s);
	FILE* f = std::fopen(name, "w");
	if (f == nullptr) {
		return;
	}
	std::fprintf(f, "t_us,type,thread,a,b\n");
	for (uint32_t i = 0; i < count; i++) {
		const auto& e  = g_events[i];
		const auto  dt = static_cast<int64_t>(e.t_ns) - g_start_ns;
		std::fprintf(f, "%lld.%03lld,%s,%u,%llu,%llu\n", static_cast<long long>(dt / 1000), static_cast<long long>((dt % 1000 + 1000) % 1000),
		             e.type < TYPE_COUNT ? kTypeNames[e.type] : "?", e.thread, static_cast<unsigned long long>(e.a),
		             static_cast<unsigned long long>(e.b));
	}
	std::fclose(f);
	std::printf("NHL27 idle-trace wrote %s (%u events)\n", name, count);
	std::fflush(stdout);
	if (g_csv_written_hook != nullptr) {
		g_csv_written_hook(unix_s);
	}
}

// Call from a submit/frame safe point after RuntimeFlags::Refresh().
inline void Poll() {
	const bool flag = (RuntimeFlags::g_published.load(std::memory_order_relaxed) & (uint64_t {1} << RuntimeFlags::kBitIdleTrace)) != 0;
	if (g_capturing.load(std::memory_order_relaxed)) {
		if (NowNs() - g_start_ns < kDurationNs && g_index.load(std::memory_order_relaxed) < kCapacity) {
			return;
		}
		bool expected = true;
		if (!g_capturing.compare_exchange_strong(expected, false)) {
			return;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5)); // let in-flight writers finish
		const auto n = std::min(g_index.load(), kCapacity);
		WriteCsv(n);
		g_prev_flag.store(flag);
		return;
	}
	const bool prev = g_prev_flag.exchange(flag);
	if (flag && !prev) {
		if (g_events == nullptr) {
			g_events = new Event[kCapacity];
		}
		g_index.store(0);
		g_start_ns = NowNs();
		g_capturing.store(true, std::memory_order_release);
	}
}

} // namespace Common::EventTrace
