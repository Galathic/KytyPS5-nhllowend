#pragma once
// Offline-only TSC accumulators for the SRT materialization path (tests/ResourceMaterializationTests.cpp
// --profile-plans). Compiled out unless KYTY_SRT_PROFILE is defined to 1 for the test target; the
// emulator build never defines it, so every macro below expands to nothing there.
#ifndef KYTY_SRT_PROFILE
#define KYTY_SRT_PROFILE 0
#endif

#if KYTY_SRT_PROFILE
#include <intrin.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace SrtProf {

enum Slot : int {
	MatTotal,       // MaterializeResourcesCore
	Refresh,        // RefreshFlatBuffer (all paths)
	EvalDesc,       // EvaluateDescriptor (sum of calls; own count in `count`)
	Reader,         // guest reader callback time (inside Read steps)
	MemoWrite,      // ExecuteTapeOp memo publish (value+generation store) -- tape executor only
	Preamble,       // flat.assign + active copy + published.clear etc.
	StepUserData,
	StepAlu,
	StepRead,       // read step excluding nothing: includes address math, reader, memo store
	StepImm,
	StepMemo,       // Memo-hit steps (copy of previously published value)
	StepFlat,
	StepActivate,
	StepCond,
	StepGather,
	StepEnd,
	Count
};

inline constexpr const char* kNames[Count] = {"MatTotal", "Refresh", "EvalDesc", "Reader", "MemoWrite", "Preamble",
                                              "Step.UserData", "Step.Alu", "Step.Read(total)", "Step.Imm",
                                              "Step.Memo", "Step.Flat", "Step.Activate", "Step.Cond",
                                              "Step.Gather", "Step.End"};

struct Data {
	std::array<uint64_t, Count> cycles {};
	std::array<uint64_t, Count> count {};
	uint64_t                    steps = 0, replays = 0;
	uint64_t                    reader_calls = 0, reader_words = 0;
	uint64_t                    memo_published = 0;
	uint64_t                    eval_gather = 0, eval_recursive = 0, eval_reused = 0;
	bool                        enabled = false;
	// Optional reader-call log (address, dwords, strict) for isolating the fixture reader cost.
	struct Call { uint64_t address; uint32_t words; bool strict; };
	std::vector<Call>*          log = nullptr;
};
inline Data g;

struct Scope {
	int      slot;
	uint64_t start;
	explicit Scope(int s) : slot(s), start(g.enabled ? __rdtsc() : 0) {}
	~Scope() {
		if (!g.enabled) return;
		g.cycles[slot] += __rdtsc() - start;
		++g.count[slot];
	}
};

// Per-step attribution: one TSC read per step; the interval since the previous tick is billed to the
// previous step's slot.
struct Stepper {
	uint64_t last = 0;
	int      prev = -1;
	Stepper() : last(g.enabled ? __rdtsc() : 0) {}
	void Tick(int next) {
		if (!g.enabled) return;
		const uint64_t now = __rdtsc();
		if (prev >= 0) {
			g.cycles[prev] += now - last;
			++g.count[prev];
		}
		last = now;
		prev = next;
		++g.steps;
	}
	void Finish() {
		if (!g.enabled) return;
		const uint64_t now = __rdtsc();
		if (prev >= 0) {
			g.cycles[prev] += now - last;
			++g.count[prev];
		}
		prev = -1;
	}
};

} // namespace SrtProf

#define SRT_PROF_CAT2(a, b) a##b
#define SRT_PROF_CAT(a, b) SRT_PROF_CAT2(a, b)
#define SRT_PROF_SCOPE(slot) SrtProf::Scope SRT_PROF_CAT(srt_prof_scope_, __LINE__)(slot)
#define SRT_PROF_STEPPER(name) SrtProf::Stepper name
#define SRT_PROF_TICK(name, slot) (name).Tick(slot)
#define SRT_PROF_FINISH(name) (name).Finish()
#define SRT_PROF_COUNT(field, n) (SrtProf::g.field += (n))
#define SRT_PROF_LOG(address, words, strict) do { if (SrtProf::g.log) SrtProf::g.log->push_back({(address), (words), (strict)}); } while (0)
#else
#define SRT_PROF_SCOPE(slot) ((void)0)
#define SRT_PROF_STEPPER(name) ((void)0)
#define SRT_PROF_TICK(name, slot) ((void)0)
#define SRT_PROF_FINISH(name) ((void)0)
#define SRT_PROF_COUNT(field, n) ((void)0)
#define SRT_PROF_LOG(address, words, strict) ((void)0)
#endif
