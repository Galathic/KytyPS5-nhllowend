#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);
using SrtUserDataObserver = void (*)(void* userdata, uint32_t index);

// Optional per-evaluation diagnostics; no global atomics on the recording/replay path.
struct SrtEvaluationCounts {
	uint64_t recursive_ops = 0;
	uint64_t tape_ops = 0;
	uint64_t cfg_schedule_steps = 0;
	uint64_t scalar_batch_words = 0;
	uint64_t cfg_gather_eligible = 0;
	uint64_t cfg_gather_hits = 0;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	// Independent from userdata, which belongs to the memory-reader callbacks.
	SrtUserDataObserver       observe_user_data = nullptr;
	void*                     user_data_observer_userdata = nullptr;
	SrtEvaluationCounts* evaluation_counts = nullptr;
	// Optional completed-span capabilities: false performs no reads, transfers,
	// faults or observations; true is equivalent to successful ordered reads of
	// every requested DWORD in that reader domain. These share reader userdata.
	SrtMemoryReader try_read_memory_span = nullptr;
	SrtMemoryReader try_read_specialization_span = nullptr;
};

enum class RuntimeValueType { Any, Integer };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// KYTY_SRT_TAPE counters (GPU thread; relaxed atomics).
struct SrtTapeStats {
	uint64_t runs = 0;          // flat refreshes served by a tape
	uint64_t failed = 0;        // tape run reported a walker-equivalent failure
	uint64_t walker = 0;        // refreshes that fell back to the recursive walker
	uint64_t built = 0;         // tapes built
	uint64_t unsupported = 0;   // plans rejected by the tape builder
	uint64_t ops = 0;           // total ops executed
	uint64_t cfg_runs = 0;      // supported CFG roots demanded (including memo hits)
	uint64_t cfg_fallback = 0;  // roots evaluated recursively; unsupported or flag off
	uint64_t cfg_scheduled = 0; // roots served by a cached ordered demand schedule
	uint64_t cfg_checked = 0;  // CFG refreshes entering immutable-read verification
	uint64_t cfg_matched = 0;  // fully covered matching recursive refreshes
	uint64_t cfg_mismatch = 0; // fully covered mismatch; validated recursive fallback
	uint64_t cfg_inconclusive = 0; // rejected refresh: no safe captured-only comparison
	uint64_t cfg_trace_hits = 0;    // CFG refreshes replayed from a recorded trace
	uint64_t cfg_trace_misses = 0;  // trace present but path/memo diverged; DFS fallback
	uint64_t cfg_trace_recorded = 0; // paths merged into a trace
	std::array<uint64_t, 8> reject {}; // rejection reasons, see SrtTapeRejectName
};
SrtTapeStats GetSrtTapeStats();
const char*  SrtTapeRejectName(size_t reason);

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {}, bool skip_tape = false);
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	size_t TryScalarReadBatch(std::span<const Value> values, bool strict,
	                          std::span<uint32_t> output);
	size_t TryFlatScalarBatch(std::span<const uint32_t> slots, uint32_t sequential_slot,
	                         std::vector<uint32_t>& flat);
	bool EvaluateInst(const Inst& inst, uint64_t& result);
	// 1 = refreshed, 0 = failed (same as walker), -1 = tape not applicable.
	int  RunTape(std::vector<uint32_t>& flat);
	void PrepareCfgTape();
	bool VerifyCfgRefresh(std::vector<uint32_t>& flat);
	int RunCfgRoot(uint32_t root, uint32_t& result);
	int RunCfgSchedule(uint32_t root, uint32_t& result);
	// 1 = refreshed from a recorded trace, -1 = no matching trace (run the DFS).
	int  ReplayCfgTrace(std::vector<uint32_t>& flat);
	void MergeCfgTrace();
	bool ExecuteTapeOp(const ResourcePlan::SrtTape::Op& op, std::vector<uint64_t>& values,
	                   uint32_t index);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	bool                            m_skip_tape = false;
	bool                            m_cfg_verifying = false;
	bool                            m_cfg_tape_enabled = false;
	bool                            m_cfg_schedule_enabled = false;
	bool                            m_scalar_batch_enabled = false;
	bool                            m_cfg_gather_enabled = false;
	bool                            m_cfg_trace_enabled = false;
	// Set while a CFG refresh records its trace; cleared by anything the replay cannot reproduce.
	bool                            m_cfg_trace_recording = false;
	std::vector<uint8_t>            m_cfg_gather_ready_sources;
	uint64_t                        m_cycle_count = 0;
	ResourcePlan::EvaluationContext& m_context;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
