#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "common/experiments.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		// Uniform checks depend on the active EXEC mask; reuse a success under the same mask.
		// Without this, shared subexpressions are revalidated once per path (exponential).
		if (require_uniform) {
			if (const auto it = m_validated_uniform.find(inst); it != m_validated_uniform.end() &&
			    std::find(it->second.begin(), it->second.end(), m_active_mask) != it->second.end()) {
				return true;
			}
		}
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			if (valid && require_uniform) m_validated_uniform[inst].push_back(m_active_mask);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
	std::unordered_map<const Inst*, std::vector<Value>> m_validated_uniform;
};


} // namespace

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_context(AcquireContext(program)) {}

SrtWalker::~SrtWalker() { --m_program.evaluation_depth; }

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 4;
	return context;
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low bits distinguish in-progress and failed evaluations in this session.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		++m_cycle_count;
		return false;
	}
	if (m_context.values[index].generation == (m_context.generation | 2u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const auto cycles_before = m_cycle_count;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		// Cyclic dependencies may become evaluable after another node completes.
		// Only retain failures independent of an in-progress evaluation.
		memo.generation = cycles_before == m_cycle_count ? m_context.generation | 2u : 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::RawReadAddress(const Inst& inst, uint64_t& address, uint64_t bytes, bool& zero) {
	zero = false;
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			zero = true;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				zero = true;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < bytes) {
			return false;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	return bytes != 0 && address <= AddressMask && bytes - 1 <= AddressMask - address;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	uint64_t address = 0;
	bool zero = false;
	if (!RawReadAddress(inst, address, sizeof(uint32_t), zero)) return false;
	if (zero) { result = 0; return true; }
	uint32_t word = 0;
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector) return false;
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::RefreshReadBatch(uint32_t slot, uint32_t count, std::vector<uint32_t>& flat) {
	const auto& read = m_program.srt_reads[slot];
	const bool clean = read.flat_offset < m_clean_flat_slots.size() && m_clean_flat_slots[read.flat_offset] != 0;
	if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) return false;
	auto& evaluator = clean ? *m_clean_evaluator : *this;
	if (read.flat_offset > flat.size() || count > flat.size() - read.flat_offset) return false;
	auto& slot_indices = m_program.srt_evaluation_indices;
	if (slot_indices.size() != m_program.srt_reads.size()) {
		slot_indices.resize(m_program.srt_reads.size());
		for (size_t i = 0; i < slot_indices.size(); ++i)
			slot_indices[i] = m_program.srt_reads[i].value.ResolveInstruction()->EvaluationIndex(m_program.evaluation_value_count);
	}
	const auto* indices = slot_indices.data() + slot;
	const auto already_evaluated = [&] {
		for (uint32_t i = 0; i < count; ++i) {
			if (indices[i] < evaluator.m_context.values.size() &&
			    (evaluator.m_context.values[indices[i]].generation & ~uint64_t{3}) == evaluator.m_context.generation)
				return true;
		}
		return false;
	};
	const auto scalar = [&] {
		for (uint32_t i = 0; i < count; ++i)
			if (!evaluator.Evaluate(m_program.srt_reads[slot + i].value, flat[read.flat_offset + i])) return false;
		return true;
	};
	// A batch must preserve prior successes/failures in this evaluation session.
	if (already_evaluated()) return scalar();
	uint64_t address = 0;
	bool zero = false;
	if (!evaluator.RawReadAddress(*read.value.ResolveInstruction(), address, count * 4u, zero) || zero) return false;
	if (already_evaluated()) return scalar();
	const auto words = std::span(flat).subspan(read.flat_offset, count);
	if (evaluator.m_runtime.read_memory != nullptr) {
		if (!evaluator.m_runtime.read_memory(evaluator.m_runtime.userdata, address, words)) return false;
	} else {
		std::memcpy(words.data(), reinterpret_cast<const void*>(address), words.size_bytes());
	}
	// Descriptor expressions can refer back to individual raw reads. Publish each
	// result into the same dense memo used by ordinary scalar evaluation.
	if (evaluator.m_context.values.size() < m_program.evaluation_value_count)
		evaluator.m_context.values.resize(m_program.evaluation_value_count);
	for (uint32_t i = 0; i < count; ++i)
		evaluator.m_context.values[indices[i]] = {words[i], evaluator.m_context.generation};
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (inst.GetOpcode() == ValueOpcode::LoadBufferU32 && m_clean_evaluator != nullptr &&
				    m_clean_evaluator->m_active_mask == m_active_mask) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
			if (binary()) {
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	const int32_t* direct = m_direct_flat != nullptr && m_program.direct_descriptor_offsets.size() >= (source + 1u) * 8u
	                            ? m_program.direct_descriptor_offsets.data() + source * 8u
	                            : nullptr;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (direct != nullptr && index < 8u && direct[index] >= 0 &&
		    static_cast<size_t>(direct[index]) < m_direct_flat->size() &&
		    (m_direct_filled == nullptr ||
		     (static_cast<size_t>(direct[index]) < m_direct_filled->size() &&
		      (*m_direct_filled)[static_cast<size_t>(direct[index])] != 0u))) {
			result.dwords[index] = (*m_direct_flat)[static_cast<size_t>(direct[index])];
			continue;
		}
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

namespace {

std::atomic<int> g_direct_srt_override {-1};

bool DirectSrtReadsEnabled() {
	const int forced = g_direct_srt_override.load(std::memory_order_relaxed);
	return forced >= 0 ? forced != 0 : Common::Experiments::Enabled(Common::Experiments::DirectSrtReads);
}

// Decides once per plan whether every SRT read is LoadAddressU32(GetAddressResource(
// GetUserData(lo), GetUserData(hi)), imm, ...) on a scalar address, and records its operands.
// KYTY_DBG_DIRECT_SRT=1: why plans fall back to the interpreter and how often they refresh.
bool DirectSrtDebug() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DBG_DIRECT_SRT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

std::string DirectShape(Value value, int depth) {
	value = value.Resolve();
	if (value.IsImmediate()) return "imm";
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return "?";
	std::string result(ValueOpcodeName(inst->GetOpcode()));
	if (inst->GetOpcode() == ValueOpcode::ReadConst || inst->GetOpcode() == ValueOpcode::GetUserData)
		return result;
	if (depth >= 3) return result + "(...)";
	result += "(";
	for (size_t i = 0; i < inst->NumArgs(); ++i) {
		if (i != 0) result += ",";
		result += DirectShape(inst->Arg(i), depth + 1);
	}
	return result + ")";
}

struct DirectDebugState {
	std::mutex                                      lock;
	std::unordered_map<const ResourcePlan*, std::string> reasons;
	std::unordered_map<const ResourcePlan*, uint64_t>    refreshes;
	uint64_t                                        counts[3] {};
};

DirectDebugState& DirectDebug() {
	static DirectDebugState state;
	return state;
}

bool RejectDirect(const ResourcePlan& program, const char* reason, Value value) {
	if (DirectSrtDebug()) {
		auto&            state = DirectDebug();
		std::scoped_lock lock {state.lock};
		state.reasons[&program] = std::string(reason) + " reads=" + std::to_string(program.srt_reads.size()) +
		                          " cf=" + std::to_string(program.control_flow.size()) + " " +
		                          DirectShape(value, 0).substr(0, 200);
	}
	return false;
}

// kind: 0 direct without control flow, 1 direct with control flow, 2 interpreter.
void CountDirectRefresh(const ResourcePlan& program, int kind) {
	auto&            state = DirectDebug();
	std::scoped_lock lock {state.lock};
	++state.counts[kind];
	if (kind == 2) ++state.refreshes[&program];
	const auto total = state.counts[0] + state.counts[1] + state.counts[2];
	if (total % 50000u != 0u) return;
	std::printf("NHL27SRT: refreshes direct=%llu direct_cf=%llu interpreted=%llu\n",
	            static_cast<unsigned long long>(state.counts[0]),
	            static_cast<unsigned long long>(state.counts[1]),
	            static_cast<unsigned long long>(state.counts[2]));
	std::vector<std::pair<uint64_t, const ResourcePlan*>> hot;
	for (const auto& [plan, count]: state.refreshes) hot.emplace_back(count, plan);
	std::sort(hot.rbegin(), hot.rend());
	for (size_t i = 0; i < hot.size() && i < 10; ++i) {
		const auto reason = state.reasons.find(hot[i].second);
		std::printf("NHL27SRT:   %llu %s\n", static_cast<unsigned long long>(hot[i].first),
		            reason != state.reasons.end() ? reason->second.c_str() : "(no reason)");
	}
	std::fflush(stdout);
}

bool BuildDirectSrtReads(const ResourcePlan& program, std::span<const uint8_t> clean_flat_slots) {
	auto& direct = program.direct_srt_reads;
	direct.clear();
	direct.reserve(program.srt_reads.size());
	const auto user_data_index = [&](Value value, uint32_t& index) {
		value = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetUserData) return false;
		const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
		if (reg < program.user_data_base) return false;
		index = reg - program.user_data_base;
		return true;
	};
	for (const auto& read: program.srt_reads) {
		const auto value = read.value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32) {
			return RejectDirect(program, "not-load", value);
		}
		if (!IsRawRead(program, *inst)) return RejectDirect(program, "not-raw", value);
		if (read.flat_offset >= program.srt_reads.size()) return RejectDirect(program, "flat", value);
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() < 2) {
			return RejectDirect(program, "handle", value);
		}
		ResourcePlan::DirectSrtRead entry;
		if (!user_data_index(handle->Arg(0), entry.low_register) ||
		    !user_data_index(handle->Arg(1), entry.high_register)) {
			return RejectDirect(program, "base", value);
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			return RejectDirect(program, "offset", value);
		}
		const auto& mem       = program.memory_info[inst->Flags<MemoryFlags>().index];
		const auto  immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		entry.relative = (immediate & ~int64_t {3}) + static_cast<int64_t>(offset.U32() & ~3u);
		entry.evaluation_index = inst->EvaluationIndex(program.evaluation_value_count);
		entry.clean = read.flat_offset < clean_flat_slots.size() && clean_flat_slots[read.flat_offset] != 0u;
		direct.push_back(entry);
	}
	auto& offsets = program.direct_descriptor_offsets;
	offsets.assign(program.descriptor_sources.size() * 8u, -1);
	for (size_t source = 0; source < program.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source];
		for (uint32_t i = 0; i < descriptor.dword_count && i < 8u; ++i) {
			const auto* inst = descriptor.dwords[i].Resolve().TryInstruction();
			if (inst == nullptr || inst->GetOpcode() != ValueOpcode::ReadConst) continue;
			const auto slot = inst->Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= program.srt_reads.size())
				continue;
			offsets[source * 8u + i] = static_cast<int32_t>(program.srt_reads[slot.U32()].flat_offset);
		}
	}
	// Group contiguous dwords read through the same pointer into runs (one reader call each).
	for (size_t head = 0; head < direct.size();) {
		size_t next = head + 1;
		while (next < direct.size() && next - head < 256 &&
		       direct[next].clean == direct[head].clean &&
		       direct[next].low_register == direct[head].low_register &&
		       direct[next].high_register == direct[head].high_register &&
		       direct[next].relative == direct[next - 1].relative + 4 &&
		       program.srt_reads[next].flat_offset == program.srt_reads[next - 1].flat_offset + 1) {
			direct[next].run = 0;
			++next;
		}
		direct[head].run = static_cast<uint32_t>(next - head);
		head = next;
	}
	return true;
}

} // namespace

void SetDirectSrtReadsOverride(int value) {
	g_direct_srt_override.store(value, std::memory_order_relaxed);
}

bool SrtWalker::DirectReadRun(uint32_t slot, uint32_t count, std::vector<uint32_t>& flat) {
	const auto& direct = m_program.direct_srt_reads;
	if (count == 0 || slot >= direct.size() || count > direct.size() - slot) return false;
	const auto& head = direct[slot];
	if (head.clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
		return false;
	auto&       evaluator = head.clean ? *m_clean_evaluator : *this;
	const auto& user_data = evaluator.m_runtime.user_data;
	if (head.low_register >= user_data.size() || head.high_register >= user_data.size()) return false;
	const auto base = ((static_cast<uint64_t>(user_data[head.high_register]) << 32u) |
	                   user_data[head.low_register]) &
	                  AddressMask;
	uint64_t   address = 0;
	const auto bytes   = static_cast<uint64_t>(count) * 4u;
	if (!AddSignedAddress(base & ~uint64_t {3}, head.relative, address) || address > AddressMask ||
	    bytes - 1 > AddressMask - address) {
		return false;
	}
	const auto first = m_program.srt_reads[slot].flat_offset;
	if (first > flat.size() || count > flat.size() - first) return false;
	const auto words = std::span(flat).subspan(first, count);
	if (evaluator.m_runtime.read_memory != nullptr) {
		if (!evaluator.m_runtime.read_memory(evaluator.m_runtime.userdata, address, words)) return false;
	} else {
		std::memcpy(words.data(), reinterpret_cast<const void*>(address), words.size_bytes());
	}
	// Descriptor and condition expressions refer back to these reads through the memo.
	auto& values = evaluator.m_context.values;
	if (values.size() < m_program.evaluation_value_count) values.resize(m_program.evaluation_value_count);
	for (uint32_t i = 0; i < count; ++i) {
		values[direct[slot + i].evaluation_index] = {words[i], evaluator.m_context.generation};
	}
	return true;
}

bool SrtWalker::RefreshDirectFlat(std::vector<uint32_t>& flat) {
	const auto& direct = m_program.direct_srt_reads;
	flat.resize(m_program.srt_reads.size());
	for (size_t slot = 0; slot < direct.size();) {
		// Without batching every slot is its own read, as in the scalar walk.
		const uint32_t count = m_runtime.batch_reads ? direct[slot].run : 1u;
		if (!DirectReadRun(static_cast<uint32_t>(slot), count, flat)) return false;
		slot += count;
	}
	m_direct_flat   = &flat;
	m_direct_filled = nullptr;
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	bool direct = false;
	if (DirectSrtReadsEnabled()) {
		if (m_program.direct_srt_state == ResourcePlan::DirectSrtState::Unknown) {
			m_program.direct_srt_state = BuildDirectSrtReads(m_program, m_clean_flat_slots)
			                                 ? ResourcePlan::DirectSrtState::Direct
			                                 : ResourcePlan::DirectSrtState::Unsupported;
		}
		direct = m_program.direct_srt_state == ResourcePlan::DirectSrtState::Direct;
		if (DirectSrtDebug()) {
			CountDirectRefresh(m_program, !direct ? 2 : m_program.control_flow.empty() ? 0 : 1);
		}
		if (direct && m_program.control_flow.empty()) {
			return RefreshDirectFlat(flat);
		}
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		return read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size();) {
			const auto count = m_runtime.batch_reads && slot < m_program.srt_batch_counts.size()
			                       ? m_program.srt_batch_counts[slot] : 1u;
			if (!(count > 1 ? RefreshReadBatch(slot, count, flat) : refresh(slot))) return false;
			slot += count;
		}
		return true;
	}
	flat.assign(m_program.srt_reads.size(), 0u);
	if (direct) {
		m_program.direct_filled.assign(m_program.srt_reads.size(), 0u);
	}
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) active.at(source) = 0u;
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) continue;
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (size_t position = 0; direct && position < block.srt_reads.size();) {
			// Direct reads: extend the run while this block's next slot continues it.
			const auto  slot  = block.srt_reads[position];
			const auto& reads = m_program.direct_srt_reads;
			if (slot >= reads.size()) return false;
			uint32_t count = 1;
			while (m_runtime.batch_reads && position + count < block.srt_reads.size() &&
			       block.srt_reads[position + count] == slot + count && slot + count < reads.size() &&
			       reads[slot + count].run == 0u && reads[slot + count].clean == reads[slot].clean) {
				++count;
			}
			if (!DirectReadRun(slot, count, flat)) return false;
			for (uint32_t i = 0; i < count; ++i) {
				const auto offset = m_program.srt_reads[slot + i].flat_offset;
				if (offset < m_program.direct_filled.size()) m_program.direct_filled[offset] = 1u;
			}
			position += count;
		}
		for (size_t position = 0; !direct && position < block.srt_reads.size();) {
			const auto slot = block.srt_reads[position];
			const auto possible = m_runtime.batch_reads && slot < m_program.srt_batch_counts.size()
			                          ? m_program.srt_batch_counts[slot] : 1u;
			uint32_t count = 1;
			// Do not fetch adjacent reads belonging to another branch or a gap in this block.
			while (count < possible && position + count < block.srt_reads.size() &&
			       block.srt_reads[position + count] == slot + count) ++count;
			if (!(count > 1 ? RefreshReadBatch(slot, count, flat) : refresh(slot))) return false;
			position += count;
		}
		uint32_t condition = 0;
		auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr &&
		    predicate.Evaluate(block.condition, condition)) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	if (direct) {
		m_direct_flat   = &flat;
		m_direct_filled = &m_program.direct_filled;
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
