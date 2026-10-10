#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
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
	context.refreshed_flat = nullptr;
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

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
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
			result = 0u;
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
	uint64_t   address   = 0;
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
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
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
	uint32_t word = 0;
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
	if (source < m_program.descriptor_gathers.size()) {
		const auto& gather = m_program.descriptor_gathers[source];
		const auto* flat = m_context.refreshed_flat;
		bool ready = gather.valid && m_active_mask.IsEmpty() &&
		             (!gather.needs_flat || (flat != nullptr && m_program.control_flow.empty()));
		if (ready && gather.needs_flat) {
			for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
				const auto& word = gather.words[index];
				if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::Flat &&
				    word.value >= flat->size()) ready = false;
			}
		}
		if (ready) {
			for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
				const auto& word = gather.words[index];
				if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::UserData) {
					if (word.value < m_program.user_data_base ||
					    word.value - m_program.user_data_base >= m_runtime.user_data.size()) return false;
					result.dwords[index] = m_runtime.user_data[word.value - m_program.user_data_base];
				} else if (word.kind == ResourcePlan::DescriptorGather::Word::Kind::Flat) {
					result.dwords[index] = (*flat)[word.value];
				} else {
					result.dwords[index] = word.value;
				}
			}
			return true;
		}
	}
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	m_context.refreshed_flat = nullptr;
	if (!m_program.srt_plan_complete) return false;
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
		const auto run_tape = RunTape(flat);
		if (run_tape >= 0) return run_tape != 0;

		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		m_context.refreshed_flat = &flat;
		return true;
	}
	flat.assign(m_program.srt_reads.size(), 0u);
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
		for (const auto slot: block.srt_reads) {
			if (!refresh(slot)) return false;
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
	return true;
}

namespace {
struct GlobalTapeStats {
	std::atomic<uint64_t> runs {0};
	std::atomic<uint64_t> failed {0};
	std::atomic<uint64_t> walker {0};
	std::atomic<uint64_t> built {0};
	std::atomic<uint64_t> unsupported {0};
	std::atomic<uint64_t> ops {0};
	std::array<std::atomic<uint64_t>, 8> reject {};
} g_tape_stats;

const char* const kRejectNames[] = {
	"ok",
	"control_flow",
	"incomplete_plan",
	"active_mask",
	"unsupported_opcode",
	"unsupported_type",
	"circular_or_complex",
	"missing_clean_eval",
};
} // namespace

const char* SrtTapeRejectName(size_t reason) {
	if (reason < std::size(kRejectNames)) return kRejectNames[reason];
	return "unknown";
}

SrtTapeStats GetSrtTapeStats() {
	SrtTapeStats s;
	s.runs = g_tape_stats.runs.load(std::memory_order_relaxed);
	s.failed = g_tape_stats.failed.load(std::memory_order_relaxed);
	s.walker = g_tape_stats.walker.load(std::memory_order_relaxed);
	s.built = g_tape_stats.built.load(std::memory_order_relaxed);
	s.unsupported = g_tape_stats.unsupported.load(std::memory_order_relaxed);
	s.ops = g_tape_stats.ops.load(std::memory_order_relaxed);
	for (size_t i = 0; i < s.reject.size(); ++i) {
		s.reject[i] = g_tape_stats.reject[i].load(std::memory_order_relaxed);
	}
	return s;
}

namespace {
bool BuildTapeForPlan(const ResourcePlan& plan, bool with_clean,
                      std::span<const uint8_t> clean_flat_slots,
                      ResourcePlan::SrtTape& tape, size_t& reject_reason) {
	if (!plan.control_flow.empty()) {
		reject_reason = 1;
		return false;
	}
	if (!plan.srt_plan_complete) {
		reject_reason = 2;
		return false;
	}

	tape.ops.clear();
	tape.slot_ops.clear();
	tape.slot_ops.reserve(plan.srt_reads.size());
	tape.clean_root = with_clean;

	std::unordered_map<const Inst*, uint32_t> inst_to_op;

	auto get_value_u64 = [](Value val, uint64_t& out) -> bool {
		val = val.Resolve();
		if (!val.IsImmediate()) return false;
		switch (val.GetType()) {
			case Type::U1: out = val.U1(); return true;
			case Type::U8: out = val.U8(); return true;
			case Type::U16: out = val.U16(); return true;
			case Type::U32: out = val.U32(); return true;
			case Type::U64: out = val.U64(); return true;
			case Type::F32: out = std::bit_cast<uint32_t>(val.F32Value()); return true;
			default: return false;
		}
	};

	std::function<int(Value, bool)> emit_val;
	emit_val = [&](Value val, bool is_clean) -> int {
		val = val.Resolve();
		if (val.IsImmediate()) {
			uint64_t imm = 0;
			if (!get_value_u64(val, imm)) return -1;
			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			ResourcePlan::SrtTape::Op op;
			op.kind = ResourcePlan::SrtTape::Kind::Immediate;
			op.imm = imm;
			tape.ops.push_back(op);
			return static_cast<int>(op_idx);
		}

		const auto* inst = val.TryInstruction();
		if (inst == nullptr) return -1;

		if (auto it = inst_to_op.find(inst); it != inst_to_op.end()) {
			if (is_clean && !tape.ops[it->second].clean) {
				auto mark_clean = [&](auto& self, uint32_t op_index) -> void {
					auto& target = tape.ops[op_index];
					if (target.clean) return;
					target.clean = 1;
					if (target.kind == ResourcePlan::SrtTape::Kind::Inst ||
					    target.kind == ResourcePlan::SrtTape::Kind::RawRead) {
						for (uint32_t arg : target.args) {
							if (arg < tape.ops.size()) {
								self(self, arg);
							}
						}
					}
				};
				mark_clean(mark_clean, it->second);
			}
			return static_cast<int>(it->second);
		}

		const auto opcode = inst->GetOpcode();
		ResourcePlan::SrtTape::Op op;
		op.opcode = opcode;
		op.clean = is_clean ? 1 : 0;
		op.memo_index = inst->EvaluationIndex(const_cast<ResourcePlan&>(plan).evaluation_value_count);

		if (opcode == ValueOpcode::GetUserData) {
			op.kind = ResourcePlan::SrtTape::Kind::UserData;
			op.args[0] = RegIndex(inst->Arg(0).ScalarRegister());
			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[inst] = op_idx;
			return static_cast<int>(op_idx);
		}
		if (opcode == ValueOpcode::GetShaderBase) {
			op.kind = ResourcePlan::SrtTape::Kind::ShaderBase;
			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[inst] = op_idx;
			return static_cast<int>(op_idx);
		}

		if (opcode == ValueOpcode::ReadConst) {
			const auto slot_val = inst->Arg(1).Resolve();
			if (!slot_val.IsImmediate() || slot_val.GetType() != Type::U32) return -1;
			uint32_t slot = slot_val.U32();
			if (slot >= plan.srt_reads.size()) return -1;

			bool target_clean = false;
			if (slot < clean_flat_slots.size() && clean_flat_slots[slot] != 0u && with_clean) {
				target_clean = true;
			}
			return emit_val(plan.srt_reads[slot].value, target_clean);
		}

		if (opcode == ValueOpcode::LoadAddressU32 || opcode == ValueOpcode::ReadConstBuffer ||
		    opcode == ValueOpcode::LoadBufferU32) {
			if (!IsRawRead(plan, *inst)) return -1;
			// The recursive evaluator checks the vector-load guard before evaluating
			// its address or touching memory. The tape currently cannot express that
			// conditional dependency, so keep only statically enabled vector reads.
			if (opcode == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				if (!guard.IsImmediate() || guard.GetType() != Type::U1 || !guard.U1()) {
					reject_reason = 4;
					return -1;
				}
			}
			op.kind = ResourcePlan::SrtTape::Kind::RawRead;
			op.aux = inst->Flags<MemoryFlags>().index;

			const auto* handle = inst->Arg(0).ResolveInstruction();
			if (handle == nullptr) return -1;
			int low_idx = emit_val(handle->Arg(0), is_clean);
			int high_idx = emit_val(handle->Arg(1), is_clean);
			int off_idx = emit_val(inst->Arg(1), is_clean);
			if (low_idx < 0 || high_idx < 0 || off_idx < 0) return -1;
			op.args[0] = static_cast<uint32_t>(low_idx);
			op.args[1] = static_cast<uint32_t>(high_idx);
			op.args[2] = static_cast<uint32_t>(off_idx);

			if (opcode == ValueOpcode::ReadConstBuffer || opcode == ValueOpcode::LoadBufferU32) {
				if (handle->NumArgs() != 4u) return -1;
				int rec_idx = emit_val(handle->Arg(2), is_clean);
				int w3_idx = emit_val(handle->Arg(3), is_clean);
				if (rec_idx < 0 || w3_idx < 0) return -1;
				op.args[3] = static_cast<uint32_t>(rec_idx);
				op.args[4] = static_cast<uint32_t>(w3_idx);
			}

			uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
			tape.ops.push_back(op);
			inst_to_op[inst] = op_idx;
			return static_cast<int>(op_idx);
		}

		switch (opcode) {
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32: {
				int a = emit_val(inst->Arg(0), is_clean);
				if (a < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				break;
			}
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalXor: {
				int a = emit_val(inst->Arg(0), is_clean);
				int b = emit_val(inst->Arg(1), is_clean);
				if (a < 0 || b < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.args[1] = static_cast<uint32_t>(b);
				break;
			}
			case ValueOpcode::CompositeExtractU64: {
				int a = emit_val(inst->Arg(0), is_clean);
				const auto idx_val = inst->Arg(1).Resolve();
				if (a < 0 || !idx_val.IsImmediate() || idx_val.GetType() != Type::U32) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.imm = idx_val.U32();
				break;
			}
			case ValueOpcode::CompositeExtractU32x2: {
				const auto* src = inst->Arg(0).ResolveInstruction();
				const auto idx_val = inst->Arg(1).Resolve();
				if (src == nullptr || !idx_val.IsImmediate() || idx_val.GetType() != Type::U32) return -1;
				uint32_t comp = idx_val.U32();
				if (comp >= 2u) return -1;
				if (src->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					return emit_val(src->Arg(comp), is_clean);
				}
				if (src->GetOpcode() == ValueOpcode::IAddCarry32) {
					int a = emit_val(src->Arg(0), is_clean);
					int b = emit_val(src->Arg(1), is_clean);
					if (a < 0 || b < 0) return -1;
					op.kind = ResourcePlan::SrtTape::Kind::Inst;
					op.args[0] = static_cast<uint32_t>(a);
					op.args[1] = static_cast<uint32_t>(b);
					op.imm = comp;
					break;
				}
				return -1;
			}
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract: {
				int a = emit_val(inst->Arg(0), is_clean);
				int b = emit_val(inst->Arg(1), is_clean);
				int c = emit_val(inst->Arg(2), is_clean);
				if (a < 0 || b < 0 || c < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.args[1] = static_cast<uint32_t>(b);
				op.args[2] = static_cast<uint32_t>(c);
				break;
			}
			case ValueOpcode::BitFieldInsert: {
				int a = emit_val(inst->Arg(0), is_clean);
				int b = emit_val(inst->Arg(1), is_clean);
				int c = emit_val(inst->Arg(2), is_clean);
				int d = emit_val(inst->Arg(3), is_clean);
				if (a < 0 || b < 0 || c < 0 || d < 0) return -1;
				op.kind = ResourcePlan::SrtTape::Kind::Inst;
				op.args[0] = static_cast<uint32_t>(a);
				op.args[1] = static_cast<uint32_t>(b);
				op.args[2] = static_cast<uint32_t>(c);
				op.args[3] = static_cast<uint32_t>(d);
				break;
			}
			default:
				// Any lazy/short-circuiting op (Select, Phi, ReadFirstLane, LogicalAnd, LogicalOr)
				// falls back to the dynamic walker.
				reject_reason = 4;
				return false;
		}

		uint32_t op_idx = static_cast<uint32_t>(tape.ops.size());
		tape.ops.push_back(op);
		inst_to_op[inst] = op_idx;
		return static_cast<int>(op_idx);
	};

	for (size_t slot = 0; slot < plan.srt_reads.size(); ++slot) {
		const auto& read = plan.srt_reads[slot];
		bool clean = read.flat_offset < clean_flat_slots.size() && clean_flat_slots[read.flat_offset] != 0u;
		if (clean && !with_clean) {
			reject_reason = 7;
			return false;
		}
		int emitted = emit_val(read.value, clean && with_clean);
		if (emitted < 0) {
			if (reject_reason == 0) reject_reason = 6;
			return false;
		}
		tape.slot_ops.push_back(static_cast<uint32_t>(emitted));
	}

	return true;
}
} // namespace

int SrtWalker::RunTape(std::vector<uint32_t>& flat) {
	static const bool tape_enabled = [] {
		const char* val = std::getenv("KYTY_SRT_TAPE");
		return val == nullptr || val[0] != '0';
	}();
	if (!tape_enabled || !m_active_mask.IsEmpty()) {
		g_tape_stats.walker.fetch_add(1, std::memory_order_relaxed);
		return -1;
	}

	const size_t tape_idx = (m_clean_evaluator != nullptr) ? 1 : 0;
	auto& tape = m_program.srt_tapes[tape_idx];

	if (tape.state == ResourcePlan::SrtTape::State::Unbuilt) {
		size_t reject_reason = 0;
		if (BuildTapeForPlan(m_program, m_clean_evaluator != nullptr, m_clean_flat_slots, tape, reject_reason)) {
			tape.state = ResourcePlan::SrtTape::State::Ready;
			g_tape_stats.built.fetch_add(1, std::memory_order_relaxed);
		} else {
			tape.state = ResourcePlan::SrtTape::State::Unsupported;
			g_tape_stats.unsupported.fetch_add(1, std::memory_order_relaxed);
			if (reject_reason < 8) {
				g_tape_stats.reject[reject_reason].fetch_add(1, std::memory_order_relaxed);
			}
		}
	}

	if (tape.state != ResourcePlan::SrtTape::State::Ready) {
		g_tape_stats.walker.fetch_add(1, std::memory_order_relaxed);
		return -1;
	}

	flat.resize(m_program.srt_reads.size());
	auto& vals = m_program.srt_tape_values;
	vals.resize(tape.ops.size());

	for (size_t i = 0; i < tape.ops.size(); ++i) {
		const auto& op = tape.ops[i];
		uint64_t result = 0;
		switch (op.kind) {
			case ResourcePlan::SrtTape::Kind::Immediate:
				result = op.imm;
				break;
			case ResourcePlan::SrtTape::Kind::UserData: {
				const uint32_t reg = op.args[0];
				if (reg < m_program.user_data_base ||
				    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return 0;
				}
				result = m_runtime.user_data[reg - m_program.user_data_base];
				break;
			}
			case ResourcePlan::SrtTape::Kind::ShaderBase:
				result = m_runtime.shader_base;
				break;
			case ResourcePlan::SrtTape::Kind::RawRead: {
				const auto& mem = m_program.memory_info[op.aux];
				const uint64_t low = vals[op.args[0]];
				const uint64_t high = vals[op.args[1]];
				const uint64_t offset = vals[op.args[2]];
				const auto base = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
				const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
				uint64_t address = 0;

				if (op.opcode == ValueOpcode::ReadConstBuffer || op.opcode == ValueOpcode::LoadBufferU32) {
					const uint64_t records = vals[op.args[3]];
					const uint64_t word3 = vals[op.args[4]];
					if (immediate < 0) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return 0;
					}
					const auto byte_offset =
					    (static_cast<uint64_t>(immediate) & ~uint64_t{3}) + (static_cast<uint32_t>(offset) & ~3u);
					const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
					if (op.opcode == ValueOpcode::LoadBufferU32) {
						if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u) {
							g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
							return 0;
						}
						if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
							vals[i] = 0;
							continue;
						}
					}
					const auto size = stride == 0u
					                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
					                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
					if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return 0;
					}
					address = (base & ~uint64_t{3}) + byte_offset;
				} else {
					const auto relative = (immediate & ~int64_t{3}) +
					                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
					if (!AddSignedAddress(base & ~uint64_t{3}, relative, address)) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return 0;
					}
				}

				if (op.clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
					g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
					return 0;
				}
				uint32_t word = 0;
				const bool is_vector = op.opcode == ValueOpcode::LoadBufferU32;
				const auto reader = (is_vector || op.clean) ? m_runtime.read_specialization_memory
				                                            : m_runtime.read_memory;
				if (reader != nullptr) {
					if (!reader(m_runtime.userdata, address, {&word, 1})) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return 0;
					}
				} else {
					if (is_vector) {
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return 0;
					}
					std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
				}
				result = word;
				break;
			}
			case ResourcePlan::SrtTape::Kind::Inst: {
				const uint64_t a = vals[op.args[0]];
				const uint64_t b = vals[op.args[1]];
				const uint64_t c = vals[op.args[2]];
				switch (op.opcode) {
					case ValueOpcode::BitCastU32F32:
					case ValueOpcode::BitCastF32U32: result = a; break;
					case ValueOpcode::ConvertF32U32:
						result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
						break;
					case ValueOpcode::ConvertU32F32: {
						const auto v = Float32(a);
						if (!std::isfinite(v) || v < 0.0f || static_cast<double>(v) > UINT32_MAX) {
							g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
							return 0;
						}
						result = static_cast<uint32_t>(v);
						break;
					}
					case ValueOpcode::FPTrunc32:
						result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
						break;
					case ValueOpcode::FPRecipIFlag32: {
						const auto exp = (a >> 23u) & 0xffu;
						if ((a & 0x807fffffu) != 0u || exp == 0u || exp >= 254u) {
							g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
							return 0;
						}
						result = (254u - exp) << 23u;
						break;
					}
					case ValueOpcode::FPIsNan32: result = std::isnan(Float32(a)); break;
					case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); break;
					case ValueOpcode::CompositeConstructU64:
						result = static_cast<uint32_t>(a) |
						         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
						break;
					case ValueOpcode::CompositeExtractU64:
						result = static_cast<uint32_t>(a >> (op.imm * 32u));
						break;
					case ValueOpcode::CompositeExtractU32x2: { // used for IAddCarry32
						const auto sum = static_cast<uint64_t>(static_cast<uint32_t>(a)) +
						                 static_cast<uint32_t>(b);
						result = op.imm == 0u ? static_cast<uint32_t>(sum)
						                      : static_cast<uint32_t>(sum >> 32u);
						break;
					}
					case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); break;
					case ValueOpcode::IAdd64: result = a + b; break;
					case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); break;
					case ValueOpcode::ISub64: result = a - b; break;
					case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); break;
					case ValueOpcode::IMul64: result = a * b; break;
					case ValueOpcode::UMulHi:
						result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
						break;
					case ValueOpcode::UMin32:
						result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
						break;
					case ValueOpcode::FPMul32:
						result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
						break;
					case ValueOpcode::FPOrdLessThanEqual32:
						result = Float32(a) <= Float32(b);
						break;
					case ValueOpcode::FPOrdGreaterThanEqual32:
						result = Float32(a) >= Float32(b);
						break;
					case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); break;
					case ValueOpcode::BitwiseAnd64: result = a & b; break;
					case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); break;
					case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); break;
					case ValueOpcode::ShiftLeftLogical32:
						result = static_cast<uint32_t>(a) << (b & 31u);
						break;
					case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); break;
					case ValueOpcode::ShiftRightLogical32:
						result = static_cast<uint32_t>(a) >> (b & 31u);
						break;
					case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); break;
					case ValueOpcode::ShiftRightArithmetic32:
						result = static_cast<uint32_t>(
						    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
						break;
					case ValueOpcode::ShiftRightArithmetic64:
						result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
						break;
					case ValueOpcode::IEqual32:
						result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
						break;
					case ValueOpcode::INotEqual32:
						result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
						break;
					case ValueOpcode::ULessThan32:
						result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
						break;
					case ValueOpcode::ULessThanEqual32:
						result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
						break;
					case ValueOpcode::UGreaterThan32:
						result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
						break;
					case ValueOpcode::SGreaterThanEqual32:
						result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
						         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
						break;
					case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); break;
					case ValueOpcode::BitFieldUExtract: {
						const auto offset = static_cast<uint32_t>(b);
						const auto width = static_cast<uint32_t>(c);
						if (offset > 32u || width > 32u - offset) {
							g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
							return 0;
						}
						const auto mask = width == 32u ? UINT32_MAX : (width == 0u ? 0u : (1u << width) - 1u);
						result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
						break;
					}
					case ValueOpcode::BitFieldSExtract: {
						const auto offset = static_cast<uint32_t>(b);
						const auto width = static_cast<uint32_t>(c);
						if (offset > 32u || width > 32u - offset) {
							g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
							return 0;
						}
						if (width == 0u) {
							result = 0;
						} else {
							const auto mask = width == 32u ? UINT32_MAX : (1u << width) - 1u;
							auto bits = (static_cast<uint32_t>(a) >> offset) & mask;
							if (width < 32u && (bits & (1u << (width - 1u))) != 0u) bits |= ~mask;
							result = bits;
						}
						break;
					}
					case ValueOpcode::BitFieldInsert: {
						const uint64_t d = vals[op.args[3]];
						const auto offset = static_cast<uint32_t>(c);
						const auto width = static_cast<uint32_t>(d);
						if (offset > 32u || width > 32u - offset) {
							g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
							return 0;
						}
						if (width == 0u) {
							result = static_cast<uint32_t>(a);
						} else {
							const auto mask = width == 32u ? UINT32_MAX : ((1u << width) - 1u) << offset;
							result = (static_cast<uint32_t>(a) & ~mask) |
							         ((static_cast<uint32_t>(b) << offset) & mask);
						}
						break;
					}
					default:
						g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
						return 0;
				}
				break;
			}
		}

		vals[i] = result;
		// Fill evaluation memo entry if present so callers querying individual values get hits
		if (op.memo_index != UINT32_MAX) {
			if (op.memo_index >= m_context.values.size()) {
				m_context.values.resize(m_program.evaluation_value_count);
			}
			m_context.values[op.memo_index].value = result;
			m_context.values[op.memo_index].generation = m_context.generation;

			if (m_clean_evaluator != nullptr && op.clean) {
				auto& clean_context = m_clean_evaluator->m_context;
				if (op.memo_index >= clean_context.values.size()) {
					clean_context.values.resize(m_program.evaluation_value_count);
				}
				clean_context.values[op.memo_index].value = result;
				clean_context.values[op.memo_index].generation = clean_context.generation;
			}
		}
	}

	for (size_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
		const auto& read = m_program.srt_reads[slot];
		if (read.flat_offset >= flat.size()) {
			g_tape_stats.failed.fetch_add(1, std::memory_order_relaxed);
			return 0;
		}
		flat[read.flat_offset] = static_cast<uint32_t>(vals[tape.slot_ops[slot]]);
	}

	m_context.refreshed_flat = &flat;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->m_context.refreshed_flat = &flat;
	}
	g_tape_stats.runs.fetch_add(1, std::memory_order_relaxed);
	g_tape_stats.ops.fetch_add(tape.ops.size(), std::memory_order_relaxed);
	return 1;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
