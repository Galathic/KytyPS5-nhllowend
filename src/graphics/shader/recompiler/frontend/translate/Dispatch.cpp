#include "common/assert.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

namespace Libs::Graphics::ShaderRecompiler::Frontend {

void Translator::FailMissingTranslation(const Decoder::Instruction& inst) {
	EXIT("opcode %s at pc 0x%08x has no IR translation",
	     Decoder::InstructionToString(inst).c_str(), inst.pc);
}

void Translator::RecordFp16Coverage(const Decoder::Instruction& inst) {
	// Reason bits: 0 op, 1 op_sel, 2 neg, 3 neg_hi, 4 abs, 5 clamp, 6 omod, 7 const/literal,
	// 8 sdwa, 9 dpp, 10 fmac_dst, 11 sel_hi, 12 other.
	const auto name = magic_enum::enum_name(inst.opcode);
	if (name.size() < 3u || name.substr(0, 2) != "V_" || name.find("CMP") != name.npos) {
		return;
	}
	if (name.find("F16") == name.npos && name.find("PKRTZ") == name.npos &&
	    name.find("MIX") == name.npos && name.substr(0, 5) != "V_PK_") {
		return;
	}
	auto& entry = fp16_coverage->ops[inst.opcode];
	if (fp16_native) {
		entry.native++;
		return;
	}
	entry.fallback++;
	const bool covered = inst.opcode == Decoder::Opcode::V_PK_ADD_F16 ||
	                     inst.opcode == Decoder::Opcode::V_PK_MUL_F16 ||
	                     inst.opcode == Decoder::Opcode::V_PK_FMA_F16 ||
	                     inst.opcode == Decoder::Opcode::V_PK_FMAC_F16;
	auto& reasons = fp16_coverage->reasons;
	if (!covered) {
		reasons[0]++;
		return;
	}
	uint32_t mask = 0;
	const auto scan = [&](const Decoder::Operand& op) {
		if (op.kind != Decoder::OperandKind::Vgpr && op.kind != Decoder::OperandKind::Sgpr) mask |= 1u << 7u;
		if (op.op_sel) mask |= 1u << 1u;
		if (!op.op_sel_hi) mask |= 1u << 11u;
		if (op.negate) mask |= 1u << 2u;
		if (op.negate_hi) mask |= 1u << 3u;
		if (op.absolute) mask |= 1u << 4u;
		if (op.sdwa_sel != 6u) mask |= 1u << 8u;
		if (op.dpp || op.dpp8) mask |= 1u << 9u;
	};
	const bool fmac = inst.opcode == Decoder::Opcode::V_PK_FMAC_F16;
	scan(inst.src0);
	scan(inst.src1);
	if (inst.opcode == Decoder::Opcode::V_PK_FMA_F16) scan(inst.src2);
	if (fmac) {
		const auto before = mask;
		scan(inst.dst);
		if (mask != before) mask |= 1u << 10u;
	}
	if (inst.dst.clamp) mask |= 1u << 5u;
	if (inst.dst.omod != 0u) mask |= 1u << 6u;
	if (mask == 0u) mask = 1u << 12u;
	for (uint32_t i = 0; i < reasons.size(); i++) {
		if ((mask >> i) & 1u) reasons[i]++;
	}
}

void Translator::TranslateInstruction(const Decoder::Instruction& inst) {
	if (fp16_coverage == nullptr) {
		TranslateInstructionImpl(inst);
		return;
	}
	fp16_native = false;
	TranslateInstructionImpl(inst);
	RecordFp16Coverage(inst);
}

void Translator::TranslateInstructionImpl(const Decoder::Instruction& inst) {
	current_opcode = inst.opcode;
	current_pc     = inst.pc;

	switch (inst.opcode) {
		case Decoder::Opcode::UNKNOWN:
		case Decoder::Opcode::COUNT:
			EXIT("decoded opcode has no IR translation at pc 0x%08x", inst.pc);
		case Decoder::Opcode::UNSUPPORTED:
			EXIT("unsupported decoded instruction: %s", Decoder::InstructionToString(inst).c_str());
		default: break;
	}

	switch (inst.family) {
		case Decoder::Family::SOP1:
		case Decoder::Family::SOP2:
		case Decoder::Family::SOPK:
		case Decoder::Family::SOPC:
		case Decoder::Family::SOPP: return EmitScalar(inst);
		case Decoder::Family::VOP1:
		case Decoder::Family::VOP2:
		case Decoder::Family::VOP3:
		case Decoder::Family::VOP3P:
		case Decoder::Family::VOPC: return EmitVector(inst);
		case Decoder::Family::SMEM:
		case Decoder::Family::MUBUF:
		case Decoder::Family::MTBUF:
		case Decoder::Family::FLAT:
		case Decoder::Family::DS:
		case Decoder::Family::MIMG: return EmitMemory(inst);
		case Decoder::Family::VINTRP: return EmitInterpolation(inst);
		case Decoder::Family::EXP: return EXP(inst);
		default: return FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
