#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_SPIRVHALFLANES_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_SPIRVHALFLANES_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// KYTY_HALF_VECTOR=1 (experiment, default off): peephole pass over finished SPIR-V that expresses
// 16-bit lane reads and writes of 32-bit registers as f16vec2 extract/insert/shuffle instead of
// integer shifts, masks and f32<->f16 pack/unpack round trips, so the driver can use op_sel /
// 16-bit register halves. Returns the number of rewrites (0 = module unchanged).
struct HalfLaneStats {
	uint32_t reads    = 0; // UnpackHalf2x16(bfe(x, 0|16, 16)).x -> f16vec2(x)[i]
	uint32_t shuffles = 0; // f16vec2(bfe(x, 0|16, 16)) feeding low-lane-only math -> swizzle
	uint32_t writes   = 0; // (old & mask) | packed half -> insert into f16vec2(old)
};

[[nodiscard]] bool HalfVectorEnabled();
uint32_t RewriteHalfLanes(std::vector<uint32_t>& spirv, HalfLaneStats* stats = nullptr);

// KYTY_LANE_DEMAND=1 (experiment, default off): every guest VGPR write is emitted as
// select(exec, new, old). This pass works out, per value, which lanes can still be observed
// (stores, branches, cross-lane operations) and turns a select into a plain copy of `new` when
// the lanes that would keep `old` are never observed. Returns the number of selects removed.
struct LaneDemandStats {
	uint32_t selects = 0;
	uint32_t removed = 0;
};

[[nodiscard]] bool LaneDemandEnabled();

// KYTY_FAST_DENORM=1 (experiment, default off): float compares and rcp/rsq/sqrt/exp2/log2 no longer
// flush f32 denormal inputs to zero by hand. f32 add/mul/fma results are not flushed either, so
// this only differs for operands below ~1.2e-38.
[[nodiscard]] bool FastDenormEnabled();
uint32_t RewriteLaneDemand(std::vector<uint32_t>& spirv, LaneDemandStats* stats = nullptr);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif
