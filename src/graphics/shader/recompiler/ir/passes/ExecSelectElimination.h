#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Every VGPR write under a possibly partial EXEC becomes Select(exec, new, old). When the result
// is only ever observed in lanes where that same exec value is set (consumed by lane-wise math,
// or by memory operations predicated on it), the old value is never read: the select is replaced
// by the new value, and the old value can die earlier (fewer live registers, no spills).
// Returns the number of selects removed.
uint32_t EliminateExecSelects(const BlockList& blocks);
// Operations whose result in a lane depends only on their operands in that lane (no memory, no
// other lanes, no side effects).
[[nodiscard]] bool IsLaneWiseOpcode(ValueOpcode op);
// A memory operation whose last operand is its exec predicate (it touches memory only where set).
[[nodiscard]] bool IsExecPredicatedMemoryOp(const Inst& inst);
// Sampling with implicit derivatives (reads the coordinates of the other lanes of its quad).
[[nodiscard]] bool ReadsNeighborLanesOpcode(ValueOpcode op);

} // namespace Libs::Graphics::ShaderRecompiler::IR
