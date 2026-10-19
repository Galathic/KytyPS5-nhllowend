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

} // namespace Libs::Graphics::ShaderRecompiler::IR
