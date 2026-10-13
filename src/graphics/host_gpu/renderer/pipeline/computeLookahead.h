#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPUTELOOKAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPUTELOOKAHEAD_H_

#include "graphics/guest_gpu/hardwareContext.h"

#include <array>
#include <cstdint>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {
struct ResourcePlan;
struct SrtRuntime;
struct ResourceSnapshot;
struct ResourceSpecialization;
} // namespace Libs::Graphics::ShaderRecompiler::IR

// Experiment ComputeLookahead (KYTY_COMPUTE_LOOKAHEAD): a worker thread materializes the
// resources of the compute dispatches that follow the one the GPU thread is executing.
//
// Before a DISPATCH_DIRECT runs, the GPU thread publishes its compute registers and the stream
// after the packet. The worker copies the next dwords through the backing, replays the compute
// SH-register writes and, for each later DISPATCH_DIRECT whose program it has seen, materializes
// the resources with readers that record every value they read. When the GPU thread reaches that
// dispatch, it adopts the result only if the program entry, shader base, workgroup counts and
// every user-data register are equal and every recorded read returns the same bytes through the
// GPU thread's own readers (which download GPU-written memory as usual). Anything else falls back
// to the normal materialization, so a wrong prediction costs time, never correctness.
//
// pengui07's first version (retired in d1b7b112) did the scan and a backing compare per range on
// the GPU thread, which cost about as much as the materialization; here the worker scans and the
// GPU thread re-reads ranges with plain loads.
namespace Libs::Graphics::ComputeLookahead {

struct Dispatch {
	uint32_t                offset_dw = 0; // from the start of the scanned window
	HW::ComputeShaderInfo   cs;
	std::array<uint32_t, 3> groups {};
};

// pm4Handlers.cpp: replays compute SH-register writes over a copy of the stream, starting from the
// registers in `cs`, and returns the DISPATCH_DIRECT packets found (at most out.size()). Stops at
// any packet that could change compute state in a way the model does not reproduce.
uint32_t ScanCompute(HW::ComputeShaderInfo cs, std::span<const uint32_t> stream, std::span<Dispatch> out);

[[nodiscard]] bool Active() noexcept;

// GPU thread, before a DISPATCH_DIRECT runs: the stream after the packet.
void SetStream(const uint32_t* next, uint32_t remaining_dw) noexcept;

// GPU thread: the registers of the dispatch being prepared.
void SetCurrent(const HW::ComputeShaderInfo* cs) noexcept;

// GPU thread, once the current dispatch has its resources: the worker starts on the following
// dispatches (registers from SetCurrent, stream from SetStream) while this one is recorded.
void PublishCurrent();

// GPU thread, program cache: the entry the current dispatch's program resolved to.
void NoteProgram(uint64_t shader_base, const void* entry, const ShaderRecompiler::IR::ResourcePlan* plan,
                 bool batch_reads);

// GPU thread: adopts a precomputed result for (entry, runtime) into snapshot/specialization.
[[nodiscard]] bool TryAdopt(const void* entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
                            ShaderRecompiler::IR::ResourceSnapshot&      snapshot,
                            ShaderRecompiler::IR::ResourceSpecialization& specialization);

// A ResourcePlan's scratch is mutable: the GPU thread and the worker never walk one plan at once.
void LockPlan(const ShaderRecompiler::IR::ResourcePlan* plan) noexcept;
void UnlockPlan(const ShaderRecompiler::IR::ResourcePlan* plan) noexcept;

} // namespace Libs::Graphics::ComputeLookahead

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPUTELOOKAHEAD_H_
