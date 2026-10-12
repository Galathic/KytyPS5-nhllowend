#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

struct ShaderStageRuntime;

// Bumped before every compute vkCmdBindPipeline outside the guest dispatch path, so the dispatch
// path can prove that its previously bound compute pipeline is still the bound one.
inline std::atomic<uint64_t> g_compute_pipeline_epoch {0};
// Same for graphics: bumped before every graphics vkCmdBindPipeline / dynamic-state command that is not
// issued by the guest draw path (KYTY_PIPELINE_MEMO invalidation).
inline std::atomic<uint64_t> g_graphics_pipeline_epoch {0};
// Bumped for every recorded guest draw and dispatch (KYTY_GUEST_BARRIER_DEDUP): two guest global
// barriers with no shader work between them (also no host compute/graphics helper pass, which bump the
// pipeline epochs above) in one command-buffer recording order nothing new.
inline std::atomic<uint64_t> g_work_epoch {0};

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderAccessDependency();
vk::MemoryBarrier       MakeShaderWriteHazardDependency();
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);
void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages);
void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
