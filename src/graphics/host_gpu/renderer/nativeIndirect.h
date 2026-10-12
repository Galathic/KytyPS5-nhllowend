#ifndef GRAPHICS_HOST_GPU_RENDERER_NATIVEINDIRECT_H
#define GRAPHICS_HOST_GPU_RENDERER_NATIVEINDIRECT_H
#include "graphics/host_gpu/vulkanCommon.h"
#include <memory>
#include <utility>
namespace Libs::Graphics {
struct GraphicContext;
class CommandBuffer;
class CommandScheduler;
class Buffer;
struct NativeIndirectDraw;
// Mirrors the CP's indexed-count clamp without downloading its argument record.
class NativeIndirectPrep {
public:
 explicit NativeIndirectPrep(GraphicContext& graphics);
 ~NativeIndirectPrep();
 void Record(CommandBuffer& buffer, const NativeIndirectDraw& source);
private:
 GraphicContext& m_graphics;
 vk::PipelineLayout m_layout = nullptr;
 vk::Pipeline m_pipeline = nullptr;
};
// Indirect dispatch group counts above the device limits (NHL 27 occasionally dispatches
// 0xffffffff x3 groups during match loads) never complete and cost a device loss. Each indirect
// dispatch reads its counts through a GPU-side copy that zeroes such records.
class DispatchArgsGuard {
public:
 DispatchArgsGuard(GraphicContext& graphics, CommandScheduler& scheduler);
 ~DispatchArgsGuard();
 // Records the guard; returns the buffer and offset to pass to vkCmdDispatchIndirect.
 std::pair<vk::Buffer, uint64_t> Record(CommandBuffer& buffer, const Buffer& args, uint64_t offset);
private:
 static constexpr uint64_t SlotBytes = 16;
 static constexpr uint64_t SlotCount = 65536;
 GraphicContext& m_graphics;
 std::unique_ptr<Buffer> m_ring;
 uint64_t m_next = 0;
 vk::PipelineLayout m_layout = nullptr;
 vk::Pipeline m_pipeline = nullptr;
};
}
#endif