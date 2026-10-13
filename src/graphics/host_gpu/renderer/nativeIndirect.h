#ifndef GRAPHICS_HOST_GPU_RENDERER_NATIVEINDIRECT_H
#define GRAPHICS_HOST_GPU_RENDERER_NATIVEINDIRECT_H
#include "graphics/host_gpu/vulkanCommon.h"
#include <cstdint>
#include <memory>
#include <span>
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
 std::pair<vk::Buffer, uint64_t> Record(CommandBuffer& buffer, const Buffer& args, uint64_t offset,
                                        uint64_t guest_address);
private:
 static constexpr uint64_t SlotBytes = 16;
 static constexpr uint64_t SlotCount = 65536;
 GraphicContext& m_graphics;
 std::unique_ptr<Buffer> m_ring;
 // Host-visible: count of zeroed records and the guest addresses of the last 16.
 std::unique_ptr<Buffer> m_trips;
 uint32_t m_reported_trips = 0;
 uint64_t m_next = 0;
 vk::PipelineLayout m_layout = nullptr;
 vk::Pipeline m_pipeline = nullptr;
};
// Color metadata (DCC/CMASK) the GPU wrote: the fast-clear check runs on the GPU and sets
// conditional-rendering predicates, instead of reading the metadata back and waiting for the GPU.
class ColorClearCheck {
public:
 ColorClearCheck(GraphicContext& graphics, CommandScheduler& scheduler);
 ~ColorClearCheck();
 // Checks `slices` slices of `slice_words` dwords at `metadata` against up to 8 candidate codes
 // (rewriting a matched slice to 0xff keys with `expand`). Returns the predicate buffer and the
 // offset of slice 0's predicates: 8 dwords per slice, one per candidate.
 std::pair<vk::Buffer, uint64_t> Record(CommandBuffer& buffer, vk::DeviceAddress metadata,
                                        uint32_t slice_words, uint32_t slices,
                                        std::span<const uint8_t> codes, bool expand);
 static constexpr uint64_t SliceBytes = 32;
private:
 static constexpr uint64_t SlotCount = 4096;
 GraphicContext& m_graphics;
 std::unique_ptr<Buffer> m_ring;
 uint64_t m_next = 0;
 vk::PipelineLayout m_layout = nullptr;
 vk::Pipeline m_pipeline = nullptr;
};
}
#endif
