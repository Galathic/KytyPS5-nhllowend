#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "common/eventTrace.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/bindEpoch.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/runtimeFlags.h"
#include "gpu_tiler_shaders/fault_buffer_gate_spv.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults = 1024;
// Pages V#-table stores may write between two reads; more stay in the bitmap for the next read.
constexpr size_t MaxPageWrites = 16384;

// Fault bitmap: one bit per cache page, read and cleared word by word by the parser.
constexpr uint32_t FaultBitmapWords = static_cast<uint32_t>(BufferCache::CACHING_NUMPAGES / 32);
// Dirty word placed after both bitmaps. RecordBdaFault stores 1 here whenever it sets a bit.
constexpr uint32_t DirtyWordIndex = static_cast<uint32_t>(BufferCache::FAULT_DIRTY_WORD);
// Parser dispatch: one invocation per bitmap word, 64 invocations per workgroup.
constexpr uint32_t ParserWorkgroups = (FaultBitmapWords + 63) / 64;

struct FaultGatePush {
	uint32_t parser_groups;
	uint32_t dirty_index;
	uint32_t force_parse;
};

} // namespace

FaultManager::Reader::Reader(GraphicContext& graphics, CommandScheduler& scheduler,
                             size_t capacity)
    : download(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
               MaxPendingFaults * capacity * sizeof(uint64_t)),
      area_size(capacity * sizeof(uint64_t)) {}

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::FAULT_BUFFER_SIZE + sizeof(uint32_t)),
      m_faults(graphics, scheduler, MaxPageFaults), m_writes(graphics, scheduler, MaxPageWrites),
      m_indirect_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                        3 * sizeof(uint32_t)) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const char* gate_env = std::getenv("KYTY_FAULT_PARSE_GATE");
	m_gate_enabled       = gate_env != nullptr && gate_env[0] == '1' && gate_env[1] == 0;

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	// Parser pipeline.
	const auto module = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");

	// Gate pipeline (same descriptor layout, plus push constants).
	const auto gate_module = CompileSPV(FAULT_BUFFER_GATE_SPV, m_graphics.device);

	vk::PushConstantRange push_range {};
	push_range.stageFlags = vk::ShaderStageFlagBits::eCompute;
	push_range.offset     = 0;
	push_range.size       = sizeof(FaultGatePush);
	vk::PipelineLayoutCreateInfo gate_layout_info {};
	gate_layout_info.setLayoutCount         = 1;
	gate_layout_info.pSetLayouts            = &m_fault_process_desc_layout;
	gate_layout_info.pushConstantRangeCount = 1;
	gate_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&gate_layout_info, nullptr,
	                                           &m_fault_gate_pipeline_layout),
	    "create fault-gate pipeline layout");

	vk::PipelineShaderStageCreateInfo gate_stage {};
	gate_stage.stage  = vk::ShaderStageFlagBits::eCompute;
	gate_stage.module = gate_module;
	gate_stage.pName  = "main";
	vk::ComputePipelineCreateInfo gate_info {};
	gate_info.stage  = gate_stage;
	gate_info.layout = m_fault_gate_pipeline_layout;
	const auto gate_result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &gate_info, nullptr, &m_fault_gate_pipeline);
	m_graphics.device.destroyShaderModule(gate_module, nullptr);
	RequireVulkanSuccess(gate_result, "create fault-gate pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_gate_pipeline, "Fault Buffer Gate");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_gate_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_gate_pipeline_layout, nullptr);
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

void FaultManager::PrepareFaultBuffer() {
	if (!m_cleared) {
		// Shaders only set bits: both bitmaps must start clear.
		m_fault_buffer.Fill(0, m_fault_buffer.Size(), 0);
		m_cleared = true;
	}
}

void FaultManager::ProcessFaultBuffer() {
	Process(m_faults, 0, [this](std::span<const uint64_t> pages) {
		RangeSet fault_ranges;
		for (const auto address: pages) {
			fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			// The guest may have unmapped the range since the GPU accessed it: an unmap does not
			// wait for this completion.
			if (m_scheduler.Context().IsMapped(start, end - start)) {
				(void)m_buffer_cache.FindBuffer(start, end - start);
			}
		});
	});
}

void FaultManager::ProcessWriteBuffer(PagesHandler&& handler) {
	Process(m_writes, BufferCache::WRITE_BITMAP_OFFSET, std::move(handler));
}

// Gated pass (KYTY_FAULT_PARSE_GATE=1, fault bitmap only):
//   pre barrier -> gate (1 invocation: read+clear dirty word, write indirect args)
//   -> barrier (shader write -> indirect read) -> parser via vkCmdDispatchIndirect -> post barrier.
// Invariant: after every gated pass the fault bitmap is all zero and the dirty word is zero.
//   - RecordBdaFault stores the bit and then dirty=1 in the same invocation, so a set bit is
//     always accompanied by dirty==1 that the gate of the next pass will observe (barrier-ordered).
//   - The gate clears dirty and, when it saw dirty==0, no bit was set since the previous parser
//     clear, so the bitmap is already zero and skipping the parser loses nothing.
//   - When dirty!=0 the parser runs over every bitmap word and clears it, except the pages the
//     download area has no room for: those keep their bits, and the readback that sees the
//     overflow forces the next gated pass to parse (m_gate_force).
//   - The first gated pass forces a parse (force_parse=1).
// The dirty word follows both bitmaps (BufferCache::FAULT_DIRTY_WORD). The download slot's count
// is zeroed by the CPU before the pass, so a skipped parser leaves count==0 in place.
void FaultManager::Process(Reader& reader, uint64_t bitmap_offset, PagesHandler&& handler) {
	Common::EventTrace::Record(Common::EventTrace::FAULT_PARSE, reader.current);
	if (const auto wait_tick = reader.ticks[reader.current]; wait_tick != 0) {
		// Ring slot reuse: the previous pass on this slot may still be in flight. Time only the
		// wait itself (a slot that already finished costs ~0).
		{
			FrameBreakdown::Timer ring_timer(FrameBreakdown::g_counters.fault_ring_wait_ns);
			m_scheduler.Wait(wait_tick);
		}
		m_scheduler.PopPendingOperations();
	}
	FrameBreakdown::Add(FrameBreakdown::g_counters.fault_parse_passes);
	const bool gate =
	    bitmap_offset == 0 && (m_gate_enabled || Common::RuntimeFlags::Current().fault_parse_gate);

	const auto area_size = reader.area_size;
	const auto offset    = reader.current * area_size;
	auto*      mapped    = reader.download.Mapped().data() + offset;
	// The parser appends after the count; the host reads only the entries it counted.
	std::memset(mapped, 0, sizeof(uint64_t));
	reader.download.Flush(offset, sizeof(uint64_t));

	constexpr auto bitmap_size = BufferCache::CACHING_NUMPAGES / 8;
	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = bitmap_offset;
	pre_barrier.size          = bitmap_size;
	if (gate) {
		// The gate clears the dirty word, so the pre barrier must also order prior writes.
		pre_barrier.offset        = 0;
		pre_barrier.size          = m_fault_buffer.Size();
		pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	}
	auto post_barrier          = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), bitmap_offset, bitmap_size},
	    {reader.download.Handle(), offset, area_size},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);

	if (!gate) {
		Libs::Graphics::g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		InvalidateRedundantBinds();
		command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
		                             m_fault_process_pipeline_layout, 0, writes);
		command.dispatch(ParserWorkgroups, 1, 1);
	} else {
		const vk::DescriptorBufferInfo gate_infos[] {
		    {m_fault_buffer.Handle(), 0, m_fault_buffer.Size()},
		    {m_indirect_buffer.Handle(), 0, m_indirect_buffer.Size()},
		};
		std::array<vk::WriteDescriptorSet, 2> gate_writes {};
		for (uint32_t index = 0; index < gate_writes.size(); ++index) {
			gate_writes[index].dstBinding      = index;
			gate_writes[index].descriptorCount = 1;
			gate_writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
			gate_writes[index].pBufferInfo     = &gate_infos[index];
		}

		const FaultGatePush push {
		    ParserWorkgroups,
		    DirtyWordIndex,
		    m_gate_primed && !m_gate_force ? 0u : 1u,
		};
		m_gate_primed = true;
		m_gate_force  = false;

		Libs::Graphics::g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		InvalidateRedundantBinds();
		command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_gate_pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_fault_gate_pipeline_layout,
		                             0, gate_writes);
		command.pushConstants(m_fault_gate_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                      sizeof(push), &push);
		command.dispatch(1, 1, 1);

		vk::BufferMemoryBarrier2 gate_barriers[2] {};
		for (auto& barrier : gate_barriers) {
			barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
			barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader |
			                       vk::PipelineStageFlagBits2::eDrawIndirect;
			barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead |
			                       vk::AccessFlagBits2::eShaderWrite |
			                       vk::AccessFlagBits2::eIndirectCommandRead;
			barrier.offset        = 0;
		}
		gate_barriers[0].buffer = m_fault_buffer.Handle();
		gate_barriers[0].size   = m_fault_buffer.Size();
		gate_barriers[1].buffer = m_indirect_buffer.Handle();
		gate_barriers[1].size   = m_indirect_buffer.Size();
		dependency.bufferMemoryBarrierCount = 2;
		dependency.pBufferMemoryBarriers    = gate_barriers;
		command.pipelineBarrier2(dependency);

		Libs::Graphics::g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		InvalidateRedundantBinds();
		command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
		                             m_fault_process_pipeline_layout, 0, writes);
		command.dispatchIndirect(m_indirect_buffer.Handle(), 0);
		dependency.bufferMemoryBarrierCount = 1;
	}

	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);

	const auto area = reader.current;
	// The write scan marks the pages the GPU wrote as GPU-owned, except those unmapped since it was
	// queued (IndirectWriteTables::NoteWrites): unmaps need not wait for it (they waited for the
	// GPU 4-6 ms each, a few times a second in play). The fault scan only caches pages still mapped.
	Common::UniqueFunction<void> scan = [this, bitmap_offset, &reader, mapped, offset, area, area_size,
	                                     handler = std::move(handler)] {
		reader.download.Invalidate(offset, area_size);
		const auto* entries = std::bit_cast<const uint64_t*>(mapped);
		// The parser counts every set bit but stores only the entries the area holds; it leaves
		// the others set for the next read.
		const auto capacity = area_size / sizeof(uint64_t) - 1;
		const auto count    = std::min<uint64_t>(static_cast<uint32_t>(entries[0]), capacity);
		if (bitmap_offset == 0) {
			if (static_cast<uint32_t>(entries[0]) > capacity) {
				m_gate_force = true;
			}
			FrameBreakdown::Add(FrameBreakdown::g_counters.fault_parse_faults, count);
			if (count == 0) {
				FrameBreakdown::Add(FrameBreakdown::g_counters.fault_parse_zero);
			}
		}
		std::vector<uint64_t> pages;
		pages.reserve(count);
		for (uint64_t index = 1; index <= count; ++index) {
			pages.push_back(BufferCache::GuestAddress(entries[index]));
		}
		if (!pages.empty()) {
			handler(pages);
		}
		reader.ticks[area] = 0;
	};
	if (&reader == &m_writes) {
		m_scheduler.DeferOperation(std::move(scan), 0, 0);
	} else {
		m_scheduler.DeferHostOperation(std::move(scan));
	}

	reader.ticks[reader.current++] = m_scheduler.CurrentTick();
	reader.current %= MaxPendingFaults;
}

} // namespace Libs::Graphics
