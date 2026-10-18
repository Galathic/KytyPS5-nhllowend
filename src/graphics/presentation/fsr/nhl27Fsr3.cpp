#include "graphics/presentation/fsr/nhl27Fsr3.h"
#include "graphics/presentation/fsr/nhl27Fsr3Constants.h"
#include "graphics/presentation/fsr/fsr3.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {
namespace {
constexpr std::array<uint64_t, 4> kPass {
	0x4618098710b83f8fULL, 0xdb7e537e65c00dd1ULL,
	0xbbb674ad1182ed8cULL, 0xa36f0b4faa20493aULL};
using S = vk::PipelineStageFlagBits2;
using A = vk::AccessFlagBits2;

void MemoryBarrier(CommandRecorder command) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask = barrier.dstStageMask = S::eAllCommands;
	barrier.srcAccessMask = barrier.dstAccessMask = A::eMemoryRead | A::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers = &barrier;
	command.pipelineBarrier2(dependency);
}
void ImageBarrier(CommandRecorder command, vk::Image image, vk::ImageLayout from,
				  vk::ImageLayout to, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask = barrier.dstStageMask = S::eAllCommands;
	barrier.srcAccessMask = from == vk::ImageLayout::eUndefined
		? vk::AccessFlags2 {} : A::eMemoryRead | A::eMemoryWrite;
	barrier.dstAccessMask = A::eMemoryRead | A::eMemoryWrite;
	barrier.oldLayout = from;
	barrier.newLayout = to;
	barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	command.pipelineBarrier2(dependency);
}
}
struct Nhl27Fsr3::Impl {
	struct Owned {
		Fsr3Image resource;
		VmaAllocation allocation = nullptr;
		vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	};
	struct Role {
		uint64_t address = 0;
		vk::Format format {};
		vk::Extent3D extent {};
		bool Matches(const TextureBinding& other) const {
			return address == other.desc.info.data.address &&
				format == other.desc.info.pixel_format && extent == other.desc.info.extent;
		}
		static Role Of(const TextureBinding& binding) {
			return {binding.desc.info.data.address, binding.desc.info.pixel_format,
					binding.desc.info.extent};
		}
	};
	GraphicContext& graphics;
	std::array<Owned, 7> images {}; // color, depth, raw motion, output, SDK depth/motion/previous
	std::unique_ptr<Fsr3> fsr;
	vk::Buffer constants = nullptr, transfer = nullptr;
	VmaAllocation constants_alloc = nullptr, transfer_alloc = nullptr;
	void* mapped = nullptr;
	uint64_t last_tick = 0, serial = 0, rejected = 0, recorded = 0;
	unsigned stage = 0;
	Nhl27Fsr3Constants captured;
	std::array<Role, 8> prepare {}, reactive {};
	Role luma;
	explicit Impl(GraphicContext& gfx) : graphics(gfx) {}
	~Impl() {
		// RenderContext shuts down/drains the scheduler before RenderExecutor's owner dies.
		ReleaseImages();
		if (constants) vmaDestroyBuffer(graphics.allocator, constants, constants_alloc);
	}
	void Reject(const char* why) {
		stage = 0;
		if (++rejected <= 8 || rejected % 600 == 0) {
			std::printf("NHL27FSR3GAME status=rejected reason=%s count=%llu\n", why,
						static_cast<unsigned long long>(rejected));
			std::fflush(stdout);
		}
	}
	void ReleaseImages() {
		fsr.reset();
		for (auto& image : images) {
			if (image.resource.image)
				vmaDestroyImage(graphics.allocator, image.resource.image, image.allocation);
			image = {};
		}
		if (transfer) vmaDestroyBuffer(graphics.allocator, transfer, transfer_alloc);
		transfer = nullptr;
		transfer_alloc = nullptr;
	}
	bool AllocateBuffer(uint64_t bytes, bool host, vk::Buffer& buffer,
						VmaAllocation& allocation, void** pointer = nullptr) {
		vk::BufferCreateInfo info {};
		info.size = bytes;
		info.usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
		VmaAllocationCreateInfo alloc {};
		alloc.usage = VMA_MEMORY_USAGE_AUTO;
		if (host) alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer native = VK_NULL_HANDLE;
		VmaAllocationInfo result {};
		if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&info),
							&alloc, &native, &allocation, &result) != VK_SUCCESS) return false;
		buffer = native;
		if (pointer) *pointer = result.pMappedData;
		return true;
	}
	std::optional<Nhl27Fsr3Constants> ReadConstants(RenderContext& context, CommandBuffer& command,
													 const PreparedBindings& bindings) {
		if (bindings.buffer_sources.empty() || bindings.buffers.empty() ||
			bindings.buffer_sources[0].size != 208 || bindings.buffers[0].range < 208 ||
			!bindings.buffers[0].buffer) return std::nullopt;
		if (!constants && !AllocateBuffer(208, true, constants, constants_alloc, &mapped))
			return std::nullopt;
		auto recorder = command.Handle();
		MemoryBarrier(recorder);
		vk::BufferCopy copy {bindings.buffers[0].offset, 0, 208};
		recorder.copyBuffer(bindings.buffers[0].buffer, constants, 1, &copy);
		vk::MemoryBarrier2 host {};
		host.srcStageMask = S::eTransfer; host.srcAccessMask = A::eTransferWrite;
		host.dstStageMask = S::eHost; host.dstAccessMask = A::eHostRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1; dependency.pMemoryBarriers = &host;
		recorder.pipelineBarrier2(dependency);
		// Intentionally synchronous diagnostic: decode the actual shader-visible GPU bytes,
		// never CPU/dirty guest backing. The wrapper stays stable across the new submission.
		context.GetCommandScheduler().Wait(context.GetCommandScheduler().CurrentTick());
		InvalidateRedundantBinds();
		if (vmaInvalidateAllocation(graphics.allocator, constants_alloc, 0, 208) != VK_SUCCESS)
			return std::nullopt;
		std::array<uint32_t, 52> words;
		std::memcpy(words.data(), mapped, 208);
		return Nhl27Fsr3Constants::Decode(words);
	}
	bool EnsureImages(const Nhl27Fsr3Constants& c) {
		const vk::Extent2D input {c.Width(), c.Height()}, output {c.OutputWidth(), c.OutputHeight()};
		if (images[0].resource.image && images[0].resource.extent == input &&
			images[3].resource.extent == output) return true;
		// ReadConstants just waited for all prior GPU work, including SDK callbacks.
		ReleaseImages();
		constexpr std::array<vk::Format, 7> formats {
			vk::Format::eB10G11R11UfloatPack32, vk::Format::eR32Sfloat,
			vk::Format::eR16G16Sfloat, vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR32Sfloat, vk::Format::eR16G16Sfloat, vk::Format::eR32Uint};
		for (size_t i = 0; i < images.size(); ++i) {
			auto& owned = images[i];
			owned.resource.format = formats[i];
			owned.resource.extent = i == 3 ? output : input;
			vk::ImageCreateInfo info {};
			info.imageType = vk::ImageType::e2D;
			info.format = formats[i];
			info.extent = {owned.resource.extent.width, owned.resource.extent.height, 1};
			info.mipLevels = info.arrayLayers = 1;
			info.samples = vk::SampleCountFlagBits::e1;
			info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
			VmaAllocationCreateInfo alloc {};
			alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			VkImage native = VK_NULL_HANDLE;
			if (vmaCreateImage(graphics.allocator, reinterpret_cast<const VkImageCreateInfo*>(&info),
							   &alloc, &native, &owned.allocation, nullptr) != VK_SUCCESS) {
				ReleaseImages();
				return false;
			}
			owned.resource.image = native;
		}
		if (!AllocateBuffer(uint64_t(input.width) * input.height * 4, false, transfer, transfer_alloc)) {
			ReleaseImages();
			return false;
		}
		fsr = std::make_unique<Fsr3>(graphics.physical_device, graphics.device);
		return true;
	}
	bool Snapshot(RenderContext& context, CommandBuffer& command, const TextureBinding& binding,
				  Owned& owned) {
		if (!binding.image_id || binding.desc.view_info.base_level != 0 ||
			binding.desc.view_info.base_layer != 0) return false;
		auto& source = context.GetTextureCache().GetImage(binding.image_id);
		if (!source.backing.image || source.backing.samples != 1 ||
			source.backing.extent != vk::Extent3D {owned.resource.extent.width,
				owned.resource.extent.height, 1}) return false;
		auto full = vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor), aspect = full;
		const auto format = source.backing.format;
		if (format == vk::Format::eD32SfloatS8Uint) {
			full = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
			aspect = vk::ImageAspectFlagBits::eDepth;
		} else if (format == vk::Format::eD32Sfloat) {
			full = aspect = vk::ImageAspectFlagBits::eDepth;
		} else if (format != owned.resource.format &&
			!(owned.resource.format == vk::Format::eR32Sfloat && format == vk::Format::eR32Uint)) {
			return false;
		}
		if (aspect != vk::ImageAspectFlagBits::eColor && owned.resource.format != vk::Format::eR32Sfloat)
			return false;
		auto recorder = command.Handle();
		MemoryBarrier(recorder);
		ImageBarrier(recorder, source.backing.image, binding.layout,
					 vk::ImageLayout::eTransferSrcOptimal, full);
		ImageBarrier(recorder, owned.resource.image, owned.layout, vk::ImageLayout::eTransferDstOptimal);
		vk::BufferImageCopy copy {};
		copy.imageExtent = source.backing.extent;
		copy.imageSubresource = {aspect, 0, 0, 1};
		recorder.copyImageToBuffer(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
								   transfer, 1, &copy);
		MemoryBarrier(recorder);
		copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		recorder.copyBufferToImage(transfer, owned.resource.image, vk::ImageLayout::eTransferDstOptimal,
								   1, &copy);
		ImageBarrier(recorder, source.backing.image, vk::ImageLayout::eTransferSrcOptimal, binding.layout, full);
		ImageBarrier(recorder, owned.resource.image, vk::ImageLayout::eTransferDstOptimal,
					 vk::ImageLayout::eShaderReadOnlyOptimal);
		owned.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
		MemoryBarrier(recorder);
		return true;
	}
};
Nhl27Fsr3::Nhl27Fsr3(GraphicContext& graphics) : m(std::make_unique<Impl>(graphics)) {}
Nhl27Fsr3::~Nhl27Fsr3() = default;
bool Nhl27Fsr3::Enabled() {
#if KYTY_ENABLE_FSR3
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FSR3_NHL27_SHADOW");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
#else
	return false;
#endif
}
void Nhl27Fsr3::OnDispatch(RenderContext& context, CommandBuffer& command, uint64_t hash,
						  const PreparedBindings& bindings) {
	if (!Enabled()) return;
	const auto& b = bindings.images;
	const auto count = b.size();
	if (hash == kPass[0]) {
		m->stage = 0;
		if (m->last_tick) context.GetCommandScheduler().Wait(m->last_tick);
		const auto constants = m->ReadConstants(context, command, bindings);
		if (!constants || count != 8) { m->Reject("prepare-constants-or-slots"); return; }
		const auto& c = *constants;
		const vk::Extent3D extent {c.Width(), c.Height(), 1};
		if (b[0].desc.info.pixel_format != vk::Format::eR32Sfloat ||
			b[1].desc.info.pixel_format != vk::Format::eR16G16Sfloat ||
			b[3].desc.info.pixel_format != vk::Format::eB10G11R11UfloatPack32 ||
			b[0].desc.info.extent != extent || b[1].desc.info.extent != extent ||
			b[3].desc.info.extent != extent || !m->EnsureImages(c)) {
			m->Reject("prepare-layout-or-allocation"); return;
		}
		if (!m->Snapshot(context, command, b[3], m->images[0]) ||
			!m->Snapshot(context, command, b[0], m->images[1]) ||
			!m->Snapshot(context, command, b[1], m->images[2])) {
			m->Reject("snapshot-layout"); return;
		}
		m->captured = c;
		for (size_t i = 0; i < 8; ++i) m->prepare[i] = Impl::Role::Of(b[i]);
		m->stage = 1;
	} else if (hash == kPass[1]) {
		if (m->stage != 1 || count != 2 || !m->prepare[7].Matches(b[0])) {
			m->Reject("luma-edge"); return;
		}
		m->luma = Impl::Role::Of(b[1]);
		m->stage = 2;
	} else if (hash == kPass[2]) {
		if (m->stage != 2 || count != 8 || !m->prepare[5].Matches(b[0]) ||
			!m->prepare[4].Matches(b[1]) || !m->prepare[2].Matches(b[2]) || !m->luma.Matches(b[3])) {
			m->Reject("reactivity-edges"); return;
		}
		for (size_t i = 0; i < 8; ++i) m->reactive[i] = Impl::Role::Of(b[i]);
		m->stage = 3;
	} else if (hash == kPass[3]) {
		if (m->stage != 3 || count != 8 || !m->reactive[6].Matches(b[0]) ||
			!m->prepare[4].Matches(b[1]) || !m->prepare[3].Matches(b[4]) ||
			!m->reactive[7].Matches(b[5]) ||
			b[7].desc.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
			b[7].desc.info.extent != vk::Extent3D {m->captured.OutputWidth(),
				m->captured.OutputHeight(), 1}) {
			m->Reject("accumulation-edges"); return;
		}
		m->stage = 0;
		const auto end = m->ReadConstants(context, command, bindings);
		if (!end || !m->captured.SameFrame(*end)) { m->Reject("constant-frame-mismatch"); return; }
		const auto& c = m->captured;
		Fsr3Frame frame {};
		frame.color = m->images[0].resource;
		frame.depth = m->images[1].resource;
		frame.motion = m->images[2].resource;
		frame.output = m->images[3].resource;
		frame.dilated_depth = m->images[4].resource;
		frame.dilated_motion = m->images[5].resource;
		frame.previous_depth = m->images[6].resource;
		frame.render_size = {c.Width(), c.Height()};
		frame.output_size = {c.OutputWidth(), c.OutputHeight()};
		frame.frame_id = ++m->serial;
		frame.jitter_x = c.Float(28); frame.jitter_y = c.Float(29);
		frame.motion_scale_x = c.Float(32) * c.Width();
		frame.motion_scale_y = c.Float(33) * c.Height();
		frame.near_plane = c.Near(); frame.far_plane = c.Far();
		frame.vertical_fov = c.FovY();
		frame.delta_ms = c.Float(36) * 1000;
		frame.pre_exposure = c.Float(37);
		frame.sharpness = 0;
		frame.hdr = frame.inverted_depth = true;
		// Actual same-frame GPU resources/coefficients are verified above. Reset-only
		// shadow dispatch cannot validate temporal direction, units, or jitter inclusion.
		frame.verified = frame.reset = true;
		auto recorder = command.Handle();
		for (size_t i = 3; i < m->images.size(); ++i) {
			auto& owned = m->images[i];
			ImageBarrier(recorder, owned.resource.image, owned.layout, vk::ImageLayout::eGeneral);
			owned.layout = vk::ImageLayout::eGeneral;
		}
		const auto guest_frame = c.Frame();
		auto* state = m.get();
		recorder.RecordExternal([state, frame, guest_frame](vk::CommandBuffer native) {
			const bool ok = state->fsr->Record(native, frame);
			if (++state->recorded <= 3 || state->recorded % 600 == 0 || !ok) {
				std::printf("NHL27FSR3GAME status=%s mode=reset-shadow guest_frame=%u "
					"render=%ux%u display=%ux%u detail=%s\n", ok ? "recorded" : "failed",
					guest_frame, frame.render_size.width, frame.render_size.height,
					frame.output_size.width, frame.output_size.height, state->fsr->Status().c_str());
				std::fflush(stdout);
			}
		});
		InvalidateRedundantBinds();
		g_compute_pipeline_epoch.fetch_add(1, std::memory_order_relaxed);
		NoteShaderWork();
		m->last_tick = context.GetCommandScheduler().CurrentTick();
	}
}
} // namespace Libs::Graphics
