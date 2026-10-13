#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/experiments.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages();
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

namespace {

// Experiment ImageRecycle limits: NHL 27 frees and recreates about seven full-screen targets that
// share memory every frame (some 125 MiB).
constexpr size_t   kRecycledImagesMax = 32;
constexpr uint64_t kRecycledBytesMax  = 768ull * 1024 * 1024;

bool RecyclableCreateInfo(const vk::ImageCreateInfo& info) {
	return info.pNext == nullptr && info.tiling == vk::ImageTiling::eOptimal &&
	       info.sharingMode == vk::SharingMode::eExclusive &&
	       info.initialLayout == vk::ImageLayout::eUndefined;
}

} // namespace

void GraphicContext::DestroyRecycledImages() {
	std::scoped_lock lock(m_recycle_mutex);
	for (const auto& entry: m_recycled_images) {
		vmaDestroyImage(allocator, entry.image, entry.allocation);
	}
	m_recycled_images.clear();
	m_recycled_bytes = 0;
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	const bool recyclable = RecyclableCreateInfo(image_info);
	if (recyclable && Common::Experiments::Enabled(Common::Experiments::ImageRecycle)) {
		std::scoped_lock lock(m_recycle_mutex);
		for (auto it = m_recycled_images.rbegin(); it != m_recycled_images.rend(); ++it) {
			if (it->flags == image_info.flags && it->image_type == image_info.imageType &&
			    it->format == image_info.format && it->extent == image_info.extent &&
			    it->mip_levels == image_info.mipLevels && it->layers == image_info.arrayLayers &&
			    it->samples == static_cast<uint32_t>(image_info.samples) &&
			    it->usage == image_info.usage) {
				image.image      = it->image;
				image.allocation = it->allocation;
				m_recycled_bytes -= it->bytes;
				m_recycled_images.erase(std::next(it).base());
				break;
			}
		}
	}
	if (static const bool log = std::getenv("KYTY_DBG_IMAGE_RECYCLE") != nullptr; log) {
		// Diagnostic: KYTY_DBG_IMAGE_RECYCLE=1 prints recycle hits per 256 image creations.
		static std::atomic<uint32_t> created {0};
		static std::atomic<uint32_t> hits {0};
		const auto hit = image.image != nullptr ? hits.fetch_add(1, std::memory_order_relaxed) + 1
		                                        : hits.load(std::memory_order_relaxed);
		if ((created.fetch_add(1, std::memory_order_relaxed) + 1) % 256 == 0) {
			std::scoped_lock lock(m_recycle_mutex);
			std::printf("NHL27RECYCLE: created=%u recycled=%u pooled=%zu\n", created.load(), hit,
			            m_recycled_images.size());
		}
	}
	if (image.image == nullptr) {
		VmaAllocationCreateInfo alloc_info {};
		alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

		vk::Image::CType native_image = VK_NULL_HANDLE;
		const auto        result       = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
		image.image = native_image;
		if (result != vk::Result::eSuccess) {
			LogMemoryBudget();
			return false;
		}
	}
	image.recyclable = recyclable;

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	// Images are deleted once the GPU no longer uses them, so a recycled one is idle; its next
	// owner starts from the undefined layout, as with a new image.
	if (image.recyclable && Common::Experiments::Enabled(Common::Experiments::ImageRecycle)) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		std::vector<RecycledImage> evicted;
		{
			std::scoped_lock lock(m_recycle_mutex);
			m_recycled_images.push_back({image.flags, image.image_type, image.format, image.extent,
			                             image.mip_levels, image.layers, image.samples, image.usage,
			                             image.image, image.allocation, allocation_info.size});
			m_recycled_bytes += allocation_info.size;
			size_t drop = 0;
			while (drop < m_recycled_images.size() &&
			       (m_recycled_images.size() - drop > kRecycledImagesMax ||
			        m_recycled_bytes > kRecycledBytesMax)) {
				m_recycled_bytes -= m_recycled_images[drop].bytes;
				++drop;
			}
			const auto end = m_recycled_images.begin() + static_cast<std::ptrdiff_t>(drop);
			evicted.assign(m_recycled_images.begin(), end);
			m_recycled_images.erase(m_recycled_images.begin(), end);
		}
		for (const auto& entry: evicted) {
			vmaDestroyImage(allocator, entry.image, entry.allocation);
		}
	} else {
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
	image.recyclable = false;
}

} // namespace Libs::Graphics
