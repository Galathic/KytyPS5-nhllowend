#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/experiments.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"

#include <chrono>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	// KYTY_PERF_STATS: time the GPU thread spends blocked on host GPU progress.
	const bool on_gpu_thread = GuestGpu::IsGpuThread();
	FrameBreakdown::Timer wait_timer(FrameBreakdown::g_counters.gpu_wait_ns, on_gpu_thread);
	FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_WAIT);
	if (on_gpu_thread) {
		FrameBreakdown::Add(FrameBreakdown::g_counters.gpu_wait_count);
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	const bool timed  = BufferCache::PerfTimersEnabled() && GuestGpu::IsGpuThread();
	const auto start  = timed ? std::chrono::steady_clock::now()
	                          : std::chrono::steady_clock::time_point {};
	Common::Timeline::Mark("host-wait", tick);
	Common::Timeline::MarkStack();
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	Common::Timeline::Mark("host-done", tick);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (timed) {
		BufferCache::PerfAddTime(BufferCache::PerfTimer::GpuWait,
		                         static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                             std::chrono::steady_clock::now() - start).count()));
	}
	Refresh();
}

} // namespace Libs::Graphics
