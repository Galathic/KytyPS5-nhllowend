#include "common/eventTrace.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/bindEpoch.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/experiments.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

struct Pm4FaultScope {
	uint64_t packet_addr = 0;
	uint32_t packet_dw   = 0;
	uint32_t header      = 0;
	bool     executing   = false;
};
static thread_local Pm4FaultScope g_pm4_fault_scope;

void NotePm4MemoryFault(uint64_t fault_vaddr, uint32_t access) noexcept {
	static const bool enabled = [] {
		const char* setting = std::getenv("KYTY_PERF_STATS");
		return setting != nullptr && setting[0] == '1';
	}();
	if (!enabled || !GuestGpu::IsGpuThread() || g_pm4_fault_scope.packet_addr == 0) return;
	// Limit log traffic without exhausting the diagnostic during startup before training.
	static thread_local auto interval = std::chrono::steady_clock::now();
	static thread_local uint32_t lines = 0;
	const auto now = std::chrono::steady_clock::now();
	if (now - interval >= std::chrono::seconds(5)) {
		interval = now;
		lines = 0;
	}
	if (lines++ >= 8) return;
	const auto& scope = g_pm4_fault_scope;
	const uint64_t bytes = static_cast<uint64_t>(scope.packet_dw) * sizeof(uint32_t);
	const bool in_packet = fault_vaddr >= scope.packet_addr &&
	                       fault_vaddr - scope.packet_addr < bytes;
	std::printf("NHL27PM4FAULT: fault=0x%llx access=%u packet=0x%llx packet_dw=%u "
	            "header=0x%08x op=0x%x custom=0x%x phase=%s in_packet=%u\n",
	            static_cast<unsigned long long>(fault_vaddr), access,
	            static_cast<unsigned long long>(scope.packet_addr), scope.packet_dw,
	            scope.header, scope.executing ? ((scope.header >> 8u) & 0xffu) : UINT32_MAX,
	            scope.executing ? KYTY_PM4_R(scope.header) : UINT32_MAX,
	            scope.executing ? "handler" : "fetch", static_cast<unsigned>(in_packet));
}

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		FrameBreakdown::GpuBusyScope busy_scope(FrameBreakdown::GB_GPU_CMDS);
		m_renderer.BdaBatchBoundary(RenderContext::BdaBoundary::Other);
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	Common::Timeline::Mark("sync-cmd", reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	SendCommand([operation = std::move(command), &done]() mutable {
		Common::Timeline::Mark("sync-cmd-run");
		operation();
		done.release();
	});
	done.acquire();
	Common::Timeline::Mark("sync-cmd-done");
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Submission submission;
	Common::EventTrace::Record(Common::EventTrace::GUEST_SUBMIT, 0, draw_commands.size());
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Common::Timeline::Mark("submit-gfx", draw_commands.size());
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Common::EventTrace::Record(Common::EventTrace::GUEST_SUBMIT, 1 + compute_queue, commands.size());
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Common::Timeline::Mark("submit-cs", commands.size());
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Common::Timeline::Mark("submit-flip", request_id);
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	Common::Timeline::Mark("suspend-wait");
	Common::EventTrace::Record(Common::EventTrace::SUSPEND_ACQ_BEGIN);
	m_suspend_point_ready->acquire();
	Common::EventTrace::Record(Common::EventTrace::SUSPEND_ACQ_END);
	Common::Timeline::Mark("suspend-go");
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_pending_instances.clear();
	m_pending_shader_offsets.clear();
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	if (FrameBreakdown::Enabled()) FrameBreakdown::SpecNotePktWrite(reinterpret_cast<uint64_t>(dst), static_cast<uint64_t>(dw_num) * 4);
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

bool CommandProcessor::TryPeekCleanBackingBytes(const volatile void* addr, void* out, uint64_t size) {
	// Read current bytes without a GPU->CPU download when they are not GPU-dirty. The
	// backing copy is authoritative for CPU-written bytes: a guest computation writes
	// its command stream, submits it, and only then does the CP consume it, so the CP
	// cannot be reading bytes a concurrent GPU write is still producing. Any exact-range
	// GPU write still refuses the peek, so the fault path keeps the final say. This is
	// the property that makes a false-sharing read cheap: a dirty neighbour on the same
	// 4 KiB page must not drag a whole-page download and queue drain behind it.
	const auto vaddr = reinterpret_cast<uint64_t>(addr);
	if (!GuestRange {vaddr, size}.Valid() || m_renderer.GetBufferCache().HasGpuDirtyBytes(vaddr, size) ||
	    !LibKernel::Memory::TryReadBacking(vaddr, out, size)) {
		return false;
	}
	return true;
}

bool CommandProcessor::TryPeekGuest(const volatile void* addr, void* out, uint64_t size) {
	if (!BufferCache::PerfNoSyncReadback() || !GuestGpu::IsGpuThread()) {
		return false;
	}
	if (!TryPeekCleanBackingBytes(addr, out, size)) {
		return false;
	}
	BufferCache::PerfCount(BufferCache::PerfEvent::WaitRegMemBackingRead);
	return true;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	// KYTY_PERF_NO_SYNC_READBACK: the polled label can share a protected page with unrelated
	// GPU-written bytes. Dereferencing it then faults into a page download that drains the GPU
	// queue, although the label itself is current in the CPU backing (direct EOP labels write it
	// there). Read the backing when the exact bytes are not GPU-dirty; otherwise take the old path.
	const T value = PeekGuest(addr);
	if (!TestWaitRegMemValue(value, ref, mask, func)) {
		SuspendPm4();
	}
	// KYTY_GUEST_SYNC: labels are written at CP time, so a guest RELEASE_MEM + WAIT_REG_MEM pair
	// no longer orders GPU work by itself. Without the automatic dispatch barriers the wait must.
	if (BufferCache::GuestSyncOnly()) {
		EmitGlobalBarrier();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}
	if (FrameBreakdown::Enabled()) {
		FrameBreakdown::SpecNotePktWrite(reinterpret_cast<uint64_t>(dst),
		                                 write_one_address ? sizeof(uint32_t)
		                                                   : static_cast<uint64_t>(dw_num) * sizeof(uint32_t));
	}

	// Labels commonly share a protected page with GPU-written data. Update both
	// copies through the existing exact-range label path, without downloading or
	// unprotecting unrelated bytes. Read each source before its corresponding
	// repeated-address write: source and destination may alias.
	if (Common::RuntimeFlags::Current().pm4_write_data_fast && GuestGpu::IsGpuThread()) {
		auto& cache = m_renderer.GetBufferCache();
		if (write_one_address) {
			for (uint32_t i = 0; i < dw_num; i++) {
				const uint32_t value = src[i];
				if (!cache.TryWriteLabel(reinterpret_cast<uint64_t>(dst), &value, sizeof(value))) {
					dst[0] = value;
				}
			}
			return;
		}
		if (dw_num <= 2) {
			std::array<uint32_t, 2> values{};
			std::memcpy(values.data(), src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
			if (cache.TryWriteLabel(reinterpret_cast<uint64_t>(dst), values.data(),
			                        static_cast<uint64_t>(dw_num) * sizeof(uint32_t))) {
				return;
			}
		}
	}

	if (write_one_address) {
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
	} else {
		memcpy(dst, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	if (FrameBreakdown::Enabled()) FrameBreakdown::SpecNotePktWrite(dst_address, num_bytes);
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	if (!dst_gds && FrameBreakdown::Enabled()) {
		FrameBreakdown::SpecNotePktWrite(dst_address_or_offset, num_bytes);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	{
		// Preserve early packet order, then report newly seen GDS regions throughout the run.
		static std::mutex gds_log_mutex;
		static uint64_t gds_packets = 0;
		static std::vector<std::array<uint64_t, 5>> gds_regions;
		if (dst_gds || src_sel == 1) {
			std::lock_guard lock(gds_log_mutex);
			const std::array<uint64_t, 5> region {
			    dst_sel, src_sel, dst_gds ? dst_address_or_offset : 0,
			    src_sel == 1 || src_sel == 2 ? src_address_or_offset_or_immediate : 0,
			    num_bytes};
			const bool unseen = std::find(gds_regions.begin(), gds_regions.end(), region) ==
			                    gds_regions.end();
			const bool report_new = unseen && gds_regions.size() < 256;
			if (report_new) gds_regions.push_back(region);
			const auto packet = ++gds_packets;
			// KYTY_DBG_GDS_TRACE=1: also log every DMA touching GDS bytes 0..15 (rate-limited).
			static const bool gds_trace = [] {
				const char* v = std::getenv("KYTY_DBG_GDS_TRACE");
				return v != nullptr && v[0] == '1';
			}();
			static uint64_t gds_low_packets = 0;
			bool            low_gds = false;
			if (gds_trace && dst_gds && dst_address_or_offset < 16) {
				const auto n = ++gds_low_packets;
				low_gds      = n <= 400 || n % 50 == 0;
			}
			if (packet <= 48 || report_new || low_gds) {
				std::printf("NHL26GDSW: dma dst_sel=%u src_sel=%u dst=0x%llx src=0x%llx bytes=%u\n",
				            static_cast<uint32_t>(dst_sel), static_cast<uint32_t>(src_sel),
				            static_cast<unsigned long long>(dst_address_or_offset),
				            static_cast<unsigned long long>(src_address_or_offset_or_immediate),
				            static_cast<uint32_t>(num_bytes));
				std::fflush(stdout);
			}
		}
	}
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

namespace {
// Draws and dispatches the GPU thread has processed (KYTY_DBG_QUEUE_TIME reports time per item).
uint64_t g_work_items = 0;

// KYTY_DBG_QUEUE_TIME=1: GPU-thread time spent processing each guest queue (0 = graphics,
// 1.. = compute rings), printed every 5 s as NHL27QUEUES.
struct QueueTimeScope {
	static bool Enabled() {
		static const bool enabled = std::getenv("KYTY_DBG_QUEUE_TIME") != nullptr;
		return enabled;
	}
	explicit QueueTimeScope(uint32_t queue): queue(queue) {
		if (Enabled()) start = std::chrono::steady_clock::now();
	}
	~QueueTimeScope() {
		if (!Enabled()) return;
		static std::array<uint64_t, 64> ns {};
		static std::array<uint64_t, 64> slices {};
		if (queue >= ns.size()) return;
		static auto last = std::chrono::steady_clock::now();
		const auto  now  = std::chrono::steady_clock::now();
		ns[queue] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
		slices[queue]++;
		if (now - last < std::chrono::seconds(5)) return;
		const double seconds = std::chrono::duration<double>(now - last).count();
		last = now;
		uint64_t total_ns = 0;
		for (const auto value: ns) total_ns += value;
		static uint64_t last_items = 0;
		const uint64_t  items      = g_work_items - last_items;
		last_items                 = g_work_items;
		std::string line = fmt::format("NHL27QUEUES items/s={:.0f} us/item={:.2f} ms/s:",
		                               static_cast<double>(items) / seconds,
		                               items != 0 ? static_cast<double>(total_ns) / 1e3 / static_cast<double>(items) : 0.0);
		for (uint32_t i = 0; i < ns.size(); i++) {
			if (slices[i] == 0) continue;
			line += fmt::format(" q{}={:.1f}({:.0f})", i, static_cast<double>(ns[i]) / 1e6 / seconds,
			                    static_cast<double>(slices[i]) / seconds);
		}
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
		ns     = {};
		slices = {};
	}
	uint32_t                              queue;
	std::chrono::steady_clock::time_point start {};
};
} // namespace

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;

	bool high_priority = false;
	for (;;) {
		// Experiment GpuThreadPriority: this thread limits the frame rate; guest threads that spin
		// should not preempt it.
		if (const bool want = Common::Experiments::Enabled(Common::Experiments::GpuThreadPriority);
		    want != high_priority) {
			Common::Thread::SetCurrentHighPriority(want);
			high_priority = want;
		}
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				Common::Timeline::Mark("gpu-idle");
				const auto idle_start = std::chrono::steady_clock::now();
				Common::EventTrace::Record(Common::EventTrace::GPU_WAIT_BEGIN);
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				Common::EventTrace::Record(Common::EventTrace::GPU_WAIT_END,
				                           gpu->m_commands.size() + gpu->m_submission_count);
				FrameBreakdown::AddIdleGap(static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - idle_start)
				        .count()));
				Common::Timeline::Mark("gpu-wake");
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					Common::Timeline::Mark("gpu-blocked");
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					Common::Timeline::Mark("gpu-unblocked");
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_BEGIN, 1, 0);
			command();
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_END, 1, 0);

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		Common::Timeline::Mark("slice-begin", submission.queue_id * 10u + static_cast<uint32_t>(submission.type));
		bool complete = false;
		{
			// KYTY_PERF_STATS: GPU thread busy = Process wall time minus its master-semaphore waits.
			FrameBreakdown::Timer process_timer(FrameBreakdown::g_counters.gpu_process_ns);
			// NHL27GPUBUSY: exclusive bucket attribution over the same window.
			struct GpuBusyWindow {
				GpuBusyWindow() { FrameBreakdown::GpuBusyBegin(); }
				~GpuBusyWindow() { FrameBreakdown::GpuBusyEnd(); }
			} gpu_busy_window;
			// kind: 2 + SubmissionType (graphics/compute/flip prep/suspend point), +0x100 when sliced/blocked
			const uint64_t kind = 2 + static_cast<uint64_t>(submission.type);
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_BEGIN, kind, gpu->m_submit_id);
			QueueTimeScope queue_time(submission.queue_id);
			complete = gpu->Process(submission);
			Common::EventTrace::Record(Common::EventTrace::GPU_CMD_END, kind | (complete ? 0u : 0x100u),
			                           gpu->m_submit_id);
		}
		Common::Timeline::Mark(complete ? "slice-done" : "slice-blocked", submission.queue_id);
		FrameBreakdown::MaybePrint();

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

// KYTY_SUBMIT_SPLIT: while recording one large guest command buffer, submit the native command
// buffer after enough draws/dispatches or recording time so the host GPU can start earlier.
// GPU-thread only. Policy (kept simple): split at packet boundaries right after a draw/dispatch
// packet completes, regardless of render-pass state. Submit() ends dynamic rendering and the next
// draw re-begins it, exactly like the existing Flip()/WaitFlipDone()/slice-end flushes.
struct SubmitSplitState {
	bool                                  enabled = false;
	uint32_t                              work    = 0;
	std::chrono::steady_clock::time_point since {};
};
static thread_local SubmitSplitState g_submit_split;
static std::atomic<uint64_t>         g_submit_split_count {0};

// Split thresholds: env KYTY_SUBMIT_SPLIT_DRAWS / KYTY_SUBMIT_SPLIT_US set the defaults; a
// "split.ctl" file in the working directory (lines "draws=N", "us=T") overrides them and is
// re-read every 500 ms so the thresholds can be tuned while the game runs.
static std::atomic<uint32_t> g_split_draws {0};
static std::atomic<int64_t>  g_split_us {0};

static void SubmitSplitRefresh() {
	// Measured best on an RX 9070 XT: 768 draws / 12 ms when each split costs this thread a
	// vkQueueSubmit, 48 draws / 0.75 ms with experiment AsyncSubmit (+4.5% paused match).
	const bool            async     = Common::Experiments::Enabled(Common::Experiments::AsyncSubmit);
	static const uint32_t env_draws = [] {
		const char* s = std::getenv("KYTY_SUBMIT_SPLIT_DRAWS");
		const long  n = s != nullptr ? std::atol(s) : 0;
		return n > 0 ? static_cast<uint32_t>(n) : 0u;
	}();
	static const int64_t env_us = [] {
		const char* s = std::getenv("KYTY_SUBMIT_SPLIT_US");
		const long  n = s != nullptr ? std::atol(s) : 0;
		return n > 0 ? static_cast<int64_t>(n) : int64_t {0};
	}();
	static auto last = std::chrono::steady_clock::time_point {};
	const auto  now  = std::chrono::steady_clock::now();
	if (last != std::chrono::steady_clock::time_point {} && now - last < std::chrono::milliseconds(500)) {
		return;
	}
	last           = now;
	uint32_t draws = env_draws != 0 ? env_draws : async ? 48u : 768u;
	int64_t  us    = env_us != 0 ? env_us : async ? int64_t {750} : int64_t {12000};
	if (FILE* file = std::fopen("split.ctl", "r"); file != nullptr) {
		char line[64];
		while (std::fgets(line, sizeof(line), file) != nullptr) {
			long value = 0;
			if (std::sscanf(line, "draws=%ld", &value) == 1 && value > 0) {
				draws = static_cast<uint32_t>(value);
			} else if (std::sscanf(line, "us=%ld", &value) == 1 && value > 0) {
				us = value;
			}
		}
		std::fclose(file);
	}
	g_split_draws.store(draws, std::memory_order_relaxed);
	g_split_us.store(us, std::memory_order_relaxed);
}

static uint32_t SubmitSplitDraws() {
	return g_split_draws.load(std::memory_order_relaxed);
}

static int64_t SubmitSplitUs() {
	return g_split_us.load(std::memory_order_relaxed);
}

static bool SubmitSplitIsWorkOpcode(uint32_t op) {
	switch (op) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

// NHL27GPUBUSY: which exclusive bucket a PM4 packet handler is charged to.
static FrameBreakdown::GpuBucket PacketBusyBucket(uint32_t op) {
	switch (op) {
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW: return FrameBreakdown::GB_DRAW_HANDLER;
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return FrameBreakdown::GB_DISPATCH_HANDLER;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_MEM_SEMAPHORE:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_SURFACE_SYNC:
		case Pm4::IT_ACQUIRE_MEM: return FrameBreakdown::GB_SYNC_PKT;
		case Pm4::IT_DMA_DATA:
		case Pm4::IT_CP_DMA:
		case Pm4::IT_COPY_DATA: return FrameBreakdown::GB_DMA_PKT;
		default: return FrameBreakdown::GB_HANDLER;
	}
}

static void SubmitSplitMaybePrint() {
	static const bool stats = [] {
		const char* s = std::getenv("KYTY_PERF_STATS");
		return s != nullptr && s[0] == '1';
	}();
	if (!stats) return;
	static auto     last = std::chrono::steady_clock::now();
	static uint64_t last_count = 0;
	const auto      now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	const auto count = g_submit_split_count.load(std::memory_order_relaxed);
	std::printf("NHL27SPLIT: splits=%llu delta=%llu draws_thr=%u us_thr=%lld\n",
	            static_cast<unsigned long long>(count),
	            static_cast<unsigned long long>(count - last_count), SubmitSplitDraws(),
	            static_cast<long long>(SubmitSplitUs()));
	std::fflush(stdout);
	last = now;
	last_count = count;
}

// KYTY_BDA_BATCH: PM4 packets after which guest memory may be ordered for the GPU or CPU. WRITE_DATA, COPY_DATA
// and draws/dispatches are deliberately not boundaries: they do not order CPU-written memory for the GPU.
static bool BdaBoundaryKind(uint32_t opcode, uint32_t header, RenderContext::BdaBoundary& kind) {
	using B = RenderContext::BdaBoundary;
	switch (opcode) {
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_WAIT_ON_CE_COUNTER:
		case Pm4::IT_WAIT_ON_DE_COUNTER_DIFF: kind = B::Wait; return true;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS: kind = B::Event; return true;
		case Pm4::IT_ACQUIRE_MEM: kind = B::Acquire; return true;
		case Pm4::IT_DMA_DATA: kind = B::Dma; return true;
		case Pm4::IT_PFP_SYNC_ME: kind = B::Sync; return true;
		case Pm4::IT_INDIRECT_BUFFER: kind = B::Ib; return true;
		case Pm4::IT_NOP:
			switch (KYTY_PM4_R(header)) {
				case Pm4::R_WAIT_FLIP_DONE: kind = B::Wait; return true;
				case Pm4::R_ACQUIRE_MEM: kind = B::Acquire; return true;
				case Pm4::R_RELEASE_MEM: kind = B::Event; return true;
				case Pm4::R_FLIP: kind = B::Flip; return true;
				default: return false;
			}
		default: return false;
	}
}

bool GuestGpu::Process(Submission& submission) {
	// Start of each PM4 submission slice (graphics, compute queue, flip preparation, suspend point).
	m_renderer.BdaBatchBoundary(submission.type == SubmissionType::FlipPreparation ? RenderContext::BdaBoundary::Flip
	                                                                                : RenderContext::BdaBoundary::Submit);
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	g_submit_split.enabled = Common::RuntimeFlags::Current().submit_split;
	if (g_submit_split.enabled) {
		SubmitSplitRefresh();
	}
	g_submit_split.work    = 0;
	g_submit_split.since   = {};
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			cp.EmitGlobalBarrier();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] {
				    Common::EventTrace::Record(Common::EventTrace::SUSPEND_RELEASE);
				    ready->release();
			    });
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	g_submit_split.enabled = false;
	SubmitSplitMaybePrint();
	m_renderer.BdaBatchBoundary(RenderContext::BdaBoundary::Submit);
	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	BufferCache::PerfTimerScope phase_timer(BufferCache::PerfTimer::Pm4);
	ProcessPm4(execution);
	phase_timer.Stop();
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	if (FrameBreakdown::Enabled()) FrameBreakdown::SpecNoteWait();
	g_current_execution->m_suspended = true;
}

// Experiment FalseShareReads: a compute dispatch can write into data embedded in the command
// buffer itself, which protects the whole page. Reading the following packets on that page
// would then fault and wait for all queued GPU work. When the page holds GPU-dirty bytes but the
// packet's own bytes are clean, parse a copy taken from the backing instead.
const uint32_t* CommandProcessor::ShadowPm4Packet(const uint32_t* packet, uint32_t remaining_dw) {
	if (!GuestGpu::IsGpuThread() || remaining_dw == 0) {
		return packet;
	}
	auto&          buffers    = m_renderer.GetBufferCache();
	const auto     address    = reinterpret_cast<uint64_t>(packet);
	const auto     page       = address & ~uint64_t {0xfff};
	const uint64_t generation = buffers.GpuDirtyGeneration();
	if (page != m_pm4_page || generation != m_pm4_page_generation) {
		m_pm4_page            = page;
		m_pm4_page_generation = generation;
		m_pm4_page_dirty = GuestRange {page, 0x1000}.Valid() && buffers.HasGpuDirtyBytes(page, 0x1000);
	}
	if (!m_pm4_page_dirty) {
		return packet;
	}
	const auto read = [&](uint64_t offset_dw, uint32_t* out, uint64_t count_dw) {
		const auto at    = address + offset_dw * 4u;
		const auto bytes = count_dw * 4u;
		return !buffers.HasGpuDirtyBytes(at, bytes) &&
		       LibKernel::Memory::TryReadBacking(at, out, bytes);
	};
	uint32_t header[2] {};
	if (!read(0, header, std::min<uint32_t>(2u, remaining_dw))) {
		return packet;
	}
	uint32_t length = header[0] == 0x80000000u
	                      ? 1u
	                      : std::min<uint32_t>(KYTY_PM4_LEN(header[0]), remaining_dw);
	// A plain NOP wraps embedded data, often the very bytes the GPU writes. Its handler reads
	// only the first payload dword unless the NOP carries a marker string.
	if (((header[0] >> 8u) & 0xffu) == Pm4::IT_NOP && KYTY_PM4_R(header[0]) == Pm4::R_ZERO &&
	    length > 2u && (header[1] & 0xffff0000u) != 0x68750000u) {
		length = 2u;
	}
	m_pm4_shadow.resize(length);
	if (!read(0, m_pm4_shadow.data(), length)) {
		return packet;
	}
	BufferCache::PerfCount(BufferCache::PerfEvent::Pm4ShadowPacket);
	return m_pm4_shadow.data();
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	static const bool clean_backing = [] {
		const char* setting = std::getenv("KYTY_PM4_CLEAN_BACKING");
		return setting != nullptr && setting[0] == '1';
	}();
	// Only bypass exact-clean header reads. Some register handlers fuse adjacent packets,
	// so their payload pointer and available stream length must remain the original ones.
	struct FaultScopeRestore {
		Pm4FaultScope previous = g_pm4_fault_scope;
		~FaultScopeRestore() { g_pm4_fault_scope = previous; }
	} fault_scope_restore;
	while (!execution.m_buffer_stack.empty()) {
		g_pm4_fault_scope = {};
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			if (execution.m_buffer_stack.size() > 1) {
				m_renderer.BdaBatchBoundary(RenderContext::BdaBoundary::Ib);  // return edge of an indirect buffer call
			}
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const guest_packet  = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto* const packet =
		    Common::Experiments::Enabled(Common::Experiments::FalseShareReads)
		        ? ShadowPm4Packet(guest_packet, remaining_dw)
		        : guest_packet;
		g_pm4_fault_scope = {reinterpret_cast<uint64_t>(packet), 1, 0, false};
		uint32_t packet_header = 0;
		if (!clean_backing || !TryPeekGuest(packet, &packet_header, sizeof(packet_header))) {
			packet_header = packet[0];
		}
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		g_pm4_fault_scope = {reinterpret_cast<uint64_t>(packet),
		                     std::min<uint32_t>(KYTY_PM4_LEN(packet_header), remaining_dw),
		                     packet_header, true};
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);
		const auto declared_dw = KYTY_PM4_LEN(packet_header);
		EXIT_NOT_IMPLEMENTED(declared_dw > remaining_dw);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(guest_packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(guest_packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = guest_packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		BufferCache::PerfSetPm4Op(opcode);
		uint32_t packet_dw;
		{
			FrameBreakdown::GpuBusyScope busy_scope(PacketBusyBucket(opcode));
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (RenderContext::BdaBoundary bda_kind; BdaBoundaryKind(opcode, packet_header, bda_kind)) {
			m_renderer.BdaBatchBoundary(bda_kind);  // after the handler, also when the packet suspended on a wait
		}
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (g_submit_split.enabled && SubmitSplitIsWorkOpcode(opcode)) {
			// Packet fully handled and cursor advanced: a flush here cannot separate a
			// draw/dispatch from any later EOP/label packet in the stream.
			const auto now = std::chrono::steady_clock::now();
			if (g_submit_split.work++ == 0) {
				g_submit_split.since = now;
			} else if (g_submit_split.work >= SubmitSplitDraws() ||
			           std::chrono::duration_cast<std::chrono::microseconds>(now - g_submit_split.since)
			                   .count() >= SubmitSplitUs()) {
				BufferFlush();
				g_submit_split.work = 0;
				g_submit_split_count.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances = num_instances;
	m_pending_instances.clear();
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const volatile uint64_t* results = reinterpret_cast<const volatile uint64_t*>(address);
			uint64_t                 snapshot[32];
			if (TryPeekGuest(address, snapshot, sizeof(snapshot))) {
				results = snapshot;
			}
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = PeekGuest(reinterpret_cast<const volatile uint64_t*>(address));
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

uint32_t CommandProcessor::ResolveNumInstances() {
	m_last_instance_args_addr = 0;
	if (m_pending_instances.empty()) return m_num_instances;
	// Read immutable bytes captured when each indirect packet executed. Guest arguments may
	// already have been overwritten, so reading their current addresses would be incorrect.
	m_renderer.GetCommandScheduler().Finish();
	for (auto it = m_pending_instances.rbegin(); it != m_pending_instances.rend(); ++it) {
		it->snapshot->Invalidate(0, it->snapshot->Size());
		const auto bytes = it->snapshot->Mapped();
		uint32_t count = 0;
		std::memcpy(&count, bytes.data() + it->ArgsSize(), sizeof(count));
		count = std::min(count, it->max_count);
		if (count == 0) continue;
		std::memcpy(&m_num_instances, bytes.data() + static_cast<uint64_t>(count - 1) *
		            it->stride + offsetof(vk::DrawIndirectCommand, instanceCount), sizeof(uint32_t));
		m_last_instance_args_addr = it->args_addr;
		const bool zero = m_num_instances == 0;
		const bool fixed = zero && InstanceZeroDiag::FixEnabled();
		if (fixed) {
			// Mirror SetNumInstances(): a zero instance count is clamped to 1.
			m_num_instances = 1;
			InstanceZeroDiag::NoteFixApplied();
		}
		if (zero && InstanceZeroDiag::Enabled() && InstanceZeroDiag::TakeLine()) {
			std::printf("NHL27INST: resolve_zero args=0x%llx count_word=%u max=%u stride=%u "
			            "indexed=%d fixed=%d\n",
			            static_cast<unsigned long long>(it->args_addr), count, it->max_count,
			            it->stride, it->indexed ? 1 : 0, fixed ? 1 : 0);
		}
		break;
	}
	m_pending_instances.clear();
	return m_num_instances;
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	++g_work_items;
	ResolveNativeShaderRegisters();
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		const bool from_pending = !m_pending_instances.empty();
		args.instance_count     = ResolveNumInstances();
		args.instance_source    = from_pending ? InstanceZeroDiag::kSourcePending
		                                       : InstanceZeroDiag::kSourcePersist;
		args.instance_args_addr = m_last_instance_args_addr;
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	++g_work_items;
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
}

void CommandProcessor::CancelNativeShaderRegister(uint32_t location) {
	for (auto& source: m_pending_shader_offsets) {
		for (auto& reg: source.shader_regs) if (reg == location) reg = Pm4::SH_NOP;
	}
	std::erase_if(m_pending_shader_offsets, [](const NativeIndirectDraw& source) {
		return std::ranges::all_of(source.shader_regs, [](uint32_t reg) { return reg == Pm4::SH_NOP; });
	});
}

void CommandProcessor::ResolveNativeShaderRegisters() {
	if (m_pending_shader_offsets.empty()) return;
	// Move first: the register handlers call GetShCtx(), which must not recurse.
	auto pending = std::move(m_pending_shader_offsets);
	m_pending_shader_offsets.clear();
	m_renderer.GetCommandScheduler().Finish();
	for (auto& source: pending) {
		source.snapshot->Invalidate(0, source.snapshot->Size());
		const auto bytes = source.snapshot->Mapped();
		uint32_t count = 0;
		std::memcpy(&count, bytes.data() + source.ArgsSize(), sizeof(count));
		count = std::min(count, source.max_count);
		if (count == 0) continue;
		const auto* record = bytes.data() + static_cast<uint64_t>(count - 1u) * source.stride;
		uint32_t vertex = 0, instance = 0, index = 0;
		std::memcpy(&vertex, record + (source.indexed ? 12 : 8), 4);
		std::memcpy(&instance, record + (source.indexed ? 16 : 12), 4);
		if (source.indexed) std::memcpy(&index, record + 8, 4);
		const auto saved_marker = m_user_data_marker;
		m_user_data_marker = static_cast<HW::UserSgprType>(source.shader_marker);
		PatchIndirectDrawOffsets(*this, {source.shader_regs[0], source.shader_regs[1], source.shader_regs[2]}, vertex, instance, index);
		m_user_data_marker = saved_marker;
	}
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	++g_work_items;
	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	++g_work_items;
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	// Report packet shapes without reading GPU-written argument or count memory.
	// These SH destinations determine which shader inputs must become GPU draw parameters.
	static const bool audit_indirect = [] {
		const char* setting = std::getenv("KYTY_PERF_STATS");
		return setting != nullptr && setting[0] == '1';
	}();
	if (audit_indirect) {
		struct PacketShape {
			uint32_t vertex, instance, index, stride, maximum;
			bool indexed, counted;
			bool operator==(const PacketShape&) const = default;
		};
		static thread_local std::vector<PacketShape> reported;
		const PacketShape shape {registers.vertex_offset, registers.instance_offset,
		                         registers.index_offset, stride_in_bytes, max_count_or_count,
		                         indexed, count_addr != nullptr};
		if (reported.size() < 32 && std::find(reported.begin(), reported.end(), shape) == reported.end()) {
			reported.push_back(shape);
			std::printf("NHL27INDIRECT: indexed=%u counted=%u max=%u stride=%u "
			            "vertex_sh=0x%x instance_sh=0x%x index_sh=0x%x\n",
			            static_cast<unsigned>(indexed), static_cast<unsigned>(count_addr != nullptr),
			            max_count_or_count, stride_in_bytes, registers.vertex_offset,
			            registers.instance_offset, registers.index_offset);
		}
	}

	static const bool native_indirect = [] {
		const char* setting = std::getenv("KYTY_NATIVE_INDIRECT");
		return setting != nullptr && setting[0] == '1';
	}();
	// Register writes require a shader variant; preserve the CPU path for those packets.
	const std::array<uint32_t, 3> shader_regs {registers.vertex_offset, registers.instance_offset, registers.index_offset};
	if (std::ranges::any_of(m_pending_shader_offsets, [&](const NativeIndirectDraw& old) {
		return std::ranges::any_of(old.shader_regs, [&](uint32_t reg) {
			return reg != Pm4::SH_NOP && std::ranges::find(shader_regs, reg) == shader_regs.end();
		});
	})) ResolveNativeShaderRegisters();
	const bool patch_supported = std::ranges::all_of(shader_regs, [](uint32_t location) {
		return location == Pm4::SH_NOP || (location >= Pm4::SPI_SHADER_USER_DATA_GS_0 && location <= Pm4::SPI_SHADER_USER_DATA_GS_31);
	});
	const bool has_patch = std::ranges::any_of(shader_regs, [](uint32_t location) { return location != Pm4::SH_NOP; });
	if (native_indirect && patch_supported && max_count_or_count != 0 && (!has_patch || count_addr == nullptr || m_user_data_marker == HW::UserSgprType::Unknown) &&
	    (!has_patch || m_renderer.GetGraphics().native_draw_parameters_enabled)) {
		if (m_pending_shader_offsets.size() >= 64) ResolveNativeShaderRegisters();
		if (m_pending_instances.size() >= 64) (void)ResolveNumInstances();
		NativeIndirectDraw source {.args_addr = m_draw_indirect_args_base_addr + data_offset,
		                           .count_addr = reinterpret_cast<uint64_t>(count_addr),
		                           .max_count = max_count_or_count, .stride = stride_in_bytes,
		                           .indexed = indexed, .index_base = m_index_base_addr,
		                           .index_count_limit = m_index_buffer_size, .index_type = m_index_type_and_size,
		                           .shader_regs = shader_regs, .shader_marker = static_cast<uint32_t>(m_user_data_marker)};
		if (m_renderer.GetRenderExecutor().DrawAutoIndirect(m_submit_id, CurrentBuffer(), source)) {
			if (source.count_addr == 0) m_pending_instances.clear();
			if (has_patch) {
				m_pending_shader_offsets.push_back(source);
				if (count_addr == nullptr) m_user_data_marker = HW::UserSgprType::Unknown;
			}
			m_pending_instances.push_back(std::move(source));
			return;
		}
	}

	ResolveNativeShaderRegisters();
	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = PeekGuest(count_addr);
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}
	m_pending_instances.clear();

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			PatchIndirectDrawOffsets(*this, registers, args->start_vertex_location,
			                         args->start_instance_location, 0);
			m_num_instances = args->instance_count;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);
		PatchIndirectDrawOffsets(*this, registers, args->base_vertex_location,
		                         args->start_instance_location, args->start_index_location);

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		m_num_instances = args->instance_count;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	++g_work_items;
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode);
	}
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	++g_work_items;
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode);
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	++g_work_items;
	ResolveNativeShaderRegisters();
	if (args.instance_count == 0) {
		const bool from_pending = !m_pending_instances.empty();
		args.instance_count     = ResolveNumInstances();
		args.instance_source    = from_pending ? InstanceZeroDiag::kSourcePending
		                                       : InstanceZeroDiag::kSourcePersist;
		args.instance_args_addr = m_last_instance_args_addr;
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	if (dst_gpu_addr != nullptr && FrameBreakdown::Enabled()) {
		FrameBreakdown::SpecNotePktWrite(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(T));
	}
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	// Opt-in label path for NHL26 menu synchronization profiling.
	static const bool direct_labels = [] {
		const char* setting = std::getenv("KYTY_EOP_DIRECT_LABELS");
		return setting != nullptr && setting[0] == '1';
	}();
	auto write_label = [&](void* destination, const void* data, uint64_t bytes) {
		Common::EventTrace::Record(Common::EventTrace::LABEL_WRITE,
		                           reinterpret_cast<uint64_t>(destination) & 0xffffffffu,
		                           m_renderer.GetCommandScheduler().CurrentTick());
		if (!direct_labels || !m_renderer.GetBufferCache().TryWriteLabel(
		        reinterpret_cast<uint64_t>(destination), data, bytes)) {
			std::memcpy(destination, data, static_cast<size_t>(bytes));
		} else {
			static std::atomic_flag reported = ATOMIC_FLAG_INIT;
			if (!reported.test_and_set(std::memory_order_relaxed)) {
				std::printf("NHL26MENU: direct EOP label write active width=%llu\n",
				            static_cast<unsigned long long>(bytes));
				std::fflush(stdout);
			}
		}
	};
	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		write_label(dst, &data, sizeof(data));

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					write_label(dst, &value, sizeof(value));

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());
	// KYTY_GUEST_BARRIER_DEDUP: a global barrier with no draw, dispatch or host helper pass (tiler, fault
	// manager, blit, indirect prep: they bump the pipeline epochs) recorded since the previous one in the
	// same command-buffer recording orders nothing new. A recording is identified by the scheduler tick.
	static uint64_t last_work = UINT64_MAX, last_compute = UINT64_MAX, last_graphics = UINT64_MAX;
	static uint64_t last_tick = UINT64_MAX;
	if (Common::RuntimeFlags::Current().guest_barrier_dedup) {
		const uint64_t work     = g_work_epoch.load(std::memory_order_relaxed);
		const uint64_t compute  = g_compute_pipeline_epoch.load(std::memory_order_relaxed);
		const uint64_t graphics = g_graphics_pipeline_epoch.load(std::memory_order_relaxed);
		const uint64_t tick     = GetScheduler().CurrentTick();
		const bool     redundant = work == last_work && compute == last_compute &&
		                           graphics == last_graphics && tick == last_tick;
		last_work = work; last_compute = compute; last_graphics = graphics; last_tick = tick;
		if (redundant) {
			FrameBreakdown::Add(FrameBreakdown::g_counters.port_guest_barriers_skipped);
			return;
		}
	} else {
		last_tick = UINT64_MAX;
	}
	FrameBreakdown::Add(FrameBreakdown::g_counters.port_guest_barriers);

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().Handle().pipelineBarrier2(dependency);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	Common::Timeline::Mark("cp-flip");
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
