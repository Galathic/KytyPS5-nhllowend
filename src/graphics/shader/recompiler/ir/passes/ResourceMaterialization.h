#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          operator==(const Image&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Retains direct descriptor decoding independently of changing addresses/payloads.
// The owner must reset this cache if its immutable plan is replaced or modified.
struct ResourceSpecializationCache {
	struct BufferEntry {
		std::array<uint32_t, 2> key {};
		ResourceSpecialization::Buffer value;
		bool valid = false;
	};
	struct ImageEntry {
		std::array<uint32_t, 7> key {};
		ResourceSpecialization::Image value;
		bool null = false;
		bool valid = false;
	};
	const ResourcePlan* owner = nullptr;
	bool active = false; // A nested materialization falls back instead of sharing mutable entries.
	std::vector<BufferEntry> buffers;
	std::vector<ImageEntry> images;
	void Reset() { owner = nullptr; active = false; buffers.clear(); images.clear(); }
};

struct ResourceSpecializationCacheStats {
	uint64_t calls = 0;
	uint64_t buffer_hits = 0, buffer_misses = 0;
	uint64_t image_hits = 0, image_misses = 0;
	uint64_t indirect_uncached = 0;
};
// Per-thread counters: diagnostics do not introduce contended atomics in draw prep.
ResourceSpecializationCacheStats GetResourceSpecializationCacheStats();

// A ResourcePlan carries mutable evaluation scratch (value memo, tapes, CFG traces, active_sources,
// Value::evaluation_index). It is single-threaded by design. The KYTY_DRAWPREP_PREFETCH worker
// materializes plans off the GPU thread, so every access to a plan's mutable state (any
// MaterializeResources call and any read/write of plan.active_sources) happens under this guard.
// Locks are striped by plan address (two plans may share a stripe; that only serializes them) and
// recursive, so nested guards on one thread are fine. A thread never holds two stripes at once.
// Inactive (no lock traffic at all) until a prefetch worker has been started.
inline std::atomic<bool> g_plan_access_enabled {false};
inline std::recursive_mutex g_plan_access_stripes[64];
class PlanAccessGuard {
public:
	explicit PlanAccessGuard(const ResourcePlan& plan) noexcept {
		if (g_plan_access_enabled.load(std::memory_order_acquire)) {
			const auto hash = std::hash<const void*> {}(&plan);
			m_mutex = &g_plan_access_stripes[(hash >> 4u) % 64u];
			m_mutex->lock();
		}
	}
	~PlanAccessGuard() {
		if (m_mutex != nullptr) m_mutex->unlock();
	}
	PlanAccessGuard(const PlanAccessGuard&)            = delete;
	PlanAccessGuard& operator=(const PlanAccessGuard&) = delete;

private:
	std::recursive_mutex* m_mutex = nullptr;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
