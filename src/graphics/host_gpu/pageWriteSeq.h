#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEWRITESEQ_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEWRITESEQ_H_

#include <atomic>
#include <cstdint>
#include <memory>

// Per guest page (4 KiB), the sequence of the last GPU operation that writes it from the emulator's
// side: writes the command processor notes for an operation (labels, DMA, clocks), and what the
// execution thread does while executing one (marking buffer ranges GPU-written, writing GPU data
// back into guest memory). The resolve thread records the sequence the execution thread had
// finished when it read guest memory ahead; a read on a page no later operation writes still holds
// when its operation executes (see GuestReadLog::StillValid).
namespace Libs::Graphics::PageWriteSeq {

namespace Detail {
constexpr uint32_t PageBits = 12;
constexpr uint32_t LeafBits = 16;
constexpr uint32_t TopBits  = 44 - PageBits - LeafBits;
using Leaf                  = std::atomic<uint64_t>[size_t {1} << LeafBits];
inline std::atomic<std::atomic<uint64_t>*> g_top[size_t {1} << TopBits] {};

inline std::atomic<uint64_t>* LeafFor(uint64_t page, bool create) {
	auto& slot = g_top[(page >> LeafBits) & ((uint64_t {1} << TopBits) - 1)];
	auto* leaf = slot.load(std::memory_order_acquire);
	if (leaf != nullptr || !create) {
		return leaf;
	}
	auto* fresh = new std::atomic<uint64_t>[size_t {1} << LeafBits] {};
	if (slot.compare_exchange_strong(leaf, fresh, std::memory_order_acq_rel)) {
		return fresh;
	}
	delete[] fresh;
	return leaf;
}
} // namespace Detail

// The operation the calling thread executes (the execution thread sets it per operation; other
// threads use the last one it published).
inline thread_local uint64_t t_operation = 0;
inline std::atomic<uint64_t> g_operation {0};

inline void SetOperation(uint64_t sequence) {
	t_operation = sequence;
	g_operation.store(sequence, std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t CurrentOperation() {
	// Writes of other threads (completions, guest calls) are not ordered with the operations: those
	// pages are checked from then on.
	return t_operation != 0 ? t_operation : ~uint64_t {0};
}

// Records that operation `sequence` writes [vaddr, vaddr + size).
inline void Raise(uint64_t vaddr, uint64_t size, uint64_t sequence) {
	if (size == 0 || vaddr >= (uint64_t {1} << 44u)) {
		return;
	}
	const uint64_t first = vaddr >> Detail::PageBits;
	const uint64_t last  = (vaddr + size - 1) >> Detail::PageBits;
	for (uint64_t page = first; page <= last; page++) {
		auto* leaf = Detail::LeafFor(page, true);
		auto& cell = leaf[page & ((uint64_t {1} << Detail::LeafBits) - 1)];
		auto  seen = cell.load(std::memory_order_relaxed);
		while (seen < sequence && !cell.compare_exchange_weak(seen, sequence, std::memory_order_release)) {
		}
	}
}

inline void RaiseCurrent(uint64_t vaddr, uint64_t size) {
	Raise(vaddr, size, CurrentOperation());
}

// Guest memory changed in a way no page records (mapping changes): reads resolved before the
// operations after the current one are checked in full.
inline std::atomic<uint64_t> g_floor {0};
inline void ForceChecks() {
	const auto floor = g_operation.load(std::memory_order_relaxed) + 1;
	auto       seen  = g_floor.load(std::memory_order_relaxed);
	while (seen < floor && !g_floor.compare_exchange_weak(seen, floor, std::memory_order_release)) {
	}
}

// The last operation recorded writing any page of [vaddr, vaddr + size).
[[nodiscard]] inline uint64_t Last(uint64_t vaddr, uint64_t size) {
	if (size == 0 || vaddr >= (uint64_t {1} << 44u)) {
		return ~uint64_t {0};
	}
	const uint64_t first  = vaddr >> Detail::PageBits;
	const uint64_t last   = (vaddr + size - 1) >> Detail::PageBits;
	uint64_t       result = 0;
	for (uint64_t page = first; page <= last; page++) {
		const auto* leaf = Detail::LeafFor(page, false);
		if (leaf != nullptr) {
			const auto value = leaf[page & ((uint64_t {1} << Detail::LeafBits) - 1)].load(std::memory_order_acquire);
			result           = value > result ? value : result;
		}
	}
	return result;
}

} // namespace Libs::Graphics::PageWriteSeq

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEWRITESEQ_H_
