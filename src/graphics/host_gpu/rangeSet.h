#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>

namespace Libs::Graphics {

class RangeSet final {
public:
	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		// Every path below covers [address, end) and keeps ranges already marked.
		FilterMark(address, end);
		// Written ranges are added again for every draw that binds them: extend a range that
		// begins at or before the address in place, and allocate a node only for a new range.
		auto it = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (previous->second >= address) {
				if (previous->second >= end) {
					return;
				}
				m_version++;
				uint64_t last = end;
				while (it != m_ranges.end() && it->first <= last) {
					last = std::max(last, it->second);
					it   = m_ranges.erase(it);
				}
				previous->second = last;
				return;
			}
		}
		uint64_t last = end;
		while (it != m_ranges.end() && it->first <= last) {
			last = std::max(last, it->second);
			it   = m_ranges.erase(it);
		}
		m_ranges.emplace_hint(it, address, last);
		m_version++;
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			m_version++;
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
		// The filter stays a superset; refresh it once enough removals made it loose.
		if (++m_filter_stale >= 256) {
			m_filter = {};
			for (const auto& [begin, last]: m_ranges) {
				FilterMark(begin, last);
			}
			m_filter_stale = 0;
		}
	}

	void Clear() {
		m_version += m_ranges.empty() ? 0u : 1u;
		m_ranges.clear();
		m_filter       = {};
		m_filter_stale = 0;
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		// O(1) reject outside the overall span (stored ranges are disjoint and ordered).
		if (m_ranges.empty() || end <= m_ranges.begin()->first || address >= m_ranges.rbegin()->second) {
			return false;
		}
		if (!FilterMayIntersect(address, end)) {
			return false;
		}
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }
	// Changes whenever the set does (not when an Add or Subtract leaves it as it was).
	[[nodiscard]] uint64_t Version() const { return m_version; }

private:
	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	// Intersects() runs for nearly every resource read and download check: one bit per 64 KiB
	// chunk (chunk number % 1024) that any range touches answers most misses without the tree
	// walk. Add() sets bits, Subtract() leaves them until the periodic rebuild.
	static constexpr uint32_t FilterShift = 16;
	static constexpr uint64_t FilterBits  = 1024;

	void FilterMark(uint64_t begin, uint64_t end) {
		const uint64_t first = begin >> FilterShift;
		const uint64_t last  = (end - 1) >> FilterShift;
		if (last - first >= FilterBits - 1) {
			m_filter.fill(~uint64_t {0});
			return;
		}
		for (uint64_t chunk = first; chunk <= last; chunk++) {
			m_filter[(chunk % FilterBits) / 64] |= uint64_t {1} << (chunk % 64);
		}
	}

	[[nodiscard]] bool FilterMayIntersect(uint64_t address, uint64_t end) const {
		const uint64_t first = address >> FilterShift;
		const uint64_t last  = (end - 1) >> FilterShift;
		if (last - first >= 8) {
			return true;
		}
		for (uint64_t chunk = first; chunk <= last; chunk++) {
			if (((m_filter[(chunk % FilterBits) / 64] >> (chunk % 64)) & 1u) != 0) {
				return true;
			}
		}
		return false;
	}

	std::map<uint64_t, uint64_t>           m_ranges;
	std::array<uint64_t, FilterBits / 64> m_filter {};
	uint32_t                               m_filter_stale = 0;
	uint64_t                               m_version      = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
