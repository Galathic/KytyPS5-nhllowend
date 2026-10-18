#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace Libs::Graphics {

// Frostbite's captured 208-byte block, distinct from the embedded SDK FSR2 layout.
// Only the verified layout/format combination is accepted. Unknown tuning words
// and pass-specific tail words are retained without assigning semantics.
struct Nhl27Fsr3Constants {
	std::array<uint32_t, 52> words {};
	float Float(size_t i) const { return std::bit_cast<float>(words[i]); }
	uint32_t Width() const { return words[4]; }
	uint32_t Height() const { return words[5]; }
	uint32_t OutputWidth() const { return words[8]; }
	uint32_t OutputHeight() const { return words[9]; }
	uint32_t Frame() const { return words[39]; }
	float Near() const { return Float(1) / (1 - Float(0)); }
	float Far() const { return Float(1) / -Float(0); }
	float FovY() const { return 2 * std::atan(Float(3)); }
	bool SameFrame(const Nhl27Fsr3Constants& other) const {
		for (size_t i = 0; i < 49; ++i)
			if (words[i] != other.words[i]) return false;
		return true;
	}
	static std::optional<Nhl27Fsr3Constants> Decode(std::span<const uint32_t> source) {
		if (source.size() != 52) return std::nullopt;
		Nhl27Fsr3Constants c;
		std::copy(source.begin(), source.end(), c.words.begin());
		if (c.Width() < 320 || c.Height() < 180 || c.OutputWidth() < c.Width() ||
			c.OutputHeight() < c.Height() || c.OutputWidth() > 8192 || c.OutputHeight() > 8192)
			return std::nullopt;
		// Fail closed for dynamic-resource layouts not yet observed.
		for (size_t i : {size_t(4), size_t(5), size_t(8), size_t(9)})
			if (c.words[i] != c.words[i + 2]) return std::nullopt;
		for (size_t i = 0; i < 4; ++i) {
			const size_t offset = 12 + i * 2;
			const float w = float(c.words[4 + i * 2]), h = float(c.words[5 + i * 2]);
			if (c.Float(offset) != w || c.Float(offset + 1) != h ||
				std::abs(c.Float(offset + 8) * w - 1) > 1e-5f ||
				std::abs(c.Float(offset + 9) * h - 1) > 1e-5f)
				return std::nullopt;
		}
		for (size_t i : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(28), size_t(29),
						size_t(30), size_t(31), size_t(32), size_t(33), size_t(34), size_t(35),
						size_t(36), size_t(37), size_t(38)})
			if (!std::isfinite(c.Float(i))) return std::nullopt;
		if (c.Float(0) >= 0 || c.Float(1) <= 0 || c.Float(2) <= 0 || c.Float(3) <= 0 ||
			!std::isfinite(c.Near()) || !std::isfinite(c.Far()) || c.Near() <= 0 ||
			c.Far() <= c.Near() || c.FovY() >= 3.141593f ||
			c.Float(32) != -0.5f || c.Float(33) != 0.5f ||
			c.Float(36) <= 0 || c.Float(36) > 1 || c.Float(37) <= 0 || c.Float(38) <= 0)
			return std::nullopt;
		for (size_t i = 28; i < 32; ++i)
			if (std::abs(c.Float(i)) > 0.50001f) return std::nullopt;
		if (std::abs(c.Float(34) - float(c.Width()) / c.OutputWidth()) > 1e-5f ||
			std::abs(c.Float(35) - float(c.Height()) / c.OutputHeight()) > 1e-5f)
			return std::nullopt;
		return c;
	}
};
} // namespace Libs::Graphics
