#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace Libs::Graphics {

// Candidate decoder only. Layout supplied from NHL's FSR2 reflection tables; a plausible match
// does not certify a shader, resource slot, depth convention or a same-frame snapshot.
struct Fsr2ConstantsCandidate {
	std::array<uint32_t, 32> words {};
	[[nodiscard]] float Float(unsigned byte_offset) const {
		return std::bit_cast<float>(words[byte_offset / 4]);
	}
	[[nodiscard]] bool Plausible() const {
		auto dimension = [](uint32_t x) { return x >= 32 && x <= 16384; };
		for (unsigned i = 0; i < 8; ++i)
			if (!dimension(words[i]))
				return false;
		if (words[0] > words[2] || words[1] > words[3] || words[0] > words[4] ||
			words[1] > words[5] || words[0] > words[6] || words[1] > words[7] || words[10] > 16)
			return false;
		for (unsigned offset = 48; offset <= 120; offset += 4)
			if (!std::isfinite(Float(offset)))
				return false;
		if (std::abs(Float(64)) > 2 || std::abs(Float(68)) > 2 || Float(72) == 0 ||
			Float(76) == 0 || std::abs(Float(72)) > 32768 || std::abs(Float(76)) > 32768 ||
			Float(96) <= 0 || Float(100) <= 0 || Float(104) <= 0.01f || Float(104) > 100 ||
			Float(108) < 1 || Float(108) > 1024 || Float(112) <= 0 || Float(112) > 10 ||
			Float(120) <= 0)
			return false;
		return std::abs(Float(80) - float(words[0]) / float(words[4])) < 0.02f &&
			   std::abs(Float(84) - float(words[1]) / float(words[5])) < 0.02f;
	}
};
static_assert(sizeof(Fsr2ConstantsCandidate) == 128);

} // namespace Libs::Graphics
