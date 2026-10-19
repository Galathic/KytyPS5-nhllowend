#pragma once
#include <cstdint>
#include <memory>

namespace Libs::Graphics {
struct GraphicContext;
struct PreparedBindings;
class RenderContext;
class CommandBuffer;

// Off-default synchronous shadow provider. It snapshots actual bound GPU
// constants/inputs, retains the guest's entire temporal chain, and resets host
// history only on inferred discontinuities (KYTY_FSR3_NHL27_HISTORY=0: every frame).
class Nhl27Fsr3 {
public:
	explicit Nhl27Fsr3(GraphicContext& graphics);
	~Nhl27Fsr3(); // RenderContext drains the scheduler before releasing this owner.
	static bool Enabled();
	// KYTY_FSR3_NHL27_HISTORY=0 restores reset-every-frame; default on.
	static bool HistoryEnabled();
	// KYTY_FSR3_NHL27_PRESENT=1 copies the FSR output over the guest's accumulation output, so the
	// game's later LUT/UI/composite passes run on it; frames FSR rejects keep the guest image.
	static bool PresentEnabled();
	void OnDispatch(RenderContext& context, CommandBuffer& command, uint64_t hash,
					const PreparedBindings& bindings);
private:
	struct Impl;
	std::unique_ptr<Impl> m;
};
} // namespace Libs::Graphics
