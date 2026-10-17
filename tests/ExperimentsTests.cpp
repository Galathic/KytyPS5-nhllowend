#include "common/experiments.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace {
void SetEnvironment(const char* name, const char* value) {
#if defined(_WIN32)
	_putenv_s(name, value);
#else
	setenv(name, value, 1);
#endif
}

bool WaitForMask(uint64_t expected) {
	const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
	do {
		if (Common::Experiments::Mask() == expected) return true;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	} while (std::chrono::steady_clock::now() < until);
	return false;
}
} // namespace

int main() {
	using namespace Common::Experiments;
	const auto file = std::filesystem::temp_directory_path() /
	                  ("kyty-experiments-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	SetEnvironment("KYTY_EXPERIMENT_FILE", file.string().c_str());
	SetEnvironment("KYTY_NATIVE_FP16", "1");
	SetEnvironment("KYTY_PROTECT_SHARED_LOCK", "1");
	SetEnvironment("KYTY_FAULT_PAGE_READS", "0");
	SetEnvironment("KYTY_CLOCK_WAIT_PREDICT", "0");
	const auto defaults = Mask();
	if (!Enabled(NativeFp16) || !Enabled(ProtectSharedLock) || Enabled(FaultPageReads) || Enabled(ClockWaitPredict)) {
		std::fprintf(stderr, "Independent environment switches collided\n");
		return 1;
	}
	// Exercise decimal file parsing above bit 31, while retaining the legacy A/B bit and a shader bit.
	const uint64_t masks[] = {uint64_t(AbLegacy) | NativeFp16 | ProtectSharedLock | ClockWaitPredict,
	                          uint64_t(GpuColorClear) | FalseShareAnyThread | CpCleanRegReads | CpSkipIdleSpin | SpeculateDirtyReads,
	                          0};
	for (const auto expected: masks) {
		{
			std::ofstream output(file, std::ios::trunc);
			output << expected << '\n';
			if (!output) return 1;
		}
		if (!WaitForMask(expected)) {
			std::filesystem::remove(file);
			std::fprintf(stderr, "Live experiment mask was truncated or aliased\n");
			return 1;
		}
	}
	std::filesystem::remove(file);
	if (!WaitForMask(defaults)) {
		std::fprintf(stderr, "Removing the experiment file did not restore environment defaults\n");
		return 1;
	}
	std::puts("Experiment environment and 64-bit live mask tests passed");
	return 0;
}
