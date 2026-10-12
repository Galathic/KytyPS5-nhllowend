#include "common/experiments.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace Common::Experiments {

namespace {

bool EnvOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] == '1';
}

uint32_t EnvironmentMask() {
	uint32_t mask = 0;
	if (EnvOn("KYTY_GUEST_SYNC")) mask |= GuestSync;
	if (EnvOn("KYTY_SKIP_REDUNDANT_BINDS")) mask |= SkipRedundantBinds;
	if (EnvOn("KYTY_DEDUP_BARRIERS")) mask |= DedupBarriers;
	if (EnvOn("KYTY_CPU_COPY_FORWARD")) mask |= CpuCopyForward;
	if (EnvOn("KYTY_FALSE_SHARE_READS")) mask |= FalseShareReads;
	if (EnvOn("KYTY_SRT_BATCH_READS")) mask |= SrtBatchReads;
	if (EnvOn("KYTY_RESOURCE_MULTI_CACHE")) mask |= ResourceMultiCache;
	if (EnvOn("KYTY_BACKING_READ_CACHE")) mask |= BackingReadCache;
	if (EnvOn("KYTY_DIRECT_SRT_READS")) mask |= DirectSrtReads;
	if (EnvOn("KYTY_TEXTURE_DESC_CACHE")) mask |= TextureDescCache;
	return mask;
}

struct State {
	std::atomic<uint32_t> mask {0};

	State() {
		const uint32_t defaults = EnvironmentMask();
		mask.store(defaults, std::memory_order_relaxed);
		const char* path_env = std::getenv("KYTY_EXPERIMENT_FILE");
		if (path_env == nullptr || path_env[0] == '\0') {
			return;
		}
		std::thread([this, path = std::string(path_env), defaults] {
			uint32_t last = defaults;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				uint32_t value = defaults;
				if (std::FILE* f = std::fopen(path.c_str(), "rb"); f != nullptr) {
					unsigned parsed = 0;
					if (std::fscanf(f, "%u", &parsed) == 1) {
						value = parsed;
					}
					std::fclose(f);
				}
				if (value != last) {
					mask.store(value, std::memory_order_relaxed);
					std::printf("Experiments: mask %u -> %u\n", last, value);
					std::fflush(stdout);
					last = value;
				}
			}
		}).detach();
	}
};

State& GetState() {
	static State state;
	return state;
}

} // namespace

bool Enabled(Bit bit) noexcept {
	return (GetState().mask.load(std::memory_order_relaxed) & bit) != 0;
}

uint32_t Mask() noexcept {
	return GetState().mask.load(std::memory_order_relaxed);
}

} // namespace Common::Experiments
