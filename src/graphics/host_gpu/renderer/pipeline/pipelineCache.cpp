#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/resourceMemo.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineRecord.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

bool ReadShaderOrdinaryBacking(void*, uint64_t address, std::span<uint32_t> values) {
	// Ordinary scalar loads previously dereferenced the protected guest mapping.
	// Read only the requested bytes through its alias, synchronizing actual GPU
	// writes rather than unrelated bytes sharing the protection page. Non-backed
	// pointers retain the original memcpy/fault behavior; strict reads still fail.
	if (!values.empty() && Libs::LibKernel::Memory::TryReadBufferBacking(
	        address, values.data(), values.size_bytes())) {
		return true;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

bool PeekShaderCleanSpan(void*, uint64_t address, std::span<uint32_t> values) {
	// This capability declines dirty or unavailable ranges without synchronizing
	// them. The walker then performs its original ordered scalar reads.
	return !values.empty() && Libs::LibKernel::Memory::TryReadCleanBufferBacking(
	    address, values.data(), values.size_bytes());
}

bool ReadShaderNativeSpan(void*, uint64_t address, std::span<uint32_t> values) {
	if (!GuestGpu::IsGpuThread() || values.empty() ||
	    !GuestRange{address, values.size_bytes()}.Valid()) return false;
	// This completes the original ordinary native reads; it does not certify the
	// range clean. Protected GPU-owned bytes follow existing fault/readback handling.
	// After access begins, failures keep native fault behavior rather than returning false.
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

template <bool strict, bool native = false>
bool CaptureShaderSpan(void* data, uint64_t address, std::span<uint32_t> values) {
	const auto reader = native ? ReadShaderNativeSpan : PeekShaderCleanSpan;
	if (!reader(nullptr, address, values)) return false;
	auto& record = *static_cast<PipelineRecord::Program*>(data);
	auto& reads = strict ? record.reads : record.raw_reads;
	for (size_t index = 0; index < values.size(); ++index) {
		reads.push_back({address + index * sizeof(uint32_t), values[index], 1u});
	}
	return true;
}


bool ReplayRecordedWords(void* data, uint64_t address, std::span<uint32_t> values, bool strict) {
    const auto& record = *static_cast<const PipelineRecord::Program*>(data);
    return PipelineRecord::ReplayWords(record, address, values, strict);
}
bool ReplayRawWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return ReplayRecordedWords(data, address, values, false);
}
bool ReplayCleanWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return ReplayRecordedWords(data, address, values, true);
}
bool CaptureRecordedWords(void* data, uint64_t address, std::span<uint32_t> values, bool strict,
                          bool ordinary_backing = false) {
    auto& record = *static_cast<PipelineRecord::Program*>(data);
    const bool valid = strict ? ReadShaderGuestMemory(nullptr, address, values) : true;
    if (!strict) {
        if (ordinary_backing) ReadShaderOrdinaryBacking(nullptr, address, values);
        else std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
    }
    auto& reads = strict ? record.reads : record.raw_reads;
    for (size_t index = 0; index < values.size(); ++index)
        reads.push_back({address + index * 4, valid ? values[index] : 0u, valid ? 1u : 0u});
    return valid;
}
bool CaptureRawWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return CaptureRecordedWords(data, address, values, false);
}
bool CaptureOrdinaryBackingWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return CaptureRecordedWords(data, address, values, false, true);
}
bool CaptureCleanWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return CaptureRecordedWords(data, address, values, true);
}

// NHL26 debugging: KYTY_DBG_DUMP_SHADERS=<hex>[,<hex>...] dumps just those shaders.
static bool ForceDumpShader(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		if (const char* v = std::getenv("KYTY_DBG_DUMP_SHADERS"); v != nullptr) {
			for (const char* p = v; *p != '\0';) {
				char*      end   = nullptr;
				const auto value = std::strtoull(p, &end, 16);
				if (end == p) break;
				list.push_back(value);
				p = (*end == ',') ? end + 1 : end;
			}
		}
		return list;
	}();
	return std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end();
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled() && !ForceDumpShader(shader_hash)) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled() && !ForceDumpShader(shader_hash)) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		ResourceMemo                               resource_memo;
		size_t                                     last_permutation = std::numeric_limits<size_t>::max();
		bool                                       native_plan_checked = false;
		bool                                       native_plan_allowed = false;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		// KYTY_PERF_STATS: shader permutation compile time (SPIR-V generation and module creation).
		FrameBreakdown::Timer compile_timer(FrameBreakdown::g_counters.compile_ns);
		FrameBreakdown::Add(FrameBreakdown::g_counters.compile_count);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		// NHL debugging: shaders whose CFG build failed under KYTY_DBG_SKIP_BAD_SHADERS=1.
		if (!failed_shaders.empty()) {
			if (const auto failed = failed_shaders.find(params.hash); failed != failed_shaders.end()) {
				if (failed->second++ < 4u) {
					std::printf("NHL26DBG: skipping use of shader 0x%016" PRIx64
					            " (CFG build failed)\n", params.hash);
					std::fflush(stdout);
				}
				return {};
			}
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		static const bool reuse_enabled = [] {
			const char* value = std::getenv("KYTY_PROGRAM_RESOURCE_CACHE");
			return value != nullptr && value[0] == '1' && value[1] == 0;
		}();
		++lookups;
		auto& last = last_sources[static_cast<size_t>(stage)];
		SourceEntry* entry = nullptr;
		if (reuse_enabled && last.key != nullptr && *last.key == lookup_key) {
			entry = last.source;
			++source_hits;
		} else if (auto found = programs.find(lookup_key); found != programs.end()) {
			entry = &found->second;
			last = {&found->first, entry};
		}
		const auto runtime_flags = Common::RuntimeFlags::Current();
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory = runtime_flags.shader_clean_reads
		                       ? ReadShaderOrdinaryBacking : nullptr,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .try_read_memory_span = runtime_flags.srt_scalar_batch
		                                ? (runtime_flags.shader_clean_reads
		                                       ? PeekShaderCleanSpan : ReadShaderNativeSpan) : nullptr,
		    .try_read_specialization_span = runtime_flags.srt_scalar_batch
		                                        ? PeekShaderCleanSpan : nullptr,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		if (replaying != nullptr) {
            runtime.read_memory = ReplayRawWords;
            runtime.read_specialization_memory = ReplayCleanWords;
            runtime.userdata = const_cast<PipelineRecord::Program*>(replaying);
            // Frozen PipelineRecords replay through their domain readers. Never
            // retain a live-memory capability after replacing its userdata.
            runtime.try_read_memory_span = nullptr;
            runtime.try_read_specialization_span = nullptr;
        }
        const bool memo_enabled = reuse_enabled && replaying == nullptr && !runtime_flags.resource_memo_bypass;
        const auto refresh = [&](SourceEntry& source, bool& reused) {
            const bool ok = source.resource_memo.Refresh(source.resource_plan, runtime,
                source.resources, source.specialization, memo_enabled, reused);
            if (!ok && replaying == nullptr) EXIT("Resource materialization failed\n");
            return ok;
        };
        const auto native_plan_safe = [&](const ShaderRecompiler::IR::ResourcePlan& plan) {
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				if (input_info.native_draw.Enabled()) {
					for (const auto& value: plan.value_storage) {
						if (value.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) continue;
						const auto reg = ShaderRecompiler::IR::RegIndex(value.Arg(0).ScalarRegister());
						if (std::ranges::find(input_info.native_draw.registers, reg) != input_info.native_draw.registers.end()) return false;
					}
				}
			}
			return true;
		};
		const auto check_native_plan = [&](SourceEntry& source) {
			if (reuse_enabled && source.native_plan_checked) return source.native_plan_allowed;
			const bool allowed = native_plan_safe(source.resource_plan);
			if (reuse_enabled) {
				// Native SGPR mappings belong to ProgramKey; the plan graph is immutable.
				source.native_plan_checked = true;
				source.native_plan_allowed = allowed;
			}
			return allowed;
		};
		if (entry != nullptr) {
			if (!check_native_plan(*entry)) return {};
			bool reused = false;
			if (!refresh(*entry, reused)) return {};
			resource_hits += reused;
			materializations += !reused;
			if (memo_enabled && !reused) ++resource_misses[static_cast<size_t>(entry->resource_memo.LastMiss())];
			if (reuse_enabled && entry->last_permutation < entry->permutations.size()) {
				const auto& candidate = entry->permutations[entry->last_permutation];
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword == ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) &&
				    candidate.specialization == entry->specialization) {
					++permutation_hits;
					input_info.stage = {.program = &candidate.program, .resources = &entry->resources};
					layout.AdvancePushData(push_data_cursor);
					PrintStats(reuse_enabled);
					return candidate.handle;
				}
			}
			if (const auto permutation = std::ranges::find_if(
			        entry->permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == entry->specialization;
			        });
			    permutation != entry->permutations.end()) {
				entry->last_permutation = static_cast<size_t>(permutation - entry->permutations.begin());
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				PrintStats(reuse_enabled);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (translated.build_failed) {
			// NHL debugging: KYTY_DBG_SKIP_BAD_SHADERS=1 turned the CFG failure into a skipped draw.
			failed_shaders.emplace(params.hash, 0u);
			return {};
		}
		if (entry == nullptr) {
			const auto found = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			entry = &found->second;
			last = {&found->first, entry};
			if (!check_native_plan(*entry)) return {};
			bool reused = false;
			if (!refresh(*entry, reused)) return {};
			++materializations;
		}

        PipelineRecord::Program recorded;
        bool recording = record != nullptr && record->IsOpen() && replaying == nullptr &&
                         PipelineRecord::IsRecordableStage(stage);
        if (recording) {
            recorded.stage = stage; recorded.hash = params.hash;
            recorded.shader_base = params.Base(); recorded.push_data_cursor = push_data_cursor;
            recorded.code.assign(params.code.begin(), params.code.end());
            recorded.back_code.assign(params.back_code.begin(), params.back_code.end());
            recorded.user_data.assign(user_data.begin(), user_data.end());
            if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) recorded.vertex = input_info;
            else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) recorded.pixel = input_info;
            else recorded.compute = input_info;
            auto capture_runtime = runtime;
            capture_runtime.userdata = &recorded;
            capture_runtime.read_memory = runtime.read_memory == ReadShaderOrdinaryBacking
                ? CaptureOrdinaryBackingWords : CaptureRawWords;
            capture_runtime.read_specialization_memory = CaptureCleanWords;
            capture_runtime.try_read_memory_span = runtime.try_read_memory_span != nullptr
                ? (runtime.try_read_memory_span == ReadShaderNativeSpan
                       ? CaptureShaderSpan<false, true> : CaptureShaderSpan<false>) : nullptr;
            capture_runtime.try_read_specialization_span = runtime.try_read_specialization_span != nullptr
                ? CaptureShaderSpan<true> : nullptr;
            ShaderRecompiler::IR::ResourceSnapshot captured;
            ShaderRecompiler::IR::ResourceSpecialization captured_specialization;
            recording = ShaderRecompiler::IR::MaterializeResources(entry->resource_plan, capture_runtime,
                                                                  captured, captured_specialization) &&
                        captured_specialization == entry->specialization;
        }
        entry->permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->specialization, push_data_cursor));
		entry->last_permutation = entry->permutations.size() - 1;
		const auto& permutation = entry->permutations.back();
        if (recording) ordinals[permutation.handle.id] = record->AppendProgram(recorded);
		input_info.stage = {.program = &permutation.program, .resources = &entry->resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}
	void PrintStats(bool enabled) {
		static const bool stats = [] {
			const char* value = std::getenv("KYTY_PERF_STATS");
			return value != nullptr && value[0] == '1';
		}();
		if (!stats || (lookups & 16383u) != 0) return;
		const auto now = std::chrono::steady_clock::now();
		if (now - last_stats < std::chrono::seconds(5)) return;
		std::printf("NHL27PROGRAM: cache=%d lookups=%llu source_hits=%llu resource_hits=%llu materialize=%llu perm_hits=%llu\n",
		    enabled, static_cast<unsigned long long>(lookups), static_cast<unsigned long long>(source_hits),
		    static_cast<unsigned long long>(resource_hits), static_cast<unsigned long long>(materializations),
		    static_cast<unsigned long long>(permutation_hits));
		if (enabled) {
			std::printf("NHL27PROGRAMMISS: cold=%llu base=%llu regs=%llu groups=%llu read_failed=%llu memory_changed=%llu\n",
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::Cold)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::ShaderBase)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::Registers)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::Workgroups)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::ReadFailed)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::MemoryChanged)]));
			const auto reuse_trace = GetResourceReuseTraceStats();
			if (reuse_trace.checked != 0 || reuse_trace.evictions != 0) {
				std::printf("NHL27REUSETRACE: evictions=%llu checked=%llu register_recur=%llu distance16=%llu distance32=%llu distance64=%llu distance128=%llu\n",
				    static_cast<unsigned long long>(reuse_trace.evictions), static_cast<unsigned long long>(reuse_trace.checked),
				    static_cast<unsigned long long>(reuse_trace.matched), static_cast<unsigned long long>(reuse_trace.distance16),
				    static_cast<unsigned long long>(reuse_trace.distance32), static_cast<unsigned long long>(reuse_trace.distance64),
				    static_cast<unsigned long long>(reuse_trace.distance128));
			}
			const auto tape = ShaderRecompiler::IR::GetSrtTapeStats();
			const auto filter = GetResourceHashFilterStats();
			if (filter.eligible != 0) {
				std::printf("NHL27RESOURCEFILTER: eligible=%llu rejected=%llu hash_match=%llu\n",
				    static_cast<unsigned long long>(filter.eligible),
				    static_cast<unsigned long long>(filter.rejected),
				    static_cast<unsigned long long>(filter.matched));
			}
			std::printf("NHL27SRTTAPE: runs=%llu failed=%llu walker=%llu built=%llu unsupported=%llu ops=%llu cfg_roots=%llu cfg_fallback=%llu cfg_checked=%llu cfg_matched=%llu cfg_mismatch=%llu cfg_inconclusive=%llu\n",
			    static_cast<unsigned long long>(tape.runs),
			    static_cast<unsigned long long>(tape.failed),
			    static_cast<unsigned long long>(tape.walker),
			    static_cast<unsigned long long>(tape.built),
			    static_cast<unsigned long long>(tape.unsupported),
			    static_cast<unsigned long long>(tape.ops),
			    static_cast<unsigned long long>(tape.cfg_runs),
			    static_cast<unsigned long long>(tape.cfg_fallback),
			    static_cast<unsigned long long>(tape.cfg_checked),
			    static_cast<unsigned long long>(tape.cfg_matched),
			    static_cast<unsigned long long>(tape.cfg_mismatch),
			    static_cast<unsigned long long>(tape.cfg_inconclusive));
		}
		last_stats = now;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	struct LastSource { const ProgramKey* key = nullptr; SourceEntry* source = nullptr; };
	std::array<LastSource, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> last_sources {};
	uint64_t lookups = 0, source_hits = 0, resource_hits = 0, materializations = 0, permutation_hits = 0;
	std::array<uint64_t, static_cast<size_t>(ResourceMemo::Miss::Count)> resource_misses {};
	std::chrono::steady_clock::time_point last_stats = std::chrono::steady_clock::now();
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
    PipelineRecord::RecordFile* record = nullptr;
    const PipelineRecord::Program* replaying = nullptr;
    std::unordered_map<uint64_t, uint32_t> ordinals;
    uint32_t Ordinal(uint64_t id) const {
        const auto found = ordinals.find(id);
        return found == ordinals.end() ? PipelineRecord::NoProgram : found->second;
    }
	// NHL debugging: shader hash -> number of skip warnings printed.
	std::unordered_map<uint64_t, uint32_t>                      failed_shaders;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	InitializeShaderRecord();
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	const char* allow_dirty = std::getenv("KYTY_PIPELINE_CACHE_DIRTY");
	if (git_hash.ends_with("-dirty") && !(allow_dirty != nullptr && allow_dirty[0] == '1')) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= 1024ull * 1024ull * 1024ull) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::AutoSave() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_unsaved_pipelines++;
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	if (m_last_save_ms == 0) {
		m_last_save_ms = now;
	}
	if (now - m_last_save_ms >= 30000 || m_unsaved_pipelines >= 128) {
		Save();
		m_unsaved_pipelines = 0;
		m_last_save_ms      = now;
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr) {
		return;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
}

void PipelineCache::InitializeShaderRecord() {
	const char* flag = std::getenv("KYTY_SHADER_PREGEN");
	if (!Config::PreGenEnabled() || (flag != nullptr && flag[0] == '0')) {
		PipelineCacheLog("Shader warm-up: disabled (--pre-gen false)");
		return;
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	const auto path = std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
	PipelineRecord::Records records;
	m_record = std::make_unique<PipelineRecord::RecordFile>();
	if (!m_record->Open(path, records)) {
		PipelineCacheLog("Shader warm-up: cannot write {}; new shaders are not recorded",
		                 Common::PathToString(path));
	}
	m_program_cache->record = m_record.get();
	WarmUp(records);
}


void PipelineCache::WarmUp(PipelineRecord::Records& records) {
    if (records.programs.empty()) {
        PipelineCacheLog("Shader warm-up: no records yet; recording during play");
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    // Shared ProgramCache/compiler state is serialized; do not race translation or insertion.
    std::vector<ShaderProgram> handles(records.programs.size());
    std::vector<ShaderRecompiler::IR::ResourceSnapshot> snapshots(records.programs.size());
    for (size_t index = 0; index < records.programs.size(); ++index) {
        auto& source = records.programs[index];
        if (source.user_data.size() > ShaderParams{}.user_data.size()) continue;
        if (source.stage == ShaderType::Vertex && source.vertex.native_draw.Enabled() &&
            !m_graphics.native_draw_parameters_enabled) continue;
        if (source.stage == ShaderType::Pixel) {
            const char* quad = std::getenv("KYTY_PIXEL_QUAD_CORRECT");
            source.pixel.pixel_quad_correct = m_graphics.pixel_quad_supported &&
                quad != nullptr && quad[0] == '1';
        }
        if (source.stage == ShaderType::Compute) {
            source.compute.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
            source.compute.lds_storage = source.compute.lds_size_dwords * 4u >
                m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
        }
        ShaderParams params;
        params.code = source.code; params.back_code = source.back_code;
        params.base = source.shader_base; params.hash = source.hash;
        params.user_data_count = static_cast<uint32_t>(source.user_data.size());
        std::copy(source.user_data.begin(), source.user_data.end(), params.user_data.begin());
        auto cursor = source.push_data_cursor;
        m_program_cache->replaying = &source;
        switch (source.stage) {
            case ShaderType::Vertex: handles[index] = m_program_cache->Get(params, source.vertex, cursor); break;
            case ShaderType::Pixel: handles[index] = m_program_cache->Get(params, source.pixel, cursor); break;
            case ShaderType::Compute: handles[index] = m_program_cache->Get(params, source.compute, cursor); break;
            default: break;
        }
        m_program_cache->replaying = nullptr;
        if (handles[index]) {
            const ShaderStageRuntime* runtime = nullptr;
            switch (source.stage) {
                case ShaderType::Vertex: runtime = &source.vertex.stage; break;
                case ShaderType::Pixel: runtime = &source.pixel.stage; break;
                case ShaderType::Compute: runtime = &source.compute.stage; break;
                default: break;
            }
            if (runtime != nullptr && runtime->resources != nullptr)
                snapshots[index] = *runtime->resources;
        }
        if (handles[index]) m_program_cache->ordinals[handles[index].id] = static_cast<uint32_t>(index);
    }
    std::unordered_map<uint64_t, const ShaderRecompiler::IR::CompiledShaderInfo*> compiled;
    for (const auto& [key, entry]: m_program_cache->programs)
        for (const auto& permutation: entry.permutations)
            compiled[permutation.handle.id] = &permutation.program;
    size_t graphics_count = 0, compute_count = 0;
    for (auto& record: records.graphics) {
        const auto vertex = handles[record.vertex_program];
        const bool ps_active = record.pixel_program != PipelineRecord::NoProgram;
        const auto pixel = ps_active ? handles[record.pixel_program] : ShaderProgram{};
        if (!vertex || (ps_active && !pixel)) continue;
        GraphicsPipelineKey key;
        key.rendering = record.rendering; key.vertex_shader_ids[0] = vertex.id;
        key.ps_shader_id = pixel.id; key.vertex_input = record.vertex_input;
        key.static_params = record.static_params;
        if (m_graphics_pipelines.contains(key)) continue;
        GraphicsPrograms programs; programs.vertex[0] = vertex; programs.pixel = pixel;
        record.vertex.stage.program = compiled.at(vertex.id);
        record.vertex.stage.resources = &snapshots[record.vertex_program];
        if (ps_active) {
            record.pixel.stage.program = compiled.at(pixel.id);
            record.pixel.stage.resources = &snapshots[record.pixel_program];
        }
        auto pipeline = std::make_unique<Pipeline>();
        CreatePipelineInternal(m_graphics, *pipeline, key.rendering, key.vertex_input,
            std::span<const ShaderVertexInputInfo>(&record.vertex, 1),
            ps_active ? &record.pixel : nullptr, programs, key.static_params, m_driver_cache);
        m_graphics_pipelines.emplace(std::move(key), std::move(pipeline)); ++graphics_count;
    }
    for (auto& record: records.compute) {
        const auto program = handles[record.program];
        if (!program || m_compute_pipelines.contains(program.id)) continue;
        record.compute.stage.program = compiled.at(program.id);
        record.compute.stage.resources = &snapshots[record.program];
        record.compute.host_subgroup_size = records.programs[record.program].compute.host_subgroup_size;
        record.compute.lds_storage = records.programs[record.program].compute.lds_storage;
        auto pipeline = std::make_unique<Pipeline>();
        CreatePipelineInternal(m_graphics, *pipeline, record.compute, program.module, m_driver_cache);
        m_compute_pipelines.emplace(program.id, std::move(pipeline)); ++compute_count;
    }
    Save();
    PipelineCacheLog("Shader warm-up: {} programs, {} graphics and {} compute pipelines in {:.1f}s",
        std::ranges::count_if(handles, [](const auto& handle) { return bool(handle); }),
        graphics_count, compute_count,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

void PipelineCache::RecordGraphicsPipeline(const GraphicsPipelineKey&             key,
                                           std::span<const ShaderVertexInputInfo> vertex_info,
                                           const ShaderPixelInputInfo*            ps_input_info) {
	if (m_record == nullptr || !m_record->IsOpen() || vertex_info.size() != 1 ||
	    vertex_info[0].logical_stage != ShaderType::Vertex) {
		return;
	}
	PipelineRecord::GraphicsPipeline record;
	record.vertex_program = m_program_cache->Ordinal(key.vertex_shader_ids[0]);
	record.pixel_program  = ps_input_info != nullptr ? m_program_cache->Ordinal(key.ps_shader_id)
	                                                 : PipelineRecord::NoProgram;
	if (record.vertex_program == PipelineRecord::NoProgram ||
	    (ps_input_info != nullptr && record.pixel_program == PipelineRecord::NoProgram)) {
		return;
	}
	record.rendering     = key.rendering;
	record.vertex_input  = key.vertex_input;
	record.static_params = key.static_params;
	record.vertex        = vertex_info[0];
	if (ps_input_info != nullptr) {
		record.pixel = *ps_input_info;
	}
	m_record->AppendGraphicsPipeline(record);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info, const NativeDrawShaderInfo& native_draw) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_graphics_programs_ns,
	                                 FrameBreakdown::DetailEnabled());
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	if (vertex_info[0].logical_stage == ShaderType::Vertex) vertex_info[0].native_draw = native_draw;
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		static const bool quad_enabled = [] {
			const char* flag = std::getenv("KYTY_PIXEL_QUAD_CORRECT");
			return flag != nullptr && flag[0] == '1';
		}();
		pixel_info.pixel_quad_correct = quad_enabled && m_graphics.pixel_quad_supported;
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	bool              build_failed = false;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
		build_failed |= !result.pixel;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		build_failed |= !result.vertex[i];
	}
	// NHL debugging: an empty vertex[0] tells the draw path to skip the draw.
	return build_failed ? GraphicsPrograms {} : result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_compute_program_ns,
	                                 FrameBreakdown::DetailEnabled());
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	RecordGraphicsPipeline(key, vertex_info, ps_input_info);
	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	AutoSave();

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

if (m_record != nullptr && m_record->IsOpen() && m_program_cache->replaying == nullptr) {
        const auto ordinal = m_program_cache->Ordinal(compute_program.id);
        if (ordinal != PipelineRecord::NoProgram) m_record->AppendComputePipeline({ordinal, input_info});
    }
    auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	AutoSave();

	return *iter->second;
}
} // namespace Libs::Graphics
