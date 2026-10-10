#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINERECORD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINERECORD_H_

#include "common/common.h"
#include "common/file.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <span>
#include <vector>

namespace Libs::Graphics::PipelineRecord {

// Shader warm-up records the inputs of every program and pipeline a game makes Kyty compile, so a
// later boot can compile them before the guest starts. Inputs are stored rather than SPIR-V: a newer
// recompiler turns them into its own output, so a record does not go stale when the emulator changes.

inline constexpr uint32_t NoProgram = UINT32_MAX;

struct MemoryRead {
	uint64_t address = 0;
	uint32_t value   = 0;
	uint32_t valid   = 0;
};

struct Program {
	ShaderType            stage            = ShaderType::Unknown;
	uint64_t              hash             = 0;
	uint64_t              shader_base      = 0;
	uint32_t              push_data_cursor = 0;
	std::vector<uint32_t> code;
	std::vector<uint32_t> back_code;
	std::vector<uint32_t> user_data;
	// Guest words read while deriving the resource specialization, so replay derives the same one:
	// `reads` come through the clean specialization reader, `raw_reads` straight from guest memory
	// (SrtRuntime::read_memory). Warm-up runs before the game is loaded, so both must be recorded.
	std::vector<MemoryRead> reads;
	std::vector<MemoryRead> raw_reads;
	// Only the member matching `stage` is recorded.
	ShaderVertexInputInfo  vertex;
	ShaderPixelInputInfo   pixel;
	ShaderComputeInputInfo compute;
};

// Missing or invalid reads fail. Replay must never dereference guest addresses.
inline bool ReplayWords(const Program& program, uint64_t address,
                        std::span<uint32_t> values, bool strict) {
	const auto& reads = strict ? program.reads : program.raw_reads;
	for (size_t index = 0; index < values.size(); ++index) {
		const auto found = std::ranges::find(reads, address + index * 4, &MemoryRead::address);
		if (found == reads.end() || !found->valid) return false;
		values[index] = found->value;
	}
	return true;
}

// Program fields are ordinals of Program records in the same file.
struct GraphicsPipeline {
	uint32_t                 vertex_program = NoProgram;
	uint32_t                 pixel_program  = NoProgram;
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;
	ShaderVertexInputInfo    vertex;
	ShaderPixelInputInfo     pixel;
};

struct ComputePipeline {
	uint32_t               program = NoProgram;
	ShaderComputeInputInfo compute;
};

struct Records {
	std::vector<Program>          programs;
	std::vector<GraphicsPipeline> graphics;
	std::vector<ComputePipeline>  compute;
};

// Tessellation and mesh programs depend on more than one guest binary and are not recorded.
[[nodiscard]] inline bool IsRecordableStage(ShaderType stage) {
	return stage == ShaderType::Vertex || stage == ShaderType::Pixel || stage == ShaderType::Compute;
}

// Append-only file of checksummed records. Loading keeps every complete record and drops a torn
// tail, so a crash while appending loses at most the record being written.
class RecordFile final {
public:
	RecordFile()  = default;
	~RecordFile() = default;
	KYTY_CLASS_NO_COPY(RecordFile);

	// Loads the existing records into `records` and opens the file for appending. Returns false when
	// new records cannot be written; the loaded records are still usable.
	bool Open(const std::filesystem::path& path, Records& records);
	[[nodiscard]] bool IsOpen() const { return m_open; }

	// Returns the ordinal of the new record, or NoProgram if it was not written.
	uint32_t AppendProgram(const Program& program);
	void     AppendGraphicsPipeline(const GraphicsPipeline& pipeline);
	void     AppendComputePipeline(const ComputePipeline& pipeline);

private:
	void Append(uint32_t type, const std::vector<uint8_t>& payload);

	Common::File m_file;
	uint64_t     m_size          = 0;
	uint32_t     m_program_count = 0;
	bool         m_open          = false;
};

} // namespace Libs::Graphics::PipelineRecord

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINERECORD_H_
