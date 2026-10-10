#include "graphics/host_gpu/renderer/pipeline/pipelineRecord.h"

#include "common/assert.h"

#include <array>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics::PipelineRecord {

namespace {

constexpr uint32_t FileMagic = 0x5250594bu; // "KYPR"
// Bump whenever a recorded structure or one of the visitors below changes.
constexpr uint32_t FileVersion = 3;
constexpr uint64_t HeaderSize  = 2 * sizeof(uint32_t);
constexpr uint64_t FrameSize   = 2 * sizeof(uint32_t) + sizeof(uint64_t);
constexpr uint64_t MaxFileSize = 1024ull * 1024ull * 1024ull;
constexpr uint32_t MaxElements = 1u << 22u;

enum class RecordType : uint32_t { Program = 1, GraphicsPipeline = 2, ComputePipeline = 3 };

class Writer {
public:
	template <typename T>
	void Field(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		Append(&value, sizeof(T));
	}

	template <typename T>
	void Array(const std::vector<T>& values) {
		static_assert(std::is_trivially_copyable_v<T>);
		EXIT_IF(values.size() > MaxElements);
		Field(static_cast<uint32_t>(values.size()));
		Append(values.data(), values.size() * sizeof(T));
	}

	[[nodiscard]] const std::vector<uint8_t>& Bytes() const { return m_bytes; }

private:
	void Append(const void* data, size_t size) {
		if (size == 0) {
			return;
		}
		const auto* bytes = static_cast<const uint8_t*>(data);
		m_bytes.insert(m_bytes.end(), bytes, bytes + size);
	}

	std::vector<uint8_t> m_bytes;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> bytes): m_bytes(bytes) {}

	template <typename T>
	void Field(T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (!m_ok || m_bytes.size() - m_offset < sizeof(T)) {
			m_ok = false;
			return;
		}
		std::memcpy(&value, m_bytes.data() + m_offset, sizeof(T));
		m_offset += sizeof(T);
	}

	template <typename T>
	void Array(std::vector<T>& values) {
		static_assert(std::is_trivially_copyable_v<T>);
		uint32_t count = 0;
		Field(count);
		if (!m_ok || count > MaxElements || (m_bytes.size() - m_offset) / sizeof(T) < count) {
			m_ok = false;
			return;
		}
		values.resize(count);
		if (count != 0) {
			std::memcpy(values.data(), m_bytes.data() + m_offset, count * sizeof(T));
		}
		m_offset += count * sizeof(T);
	}

	[[nodiscard]] bool Complete() const { return m_ok && m_offset == m_bytes.size(); }

private:
	std::span<const uint8_t> m_bytes;
	size_t                   m_offset = 0;
	bool                     m_ok     = true;
};

// The visitors list every field except the runtime `stage` member. Keep them in step with shader.h
// and pipelineCache.h, and bump FileVersion when they change. Small structs with padding are
// written field by field so identical inputs produce identical bytes.
template <typename Archive, typename Info>
void VisitVertex(Archive& archive, Info& info) {
	archive.Field(info.resources);
	archive.Field(info.resources_dst);
	archive.Field(info.buffers);
	archive.Field(info.logical_stage);
	archive.Field(info.wave_size);
	archive.Field(info.native_draw.registers);
	archive.Field(info.native_draw.indexed);
	archive.Field(info.resources_num);
	archive.Field(info.fetch_attrib_reg);
	archive.Field(info.fetch_buffer_reg);
	archive.Field(info.buffers_num);
	archive.Field(info.scratch_size_dwords);
	archive.Field(info.pa_cl_vs_out_cntl);
	archive.Field(info.clip_space.scale);
	archive.Field(info.clip_space.offset);
	archive.Field(info.clip_space.half_extent);
	archive.Field(info.clip_space.enabled);
	archive.Field(info.mesh);
	archive.Field(info.tess);
	archive.Field(info.fetch_external);
	archive.Field(info.fetch_embedded);
}

template <typename Archive, typename Info>
void VisitPixel(Archive& archive, Info& info) {
	archive.Field(info.pixel_quad_correct);
	archive.Field(info.interpolator_settings);
	archive.Field(info.input_num);
	archive.Field(info.wave_size);
	archive.Field(info.ps_system_input_base);
	archive.Field(info.custom_interpolation_mask);
	archive.Field(info.ps_perspective_center_vgpr);
	archive.Field(info.target_output_mode);
	archive.Field(info.target_shader_mask);
	archive.Field(info.target_export_mapping);
	archive.Field(info.scratch_size_dwords);
	archive.Field(info.ps_pos_x);
	archive.Field(info.ps_pos_y);
	archive.Field(info.ps_pos_z);
	archive.Field(info.ps_pos_w);
	archive.Field(info.ps_front_face);
	archive.Field(info.ps_ancillary);
	archive.Field(info.ps_no_perspective);
	archive.Field(info.ps_pixel_kill_enable);
	archive.Field(info.ps_depth_export_enable);
	archive.Field(info.ps_sample_mask_export_enable);
	archive.Field(info.ps_sample_shading);
	archive.Field(info.ps_early_z);
	archive.Field(info.ps_execute_on_noop);
	archive.Field(info.dual_source_blending);
	archive.Field(info.alpha_blend_source);
	archive.Field(info.ps_front_face_all_bits);
	archive.Field(info.ps_perspective_sample_vgpr);
	archive.Field(info.ps_perspective_centroid_vgpr);
}

template <typename Archive, typename Info>
void VisitCompute(Archive& archive, Info& info) {
	archive.Field(info.threads_num);
	archive.Field(info.lds_size_dwords);
	archive.Field(info.scratch_size_dwords);
	archive.Field(info.host_subgroup_size);
	archive.Field(info.wave_size);
	archive.Field(info.dispatch_threads_num);
	archive.Field(info.group_id);
	archive.Field(info.dispatch_thread_dimensions);
	archive.Field(info.thread_ids_num);
	archive.Field(info.workgroup_register);
	archive.Field(info.tg_size_en);
	archive.Field(info.float_mode);
	archive.Field(info.workgroup_counts);
	archive.Field(info.lds_storage);
}

template <typename Archive, typename Record>
bool VisitProgram(Archive& archive, Record& program) {
	archive.Field(program.stage);
	archive.Field(program.hash);
	archive.Field(program.shader_base);
	archive.Field(program.push_data_cursor);
	archive.Array(program.code);
	archive.Array(program.back_code);
	archive.Array(program.user_data);
	archive.Array(program.reads);
	archive.Array(program.raw_reads);
	switch (program.stage) {
		case ShaderType::Vertex: VisitVertex(archive, program.vertex); return true;
		case ShaderType::Pixel: VisitPixel(archive, program.pixel); return true;
		case ShaderType::Compute: VisitCompute(archive, program.compute); return true;
		default: return false;
	}
}

template <typename Archive, typename Record>
void VisitGraphicsPipeline(Archive& archive, Record& pipeline) {
	archive.Field(pipeline.vertex_program);
	archive.Field(pipeline.pixel_program);
	archive.Field(pipeline.rendering.color_formats);
	archive.Field(pipeline.rendering.depth_format);
	archive.Field(pipeline.rendering.stencil_format);
	archive.Field(pipeline.rendering.color_count);
	for (auto& binding: pipeline.vertex_input.bindings) {
		archive.Field(binding.stride);
		archive.Field(binding.instance);
	}
	for (auto& attribute: pipeline.vertex_input.attributes) {
		archive.Field(attribute.offset);
		archive.Field(attribute.binding);
	}
	archive.Field(pipeline.vertex_input.binding_count);
	archive.Field(pipeline.vertex_input.attribute_count);
	archive.Field(pipeline.static_params);
	VisitVertex(archive, pipeline.vertex);
	if (pipeline.pixel_program != NoProgram) {
		VisitPixel(archive, pipeline.pixel);
	}
}

template <typename Archive, typename Record>
void VisitComputePipeline(Archive& archive, Record& pipeline) {
	archive.Field(pipeline.program);
	VisitCompute(archive, pipeline.compute);
}

bool Decode(RecordType type, std::span<const uint8_t> payload, Records& records) {
	Reader     reader(payload);
	const auto program_count = records.programs.size();
	switch (type) {
		case RecordType::Program: {
			Program program;
			if (!VisitProgram(reader, program) || !reader.Complete()) {
				return false;
			}
			records.programs.push_back(std::move(program));
			return true;
		}
		case RecordType::GraphicsPipeline: {
			GraphicsPipeline pipeline;
			VisitGraphicsPipeline(reader, pipeline);
			if (!reader.Complete() || pipeline.vertex_program >= program_count ||
			    (pipeline.pixel_program != NoProgram && pipeline.pixel_program >= program_count)) {
				return false;
			}
			records.graphics.push_back(std::move(pipeline));
			return true;
		}
		case RecordType::ComputePipeline: {
			ComputePipeline pipeline;
			VisitComputePipeline(reader, pipeline);
			if (!reader.Complete() || pipeline.program >= program_count) {
				return false;
			}
			records.compute.push_back(std::move(pipeline));
			return true;
		}
	}
	return false;
}

uint32_t LoadU32(const std::vector<uint8_t>& bytes, uint64_t offset) {
	uint32_t value = 0;
	std::memcpy(&value, bytes.data() + offset, sizeof(value));
	return value;
}

} // namespace

bool RecordFile::Open(const std::filesystem::path& path, Records& records) {
	EXIT_IF(m_open);
	records = {};

	std::vector<uint8_t> bytes;
	if (Common::File::IsFileExisting(path)) {
		Common::File file(path, Common::File::Mode::Read);
		const uint64_t size = file.IsInvalid() ? 0 : file.Size();
		if (size >= HeaderSize && size <= MaxFileSize) {
			bytes.resize(size);
			uint32_t read = 0;
			file.Read(bytes.data(), static_cast<uint32_t>(size), &read);
			if (read != size) {
				bytes.clear();
			}
		}
		file.Close();
	}

	uint64_t valid_end = 0;
	if (bytes.size() >= HeaderSize && LoadU32(bytes, 0) == FileMagic &&
	    LoadU32(bytes, sizeof(uint32_t)) == FileVersion) {
		valid_end = HeaderSize;
		while (bytes.size() - valid_end >= FrameSize) {
			const uint32_t type = LoadU32(bytes, valid_end);
			const uint32_t size = LoadU32(bytes, valid_end + sizeof(uint32_t));
			uint64_t       checksum = 0;
			std::memcpy(&checksum, bytes.data() + valid_end + 2 * sizeof(uint32_t), sizeof(checksum));
			const uint64_t payload_offset = valid_end + FrameSize;
			if (bytes.size() - payload_offset < size) {
				break;
			}
			const std::span<const uint8_t> payload(bytes.data() + payload_offset, size);
			if (XXH3_64bits(payload.data(), payload.size()) != checksum ||
			    !Decode(static_cast<RecordType>(type), payload, records)) {
				break;
			}
			valid_end = payload_offset + size;
		}
	}

	if (valid_end >= HeaderSize) {
		// Keep the valid prefix and continue after it.
		if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
			if ((valid_end == bytes.size() || m_file.Truncate(valid_end)) && m_file.Seek(valid_end)) {
				m_open = true;
			} else {
				m_file.Close();
			}
		}
	} else if (Common::File::CreateDirectories(path.parent_path()) && m_file.Create(path)) {
		// Missing, unreadable, or written by another format version: start a new file.
		const std::array<uint32_t, 2> header {FileMagic, FileVersion};
		uint32_t                      written = 0;
		m_file.Write(header.data(), static_cast<uint32_t>(sizeof(header)), &written);
		if (written == sizeof(header)) {
			valid_end = HeaderSize;
			m_open    = true;
		} else {
			m_file.Close();
		}
	}
	m_size          = valid_end;
	m_program_count = static_cast<uint32_t>(records.programs.size());
	return m_open;
}

void RecordFile::Append(uint32_t type, const std::vector<uint8_t>& payload) {
	if (!m_open) {
		return;
	}
	if (m_size + FrameSize + payload.size() > MaxFileSize) {
		m_open = false;
		m_file.Close();
		return;
	}
	const auto     size     = static_cast<uint32_t>(payload.size());
	const uint64_t checksum = XXH3_64bits(payload.data(), payload.size());
	std::array<uint8_t, FrameSize> frame {};
	std::memcpy(frame.data(), &type, sizeof(type));
	std::memcpy(frame.data() + sizeof(type), &size, sizeof(size));
	std::memcpy(frame.data() + 2 * sizeof(uint32_t), &checksum, sizeof(checksum));
	uint32_t frame_written   = 0;
	uint32_t payload_written = 0;
	m_file.Write(frame.data(), static_cast<uint32_t>(frame.size()), &frame_written);
	if (size != 0) {
		m_file.Write(payload.data(), size, &payload_written);
	}
	if (frame_written != frame.size() || payload_written != size || !m_file.Flush()) {
		// The torn record is dropped on the next load; nothing after it would be readable.
		m_open = false;
		m_file.Close();
		return;
	}
	m_size += FrameSize + size;
}

uint32_t RecordFile::AppendProgram(const Program& program) {
	if (!m_open) {
		return NoProgram;
	}
	Writer writer;
	EXIT_IF(!VisitProgram(writer, program));
	Append(static_cast<uint32_t>(RecordType::Program), writer.Bytes());
	return m_open ? m_program_count++ : NoProgram;
}

void RecordFile::AppendGraphicsPipeline(const GraphicsPipeline& pipeline) {
	EXIT_IF(pipeline.vertex_program >= m_program_count ||
	        (pipeline.pixel_program != NoProgram && pipeline.pixel_program >= m_program_count));
	Writer writer;
	VisitGraphicsPipeline(writer, pipeline);
	Append(static_cast<uint32_t>(RecordType::GraphicsPipeline), writer.Bytes());
}

void RecordFile::AppendComputePipeline(const ComputePipeline& pipeline) {
	EXIT_IF(pipeline.program >= m_program_count);
	Writer writer;
	VisitComputePipeline(writer, pipeline);
	Append(static_cast<uint32_t>(RecordType::ComputePipeline), writer.Bytes());
}

} // namespace Libs::Graphics::PipelineRecord
