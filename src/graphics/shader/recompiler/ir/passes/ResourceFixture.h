#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RESOURCEFIXTURE_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RESOURCEFIXTURE_H_

// Recorded resource-materialization workload: the SRT reads a real shader plan
// performed, saved to disk so the same plan can be re-evaluated with no emulator,
// no GPU and no game.
//
// Why: a training-zone session runs on the order of 3.9M tape runs and 238M ops, and
// roughly 95% of descriptor lookups still materialize. Iterating on SrtWalker or
// resourceMemo against a recorded workload costs seconds per edit instead of a full
// game run, which is the whole point of the experiment-flag workflow around it.
//
// Scope and limits, stated here because they matter more than the format:
//   - A fixture freezes guest memory. Replay cannot catch ordering, synchronization
//     or timing bugs. Live verifier modes still own that.
//   - A read that was never captured must be reported as missing, never silently
//     served by live memory. A silent fallback would make the fixture lie.
//   - The format is shared by the emulator (writer) and the test target (reader) so
//     the two cannot drift apart.
//
// Format (little-endian, unaligned-safe):
//   magic "KYTYARB1" (8 bytes), u32 version, u32 read_count
//   u64 shader_base, u32 user_data_count, u32 flags
//   user_data_count * u32
//   read_count * { u64 address, u32 value_count, u32 strict, u32 valid,
//                  (valid ? value_count : 0) * u32 }

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

inline constexpr char     kResourceFixtureMagic[8] = {'K', 'Y', 'T', 'Y', 'A', 'R', 'B', '1'};
inline constexpr uint32_t kResourceFixtureVersion  = 1;

// One captured read: address, the exact width asked for, which reader was used, and
// the values returned. A !valid entry records a read that legitimately failed.
struct FixtureRead {
	uint64_t              address     = 0;
	bool                  strict      = false;
	bool                  valid       = true;
	std::vector<uint32_t> values;
};

class ResourceFixture {
public:
	// --- writer side -------------------------------------------------------

	void SetUserData(std::span<const uint32_t> user_data) { m_user_data.assign(user_data.begin(), user_data.end()); }
	void SetShaderBase(uint64_t base) { m_shader_base = base; }

	// Coalescing keeps the accumulator small by extending a record with an immediately
	// following read. Rebuild merges by word anyway, so turning it off only costs memory.
	// Tests disable it so appended batches line up one-to-one with records.
	void EnableCoalescing(bool enable) { m_enable_coalesce = enable; }

	void AddRead(FixtureRead read) {
		++m_total_reads;
		m_words_added += read.values.size();
		// Adjacent reads from the same reader extend the previous record so a
		// descriptor walked word by word does not become hundreds of entries.
		if (m_enable_coalesce && !m_reads.empty()) {
			auto& previous = m_reads.back();
			if (previous.strict == read.strict && previous.valid == read.valid &&
			    previous.address + previous.values.size() * sizeof(uint32_t) == read.address) {
				previous.values.insert(previous.values.end(), read.values.begin(), read.values.end());
				++m_coalesced_records;
				return;
			}
		}
		m_reads.push_back(std::move(read));
	}

	uint64_t TotalReads() const { return m_total_reads; }
	size_t   DistinctReads() const { return m_reads.size(); }
	bool     Empty() const { return m_reads.empty(); }

	// Conservation accounting. A recorder that silently drops words would produce a
	// fixture that looks fine and replays wrong, so the retained word count is tracked
	// explicitly and reported rather than inferred.
	uint64_t WordsAdded() const { return m_words_added; }
	uint64_t WordsRetained() const { return m_words.size(); }
	uint64_t CoalescedRecords() const { return m_coalesced_records; }

	const std::vector<FixtureRead>& Reads() const { return m_reads; }
	std::span<const uint32_t>       UserData() const { return m_user_data; }
	uint64_t                        ShaderBase() const { return m_shader_base; }

	// --- bounded-cost appending for the live recorder -----------------------
	// The recorder writes in append-only batches. Rewriting the whole fixture on
	// every flush made each save O(total), which is what stalled the walk path.
	// Appends are serialised by their own mutex; reads never take it.

	std::mutex& AppendMutex() const { return m_append_mutex; }

	// Writes the magic/version/shader base/user data once, in place. Rebuilding in
	// memory does not touch the file, so a flush can rewrite the header without
	// disturbing earlier batches.
	bool BeginFile(const std::string& path) const {
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		if (!out.is_open()) {
			return false;
		}
		out.write(kResourceFixtureMagic, sizeof(kResourceFixtureMagic));
		WriteU32(out, kResourceFixtureVersion);
		WriteU32(out, 0);
		WriteU64(out, m_shader_base);
		WriteU32(out, static_cast<uint32_t>(m_user_data.size()));
		WriteU32(out, 0);
		for (uint32_t word : m_user_data) {
			WriteU32(out, word);
		}
		out.flush();
		return out.good();
	}

	// Appends one batch and patches the record count. Only the new reads are written, so
	// the cost is proportional to what changed.
	//
	// Two sequential open/close cycles rather than one read/write stream: mixing reads
	// and writes on a single fstream has position subtleties that are easy to get wrong,
	// and this path is not hot enough to justify the cleverness.
	bool AppendReads(const std::string& path, const std::vector<FixtureRead>& batch) const {
		if (batch.empty()) {
			return true;
		}
		{
			std::ofstream out(path, std::ios::binary | std::ios::app);
			if (!out.is_open()) {
				return false;
			}
			for (const auto& read : batch) {
				WriteU64(out, read.address);
				WriteU32(out, static_cast<uint32_t>(read.values.size()));
				WriteU32(out, read.strict ? 1u : 0u);
				WriteU32(out, read.valid ? 1u : 0u);
				if (read.valid) {
					for (uint32_t word : read.values) {
						WriteU32(out, word);
					}
				}
			}
			out.flush();
			if (!out.good()) {
				return false;
			}
		}
		// Patch the header count last, so a torn file is detectable by a short read
		// rather than by silently claiming records that are not there.
		//
		// The count field sits at byte 12: 8 bytes of magic, then a u32 version, then
		// the u32 count. Writing it at 8 would clobber the version and make the file
		// unloadable - which is exactly what an earlier version of this did.
		{
			std::fstream header(path, std::ios::binary | std::ios::in | std::ios::out);
			if (!header.is_open()) {
				return false;
			}
			const uint32_t count = static_cast<uint32_t>(m_reads.size());
			header.seekp(12, std::ios::beg);
			WriteU32(header, count);
			header.flush();
			return header.good();
		}
	}

	// Sorts records by address and rebuilds the word index. Deduplication happens at
	// the word level, not the record level: two records that overlap are merged word by
	// word, so a record is never dropped wholesale and no captured word can be lost.
	//
	// An earlier version dropped any record narrower than the widest already seen at
	// the same address. That silently discarded new words that happened to sit beyond
	// the kept span - the conservation counters reported more dropped words than
	// retained, which is how it was caught.
	void Rebuild() {
		std::lock_guard<std::mutex> lock(m_append_mutex);
		std::stable_sort(m_reads.begin(), m_reads.end(), [](const FixtureRead& a, const FixtureRead& b) {
			if (a.address != b.address) {
				return a.address < b.address;
			}
			return a.values.size() > b.values.size();
		});
		// Merge everything into a per-word map. First write wins at a given word, so a
		// later overlapping record cannot overwrite a captured value.
		m_words.clear();
		for (const auto& read : m_reads) {
			for (size_t i = 0; i < read.values.size(); ++i) {
				const uint64_t word_address = read.address + i * sizeof(uint32_t);
				m_words.emplace(word_address, Word {read.values[i], read.valid});
			}
		}
		// Rebuild the record list from the merged word map by walking runs of
		// consecutive addresses that share validity.
		std::vector<FixtureRead> merged;
		for (const auto& [address, word] : m_words) {
			if (!merged.empty()) {
				auto& back = merged.back();
				const bool adjacent = back.address + back.values.size() * sizeof(uint32_t) == address;
				const bool same_kind = back.valid == word.valid;
				if (adjacent && same_kind) {
					back.values.push_back(word.value);
					continue;
				}
			}
			FixtureRead read;
			read.address = address;
			read.valid   = word.valid;
			read.values.push_back(word.value);
			merged.push_back(std::move(read));
		}
		m_reads = std::move(merged);
		m_index_valid = false;
	}

	// Whole-file write to a temporary path, then rename over the target. The rename is
	// atomic on the same volume, so a reader either sees the previous complete fixture or
	// the new complete one, and a crash mid-write cannot corrupt a good file.
	//
	// This replaced an incremental append scheme. Append-only is cheaper per flush but
	// needs delta tracking, and three successive attempts at that were each wrong in a
	// different way (record-count cursor, ascending-address cursor, per-word delta) -
	// the last silently under-wrote about 30% of the recorded words. Correctness first:
	// a fixture that lies is worse than a slow one, and the cost here is bounded because
	// the accumulator is word-deduplicated.
	bool SaveAtomic(const std::string& path) const {
		std::lock_guard<std::mutex> lock(m_append_mutex);
		const std::string           tmp = path + ".tmp";
		{
			std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
			if (!out.is_open()) {
				return false;
			}
			out.write(kResourceFixtureMagic, sizeof(kResourceFixtureMagic));
			WriteU32(out, kResourceFixtureVersion);
			WriteU32(out, static_cast<uint32_t>(m_reads.size()));
			WriteU64(out, m_shader_base);
			WriteU32(out, static_cast<uint32_t>(m_user_data.size()));
			WriteU32(out, 0);
			for (uint32_t word : m_user_data) {
				WriteU32(out, word);
			}
			for (const auto& read : m_reads) {
				WriteU64(out, read.address);
				WriteU32(out, static_cast<uint32_t>(read.values.size()));
				WriteU32(out, read.strict ? 1u : 0u);
				WriteU32(out, read.valid ? 1u : 0u);
				if (read.valid) {
					for (uint32_t word : read.values) {
						WriteU32(out, word);
					}
				}
			}
			out.flush();
			if (!out.good()) {
				return false;
			}
		}
		std::error_code ec;
		std::filesystem::rename(tmp, path, ec);
		if (ec) {
			std::filesystem::remove(tmp, ec);
			return false;
		}
		return true;
	}

	// Whole-file write, used by unit tests and any non-append caller.
	bool Save(const std::string& path) const {
		std::lock_guard<std::mutex> lock(m_append_mutex);
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		if (!out.is_open()) {
			return false;
		}
		out.write(kResourceFixtureMagic, sizeof(kResourceFixtureMagic));
		WriteU32(out, kResourceFixtureVersion);
		WriteU32(out, static_cast<uint32_t>(m_reads.size()));
		WriteU64(out, m_shader_base);
		WriteU32(out, static_cast<uint32_t>(m_user_data.size()));
		WriteU32(out, 0);
		for (uint32_t word : m_user_data) {
			WriteU32(out, word);
		}
		for (const auto& read : m_reads) {
			WriteU64(out, read.address);
			WriteU32(out, static_cast<uint32_t>(read.values.size()));
			WriteU32(out, read.strict ? 1u : 0u);
			WriteU32(out, read.valid ? 1u : 0u);
			if (read.valid) {
				for (uint32_t word : read.values) {
					WriteU32(out, word);
				}
			}
		}
		out.flush();
		return out.good();
	}

	// --- reader side -------------------------------------------------------

	bool Load(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		if (!in.is_open()) {
			return false;
		}
		char magic[sizeof(kResourceFixtureMagic)] = {};
		in.read(magic, sizeof(magic));
		if (!in.good() || std::memcmp(magic, kResourceFixtureMagic, sizeof(magic)) != 0) {
			return false;
		}
		const uint32_t version = ReadU32(in);
		if (version != kResourceFixtureVersion) {
			return false; // Refuse an unknown revision rather than guess.
		}
		const uint32_t read_count = ReadU32(in);
		m_shader_base             = ReadU64(in);
		const uint32_t user_count = ReadU32(in);
		(void)ReadU32(in);
		m_user_data.clear();
		m_user_data.reserve(user_count);
		for (uint32_t i = 0; i < user_count; ++i) {
			m_user_data.push_back(ReadU32(in));
		}
		m_reads.clear();
		m_reads.reserve(read_count);
		for (uint32_t i = 0; i < read_count; ++i) {
			FixtureRead read;
			read.address     = ReadU64(in);
			const uint32_t value_count = ReadU32(in);
			read.strict      = ReadU32(in) != 0;
			read.valid       = ReadU32(in) != 0;
			if (read.valid) {
				read.values.reserve(value_count);
				for (uint32_t v = 0; v < value_count; ++v) {
					read.values.push_back(ReadU32(in));
				}
			} else {
				read.values.assign(value_count, 0u);
			}
			m_reads.push_back(std::move(read));
		}
		m_index_valid = false;
		return in.good() || in.eof();
	}

	// --- replay runtime backing -------------------------------------------

	// Serves captured responses only. Reports why it could not serve:
	//   *failed  = the fixture HAS a record and that read failed. Replay is faithful.
	//   *missing = the fixture has NO record, so the answer is unknown. Any use of the
	//              materialization result is then inconclusive, not merely failed.
	// Conflating those two would let a partial fixture look like a working one.
	//
	// Lookup is per word and reader-agnostic: the fixture records the memory access
	// trace, and which reader a plan uses is a property of the plan, not of the
	// bytes. Recording a strict label here would be wrong for any runtime whose
	// ordinary reader is the specialization reader (CleanRuntime does exactly that).
	// A request is served only if every word it asks for was captured, which also
	// means a coalesced span can answer a narrower read from inside it.
	bool Read(void* userdata, uint64_t address, std::span<uint32_t> values, bool strict, bool* missing,
	          bool* failed = nullptr) {
		(void)userdata;
		(void)strict;
		if (missing != nullptr) {
			*missing = false;
		}
		if (failed != nullptr) {
			*failed = false;
		}
		BuildIndex();
		std::array<uint32_t, 16> staged {};
		for (size_t i = 0; i < values.size(); ++i) {
			const uint64_t word_address = address + i * sizeof(uint32_t);
			const auto     it           = m_words.find(word_address);
			if (it == m_words.end()) {
				if (missing != nullptr) {
					*missing = true;
				}
				return false;
			}
			if (!it->second.valid) {
				if (failed != nullptr) {
					*failed = true;
				}
				return false;
			}
			if (i < staged.size()) {
				staged[i] = it->second.value;
			} else {
				values[i] = it->second.value;
			}
		}
		const size_t staged_count = values.size() < staged.size() ? values.size() : staged.size();
		for (size_t i = 0; i < staged_count; ++i) {
			values[i] = staged[i];
		}
		return true;
	}

	size_t MissingRequests() const { return m_missing; }
	void   NoteMissing() { ++m_missing; }

private:
	struct Word {
		uint32_t value = 0;
		bool     valid = false;
	};

	void BuildIndex() {
		if (m_index_valid) {
			return;
		}
		m_words.clear();
		for (const auto& read : m_reads) {
			for (size_t i = 0; i < read.values.size(); ++i) {
				const uint64_t word_address = read.address + i * sizeof(uint32_t);
				// First writer wins, so a later record cannot silently overwrite a
				// captured value with a different one.
				m_words.emplace(word_address, Word {read.values[i], read.valid});
			}
		}
		m_index_valid = true;
	}

	template <typename Stream>
	static void WriteU32(Stream& out, uint32_t value) {
		out.write(reinterpret_cast<const char*>(&value), sizeof(value));
	}
	template <typename Stream>
	static void WriteU64(Stream& out, uint64_t value) {
		out.write(reinterpret_cast<const char*>(&value), sizeof(value));
	}
	static uint32_t ReadU32(std::ifstream& in) {
		uint32_t value = 0;
		in.read(reinterpret_cast<char*>(&value), sizeof(value));
		return value;
	}
	static uint64_t ReadU64(std::ifstream& in) {
		uint64_t value = 0;
		in.read(reinterpret_cast<char*>(&value), sizeof(value));
		return value;
	}

	std::vector<FixtureRead>             m_reads;
	std::vector<uint32_t>                m_user_data;
	uint64_t                             m_shader_base  = 0;
	uint64_t                             m_total_reads  = 0;
	size_t                               m_missing      = 0;
	bool                                 m_index_valid  = false;
	bool                                 m_enable_coalesce = true;
	std::map<uint64_t, Word>             m_words;
	// Serialises accumulator mutation against concurrent appends. Reads never touch
	// it; it is held only briefly by AddRead and for the duration of a flush.
	mutable std::mutex                   m_append_mutex;
	// Conservation counters, reported by the recorder so a silently lossy fixture is
	// visible rather than assumed correct.
	uint64_t                             m_words_added       = 0;
	uint64_t                             m_words_dropped     = 0;
	uint64_t                             m_dropped_records   = 0;
	uint64_t                             m_coalesced_records = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RESOURCEFIXTURE_H_ */
