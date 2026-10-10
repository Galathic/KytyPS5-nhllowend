#ifndef KYTY_GRAPHICS_RESOURCE_MEMO_H_
#define KYTY_GRAPHICS_RESOURCE_MEMO_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

// A resource snapshot depends on register inputs AND guest memory. Validate the
// actual reads with their original readers; unchanged pointers alone are insufficient.
class ResourceMemoEntry {
    using Runtime = ShaderRecompiler::IR::SrtRuntime;
    struct Read {
        uint64_t address;
        bool strict;
        std::vector<uint32_t> values;
        bool valid = true;
    };
    struct Capture {
        ResourceMemoEntry& memo;
        const Runtime& source;
    };

    static bool ReadValues(const Runtime& runtime, bool strict, uint64_t address,
                           std::span<uint32_t> values) {
        const auto reader = strict ? runtime.read_specialization_memory : runtime.read_memory;
        if (reader != nullptr) return reader(runtime.userdata, address, values);
        if (strict) return false;
        std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
        return true;
    }
    template <bool strict>
    static bool CaptureRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
        auto& capture = *static_cast<Capture*>(userdata);
        const bool valid = ReadValues(capture.source, strict, address, values);
        auto& reads = capture.memo.m_reads;
        auto* previous = capture.memo.m_read_count == 0 ? nullptr
            : &reads[capture.memo.m_read_count - 1];
        if (valid && previous != nullptr && previous->valid && previous->strict == strict &&
            previous->address + previous->values.size() * sizeof(uint32_t) == address) {
            previous->values.insert(previous->values.end(), values.begin(), values.end());
        } else {
            if (capture.memo.m_read_count == reads.size()) reads.emplace_back();
            auto& read = reads[capture.memo.m_read_count++];
            read.address = address;
            read.strict = strict;
            read.valid = valid;
            if (valid) read.values.assign(values.begin(), values.end());
            else read.values.assign(values.size(), 0u);
        }
        return valid;
    }

public:
    enum class Miss { None, Cold, ShaderBase, Registers, Workgroups, ReadFailed, MemoryChanged, Count };
    Miss LastMiss() const { return m_last_miss; }
    void Invalidate() { m_valid = false; }
    size_t RecordedWords() const {
        size_t total = 0;
        for (const auto& read: std::span(m_reads).first(m_read_count)) total += read.values.size();
        return total;
    }

    bool Refresh(const ShaderRecompiler::IR::ResourcePlan& plan, const Runtime& runtime,
                 ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                 ShaderRecompiler::IR::ResourceSpecialization& specialization,
                 bool enabled, bool& reused, bool probe_only = false) {
        reused = false;
        m_last_miss = Miss::None;
        bool inputs_equal = enabled;
        if (enabled) {
            if (!m_valid || m_plan != &plan) m_last_miss = Miss::Cold;
            else if (m_shader_base != runtime.shader_base) m_last_miss = Miss::ShaderBase;
            else if (!RegisterInputsEqual(runtime.user_data)) m_last_miss = Miss::Registers;
            else if (!std::ranges::equal(m_workgroups, runtime.workgroup_counts)) m_last_miss = Miss::Workgroups;
            inputs_equal = m_last_miss == Miss::None;
        }
        if (inputs_equal) {
            bool equal = true;
            for (const auto& read: std::span(m_reads).first(m_read_count)) {
                m_check.resize(read.values.size());
                const bool valid = ReadValues(runtime, read.strict, read.address, m_check);
                if (valid != read.valid) {
                    m_last_miss = Miss::ReadFailed;
                    equal = false;
                    break;
                }
                if (valid && m_check != read.values) {
                    m_last_miss = Miss::MemoryChanged;
                    equal = false;
                    break;
                }
            }
            if (equal) {
                // Non-resource SGPRs can vary per draw. They still feed push data,
                // but do not require walking the descriptor graph again.
                snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
                reused = true;
                return true;
            }
        }
        if (probe_only) return false;
        m_valid = false;
        m_read_count = 0;
        if (!enabled) {
            return ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
        }
        Capture capture{*this, runtime};
        auto observed = runtime;
        observed.userdata = &capture;
        observed.read_memory = CaptureRead<false>;
        observed.read_specialization_memory = CaptureRead<true>;
        if (!ShaderRecompiler::IR::MaterializeResources(plan, observed, snapshot, specialization)) return false;
        if (m_plan != &plan) {
            m_register_inputs.clear();
            for (const auto& value: plan.value_storage) {
                if (value.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) continue;
                const auto reg = ShaderRecompiler::IR::RegIndex(value.Arg(0).ScalarRegister());
                if (reg >= plan.user_data_base) m_register_inputs.push_back(reg - plan.user_data_base);
            }
            std::ranges::sort(m_register_inputs);
            m_register_inputs.erase(std::unique(m_register_inputs.begin(), m_register_inputs.end()),
                                    m_register_inputs.end());
        }
        m_plan = &plan;
        m_shader_base = runtime.shader_base;
        m_user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
        m_workgroups.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
        m_valid = true;
        return true;
    }

private:
    Miss m_last_miss = Miss::None;
    bool RegisterInputsEqual(std::span<const uint32_t> current) const {
        if (current.size() != m_user_data.size()) return false;
        for (const auto index: m_register_inputs) {
            if (index >= current.size() || current[index] != m_user_data[index]) return false;
        }
        return true;
    }
    const ShaderRecompiler::IR::ResourcePlan* m_plan = nullptr;
    uint64_t m_shader_base = 0;
    bool m_valid = false;
    std::vector<uint32_t> m_user_data;
    std::vector<uint32_t> m_workgroups;
    std::vector<uint32_t> m_register_inputs;
    std::vector<Read> m_reads;
    size_t m_read_count = 0;
    std::vector<uint32_t> m_check;
};

// Keep multiple validated states so alternating materials do not evict each other.
// Never cache solely by pointer: each candidate still replays its dependency reads.
class ResourceMemo {
public:
    explicit ResourceMemo(size_t capacity = 0) : m_capacity(capacity) {
        if (m_capacity == 0) {
            const char* flag = std::getenv("KYTY_RESOURCE_MULTI_CACHE");
            m_capacity = flag != nullptr && flag[0] == '1' ? 16 : 1;
        }
        m_capacity = std::clamp(m_capacity, size_t{1}, size_t{16});
    }
    using Miss = ResourceMemoEntry::Miss;
    Miss LastMiss() const { return m_last_miss; }
    bool Refresh(const ShaderRecompiler::IR::ResourcePlan& plan,
                 const ShaderRecompiler::IR::SrtRuntime& runtime,
                 ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                 ShaderRecompiler::IR::ResourceSpecialization& specialization,
                 bool enabled, bool& reused) {
        if (!enabled) {
            m_entries.clear();
            m_last_hit = SIZE_MAX;
            reused = false;
            return ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization);
        }
        if (m_capacity == 1) {
            // With one state the caller already owns the cached snapshot. Refresh it
            // directly rather than maintaining and copying a second full snapshot.
            if (m_entries.empty()) m_entries.emplace_back();
            auto& memo = m_entries.front().memo;
            if (m_output != &snapshot || m_specialization != &specialization) memo.Invalidate();
            const bool ok = memo.Refresh(plan, runtime, snapshot, specialization, true, reused);
            m_last_miss = memo.LastMiss();
            m_output = &snapshot;
            m_specialization = &specialization;
            if (!ok || memo.RecordedWords() > 4096) {
                m_entries.clear();
                m_output = m_specialization = nullptr;
            }
            return ok;
        }
        const auto count = m_entries.size();
        for (size_t step = 0; step < count; ++step) {
            const size_t index = m_last_hit < count ? (m_last_hit + step) % count : step;
            auto& entry = m_entries[index];
            if (!entry.memo.Refresh(plan, runtime, entry.snapshot, entry.specialization,
                                    true, reused, true)) {
                if (step == 0) m_last_miss = entry.memo.LastMiss();
                continue;
            }
            if (m_last_hit != index || m_output != &snapshot || m_specialization != &specialization) {
                snapshot = entry.snapshot;
                specialization = entry.specialization;
            } else {
                snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
            }
            m_last_hit = index;
            m_output = &snapshot;
            m_specialization = &specialization;
            m_last_miss = Miss::None;
            return true;
        }
        if (count == 0) m_last_miss = Miss::Cold;
        // Recycle the victim in place. Its dependency list and captured-read storage
        // survive misses instead of rebuilding/freeing them for every object.
        const size_t selected = count < m_capacity ? count : m_next++ % m_capacity;
        if (count < m_capacity) m_entries.emplace_back();
        auto& entry = m_entries[selected];
        if (!entry.memo.Refresh(plan, runtime, entry.snapshot, entry.specialization, true, reused)) {
            m_entries.clear();
            m_last_hit = SIZE_MAX;
            return false;
        }
        if (entry.memo.RecordedWords() > 4096) {
            snapshot = std::move(entry.snapshot);
            specialization = std::move(entry.specialization);
            m_entries.erase(m_entries.begin() + selected);
            m_last_hit = SIZE_MAX;
            return true;
        }
        snapshot = entry.snapshot;
        specialization = entry.specialization;
        m_last_hit = selected;
        m_output = &snapshot;
        m_specialization = &specialization;
        return true;
    }
private:
    struct Entry {
        ResourceMemoEntry memo;
        ShaderRecompiler::IR::ResourceSnapshot snapshot;
        ShaderRecompiler::IR::ResourceSpecialization specialization;
    };
    std::vector<Entry> m_entries;
    size_t m_capacity;
    size_t m_last_hit = SIZE_MAX, m_next = 0;
    const void* m_output = nullptr;
    const void* m_specialization = nullptr;
    Miss m_last_miss = Miss::None;
};

} // namespace Libs::Graphics
#endif
