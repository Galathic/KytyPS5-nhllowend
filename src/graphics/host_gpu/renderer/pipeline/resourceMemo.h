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
        if (valid && !reads.empty() && reads.back().valid && reads.back().strict == strict &&
            reads.back().address + reads.back().values.size() * sizeof(uint32_t) == address) {
            reads.back().values.insert(reads.back().values.end(), values.begin(), values.end());
        } else {
            reads.push_back({address, strict,
                valid ? std::vector<uint32_t>(values.begin(), values.end())
                      : std::vector<uint32_t>(values.size(), 0u), valid});
        }
        return valid;
    }

public:
    enum class Miss { None, Cold, ShaderBase, Registers, Workgroups, ReadFailed, MemoryChanged, Count };
    Miss LastMiss() const { return m_last_miss; }
    size_t RecordedWords() const {
        size_t total = 0;
        for (const auto& read: m_reads) total += read.values.size();
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
            for (const auto& read: m_reads) {
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
        m_reads.clear();
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
        Entry fresh;
        if (!fresh.memo.Refresh(plan, runtime, snapshot, specialization, true, reused)) {
            m_entries.clear();
            m_last_hit = SIZE_MAX;
            return false;
        }
        // Large descriptor walks must not multiply memory use sixteenfold.
        if (fresh.memo.RecordedWords() > 4096) {
            m_last_hit = SIZE_MAX;
            return true;
        }
        fresh.snapshot = snapshot;
        fresh.specialization = specialization;
        if (m_entries.size() < m_capacity) {
            m_last_hit = m_entries.size();
            m_entries.push_back(std::move(fresh));
        } else {
            m_last_hit = m_next++ % m_capacity;
            m_entries[m_last_hit] = std::move(fresh);
        }
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
