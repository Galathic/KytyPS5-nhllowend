#ifndef KYTY_GRAPHICS_RESOURCE_MEMO_H_
#define KYTY_GRAPHICS_RESOURCE_MEMO_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <span>
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
    struct UserDataCapture {
        ResourceMemoEntry& memo;
        ShaderRecompiler::IR::SrtUserDataObserver forward;
        void* forward_userdata;
    };
    struct ReadReplay {
        std::span<const Read> reads;
        bool inconclusive = false;
    };

    static bool DynamicDependenciesEnabled() {
        static const bool enabled = [] {
            const char* flag = std::getenv("KYTY_RESOURCE_DYNAMIC_DEPS");
            return flag != nullptr && flag[0] == '1';
        }();
        return enabled;
    }
    static bool VerifyHitsEnabled() {
        static const bool enabled = [] {
            const char* flag = std::getenv("KYTY_RESOURCE_MEMO_VERIFY");
            return flag != nullptr && flag[0] == '1';
        }();
        return enabled;
    }
    static void ObserveUserData(void* userdata, uint32_t index) {
        auto& capture = *static_cast<UserDataCapture*>(userdata);
        auto& memo = capture.memo;
        if (index < memo.m_user_data_seen.size() && memo.m_user_data_seen[index] == 0) {
            memo.m_user_data_seen[index] = 1;
            memo.m_observed_register_inputs.push_back(index);
        }
        if (capture.forward != nullptr) capture.forward(capture.forward_userdata, index);
    }

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
    template <bool strict>
    static bool ReplayRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
        auto& replay = *static_cast<ReadReplay*>(userdata);
        if (values.size() > (UINT64_MAX - address) / sizeof(uint32_t)) {
            replay.inconclusive = true;
            return false;
        }
        const auto end = address + values.size_bytes();
        bool found = false;
        bool valid = false;
        for (const auto& read: replay.reads) {
            if (read.strict != strict || read.values.size() >
                    (UINT64_MAX - read.address) / sizeof(uint32_t)) continue;
            const auto read_end = read.address + read.values.size() * sizeof(uint32_t);
            if (address < read.address || end > read_end || ((address - read.address) & 3u) != 0u) continue;
            const auto offset = static_cast<size_t>((address - read.address) / sizeof(uint32_t));
            const auto captured = std::span(read.values).subspan(offset, values.size());
            if (!found) {
                std::ranges::copy(captured, values.begin());
                valid = read.valid;
                found = true;
            } else if (valid != read.valid || !std::ranges::equal(captured, values)) {
                replay.inconclusive = true;
                return false;
            }
        }
        if (!found) {
            replay.inconclusive = true;
            return false;
        }
        return valid;
    }
    static bool SameSnapshot(const ShaderRecompiler::IR::ResourceSnapshot& left,
                             const ShaderRecompiler::IR::ResourceSnapshot& right) {
        return left.buffers == right.buffers && left.images == right.images &&
               left.samplers == right.samplers && left.flattened_srt == right.flattened_srt &&
               left.user_data == right.user_data &&
               left.specialization_reads == right.specialization_reads &&
               left.uniform_fill == right.uniform_fill;
    }
    bool VerifyCachedHit(const ShaderRecompiler::IR::ResourcePlan& plan, const Runtime& runtime,
                         const ShaderRecompiler::IR::ResourceSnapshot& cached_snapshot,
                         const ShaderRecompiler::IR::ResourceSpecialization& cached_specialization,
                         bool& inconclusive) {
        ++m_verify_checks;
        ReadReplay replay{std::span(m_reads).first(m_read_count)};
        auto replay_runtime = runtime;
        replay_runtime.userdata = &replay;
        replay_runtime.read_memory = ReplayRead<false>;
        replay_runtime.read_specialization_memory = ReplayRead<true>;
        // Verification must not replay dependency notifications to an outer collector.
        replay_runtime.observe_user_data = nullptr;
        replay_runtime.user_data_observer_userdata = nullptr;

        const auto active_sources = plan.active_sources;
        auto expected_snapshot = cached_snapshot;
        expected_snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
        ShaderRecompiler::IR::ResourceSnapshot fresh_snapshot;
        ShaderRecompiler::IR::ResourceSpecialization fresh_specialization;
        const bool materialized = ShaderRecompiler::IR::MaterializeResources(
            plan, replay_runtime, fresh_snapshot, fresh_specialization);
        const auto fresh_active_sources = plan.active_sources;
        plan.active_sources = active_sources;
        if (replay.inconclusive) {
            ++m_verify_inconclusive;
            inconclusive = true;
            return false;
        }
        if (!materialized || !SameSnapshot(expected_snapshot, fresh_snapshot) ||
            cached_specialization != fresh_specialization || fresh_active_sources != m_active_sources) {
            ++m_verify_mismatches;
            return false;
        }
        ++m_verify_matches;
        return true;
    }

public:
    enum class Miss {
        None, Cold, ShaderBase, Registers, Workgroups, ReadFailed, MemoryChanged,
        VerifyMismatch, VerifyInconclusive, Count
    };
    Miss LastMiss() const { return m_last_miss; }
    size_t RegisterDependencyCount() const { return m_register_inputs.size(); }
    size_t StaticRegisterDependencyCount() const { return m_static_register_count; }
    uint64_t VerifyChecks() const { return m_verify_checks; }
    uint64_t VerifyMatches() const { return m_verify_matches; }
    uint64_t VerifyMismatches() const { return m_verify_mismatches; }
    uint64_t VerifyInconclusive() const { return m_verify_inconclusive; }
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
        const bool dynamic_dependencies = DynamicDependenciesEnabled();
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
                if (VerifyHitsEnabled()) {
                    bool inconclusive = false;
                    if (!VerifyCachedHit(plan, runtime, snapshot, specialization, inconclusive)) {
                        m_valid = false;
                        m_last_miss = inconclusive ? Miss::VerifyInconclusive : Miss::VerifyMismatch;
                        inputs_equal = false;
                    }
                }
            }
            if (equal && inputs_equal) {
                // Non-resource SGPRs can vary per draw. They still feed push data,
                // but do not require walking the descriptor graph again.
                snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
                // A caller may be collecting dependencies around this memo. Replay
                // the validated dependency set on hits so it sees the same roots.
                if (runtime.observe_user_data != nullptr) {
                    for (const auto index: m_register_inputs) {
                        runtime.observe_user_data(runtime.user_data_observer_userdata, index);
                    }
                }
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
        UserDataCapture user_data_capture{*this, runtime.observe_user_data,
                                          runtime.user_data_observer_userdata};
        if (dynamic_dependencies) {
            for (const auto index: m_observed_register_inputs) {
                if (index < m_user_data_seen.size()) m_user_data_seen[index] = 0;
            }
            m_observed_register_inputs.clear();
            if (m_user_data_seen.size() != runtime.user_data.size()) {
                m_user_data_seen.assign(runtime.user_data.size(), 0);
            }
            m_observed_register_inputs.reserve(runtime.user_data.size());
            observed.observe_user_data = ObserveUserData;
            observed.user_data_observer_userdata = &user_data_capture;
        }
        if (!ShaderRecompiler::IR::MaterializeResources(plan, observed, snapshot, specialization)) return false;
        if (dynamic_dependencies) {
            m_register_inputs = m_observed_register_inputs;
        }
        if (m_plan != &plan) {
            m_static_register_inputs.clear();
            for (const auto& value: plan.value_storage) {
                if (value.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) continue;
                const auto reg = ShaderRecompiler::IR::RegIndex(value.Arg(0).ScalarRegister());
                if (reg >= plan.user_data_base) m_static_register_inputs.push_back(reg - plan.user_data_base);
            }
            std::ranges::sort(m_static_register_inputs);
            m_static_register_inputs.erase(
                std::unique(m_static_register_inputs.begin(), m_static_register_inputs.end()),
                m_static_register_inputs.end());
            m_static_register_count = m_static_register_inputs.size();
        }
        if (!dynamic_dependencies && m_plan != &plan) {
            m_register_inputs.clear();
            m_register_inputs = m_static_register_inputs;
        }
        m_plan = &plan;
        m_shader_base = runtime.shader_base;
        m_user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
        m_workgroups.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
        if (VerifyHitsEnabled()) m_active_sources = plan.active_sources;
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
    std::vector<uint8_t> m_active_sources;
    std::vector<uint32_t> m_register_inputs;
    std::vector<uint32_t> m_observed_register_inputs;
    std::vector<uint8_t> m_user_data_seen;
    std::vector<uint32_t> m_static_register_inputs;
    size_t m_static_register_count = 0;
    std::vector<Read> m_reads;
    size_t m_read_count = 0;
    std::vector<uint32_t> m_check;
    uint64_t m_verify_checks = 0;
    uint64_t m_verify_matches = 0;
    uint64_t m_verify_mismatches = 0;
    uint64_t m_verify_inconclusive = 0;
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
