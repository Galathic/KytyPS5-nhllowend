// CPU-only comparison using a disposable copy of a shader warm-up record.
// No Vulkan device is created and no guest memory is dereferenced.
int ResourceReplayBenchmark(const char* path) {
  using namespace ShaderRecompiler;
  using namespace ShaderRecompiler::IR;
  PipelineRecord::Records records;
  PipelineRecord::RecordFile file;
  if (!file.Open(std::filesystem::path(path), records) || records.programs.empty()) return 2;
  const auto ranges = [](const ResourceSnapshot& snapshot) {
    auto result = snapshot.specialization_reads;
    for (auto& [address, bytes] : result) bytes += address;
    std::ranges::sort(result);
    size_t count = 0;
    for (const auto range : result) {
      if (count != 0 && range.first <= result[count - 1].second)
        result[count - 1].second = std::max(result[count - 1].second, range.second);
      else result[count++] = range;
    }
    result.resize(count);
    return result;
  };
  const auto same = [&](const ResourceSnapshot& a, const ResourceSnapshot& b) {
    return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
           a.flattened_srt == b.flattened_srt && a.user_data == b.user_data &&
           a.uniform_fill == b.uniform_fill && ranges(a) == ranges(b);
  };
  struct Memory {
    std::unordered_map<uint64_t, PipelineRecord::MemoryRead> ordinary, strict;
    static bool Read(const std::unordered_map<uint64_t, PipelineRecord::MemoryRead>& memory,
                     uint64_t address, std::span<uint32_t> words) {
      for (size_t i = 0; i < words.size(); ++i) {
        const auto found = memory.find(address + i * 4);
        if (found == memory.end() || !found->second.valid) return false;
        words[i] = found->second.value;
      }
      return true;
    }
  };
  unsigned checked = 0, eligible = 0, skipped = 0;
  double total_scalar = 0, total_batch = 0;
  for (auto& record : records.programs) {
    CompileOptions options;
    options.stage = record.stage;
    options.shader_hash = record.hash;
    options.user_data = record.user_data;
    options.back_code = record.back_code;
    options.dump_ir = false;
    switch (record.stage) {
      case ShaderType::Vertex:
        options.wave_size = record.vertex.wave_size;
        options.user_data_base = 8;
        options.input_info.vertex = &record.vertex;
        break;
      case ShaderType::Pixel:
        options.wave_size = record.pixel.wave_size;
        options.input_info.pixel = &record.pixel;
        break;
      case ShaderType::Compute:
        options.wave_size = record.compute.wave_size;
        options.input_info.compute = &record.compute;
        break;
      default: ++skipped; continue;
    }
    auto translated = TranslateProgram(record.code, options);
    if (translated.build_failed) { ++skipped; continue; }
    auto plan = ExtractResourcePlan(translated.program);
    Memory memory;
    for (const auto read : record.raw_reads) memory.ordinary.emplace(read.address, read);
    for (const auto read : record.reads) memory.strict.emplace(read.address, read);
    SrtRuntime runtime{
      .user_data = record.user_data,
      .shader_base = record.shader_base,
      .read_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        return Memory::Read(static_cast<Memory*>(data)->ordinary, address, words);
      },
      .userdata = &memory,
      .read_specialization_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        return Memory::Read(static_cast<Memory*>(data)->strict, address, words);
      },
      .workgroup_counts = record.compute.workgroup_counts
    };
    ResourceSnapshot scalar, batch;
    ResourceSpecialization scalar_spec, batch_spec;
    const bool scalar_ok = MaterializeResources(plan, runtime, scalar, scalar_spec);
    runtime.batch_reads = true;
    const bool batch_ok = MaterializeResources(plan, runtime, batch, batch_spec);
    if (scalar_ok != batch_ok || (scalar_ok && (!same(scalar, batch) || scalar_spec != batch_spec))) {
      std::fprintf(stderr, "ResourceReplay: mismatch hash=%016llx stage=%u\n",
                   static_cast<unsigned long long>(record.hash), static_cast<unsigned>(record.stage));
      return 1;
    }
    if (!scalar_ok) { ++skipped; continue; }
    ++checked;
    const auto batches = std::ranges::count_if(plan.srt_batch_counts, [](uint32_t count) { return count > 1; });
    if (batches == 0) continue;
    ++eligible;
    const auto measure = [&](bool batching) {
      runtime.batch_reads = batching;
      const auto begin = std::chrono::steady_clock::now();
      for (unsigned i = 0; i < 20; ++i)
        if (!MaterializeResources(plan, runtime, batch, batch_spec)) return -1.0;
      return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 20;
    };
    const auto scalar_us = measure(false);
    const auto batch_us = measure(true);
    if (scalar_us < 0 || batch_us < 0) return 1;
    total_scalar += scalar_us;
    total_batch += batch_us;
    std::printf("ResourceReplay: hash=%016llx stage=%u scalar_us=%.2f batch_us=%.2f groups=%zu srt=%zu\n",
                static_cast<unsigned long long>(record.hash), static_cast<unsigned>(record.stage),
                scalar_us, batch_us, static_cast<size_t>(batches), plan.srt_reads.size());
  }
  std::printf("ResourceReplay: checked=%u eligible=%u skipped=%u sum_scalar_us=%.2f sum_batch_us=%.2f\n",
              checked, eligible, skipped, total_scalar, total_batch);
  return 0;
}
