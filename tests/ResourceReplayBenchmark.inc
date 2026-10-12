// CPU-only comparison using a disposable copy of a shader warm-up record.
// No Vulkan device is created and no guest memory is dereferenced.
int ResourceReplayBenchmark(const char* path, bool gpu_scalar_data = false,
                            bool compile_spirv = false) {
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
           (gpu_scalar_data || a.flattened_srt == b.flattened_srt) && a.user_data == b.user_data &&
           a.uniform_fill == b.uniform_fill && (gpu_scalar_data || ranges(a) == ranges(b));
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
    ResourcePlan gpu_plan;
    TranslateResult gpu_translated;
    if (gpu_scalar_data) {
      options.gpu_scalar_data = true;
      gpu_translated = TranslateProgram(record.code, options);
      if (gpu_translated.build_failed) return 1;
      gpu_plan = ExtractResourcePlan(gpu_translated.program);
    }
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
    runtime.batch_reads = gpu_scalar_data;
    const bool scalar_ok = MaterializeResources(plan, runtime, scalar, scalar_spec);
    runtime.batch_reads = true;
    const auto& candidate_plan = gpu_scalar_data ? gpu_plan : plan;
    const bool batch_ok = MaterializeResources(candidate_plan, runtime, batch, batch_spec);
    if (scalar_ok != batch_ok || (scalar_ok && (!same(scalar, batch) || scalar_spec != batch_spec))) {
      std::fprintf(stderr, "ResourceReplay: mismatch hash=%016llx stage=%u\n",
                   static_cast<unsigned long long>(record.hash), static_cast<unsigned>(record.stage));
      return 1;
    }
    if (!scalar_ok) { ++skipped; continue; }
    ++checked;
    const auto batches = std::ranges::count_if(plan.srt_batch_counts, [](uint32_t count) { return count > 1; });
    const bool changed = gpu_scalar_data ? plan.srt_reads.size() != gpu_plan.srt_reads.size() : batches != 0;
    if (compile_spirv) {
      auto compiled = CompileProgram(std::move(gpu_translated), options, batch_spec, record.push_data_cursor);
      ValidateSpirv("ResourceReplayGPUScalar", compiled.spirv);
      eligible += changed;
      continue;
    }
    if (!changed) continue;
    ++eligible;
    const auto measure = [&](bool batching) {
      runtime.batch_reads = gpu_scalar_data || batching;
      const auto& timed_plan = gpu_scalar_data && batching ? gpu_plan : plan;
      const auto begin = std::chrono::steady_clock::now();
      for (unsigned i = 0; i < 20; ++i)
        if (!MaterializeResources(timed_plan, runtime, batch, batch_spec)) return -1.0;
      return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 20;
    };
    const auto scalar_us = measure(false);
    const auto batch_us = measure(true);
    if (scalar_us < 0 || batch_us < 0) return 1;
    total_scalar += scalar_us;
    total_batch += batch_us;
    std::printf("ResourceReplay: hash=%016llx stage=%u scalar_us=%.2f candidate_us=%.2f starts=%zu srt=%zu candidate_srt=%zu\n",
                static_cast<unsigned long long>(record.hash), static_cast<unsigned>(record.stage),
                scalar_us, batch_us, static_cast<size_t>(batches), plan.srt_reads.size(), candidate_plan.srt_reads.size());
  }
  std::printf("ResourceReplay: mode=%s checked=%u eligible=%u skipped=%u sum_scalar_us=%.2f sum_candidate_us=%.2f\n",
              compile_spirv ? "gpu-scalar-spirv" : gpu_scalar_data ? "gpu-scalar-data" : "batch",
              checked, eligible, skipped, total_scalar, total_batch);
  return 0;
}

// CPU-only per-phase cost of batched materialization for every replayable record:
// --resource-replay-phases <copied .shaders file> [iterations]. Memory reads come from the
// recorded words, so the absolute cost of reads differs from live guest-memory reads.
int ResourceReplayPhases(const char* path, unsigned iterations, int direct) {
  using namespace ShaderRecompiler;
  using namespace ShaderRecompiler::IR;
  PipelineRecord::Records records;
  PipelineRecord::RecordFile file;
  if (!file.Open(std::filesystem::path(path), records) || records.programs.empty()) return 2;
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
  struct ReadStats { uint64_t calls = 0, words = 0; };
  MaterializationTimings totals;
  uint64_t total_ns = 0, read_calls = 0, read_words = 0;
  unsigned checked = 0, skipped = 0, mismatches = 0, direct_plans = 0;
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
    struct Context { Memory memory; ReadStats stats; } context;
    for (const auto read : record.raw_reads) context.memory.ordinary.emplace(read.address, read);
    for (const auto read : record.reads) context.memory.strict.emplace(read.address, read);
    SrtRuntime runtime{
      .user_data = record.user_data,
      .shader_base = record.shader_base,
      .read_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        auto& c = *static_cast<Context*>(data);
        ++c.stats.calls;
        c.stats.words += words.size();
        return Memory::Read(c.memory.ordinary, address, words);
      },
      .userdata = &context,
      .read_specialization_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        auto& c = *static_cast<Context*>(data);
        ++c.stats.calls;
        c.stats.words += words.size();
        return Memory::Read(c.memory.strict, address, words);
      },
      .workgroup_counts = record.compute.workgroup_counts,
      .batch_reads = true,
    };
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    // The direct SRT path must reproduce the interpreter exactly (captured specialization reads
    // as merged address ranges: the direct path may split or repeat reads of the same bytes).
    const auto merged = [](const ResourceSnapshot& snapshot) {
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
    SetDirectSrtReadsOverride(0);
    ResourceSnapshot reference;
    ResourceSpecialization reference_spec;
    const bool reference_ok = MaterializeResources(plan, runtime, reference, reference_spec);
    SetDirectSrtReadsOverride(1);
    const bool direct_ok = MaterializeResources(plan, runtime, snapshot, specialization);
    if (reference_ok != direct_ok ||
        (reference_ok && (reference.buffers != snapshot.buffers || reference.images != snapshot.images ||
                          reference.samplers != snapshot.samplers ||
                          reference.flattened_srt != snapshot.flattened_srt ||
                          reference.user_data != snapshot.user_data ||
                          reference.uniform_fill != snapshot.uniform_fill ||
                          merged(reference) != merged(snapshot) ||
                          reference_spec != specialization))) {
      if (mismatches < 8)
        std::fprintf(stderr, "ResourcePhases: direct SRT mismatch hash=%016llx stage=%u ok=%d/%d buffers=%d images=%d samplers=%d flat=%d user=%d fill=%d reads=%d(%zu/%zu) spec=%d\n",
                     static_cast<unsigned long long>(record.hash), static_cast<unsigned>(record.stage),
                     reference_ok ? 1 : 0, direct_ok ? 1 : 0, reference.buffers != snapshot.buffers,
                     reference.images != snapshot.images, reference.samplers != snapshot.samplers,
                     reference.flattened_srt != snapshot.flattened_srt, reference.user_data != snapshot.user_data,
                     reference.uniform_fill != snapshot.uniform_fill,
                     merged(reference) != merged(snapshot),
                     reference.specialization_reads.size(), snapshot.specialization_reads.size(),
                     reference_spec != specialization);
      ++mismatches;
    }
    direct_plans += plan.direct_srt_state == ResourcePlan::DirectSrtState::Direct;
    SetDirectSrtReadsOverride(direct);
    if (!MaterializeResources(plan, runtime, snapshot, specialization)) { ++skipped; continue; }
    ++checked;
    context.stats = {};
    MaterializationTimings timings;
    runtime.materialization_timings = &timings;
    const auto begin = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) {
      if (!MaterializeResources(plan, runtime, snapshot, specialization)) return 1;
    }
    total_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - begin).count());
    for (size_t i = 0; i < timings.ns.size(); ++i) totals.ns[i] += timings.ns[i];
    read_calls += context.stats.calls;
    read_words += context.stats.words;
  }
  static const char* const names[] = {"setup", "flat", "scalar_ranges", "fill", "buffers", "images", "samplers", "specialization"};
  SetDirectSrtReadsOverride(-1);
  std::printf("ResourcePhases: direct=%d direct_plans=%u mismatches=%u\n", direct, direct_plans, mismatches);
  std::printf("ResourcePhases: checked=%u skipped=%u iterations=%u total_us/pass=%.1f reads/pass=%.0f words/pass=%.0f\n",
              checked, skipped, iterations, total_ns / 1e3 / iterations,
              static_cast<double>(read_calls) / iterations, static_cast<double>(read_words) / iterations);
  for (size_t i = 0; i < totals.ns.size(); ++i)
    std::printf("ResourcePhases:   %-15s us/pass=%.1f\n", names[i], totals.ns[i] / 1e3 / iterations);
  return mismatches == 0 ? 0 : 1;
}

// Histogram of expression shapes evaluated by SRT reads and descriptor dwords:
// --resource-replay-shapes <copied .shaders file>
int ResourceReplayShapes(const char* path) {
  using namespace ShaderRecompiler;
  using namespace ShaderRecompiler::IR;
  PipelineRecord::Records records;
  PipelineRecord::RecordFile file;
  if (!file.Open(std::filesystem::path(path), records) || records.programs.empty()) return 2;
  std::map<std::string, uint64_t> srt_shapes, descriptor_shapes;
  std::function<std::string(Value, int)> shape = [&](Value value, int depth) -> std::string {
    value = value.Resolve();
    if (value.IsImmediate()) return "imm";
    auto* inst = value.TryInstruction();
    if (inst == nullptr) return "?";
    std::string name(ValueOpcodeName(inst->GetOpcode()));
    if (inst->GetOpcode() == ValueOpcode::ReadConst) return "ReadConst";
    if (depth >= 4) return name + "(...)";
    std::string result = name + "(";
    for (size_t i = 0; i < inst->NumArgs(); ++i) {
      if (i != 0) result += ",";
      result += shape(inst->Arg(i), depth + 1);
    }
    return result + ")";
  };
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
      default: continue;
    }
    auto translated = TranslateProgram(record.code, options);
    if (translated.build_failed) continue;
    auto plan = ExtractResourcePlan(translated.program);
    for (const auto& read : plan.srt_reads) ++srt_shapes[shape(read.value, 0)];
    for (const auto& source : plan.descriptor_sources)
      for (uint32_t i = 0; i < source.dword_count; ++i) ++descriptor_shapes[shape(source.dwords[i], 0)];
  }
  const auto print = [](const char* title, const std::map<std::string, uint64_t>& shapes) {
    std::vector<std::pair<uint64_t, std::string>> sorted;
    uint64_t total = 0;
    for (const auto& [s, n] : shapes) { sorted.emplace_back(n, s); total += n; }
    std::ranges::sort(sorted, std::greater{});
    std::printf("%s: total=%llu distinct=%zu\n", title, static_cast<unsigned long long>(total), sorted.size());
    for (size_t i = 0; i < sorted.size() && i < 25; ++i)
      std::printf("  %8llu %5.1f%% %s\n", static_cast<unsigned long long>(sorted[i].first),
                  100.0 * sorted[i].first / total, sorted[i].second.substr(0, 220).c_str());
  };
  print("SrtShapes", srt_shapes);
  print("DescriptorShapes", descriptor_shapes);
  return 0;
}

// Prints the translated IR and uniform-fill/copy analysis of one recorded shader:
// --resource-replay-dump <copied .shaders file> <hex hash>
int ResourceReplayDump(const char* path, uint64_t hash) {
  using namespace ShaderRecompiler;
  using namespace ShaderRecompiler::IR;
  PipelineRecord::Records records;
  PipelineRecord::RecordFile file;
  if (!file.Open(std::filesystem::path(path), records) || records.programs.empty()) return 2;
  for (auto& record : records.programs) {
    if (record.hash != hash) continue;
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
      default: continue;
    }
    auto translated = TranslateProgram(record.code, options);
    if (translated.build_failed) { std::printf("build failed\n"); return 1; }
    std::printf("%s\n", ProgramToString(translated.program).c_str());
    const auto plan = ExtractResourcePlan(translated.program);
    std::printf("uniform_fill kind=%u words=%u uniform_copy valid=%d buffers=%zu images=%zu\n",
                static_cast<unsigned>(plan.uniform_fill.fill.kind), plan.uniform_fill.fill.words,
                plan.uniform_copy.valid ? 1 : 0, plan.info.buffers.size(), plan.info.images.size());
    return 0;
  }
  std::printf("hash not found\n");
  return 1;
}
