void TestResourceSpecializationCacheTransitions() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using namespace Libs::Graphics;
  Program program;
  program.stage = ShaderType::Compute;
  program.user_data_count = 16;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto& block = AddValueBlock(program);
  for (const auto [start, count] : {std::pair{0u, 4u}, {4u, 8u}, {12u, 4u}}) {
    DescriptorSource source;
    source.dword_count = count;
    source.dwords.fill(Value(0u));
    for (uint32_t i = 0; i < count; ++i) {
      auto& reg = block.AppendNewInst(ValueOpcode::GetUserData,
                                      {Value(static_cast<ScalarReg>(start + i))});
      source.dwords[i] = Value(&reg);
    }
    program.descriptor_sources.push_back(source);
  }
  program.info.buffers.push_back({.source = 0, .formatted = true});
  program.info.images.push_back({.source = 1,
    .resource_class = ImageResourceClass::Sampled,
    .numeric_class = Prospero::TextureNumericClass::Float,
    .dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D, .mip_mode = ImageMipMode::Dynamic});
  program.info.samplers.push_back({.source = 2, .gather_lod = true});
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 16> data {};
  data[0] = 0x1000; data[1] = 16u << 16; data[2] = 64;
  data[3] = DstSel(4, 5, 6, 7) | (static_cast<uint32_t>(Prospero::BufferFormat::k32Float) << 12);
  data[4] = 0x2000;
  data[5] = static_cast<uint32_t>(Prospero::BufferFormat::k32Float) << 20;
  data[7] = DstSel(4, 5, 6, 7) | (3u << 16) |
            (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28);
  ResourceSpecializationCache cache;
  ResourceSnapshot actual, expected;
  ResourceSpecialization actual_spec, expected_spec;
  const auto run = [&](bool success = true) {
    SrtRuntime runtime {.user_data = data};
    const bool reference = MaterializeResources(plan, runtime, expected, expected_spec);
    runtime.specialization_cache = &cache;
    const bool candidate = MaterializeResources(plan, runtime, actual, actual_spec);
    Check(reference == candidate && reference == success, "specialization cache changed success/failure");
    if (success) {
      Check(actual.buffers == expected.buffers && actual.images == expected.images &&
            actual.samplers == expected.samplers && actual.flattened_srt == expected.flattened_srt &&
            actual.user_data == expected.user_data && actual.uniform_fill == expected.uniform_fill &&
            actual.specialization_reads == expected.specialization_reads && actual_spec == expected_spec,
            "specialization cache retained stale outputs across descriptor transition");
    }
  };
  run();
  const auto before = GetResourceSpecializationCacheStats();
  const auto stable = actual_spec;
  data[0] += 0x1000; data[1] |= 1; data[2] = 128; data[4] += 0x1000;
  data[12] = 0x1234; // Sampler payload remains live even when specialization hits.
  run();
  const auto hits = GetResourceSpecializationCacheStats();
  Check(actual_spec == stable && actual.buffers[0].dwords[0] == data[0] &&
        actual.buffers[0].dwords[2] == data[2] && actual.images[0].dwords[0] == data[4] &&
        actual.samplers[0].dwords[0] == data[12] &&
        hits.buffer_hits == before.buffer_hits + 1 && hits.image_hits == before.image_hits + 1,
        "address/length-only updates missed component cache or froze descriptor payloads");
  data[1] = 32u << 16; run();
  Check(actual_spec != stable, "changed buffer stride retained old specialization");
  data[1] = 0; data[3] |= 1u << 28; run();
  data[3] ^= 1u << 28; run();
  data[5] = static_cast<uint32_t>(Prospero::BufferFormat::k32UInt) << 20; run();
  data[7] = (data[7] & ~(0xfu << 16)) | (1u << 16); run();
  data[4] = 0; run(); // Nullness changes although the retained image words match.
  run();
  data[4] = 0x4000;
  data[5] = static_cast<uint32_t>(Prospero::BufferFormat::kFmask8_S2_F1) << 20;
  run(); run();
  Check(actual.images.empty() && actual_spec.images[0].fmask,
        "cached FMASK did not compact the fresh runtime image snapshot");
  data[4] = 0; run();
  Check(actual.images.size() == 1 && !actual_spec.images[0].fmask,
        "FMASK-to-null transition retained stale image metadata");
  data[4] = 0x5000;
  data[5] = static_cast<uint32_t>(Prospero::BufferFormat::k32Float) << 20; run();
  data[14] = 2u << 26; run(false);
  Check(cache.owner == nullptr && !cache.active, "failed refresh retained partially published cache entries");
  data[14] = 0; run();
  auto other_plan = ExtractResourcePlan(program);
  SrtRuntime other_runtime {.user_data = data, .specialization_cache = &cache};
  Check(MaterializeResources(other_plan, other_runtime, actual, actual_spec) &&
        cache.owner == &other_plan, "cache was not reset on immutable plan replacement");
  std::puts("ResourceSpecializationCacheTransitions ok");
}
