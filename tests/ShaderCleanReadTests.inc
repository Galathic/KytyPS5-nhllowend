// Exercise the actual PipelineCache runtime reader through a scalar-loaded
// buffer descriptor, rather than duplicating its callback in the test.
if (Common::RuntimeFlags::Current().shader_clean_reads ||
    Common::RuntimeFlags::Current().srt_scalar_batch) {
  const auto descriptor_address = base + 0x800;
  ShaderBufferResource descriptor{};
  descriptor.UpdateAddress48(base + 0x900);
  descriptor.fields[2] = 16;
  descriptor.fields[3] = DstSel(4, 5, 6, 7) |
      (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
  for (uint32_t i = 0; i < 4; ++i) {
    Require(name, "ordinary descriptor initialization",
            cache.TryWriteLabel(descriptor_address + i * 4, &descriptor.fields[i], 4),
            "ordinary descriptor initialization failed");
  }
  std::vector<u32> code{
      EncodeSmem0(0x02, 8), EncodeSmem1(0, 125),
      EncodeVop1(0x01, 1, InlineU32(1)),
      EncodeMubuf0(0x1c, 0, false, false), EncodeMubuf1(1, 2, 0)};
  AppendEnd(&code);
  ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
      {.type = Prospero::ShaderBinaryType::kCs,
       .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
  HW::Shader test_shaders{};
  test_shaders.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                            .num_thread_x = 1, .num_thread_y = 1, .num_thread_z = 1,
                            .wave_size = 64, .user_sgpr = 2});
  test_shaders.SetCsUserSgpr(0, static_cast<u32>(descriptor_address), HW::UserSgprType::Unknown);
  test_shaders.SetCsUserSgpr(1, static_cast<u32>(descriptor_address >> 32u),
                           HW::UserSgprType::Unknown);
  auto neighbor = cache.ObtainBuffer(base, 4, true, false);
  neighbor.first->Fill(neighbor.second, 4, pattern);
  const bool native_span = !Common::RuntimeFlags::Current().shader_clean_reads;
  if (native_span) {
    // Establish the original native-read behavior on this protected page:
    // ordinary reads can fault and synchronize GPU-owned neighboring bytes.
    std::array<uint32_t, 4> baseline{};
    std::memcpy(baseline.data(), reinterpret_cast<const void*>(descriptor_address), sizeof(baseline));
    Require(name, "ordinary native baseline",
            std::equal(std::begin(descriptor.fields), std::end(descriptor.fields), baseline.begin()) &&
                !cache.HasGpuDirtyBytes(base, 4),
            "native baseline did not expose current descriptor and resolve protected neighbor");
    neighbor = cache.ObtainBuffer(base, 4, true, false);
    neighbor.first->Fill(neighbor.second, 4, pattern);
  }
  const auto tick_before = scheduler.CurrentTick();
  ShaderComputeInputInfo input{};
  input.workgroup_counts[0] = input.workgroup_counts[1] = input.workgroup_counts[2] = 1;
  const auto program = context.GetPipelineCache().GetComputeProgram(
      test_shaders.GetCs(), processor.GetCtx().GetShaderRegisters(), input);
  Require(name, "ordinary clean descriptor under GPU neighbor",
          program.id != 0 && input.stage.resources != nullptr &&
              input.stage.resources->buffers.size() == 1 &&
              std::equal(std::begin(descriptor.fields), std::end(descriptor.fields),
                         input.stage.resources->buffers[0].dwords.begin()) &&
              (native_span ? (scheduler.CurrentTick() > tick_before && !cache.HasGpuDirtyBytes(base, 4))
                           : (scheduler.CurrentTick() == tick_before && cache.HasGpuDirtyBytes(base, 4))),
          "ordinary descriptor values or selected native/backing synchronization behavior changed");
  // A CPU-side update of the same backing must be observed at the next materialization.
  descriptor.fields[2] = 8;
  Require(name, "ordinary descriptor mutation",
          cache.TryWriteLabel(descriptor_address + 8, &descriptor.fields[2], 4),
          "descriptor mutation failed");
  context.GetPipelineCache().GetComputeProgram(
      test_shaders.GetCs(), processor.GetCtx().GetShaderRegisters(), input);
  Require(name, "ordinary descriptor reread", input.stage.resources->buffers.size() == 1 &&
              input.stage.resources->buffers[0].dwords[2] == descriptor.fields[2],
          "ordinary reader retained stale descriptor values");
  if (native_span) {
    // True overlap with GPU-written ordinary data also retains the baseline
    // fault/readback path; completed spans cannot certify it clean or use stale backing.
    auto gpu_records = cache.ObtainBuffer(descriptor_address + 8, 4, true, false);
    gpu_records.first->Fill(gpu_records.second, 4, 12u);
    context.GetPipelineCache().GetComputeProgram(
        test_shaders.GetCs(), processor.GetCtx().GetShaderRegisters(), input);
    Require(name, "ordinary native dirty descriptor", input.stage.resources->buffers.size() == 1 &&
                input.stage.resources->buffers[0].dwords[2] == 12u &&
                !cache.HasGpuDirtyBytes(descriptor_address + 8, 4),
            "native completed span failed to synchronize actual GPU-written descriptor");
  }
  cache.ReadMemory(base, 0x1000);
}
