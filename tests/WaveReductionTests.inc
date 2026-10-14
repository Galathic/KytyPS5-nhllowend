// Exercise the emitted reduction on the GPU with partially populated subgroups,
// sparse control flow, signed inputs and all six associative operations.
void CheckWaveReduction(VulkanHarness& vulkan) {
  using namespace ShaderRecompiler::IR;
  using namespace ShaderRecompiler::CFG;
  constexpr std::array operations{WaveReduceOp::UMin, WaveReduceOp::SMin,
      WaveReduceOp::UMax, WaveReduceOp::SMax, WaveReduceOp::Or, WaveReduceOp::And};
  const auto Combine = [](WaveReduceOp op, u32 lhs, u32 rhs) -> u32 {
    switch (op) {
      case WaveReduceOp::UMin: return std::min(lhs, rhs);
      case WaveReduceOp::SMin: return std::bit_cast<u32>(std::min(
          std::bit_cast<int32_t>(lhs), std::bit_cast<int32_t>(rhs)));
      case WaveReduceOp::UMax: return std::max(lhs, rhs);
      case WaveReduceOp::SMax: return std::bit_cast<u32>(std::max(
          std::bit_cast<int32_t>(lhs), std::bit_cast<int32_t>(rhs)));
      case WaveReduceOp::Or: return lhs | rhs;
      case WaveReduceOp::And: return lhs & rhs;
    }
    std::abort();
  };
  const auto Input = [](u32 lane) { return (lane + 1u) ^ ((lane & 1u) << 31u); };
  for (const u32 threads : {1u, 7u, 19u, 32u}) {
    for (const bool sparse : {false, true}) {
      TestCase test;
      const std::string name = "WaveReduction" + std::to_string(threads) + (sparse ? "Sparse" : "Full");
      test.name = name.c_str();
      test.initial.assign(threads * operations.size(), 0xdeadbeefu);
      test.expected = test.initial;
      CompiledShader compiled;
      auto& program = compiled.program;
      program.stage = ShaderType::Compute;
      program.wave_size = 32;
      program.srt_plan_complete = true;
      program.resource_tracking_complete = true;
      program.shader_info_complete = true;
      for (u32 id = 0; id < 3; ++id) {
        program.block_storage.push_back(std::make_unique<Block>());
        program.blocks.push_back(program.block_storage.back().get());
        program.blocks.back()->id = id;
      }
      auto* entry = program.blocks[0];
      auto* body = program.blocks[1];
      auto* end = program.blocks[2];
      entry->AddBranch(body);
      entry->AddBranch(end);
      body->AddBranch(end);
      auto& lane = entry->AppendNewInst(ValueOpcode::LaneId);
      auto& odd = entry->AppendNewInst(ValueOpcode::BitwiseAnd32, {Value(&lane), Value(1u)});
      auto& active = entry->AppendNewInst(ValueOpcode::IEqual32, {Value(&odd), Value(0u)});
      entry->condition = sparse ? Value(&active) : Value(true);
      entry->terminator = {.kind = TerminatorKind::ConditionalBranch,
          .true_block = body, .false_block = end, .merge_block = end};
      body->terminator = {.kind = TerminatorKind::Branch, .true_block = end};
      auto& magnitude = body->AppendNewInst(ValueOpcode::IAdd32, {Value(&lane), Value(1u)});
      auto& sign = body->AppendNewInst(ValueOpcode::ShiftLeftLogical32, {Value(&odd), Value(31u)});
      auto& input = body->AppendNewInst(ValueOpcode::BitwiseXor32, {Value(&magnitude), Value(&sign)});
      auto& output = body->AppendNewInst(ValueOpcode::GetBufferResource,
          {Value(0u), Value(0u), Value(0u), Value(0u)});
      output.SetFlags<u32>(0u);
      program.info.buffers.push_back({.packed_stride = 1, .written = true});
      program.memory_info.push_back({.kind = ResourceKind::Buffer, .offen = true});
      auto& base = body->AppendNewInst(ValueOpcode::IMul32, {Value(&lane), Value(24u)});
      for (u32 index = 0; index < operations.size(); ++index) {
        const auto op = operations[index];
        auto& reduced = body->AppendNewInst(ValueOpcode::WaveReduceU32, {Value(&input)});
        reduced.SetFlags(op);
        auto& offset = body->AppendNewInst(ValueOpcode::IAdd32, {Value(&base), Value(index * 4u)});
        auto& store = body->AppendNewInst(ValueOpcode::StoreBufferU32,
            {Value(&output), Value(0u), Value(&offset), Value(0u), Value(&reduced), Value(true)});
        store.SetFlags(MemoryFlags{0u, 0u});
        u32 expected = Input(0);
        for (u32 lane_id = 1; lane_id < threads; ++lane_id) {
          if (!sparse || (lane_id & 1u) == 0) expected = Combine(op, expected, Input(lane_id));
        }
        for (u32 lane_id = 0; lane_id < threads; ++lane_id) {
          if (!sparse || (lane_id & 1u) == 0) test.expected[lane_id * operations.size() + index] = expected;
        }
      }
      ShaderComputeInputInfo compute{};
      compute.wave_size = 32;
      compute.host_subgroup_size = vulkan.SubgroupSize();
      compute.threads_num[0] = threads;
      compute.threads_num[1] = compute.threads_num[2] = 1;
      compiled.spirv = ShaderRecompiler::Spirv::EmitProgram(program, {.compute = &compute});
      compiled.packed_user_data.resize(program.bindings.ShaderDataDwords());
      ValidateSpirv(test.name, compiled.spirv);
      spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
      std::string text;
      Require(test.name, "lowering disassembly", tools.Disassemble(compiled.spirv, &text), "cannot disassemble");
      Require(test.name, "reduction operation", CountText(text, " Reduce ") == operations.size(),
          "each reduction must use the literal Reduce group operation");
      auto buffer = vulkan.CreateStorageBuffer(test.name, test.initial, test.initial.size());
      vulkan.Dispatch(test, compiled, buffer);
      const auto actual = vulkan.ReadBuffer(test.name, buffer, test.expected.size());
      vulkan.DestroyBuffer(&buffer);
      CompareWords(test, "partial subgroup and sparse control flow", test.expected, actual);
      std::printf("[compute] %-32s ok\n", test.name);
    }
  }
}
