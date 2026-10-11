namespace {
struct ScalarBatchReadState {
  uint32_t mode;
  std::vector<uint64_t> ordinary, strict;
  std::vector<uint32_t> ordinary_sizes, strict_sizes, observed;
};
template<bool strict>
bool ScalarBatchTestRead(void* data, uint64_t address, std::span<uint32_t> words) {
  auto& s = *static_cast<ScalarBatchReadState*>(data);
  auto& addresses = strict ? s.strict : s.ordinary;
  auto& sizes = strict ? s.strict_sizes : s.ordinary_sizes;
  addresses.push_back(address); sizes.push_back(static_cast<uint32_t>(words.size()));
  if (words.size() != 1u || address < 0x1000u || address >= 0x1010u ||
      (!strict && s.mode == 2u && address == 0x1004u)) return false;
  words[0] = (strict ? 110u : 10u) + static_cast<uint32_t>((address - 0x1000u) / 4u);
  return true;
}
template<bool strict>
bool ScalarBatchTestPeek(void* data, uint64_t address, std::span<uint32_t> words) {
  auto& s = *static_cast<ScalarBatchReadState*>(data);
  if (s.mode != 0u || address != 0x1000u || words.size() != 4u) return false;
  auto& addresses = strict ? s.strict : s.ordinary;
  auto& sizes = strict ? s.strict_sizes : s.ordinary_sizes;
  addresses.push_back(address); sizes.push_back(4u);
  for (uint32_t i = 0; i < 4u; ++i) words[i] = (strict ? 110u : 10u) + i;
  return true;
}
void TestScalarBatchDomainsFailureAndActivity() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct PublishedGuard {
    uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved, std::memory_order_relaxed); }
  } guard;
  auto flags = Common::RuntimeFlags::Current();
  flags.srt_scalar_batch = true;
  Common::RuntimeFlags::g_published.store((1ull << Common::RuntimeFlags::kRevisionShift) | Common::RuntimeFlags::Bits(flags), std::memory_order_relaxed);
  const char* verifier = std::getenv("KYTY_SRT_TAPE_VERIFY");
  const bool batch_allowed = verifier == nullptr || verifier[0] != '1';
  for (bool cfg : {false, true}) for (uint32_t mode : {0u, 1u, 2u}) {
    ResourcePlan plan;
    plan.stage = Libs::Graphics::ShaderType::Compute;
    plan.srt_plan_complete = plan.resource_tracking_complete = true;
    const auto user = [&](uint32_t index) {
      auto& inst = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
      inst.SetArg(0, Value(static_cast<ScalarReg>(index)));
      return Value(&inst);
    };
    const auto low = user(0), offset = user(1), condition = user(2);
    auto& handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    handle.SetArg(0, low); handle.SetArg(1, Value(0u));
    for (uint32_t i = 0; i < 12u; ++i) {
      MemoryInfo memory;
      memory.kind = ResourceKind::ScalarAddress; memory.planning_only = true;
      memory.offset = (i % 4u) * 4u; memory.component_index = i % 4u;
      if (i >= 8u) memory.offset += 0xdead0000u;
      plan.memory_info.push_back(memory);
      auto& inst = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
      inst.SetFlags(MemoryFlags{.index = i, .pc = 0x40u + (i / 4u) * 4u});
      inst.SetArg(0, Value(&handle)); inst.SetArg(1, offset);
      inst.SetArg(2, Value(0u)); inst.SetArg(3, Value(true));
      plan.srt_reads.push_back({Value(&inst), i});
    }
    // Without CFG, exclude inactive poison slots altogether.
    if (!cfg) plan.srt_reads.resize(8u);
    plan.clean_flat_slots.assign(plan.srt_reads.size(), 0u);
    std::fill(plan.clean_flat_slots.begin() + 4u, plan.clean_flat_slots.begin() + 8u, 1u);
    if (cfg) {
      plan.control_flow.resize(3);
      plan.control_flow[0].condition = condition; plan.control_flow[0].successors = {1u, 2u};
      plan.control_flow[1].srt_reads = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
      plan.control_flow[2].srt_reads = {8u, 9u, 10u, 11u};
    }
    ScalarBatchReadState state{mode};
    std::array<uint32_t, 3> user_data{0x1000u, 0u, 1u};
    SrtEvaluationCounts counts;
    const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = ScalarBatchTestRead<false>, .userdata = &state,
      .read_specialization_memory = ScalarBatchTestRead<true>,
      .observe_user_data = +[](void* data, uint32_t index) {
        static_cast<ScalarBatchReadState*>(data)->observed.push_back(index);
      }, .user_data_observer_userdata = &state, .evaluation_counts = &counts,
      .try_read_memory_span = ScalarBatchTestPeek<false>,
      .try_read_specialization_span = ScalarBatchTestPeek<true>};
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    std::vector<uint32_t> flat;
    Check(walker.RefreshFlatBuffer(flat) == (mode != 2u), "scalar batch changed failure outcome");
    if (mode == 2u) {
      Check(state.ordinary == std::vector<uint64_t>{0x1000u, 0x1004u} && state.strict.empty(),
            "unavailable scalar peek speculated past the first failed word");
    } else {
      const bool batched = mode == 0u && batch_allowed;
      const std::vector<uint64_t> expected = batched ? std::vector<uint64_t>{0x1000u}
        : std::vector<uint64_t>{0x1000u, 0x1004u, 0x1008u, 0x100cu};
      Check(state.ordinary == expected && state.strict == expected,
            "scalar batching mixed reader domains, touched inactive memory, or failed to batch");
      for (uint32_t i = 0; i < 4u; ++i) {
        Check(flat[i] == 10u + i && flat[4u + i] == 110u + i,
              "scalar batching changed ordinary or specialization values");
      }
      Check(counts.scalar_batch_words == (batched ? 8u : 0u), "scalar batch counter missed actual execution");
      // Each reader domain evaluates shared address SGPRs once, not per word.
      if (batch_allowed) Check(std::ranges::count(state.observed, 0u) == 2 && std::ranges::count(state.observed, 1u) == 2,
                               "scalar batching changed shared per-domain register observations");
    }
  }
}
} // namespace
