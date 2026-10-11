// Focused ordered-demand schedule regression: include before the test main.
namespace {
void TestCfgFlagSnapshotSurvivesReaderPublication() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct PublishedGuard {
    uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved, std::memory_order_relaxed); }
  } published_guard;
  const auto set_cfg = [](bool enabled) {
    auto flags = Common::RuntimeFlags::Current(); flags.srt_cfg_tape = enabled;
    Common::RuntimeFlags::g_published.store((1ull << Common::RuntimeFlags::kRevisionShift) | Common::RuntimeFlags::Bits(flags),
                                           std::memory_order_relaxed);
  };
  ResourcePlan plan;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info = {{.kind = ResourceKind::ScalarAddress, .offset = 0u, .planning_only = true},
                      {.kind = ResourceKind::ScalarAddress, .offset = 4u, .planning_only = true}};
  auto& handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x1000u)); handle.SetArg(1, Value(0u));
  for (uint32_t i = 0; i < 2u; ++i) {
    auto& read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = i}); read.SetArg(0, Value(&handle));
    read.SetArg(1, Value(0u)); read.SetArg(2, Value(0u)); read.SetArg(3, Value(true));
    plan.srt_reads.push_back({Value(&read), i});
  }
  plan.control_flow.resize(1); plan.control_flow[0].srt_reads = {0u, 1u};
  struct Reads { bool publish = false; uint32_t calls = 0; } reads;
  const SrtRuntime runtime{
    .read_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
      auto& state = *static_cast<Reads*>(data); ++state.calls;
      if (state.publish) {
        auto flags = Common::RuntimeFlags::Current(); flags.srt_cfg_tape = true;
        Common::RuntimeFlags::g_published.store((1ull << Common::RuntimeFlags::kRevisionShift) | Common::RuntimeFlags::Bits(flags),
                                               std::memory_order_relaxed);
      }
      if (words.size() != 1u) return false;
      words[0] = static_cast<uint32_t>(address); return true;
    }, .userdata = &reads};
  std::vector<uint32_t> flat;
  set_cfg(true);
  Check(SrtWalker(plan, runtime).RefreshFlatBuffer(flat), "CFG snapshot fixture failed to warm tape");
  set_cfg(false);
  // Keep the warmed depth-zero context occupied: this walker gets fresh scratch
  // while the plan's CFG tape remains ready from the previous session.
  SrtWalker outer(plan, runtime);
  SrtEvaluationCounts counts;
  auto counted = runtime; counted.evaluation_counts = &counts;
  SrtWalker walker(plan, counted);
  reads.publish = true; reads.calls = 0;
  Check(walker.RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{0x1000u, 0x1004u}
          && reads.calls == 2u && counts.tape_ops == 0u,
        "reader publication changed a walker's latched CFG choice or used unprepared scratch");
}

void TestCfgScheduleReadFailureAndSharedDag() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto flags = Common::RuntimeFlags::Current();
  for (const uint64_t failing_address : {uint64_t{0}, uint64_t{0x1000}, uint64_t{0x1004}}) {
    ResourcePlan plan;
    plan.srt_plan_complete = plan.resource_tracking_complete = true;
    plan.memory_info = {{.kind = ResourceKind::ScalarAddress, .offset = 0u, .planning_only = true},
                        {.kind = ResourceKind::ScalarAddress, .offset = 4u, .planning_only = true}};
    auto& handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    handle.SetArg(0, Value(0x1000u)); handle.SetArg(1, Value(0u));
    const auto read = [&](uint32_t memory_index) {
      auto& inst = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
      inst.SetFlags(MemoryFlags{.index = memory_index});
      inst.SetArg(0, Value(&handle)); inst.SetArg(1, Value(0u));
      inst.SetArg(2, Value(0u)); inst.SetArg(3, Value(true));
      return Value(&inst);
    };
    auto& sum = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
    sum.SetArg(0, read(0)); sum.SetArg(1, read(1));
    Value root(&sum);
    // An exponentially shared graph must compile to a bounded DAG schedule.
    for (uint32_t i = 0; i < 20; ++i) {
      auto& node = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
      node.SetArg(0, root); node.SetArg(1, root); root = Value(&node);
    }
    plan.srt_reads = {{root, 0u}, {root, 1u}};
    plan.control_flow.resize(1);
    plan.control_flow[0].srt_reads = {0u, 1u};
    struct Reads { uint64_t failing; std::vector<uint64_t> order; } reads{failing_address, {}};
    SrtEvaluationCounts counts;
    const SrtRuntime runtime{
      .read_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        auto& state = *static_cast<Reads*>(data);
        state.order.push_back(address);
        if (words.size() != 1u || address == state.failing) return false;
        words[0] = address == 0x1000u ? 3u : 5u;
        return true;
      }, .userdata = &reads, .evaluation_counts = &counts};
    std::vector<uint32_t> flat;
    SrtWalker walker(plan, runtime);
    const bool success = walker.RefreshFlatBuffer(flat);
    Check(success == (failing_address == 0), "scheduled DAG changed read failure outcome");
    const std::vector<uint64_t> expected = failing_address == 0x1000u
      ? std::vector<uint64_t>{0x1000u} : std::vector<uint64_t>{0x1000u, 0x1004u};
    Check(reads.order == expected, "scheduled DAG speculated, reordered, or repeated a raw read");
    uint32_t result = 0;
    Check(walker.Evaluate(root, result) == success && reads.order == expected,
          "scheduled DAG did not preserve successful and failed instruction memos");
    if (success) Check(result == (8u << 20u) && flat == std::vector<uint32_t>{result, result},
                       "scheduled DAG changed repeated-root output");
    if (flags.srt_cfg_tape && flags.srt_cfg_schedule) {
      Check(counts.cfg_schedule_steps > 0, "schedule regression silently used the demand stack");
      const auto& tape = plan.cfg_srt_tapes[0];
      const auto& schedule = tape.demand_schedules[tape.slot_ops[0]];
      Check(schedule.state == ResourcePlan::SrtTape::State::Ready && schedule.steps.size() < 100u,
            "shared DAG schedule expanded exponentially or fell back");
    }
  }
}
} // namespace
