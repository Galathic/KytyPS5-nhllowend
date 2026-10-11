namespace {
void TestResourceMemoRuntimeDependencySwitch() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct PublishedGuard {
    uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved, std::memory_order_relaxed); }
  } published_guard;
  const auto set_flags = [](bool dynamic, bool verify) {
    auto flags = Common::RuntimeFlags::Current();
    flags.resource_dynamic_deps = dynamic;
    flags.resource_memo_verify = verify;
    Common::RuntimeFlags::g_published.store((1ull << Common::RuntimeFlags::kRevisionShift) |
                                               Common::RuntimeFlags::Bits(flags),
                                           std::memory_order_relaxed);
  };
  ResourcePlan plan;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  const auto user = [&](uint32_t index) {
    auto& inst = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
    inst.SetArg(0, Value(static_cast<ScalarReg>(index)));
    return Value(&inst);
  };
  const auto selected = user(0), inactive = user(1), condition = user(2);
  plan.descriptor_sources.resize(2);
  for (auto& source : plan.descriptor_sources) {
    source.dword_count = 4; source.dwords.fill(Value(0u));
  }
  plan.descriptor_sources[0].dwords[0] = selected;
  plan.descriptor_sources[1].dwords[0] = inactive;
  plan.info.buffers = {{.source = 0}, {.source = 1}};
  plan.control_flow.resize(3);
  plan.control_flow[0].condition = condition;
  plan.control_flow[0].successors = {1u, 2u};
  plan.control_flow[1].sources = {0u}; plan.control_flow[2].sources = {1u};
  std::array<uint32_t, 4> regs{0x1000u, 0x2000u, 1u, 17u};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemoEntry memo;
  bool reused = false;
  const auto refresh = [&] {
    return memo.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused);
  };
  set_flags(true, false);
  Check(refresh() && !reused && memo.RegisterDependencyCount() == 2u,
        "dynamic runtime fixture did not capture active register dependencies");
  ++regs[1];
  Check(refresh() && reused, "dynamic runtime fixture missed on an inactive register");
  set_flags(false, false);
  Check(refresh() && !reused && memo.RegisterDependencyCount() == 3u,
        "dynamic-to-static runtime switch reused the narrower dependency set");
  ++regs[1];
  Check(refresh() && !reused, "static mode retained dynamic dependencies after recapture");
  set_flags(true, false);
  Check(refresh() && !reused && memo.RegisterDependencyCount() == 2u,
        "static-to-dynamic runtime switch failed to recapture consumed dependencies");
  ++regs[1];
  Check(refresh() && reused, "dynamic mode failed to resume narrowed dependencies");
  set_flags(true, true);
  Check(refresh() && reused && memo.VerifyMatches() == 1u && memo.VerifyMismatches() == 0u,
        "enabling hit verification after capture lost the entry's activity state");

  Libs::Graphics::ResourceMemo alternating(2);
  set_flags(true, false);
  Check(alternating.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused)
          && !reused, "alternating runtime fixture first state failed");
  regs[2] = 0u;
  Check(alternating.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused)
          && !reused && plan.active_sources == std::vector<uint8_t>{0u, 1u},
        "alternating runtime fixture second state failed");
  regs[2] = 1u; regs[3] = 99u;
  Check(alternating.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused)
          && reused && plan.active_sources == std::vector<uint8_t>{1u, 0u}
          && snapshot.user_data[3] == 99u,
        "memo hit failed to restore cached activity or refresh non-resource push data");
  regs[3] = 101u;
  Check(alternating.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused)
          && reused && snapshot.user_data[3] == 101u,
        "repeated last-entry hit retained stale push data");
}

void TestResourceMemoReuseTrace() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct PublishedGuard {
    uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved, std::memory_order_relaxed); }
  } published_guard;
  auto flags = Common::RuntimeFlags::Current();
  flags.resource_dynamic_deps = true;
  flags.resource_memo_verify = false;
  Common::RuntimeFlags::g_published.store((1ull << Common::RuntimeFlags::kRevisionShift) |
                                             Common::RuntimeFlags::Bits(flags),
                                         std::memory_order_relaxed);
  auto plan = UserDataBufferPlan();
  std::array<uint32_t, 2> regs{0u, 10u};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemo memo(16, true);
  bool reused = false;
  const auto refresh = [&] {
    return memo.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused);
  };
  const auto before = Libs::Graphics::GetResourceReuseTraceStats();
  for (uint32_t value = 0; value != 17; ++value) {
    regs[0] = value;
    Check(refresh() && !reused, "reuse-trace fixture failed to fill/evict the 16-entry memo");
  }
  const auto after_evictions = Libs::Graphics::GetResourceReuseTraceStats();
  regs[0] = 999u;
  Check(refresh() && !reused, "reuse-trace fixture changed-register miss failed");
  const auto before_recurrence = Libs::Graphics::GetResourceReuseTraceStats();
  Check(before_recurrence.matched == after_evictions.matched,
        "reuse-trace fixture falsely matched a changed relevant register");
  regs[0] = 0u;
  ++regs[1]; // Irrelevant user data does not change the captured dependency state.
  Check(refresh() && !reused, "reuse-trace fixture recurrence materialization failed");
  const auto after_recurrence = Libs::Graphics::GetResourceReuseTraceStats();
  Check(after_recurrence.checked == before_recurrence.checked + 1 &&
        after_recurrence.matched == before_recurrence.matched + 1 &&
        after_recurrence.distance16 == before_recurrence.distance16 + 1,
        "reuse-trace fixture did not find the evicted exact register state");
  flags = Common::RuntimeFlags::Current();
  flags.resource_dynamic_deps = false;
  Common::RuntimeFlags::g_published.store((2ull << Common::RuntimeFlags::kRevisionShift) |
                                             Common::RuntimeFlags::Bits(flags),
                                         std::memory_order_relaxed);
  regs[0] = 0u;
  Check(refresh() && !reused, "reuse-trace fixture policy-flip materialization failed");
  const auto after_flip = Libs::Graphics::GetResourceReuseTraceStats();
  Check(after_flip.matched == after_recurrence.matched && before.evictions <= after_flip.evictions,
        "reuse-trace fixture retained history across dependency-policy flip");
}

void TestResourceMemoHashFilterDependencies() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct PublishedGuard {
    uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved, std::memory_order_relaxed); }
  } published_guard;
  auto flags = Common::RuntimeFlags::Current();
  flags.resource_memo_verify = false;
  flags.resource_hash_filter = true;
  uint64_t revision = 1;
  const auto set_policy = [&](bool dynamic) {
    flags.resource_dynamic_deps = dynamic;
    Common::RuntimeFlags::g_published.store(
        (revision++ << Common::RuntimeFlags::kRevisionShift) | Common::RuntimeFlags::Bits(flags),
        std::memory_order_relaxed);
  };
  ResourcePlan plan;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  const auto user = [&](uint32_t index) {
    auto& inst = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
    inst.SetArg(0, Value(static_cast<ScalarReg>(index)));
    return Value(&inst);
  };
  const auto selected = user(0), inactive = user(1), condition = user(2);
  plan.descriptor_sources.resize(2);
  for (auto& source: plan.descriptor_sources) {
    source.dword_count = 4; source.dwords.fill(Value(0u));
  }
  plan.descriptor_sources[0].dwords[0] = selected;
  plan.descriptor_sources[1].dwords[0] = inactive;
  plan.info.buffers = {{.source = 0}, {.source = 1}};
  plan.control_flow.resize(3);
  plan.control_flow[0].condition = condition;
  plan.control_flow[0].successors = {1u, 2u};
  plan.control_flow[1].sources = {0u}; plan.control_flow[2].sources = {1u};
  std::array<uint32_t, 4> regs{0x1000u, 0x2000u, 1u, 7u};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemo memo(2);
  bool reused = false;
  const auto refresh = [&] {
    return memo.Refresh(plan, SrtRuntime{.user_data = regs}, snapshot, specialization, true, reused);
  };

  set_policy(true);
  Check(refresh() && !reused, "hash-filter fixture first dynamic branch failed");
  ++regs[3];
  Check(refresh() && reused && plan.active_sources == std::vector<uint8_t>{1u, 0u} &&
            snapshot.user_data[3] == regs[3],
        "hash filter treated an inactive register as a dependency");
  regs[2] = 0u;
  Check(refresh() && !reused && plan.active_sources == std::vector<uint8_t>{0u, 1u},
        "hash-filter fixture second dynamic dependency set failed");
  regs[2] = 1u;
  ++regs[1];
  Check(refresh() && reused && plan.active_sources == std::vector<uint8_t>{1u, 0u},
        "hash-filter dependency-set grouping lost a dynamic branch state");

  set_policy(false);
  Check(refresh() && !reused && memo.LastMiss() != Libs::Graphics::ResourceMemo::Miss::None,
        "hash filter reused a key from before the dependency-policy flip");
  set_policy(true);
  regs[2] = 0u;
  Check(refresh() && !reused,
        "hash filter reused stale group metadata after a dynamic-policy flip");
  Check(refresh() && reused && plan.active_sources == std::vector<uint8_t>{0u, 1u},
        "hash-filter group rebuild did not recover the new dynamic state");
}
} // namespace
