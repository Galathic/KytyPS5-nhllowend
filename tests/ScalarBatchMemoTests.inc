namespace {
struct ScalarBatchMemoMemory {
  std::array<uint32_t, 4> ordinary{10, 11, 12, 13};
  std::array<uint32_t, 4> strict{110, 111, 112, 113};
  bool allow_span = true;
  bool fail_strict = false;
  bool scalar_only = false;
  uint32_t reads = 0, successes = 0;
};
template <bool strict>
bool ScalarBatchMemoRead(void* data, uint64_t address, std::span<uint32_t> words) {
  auto& memory = *static_cast<ScalarBatchMemoMemory*>(data);
  ++memory.reads;
  if (address < 0x1000u || (address & 3u) ||
      address >= 0x1010u || words.size() > (0x1010u - address) / 4u ||
      (strict && memory.fail_strict) || (memory.scalar_only && words.size() != 1)) return false;
  const auto& source = strict ? memory.strict : memory.ordinary;
  std::ranges::copy(std::span(source).subspan((address - 0x1000u) / 4u, words.size()), words.begin());
  return true;
}
template <bool strict>
bool ScalarBatchMemoPeek(void* data, uint64_t address, std::span<uint32_t> words) {
  auto& memory = *static_cast<ScalarBatchMemoMemory*>(data);
  if (!memory.allow_span || (strict && memory.fail_strict) ||
      address != (strict ? 0x1008u : 0x1000u) || words.size() != 2) return false;
  const auto& source = strict ? memory.strict : memory.ordinary;
  std::ranges::copy(std::span(source).subspan((address - 0x1000u) / 4u, words.size()), words.begin());
  ++memory.successes;
  return true;
}
void TestScalarBatchMemoCaptureHitAndFlagFlip() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct PublishedGuard {
    uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
    ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved, std::memory_order_relaxed); }
  } guard;
  auto flags = Common::RuntimeFlags::Current();
  flags.srt_scalar_batch = true;
  flags.resource_memo_verify = true;
  const auto publish = [&] {
    Common::RuntimeFlags::g_published.store((1ull << Common::RuntimeFlags::kRevisionShift) | Common::RuntimeFlags::Bits(flags),
                                          std::memory_order_relaxed);
  };
  publish();
  auto plan = SrtPlan(0x1000u);
  plan.capture_specialization_reads = true;
  auto* first = plan.srt_reads[0].value.ResolveInstruction();
  auto* handle = first->Arg(0).ResolveInstruction();
  for (uint32_t i = 1; i < 4; ++i) {
    MemoryInfo memory;
    memory.kind = ResourceKind::ScalarAddress;
    memory.planning_only = true;
    memory.offset = i * 4u;
    memory.component_index = i;
    plan.memory_info.push_back(memory);
    auto& read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = i, .pc = 0x40u});
    read.SetArg(0, Value(handle)); read.SetArg(1, Value(0u));
    read.SetArg(2, Value(0u)); read.SetArg(3, Value(true));
    plan.srt_reads.push_back({Value(&read), i});
  }
  plan.memory_info[0].component_index = 0;
  plan.clean_flat_slots = {0, 0, 1, 1};
  ScalarBatchMemoMemory memory;
  SrtRuntime runtime{.read_memory = ScalarBatchMemoRead<false>, .userdata = &memory,
      .read_specialization_memory = ScalarBatchMemoRead<true>,
      .try_read_memory_span = ScalarBatchMemoPeek<false>,
      .try_read_specialization_span = ScalarBatchMemoPeek<true>};
  Libs::Graphics::ResourceMemo memo(1);
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  bool reused = true;
  const auto check_values = [&] {
    Check(snapshot.flattened_srt == std::vector<uint32_t>{memory.ordinary[0], memory.ordinary[1],
        memory.strict[2], memory.strict[3]}, "memo span capture mixed ordinary and strict domains");
  };
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "memo batch capture reused an absent entry");
  check_values();
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "memo verifier could not replay successful span captures");
  // Narrow recursive verifier reads must be served from wide captures; both
  // flag directions must remain valid despite changed recorded segmentation.
  flags.srt_scalar_batch = false; publish();
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "memo verifier failed wide capture to scalar flag flip");
  check_values();
  memory.ordinary[0] = 77;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "memo flag flip skipped fresh memory validation");
  check_values();
  flags.srt_scalar_batch = true; publish();
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "memo verifier failed scalar capture to batch flag flip");
  memory.allow_span = false;
  const auto successes = memory.successes;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused &&
        memory.successes == successes,
        "refused optional span contaminated memo validation or verifier replay");
  memory.strict[2] = 177;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "declined span reused stale strict values");
  check_values();
  memory.fail_strict = true;
  Check(!memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "memo span refusal bypassed real strict-reader failure");
  // Width-sensitive readers must retain narrow captures and hit validation
  // when batching is disabled or its capability refuses without observing.
  memory.fail_strict = false;
  memory.scalar_only = true;
  flags.srt_scalar_batch = false; publish();
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "scalar-only memo capture failed with batching disabled");
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "scalar-only memo validation widened adjacent reader calls");
  flags.srt_scalar_batch = true; publish();
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "refused span broke scalar-only memo validation or verifier fallback");
}
} // namespace
