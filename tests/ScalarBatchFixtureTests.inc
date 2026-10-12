namespace {
struct ScalarBatchFixtureReadState {
 bool allow_span = true;
 uint32_t span_attempts = 0, span_successes = 0;
 std::vector<std::pair<uint64_t,uint32_t>> ordinary, strict;
};
template<bool strict>
bool ScalarBatchFixtureRead(void* data,uint64_t address,std::span<uint32_t> values) {
 auto& state = *static_cast<ScalarBatchFixtureReadState*>(data);
 auto& reads = strict ? state.strict : state.ordinary;
 if (values.size() != 1u || address < 0x1000u || address >= 0x1010u || (address & 3u)) return false;
 const uint32_t value = (strict ? 110u : 10u) + static_cast<uint32_t>((address-0x1000u)/4u);
 reads.emplace_back(address,value); values[0] = value; return true;
}
template<bool strict>
bool ScalarBatchFixturePeek(void* data,uint64_t address,std::span<uint32_t> values) {
 auto& state = *static_cast<ScalarBatchFixtureReadState*>(data);
 ++state.span_attempts;
 const uint64_t expected_address = strict ? 0x1008u : 0x1000u;
 if (!state.allow_span || address != expected_address || values.size() != 2u) return false;
 const uint32_t first = (strict ? 110u : 10u) + static_cast<uint32_t>((address-0x1000u)/4u);
 values[0] = first; values[1] = first+1u; ++state.span_successes; return true;
}

void TestScalarBatchFixtureCaptureAndReplay() {
 using namespace Libs::Graphics::ShaderRecompiler::IR;
 struct PublishedGuard {
  uint64_t saved = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
  ~PublishedGuard() { Common::RuntimeFlags::g_published.store(saved,std::memory_order_relaxed); }
 } guard;
 auto flags = Common::RuntimeFlags::Current(); flags.srt_scalar_batch = true;
 Common::RuntimeFlags::g_published.store((1ull<<Common::RuntimeFlags::kRevisionShift)|Common::RuntimeFlags::Bits(flags),std::memory_order_relaxed);
 const char* verifier_env=std::getenv("KYTY_SRT_TAPE_VERIFY");
 const bool expect_batch=verifier_env==nullptr || verifier_env[0]!='1';

 auto plan = SrtPlan(0x1000u);
 plan.capture_specialization_reads = true;
 plan.srt_reads.resize(1);
 auto* first = plan.srt_reads[0].value.ResolveInstruction();
 auto* handle = first->Arg(0).ResolveInstruction();
 for (uint32_t i=1; i<4u; ++i) {
  MemoryInfo memory; memory.kind=ResourceKind::ScalarAddress; memory.planning_only=true;
  memory.offset=i*4u; memory.component_index=i; plan.memory_info.push_back(memory);
  auto& read=plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  read.SetFlags(MemoryFlags{.index=i,.pc=0x40u});
  read.SetArg(0,Value(handle)); read.SetArg(1,Value(0u)); read.SetArg(2,Value(0u)); read.SetArg(3,Value(true));
  plan.srt_reads.push_back({Value(&read),i});
 }
 plan.clean_flat_slots={0u,0u,1u,1u};

 ScalarBatchFixtureReadState state;
 const SrtRuntime runtime{
  .read_memory=ScalarBatchFixtureRead<false>, .userdata=&state,
  .read_specialization_memory=ScalarBatchFixtureRead<true>,
  .try_read_memory_span=ScalarBatchFixturePeek<false>,
  .try_read_specialization_span=ScalarBatchFixturePeek<true>};
 ResourceSnapshot snapshot; ResourceSpecialization specialization;
 const char* path="scalar-batch-fixture.rpf";
 Check(CaptureResourcePlan(plan,runtime,snapshot,specialization,path),"batch fixture capture succeeded");
 ResourcePlanFixture fixture;
 Check(fixture.Load(path),"batch fixture archive loaded");
 std::remove(path);
 Check(state.span_attempts==(expect_batch?2u:0u) && state.span_successes==(expect_batch?2u:0u) &&
       fixture.reads.size()==(expect_batch?2u:4u) &&
       (expect_batch ? fixture.reads[0].words.size()==2u && fixture.reads[1].words.size()==2u :
        std::ranges::all_of(fixture.reads,[](const auto& r){ return r.words.size()==1u; })),
       "successful spans or verifier-disabled scalar reads preserve both domains");
 Check(std::ranges::any_of(fixture.reads,[](const auto& r){return !r.strict && r.address==0x1000u;}) &&
       std::ranges::any_of(fixture.reads,[](const auto& r){return r.strict && r.address==0x1008u;}),
       "captured read domains remain distinct");
 ResourcePlanFixture::ReplayBacking failed_peek(fixture);
 failed_peek.words.erase({false,0x1004u});
 failed_peek.Reset();
 std::array<uint32_t,2> untouched{0xdeadbeefu,0xdeadbeefu};
 Check(!failed_peek.TryRead(false,0x1000u,untouched) && untouched[0]==0xdeadbeefu &&
       untouched[1]==0xdeadbeefu && failed_peek.result.requests==0u && failed_peek.result.words==0u &&
       failed_peek.result.ordinary_calls==0u && failed_peek.result.missing==0u,
       "failed replay peek changes neither outputs nor reader diagnostics");
 const auto batched_replay=fixture.Replay();
 Check(batched_replay.matched && batched_replay.scalar_batch_words==(expect_batch?4u:0u) &&
       batched_replay.ordinary_calls==(expect_batch?1u:2u) && batched_replay.strict_calls==(expect_batch?1u:2u),
       "fixture replay exposes batching and reader callback reduction");

 // A broad capture must also satisfy narrow scalar requests when batching is off.
 flags.srt_scalar_batch=false;
 Common::RuntimeFlags::g_published.store((2ull<<Common::RuntimeFlags::kRevisionShift)|Common::RuntimeFlags::Bits(flags),std::memory_order_relaxed);
 const auto scalar_replay=fixture.Replay();
 Check(scalar_replay.matched && scalar_replay.scalar_batch_words==0u &&
       scalar_replay.ordinary_calls==2u && scalar_replay.strict_calls==2u,
       "broad captured spans serve narrow scalar requests with reader calls restored");

 // Conversely, a union of narrow records must satisfy a broad replay peek.
 const auto broad_reads=fixture.reads;
 fixture.reads.clear();
 for (const auto& read : broad_reads) for (size_t i=0;i<read.words.size();++i)
  fixture.reads.push_back({.address=read.address+i*sizeof(uint32_t),.strict=read.strict,.valid=read.valid,.words={read.words[i]}});
 flags.srt_scalar_batch=true;
 Common::RuntimeFlags::g_published.store((3ull<<Common::RuntimeFlags::kRevisionShift)|Common::RuntimeFlags::Bits(flags),std::memory_order_relaxed);
 const auto replay_from_narrow=fixture.Replay();
 Check(replay_from_narrow.matched && replay_from_narrow.scalar_batch_words==(expect_batch?4u:0u) &&
       replay_from_narrow.ordinary_calls==(expect_batch?1u:2u) &&
       replay_from_narrow.strict_calls==(expect_batch?1u:2u),
       "narrow captured words serve broad replay spans");
 fixture.reads=broad_reads;

 // Failed peek attempts leave no captured record; successful scalar fallback
 // remains replayable in both reader domains.
 state={}; state.allow_span=false;
 Check(CaptureResourcePlan(plan,runtime,snapshot,specialization,path),"failed peek falls back to scalar capture");
 ResourcePlanFixture fallback;
 Check(fallback.Load(path),"scalar fallback archive loaded");
 std::remove(path);
 Check(state.span_attempts==(expect_batch?2u:0u) && state.span_successes==0u && fallback.reads.size()==4u &&
       state.ordinary.size()==2u && state.strict.size()==2u &&
       std::ranges::all_of(fallback.reads,[](const auto& r){ return r.valid && r.words.size()==1u; }),
       "failed span peeks are not recorded as reads");
 flags.srt_scalar_batch=false;
 Common::RuntimeFlags::g_published.store((4ull<<Common::RuntimeFlags::kRevisionShift)|Common::RuntimeFlags::Bits(flags),std::memory_order_relaxed);
 Check(fallback.Replay().matched,"fallback scalar capture replays correctly");
}
} // namespace
