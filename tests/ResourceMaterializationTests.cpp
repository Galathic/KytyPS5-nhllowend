#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/host_gpu/renderer/pipeline/resourceMemo.h"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  program.block_info.push_back({.id = 0});
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan SrtPlan(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestCompiledDescriptorGatherMatchesWalker() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.user_data_base = 8;
  auto &block = AddValueBlock(program);
  auto &reg8 = block.AppendNewInst(ValueOpcode::GetUserData,
                                   {Value(static_cast<ScalarReg>(8))});
  auto &reg10 = block.AppendNewInst(ValueOpcode::GetUserData,
                                    {Value(static_cast<ScalarReg>(10))});
  DescriptorSource descriptor;
  descriptor.dword_count = 8;
  descriptor.dwords = {Value(&reg8), Value(0x12345678u), Value(&reg10), Value(&reg8),
                      Value::F32(1.25f), Value(uint64_t{0xfedcba9876543210u}),
                      Value(true), Value(0u)};
  program.descriptor_sources.push_back(descriptor);
  auto plan = ExtractResourcePlan(program);
  Check(plan.descriptor_gathers.size() == 1 && plan.descriptor_gathers[0].valid,
        "literal/user-data descriptor did not compile a gather");
  const auto compare = [&](std::span<const uint32_t> words, bool expected_success) {
    const SrtRuntime runtime{.user_data = words};
    DescriptorValue fast, slow;
    bool fast_ok;
    { SrtWalker walker(plan, runtime); fast_ok = walker.EvaluateDescriptor(0, fast); }
    plan.descriptor_gathers[0].valid = false;
    bool slow_ok;
    { SrtWalker walker(plan, runtime); slow_ok = walker.EvaluateDescriptor(0, slow); }
    plan.descriptor_gathers[0].valid = true;
    Check(fast_ok == expected_success && fast_ok == slow_ok && fast == slow,
          "compiled gather changed descriptor values or failure behavior");
  };
  std::array<uint32_t, 3> words{0x100u, 0x200u, 0x300u};
  compare(words, true);
  words = {0x400u, 0x500u, 0x600u};
  compare(words, true);
  compare(std::span(words).first(2), false);
  // Store absolute register indices; changing the plan's base must not leave
  // stale relative offsets compiled into the gather.
  plan.user_data_base = 10;
  compare(words, false);
  plan.user_data_base = 0;
  std::array<uint32_t, 11> rebased{};
  rebased[8] = 0x700u;
  rebased[10] = 0x800u;
  compare(rebased, true);
  DescriptorValue masked;
  {
    SrtWalker walker(plan, {.user_data = rebased}, {}, nullptr,
                     plan.descriptor_sources[0].dwords[0]);
    Check(walker.EvaluateDescriptor(0, masked) && masked.dwords[0] == 1u &&
              masked.dwords[3] == 1u && masked.dwords[2] == 0x800u,
          "compiled gather bypassed the active-mask evaluation context");
  }
  auto memory_plan = SrtPlan(0x1000u);
  Check(memory_plan.descriptor_gathers[0].valid &&
            memory_plan.descriptor_gathers[0].needs_flat,
        "SRT descriptor lost its refresh requirement");
}

void TestFlatDescriptorGatherRequiresOwnCompleteRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x1000u);
  struct Reads { uint32_t value = 0x100u; uint32_t count = 0; bool fail = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void* data, uint64_t address, std::span<uint32_t> words) {
        auto& reads = *static_cast<Reads*>(data);
        ++reads.count;
        if (reads.fail || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }, .userdata = &reads};
  DescriptorValue descriptor;
  std::vector<uint32_t> flat;
  {
    SrtWalker walker(plan, runtime);
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == reads.value &&
              reads.count == 1,
          "unrefreshed flat gather bypassed ordinary memory evaluation");
  }
  for (const auto value: {0x200u, 0x300u}) {
    reads.value = value;
    reads.count = 0;
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(flat) && reads.count == 1 && flat[0] == value,
          "SRT refresh retained values from a prior evaluation session");
    // Indirect descriptor materialization can append and reallocate this vector.
    flat.reserve(flat.capacity() + 128);
    flat.push_back(0xdeadbeefu);
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == value &&
              reads.count == 1,
          "flat descriptor gather lost refreshed values after vector reallocation");
    flat.clear();
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == value &&
              reads.count == 1,
          "short flat storage did not safely fall back to the same-session memo");
  }
  reads.fail = true;
  {
    SrtWalker walker(plan, runtime);
    Check(!walker.RefreshFlatBuffer(flat) && !walker.EvaluateDescriptor(0, descriptor),
          "failed refresh reused a prior session's flat descriptor");
  }
  reads.fail = false;
  {
    const auto mask = plan.descriptor_sources[0].dwords[0];
    SrtWalker walker(plan, runtime, {}, nullptr, mask);
    Check(walker.RefreshFlatBuffer(flat) && walker.EvaluateDescriptor(0, descriptor) &&
              descriptor.dwords[0] == 1u,
          "flat gather bypassed active-mask semantics");
  }
  // A clean writable-descriptor session must not consume another walker's
  // ordinary flattened values. Its own strict reader supplies the descriptor.
  const SrtRuntime mixed{
      .read_memory = runtime.read_memory, .userdata = &reads,
      .read_specialization_memory = +[](void*, uint64_t address, std::span<uint32_t> words) {
        if (address != 0x1000u || words.size() != 1) return false;
        words[0] = 0x555u;
        return true;
      }};
  {
    SrtWalker ordinary(plan, mixed);
    Check(ordinary.RefreshFlatBuffer(flat), "ordinary SRT refresh failed");
    SrtWalker clean(plan, CleanRuntime(mixed));
    Check(clean.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == 0x555u,
          "unrefreshed clean walker consumed ordinary flat values");
  }
  // Control-flow refresh may leave unreachable slots zero. Such sessions must
  // retain the interpreter even after a successful refresh.
  plan.control_flow.emplace_back();
  plan.control_flow[0].srt_reads.push_back(0);
  plan.control_flow[0].sources.push_back(0);
  {
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(flat), "control-flow SRT refresh failed");
    flat[0] = 0xbadu;
    Check(walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == reads.value,
          "control-flow session used the unconditional flat gather");
  }
}

void TestResourceMemoTracksMemoryAndRuntime() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  uint32_t word = 0x12345678u;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&word));
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Libs::Graphics::ResourceMemo memo;
  SrtRuntime runtime{};
  bool reused = true;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "memo first refresh reused an absent snapshot");
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "unchanged memory-backed SRT was not reused");
  word = 0xabcdef01u;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.flattened_srt[0] == word,
        "same-pointer SRT update reused stale values");
  runtime.shader_base = 0x1000;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "shader-base change reused an old snapshot");
  std::array<uint32_t, 3> groups{1, 2, 3};
  runtime.workgroup_counts = groups;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "new workgroup counts reused an old snapshot");
  groups[0] = 4;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "same-pointer workgroup update reused stale values");
  Check(memo.Refresh(plan, runtime, snapshot, specialization, false, reused) && !reused,
        "disabled memo reused cached state");
  auto user_plan = UserDataBufferPlan();
  std::array<uint32_t, 1> user{0x8000};
  runtime.user_data = user;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && !reused,
        "different resource plan reused an old snapshot");
  user[0] = 0x9000;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && !reused &&
            snapshot.buffers[0].dwords[0] == user[0] && snapshot.user_data[0] == user[0],
        "same-pointer register update reused stale descriptor or push payload");
  std::array<uint32_t, 2> payload{user[0], 7};
  runtime.user_data = payload;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && !reused,
        "changed user-data length reused an old snapshot");
  payload[1] = 9;
  Check(memo.Refresh(user_plan, runtime, snapshot, specialization, true, reused) && reused &&
            snapshot.user_data[1] == 9 && snapshot.buffers[0].dwords[0] == payload[0],
        "non-resource register update failed to reuse resources or update push data");
}

void TestResourceMemoAlternatingObjects() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // Alternating objects must reuse both entries, and still reject changed memory.
  auto alternating_plan = UserDataBufferPlan();
  ResourceSnapshot alternating_snapshot;
  ResourceSpecialization alternating_specialization;
  Libs::Graphics::ResourceMemo alternating_memo(16);
  std::array<uint32_t, 2> registers{0x8000, 7};
  SrtRuntime alternating_runtime{.user_data = registers};
  bool alternating_reused = false;
  for (uint32_t address: {0x8000u, 0x9000u, 0x8000u, 0x9000u}) {
    registers[0] = address;
    Check(alternating_memo.Refresh(alternating_plan, alternating_runtime,
          alternating_snapshot, alternating_specialization, true, alternating_reused),
          "alternating memo refresh failed");
    Check(alternating_snapshot.buffers[0].dwords[0] == address,
          "multi-entry memo restored the wrong object's descriptor");
  }
  Check(alternating_reused, "alternating objects did not reuse multiple entries");
  registers[1] = 13;
  Check(alternating_memo.Refresh(alternating_plan, alternating_runtime,
        alternating_snapshot, alternating_specialization, true, alternating_reused) &&
        alternating_reused && alternating_snapshot.user_data[1] == 13,
        "multi-entry hit failed to refresh non-resource push registers");

}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&dword));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestFailedRuntimeReadIsMemoizedPerRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
      {Value(0x1000u), Value(0u)});
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  read.SetFlags(MemoryFlags{.index = 0});
  struct Reads { uint32_t count = 0; bool available = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.count;
        if (!state.available) return false;
        words[0] = 42u;
        return true;
      },
      .userdata = &reads};
  uint32_t result = 0;
  {
    SrtWalker walker(program, runtime);
    for (uint32_t i = 0; i < 32; ++i)
      Check(!walker.Evaluate(Value(&read), result), "unavailable runtime read was accepted");
    Check(reads.count == 1, "failed runtime read was retried within one refresh");
    // Another evaluation context must not inherit the failed result.
    reads.available = true;
    SrtWalker nested(program, runtime);
    Check(nested.Evaluate(Value(&read), result) && result == 42u && reads.count == 2,
          "nested runtime evaluation inherited a failed result");
  }
  Check(SrtWalker(program, runtime).Evaluate(Value(&read), result) &&
            result == 42u && reads.count == 3,
        "next refresh retained a failed runtime result");
}
void TestCyclicRuntimeFailureCanRecover() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  auto &root = block.AppendNewInst(ValueOpcode::ConditionRef, {Value(false)});
  auto &inverse = block.AppendNewInst(ValueOpcode::LogicalNot, {Value(&root)});
  auto &either = block.AppendNewInst(ValueOpcode::LogicalOr, {Value(&inverse), Value(true)});
  root.SetArg(0, Value(&either));
  SrtWalker walker(program, {});
  uint32_t result = 0;
  Check(walker.Evaluate(Value(&root), result) && result == 1u,
        "short-circuit evaluation failed to resolve a cyclic predicate");
  Check(walker.Evaluate(Value(&inverse), result) && result == 0u,
        "temporary cyclic failure prevented later predicate evaluation");
}
void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t strict = 0; uint32_t ordinary = 0; bool clean = true; } reads;
  const SrtRuntime runtime{
      .read_memory = [](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 999;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = ExtractResourcePlan(program);
    Check(plan.control_flow.empty() && plan.requires_specialization_memory &&
              plan.capture_specialization_reads,
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.strict == 1 && reads.ordinary == 0 &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "vector descriptor input was not read and captured exactly once");
    reads.clean = false;
    reads.strict = 0;
    Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.strict == 1 && reads.ordinary == 0,
          "dirty vector descriptor input fell back to an ordinary memory read");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  reads.strict = 0;
  uint32_t result = 99;
  Check(SrtWalker(program, runtime).Evaluate(Value(&inactive), result) &&
            result == 0 && reads.strict == 0 && reads.ordinary == 0,
        "literal false EXEC read vector memory");
  count.SetArg(4, Value(true));
  handle.SetArg(3, Value(0x204u));
  Check(SrtWalker(program, runtime).Evaluate(Value(&count), result) &&
            result == 0 && reads.strict == 0 && reads.ordinary == 0,
        "invalid vector buffer format read memory");
  handle.SetArg(3, Value(0x16204u));
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestGuardedVectorReadTapeFallback() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.user_data_base = 8;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.requires_specialization_memory = true;
  plan.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &condition = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
  condition.SetArg(0, Value(static_cast<ScalarReg>(8)));
  auto &guard = plan.value_storage.emplace_back(ValueOpcode::INotEqual32);
  guard.SetArg(0, Value(&condition));
  guard.SetArg(1, Value(0u));
  auto &handle = plan.value_storage.emplace_back(
      ValueOpcode::GetBufferResource);
  handle.SetArg(0, Value(0x1000u));
  handle.SetArg(1, Value(4u << 16u));
  handle.SetArg(2, Value(1u));
  handle.SetArg(3, Value(0x16204u));
  auto &load = plan.value_storage.emplace_back(ValueOpcode::LoadBufferU32);
  load.SetFlags(MemoryFlags{.index = 0});
  load.SetArg(0, Value(&handle));
  load.SetArg(1, Value(0u));
  load.SetArg(2, Value(0u));
  load.SetArg(3, Value(0u));
  load.SetArg(4, Value(&guard));
  plan.srt_reads.push_back({Value(&load), 0});
  plan.clean_flat_slots.resize(1u);

  struct ReadState {
    uint32_t calls = 0;
    uint32_t value = 0x76543210u;
  } reads;
  const auto reader = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &state = *static_cast<ReadState *>(data);
    ++state.calls;
    if (address != 0x1000u || words.size() != 1u) return false;
    words[0] = state.value;
    return true;
  };
  const SrtRuntime runtime{
      .user_data = {},
      .read_memory = reader,
      .userdata = &reads,
      .read_specialization_memory = reader,
  };
  std::array<uint32_t, 1> user_data{};
  auto guarded_runtime = runtime;
  guarded_runtime.user_data = user_data;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;

  user_data[0] = 0;
  const bool false_ok = MaterializeResources(plan, guarded_runtime, snapshot, specialization);
  Check(false_ok &&
            snapshot.flattened_srt.size() == 1u && snapshot.flattened_srt[0] == 0u &&
            reads.calls == 0u,
        "false vector-load guard touched memory or returned a nonzero value");

  user_data[0] = 1;
  const bool true_ok = MaterializeResources(plan, guarded_runtime, snapshot, specialization);
  Check(true_ok &&
            snapshot.flattened_srt.size() == 1u &&
            snapshot.flattened_srt[0] == reads.value && reads.calls == 1u,
        "true vector-load guard did not read the descriptor exactly once");
}

void TestTapeVerifyReplaysReaderDomains() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *verify = std::getenv("KYTY_SRT_TAPE_VERIFY");
  const char *tape = std::getenv("KYTY_SRT_TAPE");
  if (verify == nullptr || verify[0] != '1' || (tape != nullptr && tape[0] == '0')) return;

  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = true;
  plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  plan.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &scalar_handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  scalar_handle.SetArg(0, Value(0x2000u));
  scalar_handle.SetArg(1, Value(0u));
  auto &scalar_read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  scalar_read.SetFlags(MemoryFlags{.index = 0});
  scalar_read.SetArg(0, Value(&scalar_handle));
  scalar_read.SetArg(1, Value(0u));
  scalar_read.SetArg(2, Value(0u));
  scalar_read.SetArg(3, Value(true));
  auto &buffer_handle = plan.value_storage.emplace_back(ValueOpcode::GetBufferResource);
  buffer_handle.SetArg(0, Value(0x3000u));
  buffer_handle.SetArg(1, Value(4u << 16u));
  buffer_handle.SetArg(2, Value(1u));
  buffer_handle.SetArg(3, Value(0x16204u));
  auto &vector_read = plan.value_storage.emplace_back(ValueOpcode::LoadBufferU32);
  vector_read.SetFlags(MemoryFlags{.index = 1});
  vector_read.SetArg(0, Value(&buffer_handle));
  vector_read.SetArg(1, Value(0u));
  vector_read.SetArg(2, Value(0u));
  vector_read.SetArg(3, Value(0u));
  vector_read.SetArg(4, Value(true));
  plan.srt_reads.push_back({Value(&scalar_read), 0});
  plan.srt_reads.push_back({Value(&vector_read), 1});

  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.ordinary;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x12345678u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.strict;
        if (address != 0x3000u || words.size() != 1u) return false;
        words[0] = 0x87654321u;
        return true;
      }};
  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, runtime).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x12345678u, 0x87654321u} &&
            reads.ordinary == 1u && reads.strict == 1u,
        "tape verifier reread memory or crossed ordinary/strict reader domains");
  bool has_scalar_source = false;
  bool has_vector_source = false;
  for (const auto &op : plan.srt_tapes[0].ops) {
    has_scalar_source |= op.source == &scalar_read;
    has_vector_source |= op.source == &vector_read;
  }
  Check(has_scalar_source && has_vector_source,
        "Tape verifier lost source instruction provenance");

  reads = {};
  auto failed_runtime = runtime;
  failed_runtime.read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
    ++static_cast<Reads *>(data)->ordinary;
    return false;
  };
  Check(!SrtWalker(plan, failed_runtime).RefreshFlatBuffer(flat) &&
            reads.ordinary == 1u && reads.strict == 0u,
        "tape verifier changed failed-reader behavior");
}

void TestTapeVerifyMismatchFallbackAndUnmatchedReplay() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *verify = std::getenv("KYTY_SRT_TAPE_VERIFY");
  const char *tape = std::getenv("KYTY_SRT_TAPE");
  if (verify == nullptr || verify[0] != '1' || (tape != nullptr && tape[0] == '0')) return;
  const auto add_read = [](ResourcePlan &plan, uint64_t address, uint32_t index) {
    plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
    auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    handle.SetArg(0, Value(static_cast<uint32_t>(address)));
    handle.SetArg(1, Value(static_cast<uint32_t>(address >> 32u)));
    auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = index});
    read.SetArg(0, Value(&handle));
    read.SetArg(1, Value(0u));
    read.SetArg(2, Value(0u));
    read.SetArg(3, Value(true));
    return Value(&read);
  };
  struct MutatingRead {
    ResourcePlan *plan = nullptr;
    Value replacement;
    uint32_t calls = 0;
  };
  const auto reader = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &state = *static_cast<MutatingRead *>(data);
    ++state.calls;
    if (address != 0x4000u || words.size() != 1u) return false;
    words[0] = 0xabcdef01u;
    state.plan->srt_reads[0].value = state.replacement;
    return true;
  };

  ResourcePlan mismatch_plan;
  mismatch_plan.stage = Libs::Graphics::ShaderType::Compute;
  mismatch_plan.srt_plan_complete = true;
  mismatch_plan.resource_tracking_complete = true;
  const auto original = add_read(mismatch_plan, 0x4000u, 0u);
  mismatch_plan.srt_reads.push_back({original, 0u});
  mismatch_plan.srt_reads.push_back({original, 1u});
  auto &descriptor_expression = mismatch_plan.value_storage.emplace_back(ValueOpcode::IAdd32);
  descriptor_expression.SetArg(0, original);
  descriptor_expression.SetArg(1, Value(1u));
  DescriptorSource descriptor_source;
  descriptor_source.dword_count = 1;
  descriptor_source.dwords[0] = Value(&descriptor_expression);
  mismatch_plan.descriptor_sources.push_back(descriptor_source);
  MutatingRead mismatch_state{&mismatch_plan, Value(0x55aa55aau)};
  const SrtRuntime mismatch_runtime{.read_memory = reader, .userdata = &mismatch_state};
  std::vector<uint32_t> flat;
  SrtWalker mismatch_walker(mismatch_plan, mismatch_runtime);
  Check(mismatch_walker.RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x55aa55aau, 0xabcdef01u} && mismatch_state.calls == 1u,
        "tape verifier did not fall back to the recursive value after a synthetic mismatch");
  DescriptorValue descriptor;
  Check(mismatch_walker.EvaluateDescriptor(0u, descriptor) &&
            descriptor.dwords[0] == 0xabcdef02u && mismatch_state.calls == 1u,
        "mismatch fallback did not transfer the validated recursive memo into descriptor evaluation");

  ResourcePlan unmatched_plan;
  unmatched_plan.stage = Libs::Graphics::ShaderType::Compute;
  unmatched_plan.srt_plan_complete = true;
  unmatched_plan.resource_tracking_complete = true;
  const auto first = add_read(unmatched_plan, 0x4000u, 0u);
  const auto second = add_read(unmatched_plan, 0x5000u, 1u);
  unmatched_plan.srt_reads.push_back({first, 0u});
  MutatingRead unmatched_state{&unmatched_plan, second};
  const SrtRuntime unmatched_runtime{.read_memory = reader, .userdata = &unmatched_state};
  Check(SrtWalker(unmatched_plan, unmatched_runtime).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0xabcdef01u} && unmatched_state.calls == 1u,
        "unmatched replay was treated as a mismatch or reread live memory");
}

void TestTapeSeparatesCleanAndOrdinaryReadDomains(bool cfg = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x2000u));
  handle.SetArg(1, Value(0u));
  auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  read.SetFlags(MemoryFlags{.index = 0});
  read.SetArg(0, Value(&handle));
  read.SetArg(1, Value(0u));
  read.SetArg(2, Value(0u));
  read.SetArg(3, Value(true));
  plan.srt_reads.push_back({Value(&read), 0u});
  plan.srt_reads.push_back({Value(&read), 1u});
  plan.clean_flat_slots = {0u, 1u};
  if (cfg) {
    plan.control_flow.emplace_back();
    plan.control_flow[0].srt_reads = {0u, 1u};
  }

  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.ordinary;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x11111111u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &state = *static_cast<Reads *>(data);
        ++state.strict;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x22222222u;
        return true;
      }};
  SrtWalker clean(plan, CleanRuntime(runtime));
  SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x11111111u, 0x22222222u},
        "Tape merged ordinary and clean versions of one shared IR read");
  uint32_t ordinary_value = 0;
  uint32_t strict_value = 0;
  Check(walker.Evaluate(Value(&read), ordinary_value) && ordinary_value == 0x11111111u &&
            clean.Evaluate(Value(&read), strict_value) && strict_value == 0x22222222u &&
            reads.ordinary == 1u && reads.strict == 1u,
        "Tape published a read result into the wrong evaluator memo domain");
}

void TestVectorLoadUsesCleanDependencies(bool through_read_const, bool cfg = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  plan.memory_info.push_back({.kind = ResourceKind::Buffer, .data_dwords = 1,
                              .data_bits = 32});

  auto &scalar_handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  scalar_handle.SetArg(0, Value(0x2000u));
  scalar_handle.SetArg(1, Value(0u));
  auto &address_word = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  address_word.SetFlags(MemoryFlags{.index = 0});
  address_word.SetArg(0, Value(&scalar_handle));
  address_word.SetArg(1, Value(0u));
  address_word.SetArg(2, Value(0u));
  address_word.SetArg(3, Value(true));
  Value low(&address_word);
  if (through_read_const) {
    auto &srt = plan.value_storage.emplace_back(ValueOpcode::GetSrtResource);
    auto &read_const = plan.value_storage.emplace_back(ValueOpcode::ReadConst);
    read_const.SetArg(0, Value(&srt));
    read_const.SetArg(1, Value(0u));
    low = Value(&read_const);
  }

  auto &buffer_handle = plan.value_storage.emplace_back(ValueOpcode::GetBufferResource);
  buffer_handle.SetArg(0, low);
  buffer_handle.SetArg(1, Value(4u << 16u));
  buffer_handle.SetArg(2, Value(1u));
  buffer_handle.SetArg(3, Value(0x16204u));
  auto &vector_read = plan.value_storage.emplace_back(ValueOpcode::LoadBufferU32);
  vector_read.SetFlags(MemoryFlags{.index = 1});
  vector_read.SetArg(0, Value(&buffer_handle));
  vector_read.SetArg(1, Value(0u));
  vector_read.SetArg(2, Value(0u));
  vector_read.SetArg(3, Value(0u));
  vector_read.SetArg(4, Value(true));
  plan.srt_reads.push_back({Value(&address_word), 0u});
  plan.srt_reads.push_back({Value(&vector_read), 1u});
  if (cfg) {
    plan.control_flow.emplace_back();
    plan.control_flow[0].srt_reads = {0u, 1u};
  }

  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        if (address != 0x2000u || words.size() != 1u) return false;
        words[0] = 0x3000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->strict;
        if (words.size() != 1u) return false;
        if (address == 0x2000u) words[0] = 0x4000u;
        else if (address == 0x3000u) words[0] = 0xaaaaaaaau;
        else if (address == 0x4000u) words[0] = 0xbbbbbbbbu;
        else return false;
        return true;
      }};
  SrtWalker clean(plan, CleanRuntime(runtime));
  SrtWalker walker(plan, runtime, {}, &clean);
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x3000u, 0xbbbbbbbbu},
        through_read_const
            ? "clean vector-load ReadConst dependency used the ordinary reader"
            : "clean vector-load address dependency used the ordinary reader");
}

void TestCfgTapeVerifierRejectsCorruption() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const char *cfg = std::getenv("KYTY_SRT_CFG_TAPE");
  const char *verify = std::getenv("KYTY_SRT_TAPE_VERIFY");
  if (cfg == nullptr || cfg[0] != '1' || verify == nullptr || verify[0] != '1') return;
  const auto verifier_before = GetSrtTapeStats();
  auto plan = SrtPlan(0x2000u);
  const Value raw = plan.srt_reads[0].value;
  auto &sum = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
  sum.SetArg(0, raw); sum.SetArg(1, Value(1u));
  plan.srt_reads = {{Value(&sum), 0u}, {raw, 1u}};
  plan.clean_flat_slots = {0u, 1u};
  plan.control_flow.resize(1);
  plan.control_flow[0].srt_reads = {0u, 1u};
  plan.control_flow[0].sources = {0u};
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; } reads;
  const SrtRuntime runtime {
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary; words[0] = 10u; return true;
      }, .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->strict; words[0] = 20u; return true;
      }};
  std::vector<uint32_t> flat;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    Check(walker.RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{11u, 20u},
          "CFG verifier warmup failed");
  }
  auto &tape = plan.cfg_srt_tapes[1];
  bool corrupted = false;
  for (const auto &op: tape.ops) {
    if (op.source == &sum) {
      tape.ops[op.args[1]].imm = 8u; // Invalid compiled arithmetic, original IR unchanged.
      corrupted = true;
      break;
    }
  }
  Check(corrupted, "CFG verifier fixture did not corrupt a supported compiled root");
  reads = {};
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    uint32_t value = 0;
    // This live read predates the capture; replay must receive its initial memo.
    Check(walker.Evaluate(raw, value) && value == 10u, "CFG verifier preloaded input failed");
    Check(walker.RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{11u, 20u} &&
              reads.ordinary == 1u && reads.strict == 1u &&
              tape.state == ResourcePlan::SrtTape::State::Unsupported,
          "CFG verifier accepted corrupt arithmetic, reread inputs, or failed to disable plan");
    Check(walker.Evaluate(Value(&sum), value) && value == 11u &&
              clean.Evaluate(raw, value) && value == 20u &&
              reads.ordinary == 1u && reads.strict == 1u,
          "CFG mismatch fallback did not restore both validated memo domains");
  }

  ResourcePlan activation;
  activation.srt_plan_complete = activation.resource_tracking_complete = true;
  activation.descriptor_sources.resize(2);
  activation.control_flow.resize(3);
  activation.control_flow[0].condition = Value(true);
  activation.control_flow[0].successors = {1u, 2u};
  activation.control_flow[1].sources = {0u};
  activation.control_flow[2].sources = {1u};
  Check(SrtWalker(activation, runtime).RefreshFlatBuffer(flat), "CFG activation warmup failed");
  auto &activation_tape = activation.cfg_srt_tapes[0];
  activation_tape.ops[activation_tape.condition_ops[0]].imm = 0u;
  reads = {};
  Check(SrtWalker(activation, runtime).RefreshFlatBuffer(flat) &&
            activation.active_sources == std::vector<uint8_t>{1u, 0u} &&
            activation_tape.state == ResourcePlan::SrtTape::State::Unsupported &&
            reads.ordinary == 0u && reads.strict == 0u,
        "CFG verifier ignored an activation-only mismatch");

  auto unmatched = SrtPlan(0x2000u);
  unmatched.control_flow.resize(1);
  unmatched.control_flow[0].srt_reads = {0u};
  Check(SrtWalker(unmatched, runtime).RefreshFlatBuffer(flat), "CFG unmatched warmup failed");
  auto &unmatched_tape = unmatched.cfg_srt_tapes[0];
  for (const auto &op: unmatched_tape.ops) {
    if (op.kind == ResourcePlan::SrtTape::Kind::RawRead) {
      unmatched_tape.ops[op.args[0]].imm = 0x3000u;
      break;
    }
  }
  reads = {};
  flat = {0x55u};
  Check(!SrtWalker(unmatched, runtime).RefreshFlatBuffer(flat) && flat == std::vector<uint32_t>{0x55u} &&
            reads.ordinary == 1u && unmatched_tape.state == ResourcePlan::SrtTape::State::Unsupported,
        "unmatched CFG replay reread live memory or published an unvalidated result");

  auto unusual = SrtPlan(0x2000u);
  unusual.clean_flat_slots = {1u};
  unusual.control_flow.resize(1);
  unusual.control_flow[0].srt_reads = {0u};
  reads = {};
  {
    // Deliberately not CleanRuntime(runtime): its ordinary callback is distinct.
    SrtWalker different_clean(unusual, runtime);
    SrtWalker walker(unusual, runtime, unusual.clean_flat_slots, &different_clean);
    Check(!walker.RefreshFlatBuffer(flat) && reads.ordinary == 0u && reads.strict == 0u &&
              unusual.cfg_srt_tapes[1].state == ResourcePlan::SrtTape::State::Unsupported,
          "CFG verifier accepted a distinct clean runtime as the same capture domain");
  }
  const auto verifier_after = GetSrtTapeStats();
  Check(verifier_after.cfg_checked - verifier_before.cfg_checked == 7u &&
            verifier_after.cfg_matched - verifier_before.cfg_matched == 3u &&
            verifier_after.cfg_mismatch - verifier_before.cfg_mismatch == 2u &&
            verifier_after.cfg_inconclusive - verifier_before.cfg_inconclusive == 2u,
        "CFG verifier counters did not report matching, mismatched and rejected refreshes");
}

void TestCfgTapeDemandRoots() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto user = [&](uint32_t reg) {
    auto &inst = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
    inst.SetArg(0, Value(static_cast<ScalarReg>(reg)));
    return Value(&inst);
  };
  const auto raw = [&](Value address) {
    auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
    handle.SetArg(0, address); handle.SetArg(1, Value(0u));
    auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
    read.SetFlags(MemoryFlags{.index = 0});
    read.SetArg(0, Value(&handle)); read.SetArg(1, Value(0u));
    read.SetArg(2, Value(0u)); read.SetArg(3, Value(true));
    return Value(&read);
  };
  const auto shared = raw(user(0));
  const auto poisoned = raw(Value(0xdead0000u));
  auto &selection = plan.value_storage.emplace_back(ValueOpcode::SelectU32);
  selection.SetArg(0, Value(true)); selection.SetArg(1, shared); selection.SetArg(2, poisoned);
  // Unsupported lazy root follows supported shared roots; never execute its poison arm.
  plan.srt_reads = {{shared, 0u}, {poisoned, 1u}, {shared, 2u}, {Value(&selection), 3u}};
  plan.clean_flat_slots = {0u, 0u, 1u, 0u};
  plan.control_flow.resize(3);
  plan.control_flow[0].condition = user(1);
  plan.control_flow[0].successors = {1u, 2u};
  plan.control_flow[1].srt_reads = {0u, 2u, 3u};
  plan.control_flow[1].sources = {0u};
  plan.control_flow[2].srt_reads = {1u};
  plan.control_flow[2].sources = {1u};
  plan.descriptor_sources.resize(2);
  struct Reads {
    std::vector<uint64_t> order;
    uint32_t value = 7;
    bool allow_poison = false;
  } reads;
  const auto ordinary = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &s = *static_cast<Reads *>(data);
    s.order.push_back(address);
    if (words.size() != 1u || (address == 0xdead0000u && !s.allow_poison)) return false;
    words[0] = s.value;
    return true;
  };
  const auto strict = +[](void *data, uint64_t address, std::span<uint32_t> words) {
    auto &s = *static_cast<Reads *>(data);
    s.order.push_back(address | (1ull << 63u));
    if (words.size() != 1u) return false;
    words[0] = s.value + 100u;
    return true;
  };
  std::array<uint32_t, 2> sgpr {0x1000u, 1u};
  const SrtRuntime runtime {.user_data = sgpr, .read_memory = ordinary, .userdata = &reads,
                            .read_specialization_memory = strict};
  std::vector<uint32_t> flat;
  const auto stats_before = GetSrtTapeStats();
  for (uint32_t generation = 0; generation < 2u; ++generation) {
    reads.order.clear();
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    uint32_t preloaded = 0;
    Check(walker.Evaluate(shared, preloaded), "CFG memo warmup failed");
    Check(walker.RefreshFlatBuffer(flat) &&
              flat == std::vector<uint32_t>{reads.value, 0u, reads.value + 100u, reads.value} &&
              reads.order == std::vector<uint64_t>{sgpr[0], sgpr[0] | (1ull << 63u)} &&
              plan.active_sources == std::vector<uint8_t>{1u, 0u},
          "CFG demand tape eagerly read poison, merged domains, or ignored recursive memo");
    sgpr[0] += 0x1000u;
    ++reads.value;
  }
  const char *cfg_flag = std::getenv("KYTY_SRT_CFG_TAPE");
  if (cfg_flag != nullptr && cfg_flag[0] == '1') {
    const auto &tape = plan.cfg_srt_tapes[1];
    Check(tape.state == ResourcePlan::SrtTape::State::Ready &&
              tape.slot_ops[0] != UINT32_MAX && tape.slot_ops[3] == UINT32_MAX &&
              GetSrtTapeStats().cfg_runs > stats_before.cfg_runs &&
              GetSrtTapeStats().cfg_fallback > stats_before.cfg_fallback,
          "CFG test silently bypassed supported or independently unsupported roots");
  }
  // A runtime-unknown condition must preserve traversal of both successors.
  auto &unknown = plan.value_storage.emplace_back(ValueOpcode::GetBuiltin);
  unknown.SetArg(0, Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)));
  unknown.SetArg(1, Value(0u));
  plan.control_flow[0].condition = Value(&unknown);
  // A new plan graph requires a fresh compiled aggregate, as in extracted shader variants.
  for (auto &tape: plan.cfg_srt_tapes) tape = {};
  reads.allow_poison = true;
  reads.order.clear();
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    Check(walker.RefreshFlatBuffer(flat) && flat[1] == reads.value &&
              plan.active_sources == std::vector<uint8_t>{1u, 1u} &&
              reads.order == std::vector<uint64_t>{0xdead0000u, sgpr[0], sgpr[0] | (1ull << 63u)},
          "unknown CFG predicate did not traverse both successors in original read order");
  }
  reads.allow_poison = false;
  reads.order.clear();
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, plan.clean_flat_slots, &clean);
    uint32_t ignored = 0;
    Check(!walker.Evaluate(poisoned, ignored) && !walker.RefreshFlatBuffer(flat) &&
              reads.order == std::vector<uint64_t>{0xdead0000u},
          "CFG tape retried a memoized failed read");
  }
}

void TestCfgTapeCompilationCycleFallsBack() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  auto &cycle = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
  cycle.SetArg(0, Value(&cycle)); cycle.SetArg(1, Value(1u));
  plan.srt_reads = {{Value(4u), 0u}, {Value(&cycle), 1u}};
  plan.control_flow.resize(1);
  plan.control_flow[0].srt_reads = {0u, 1u};
  std::vector<uint32_t> flat;
  Check(!SrtWalker(plan, {}).RefreshFlatBuffer(flat) && flat[0] == 4u,
        "cyclic CFG compilation did not leave supported sibling root usable");
}

void TestTapeUnsupportedRootFallsBack() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  auto &selection = plan.value_storage.emplace_back(ValueOpcode::SelectU32);
  selection.SetArg(0, Value(true));
  selection.SetArg(1, Value(0x22222222u));
  selection.SetArg(2, Value(0x33333333u));
  // Keep a valid Tape op at index zero. Before the fix, the unsupported Select
  // returned `false` (integer zero), silently aliasing this first slot's op.
  plan.srt_reads.push_back({Value(0x11111111u), 0u});
  plan.srt_reads.push_back({Value(&selection), 1u});

  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, {}).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x11111111u, 0x22222222u},
        "unsupported Tape root did not fall back to the recursive evaluator");
}

void TestTapePreservesWideMemoryIndex() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  ResourcePlan plan;
  plan.stage = Libs::Graphics::ShaderType::Compute;
  plan.srt_plan_complete = plan.resource_tracking_complete = true;
  plan.memory_info.resize(257);
  plan.memory_info[0] = {.kind = ResourceKind::ScalarAddress, .offset = 0,
                        .planning_only = true};
  plan.memory_info[256] = {.kind = ResourceKind::ScalarAddress, .offset = 4,
                           .planning_only = true};
  auto &handle = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
  handle.SetArg(0, Value(0x2000u));
  handle.SetArg(1, Value(0u));
  auto &read = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
  read.SetFlags(MemoryFlags{.index = 256});
  read.SetArg(0, Value(&handle));
  read.SetArg(1, Value(0u));
  read.SetArg(2, Value(0u));
  read.SetArg(3, Value(true));
  plan.srt_reads.push_back({Value(&read), 0u});

  const SrtRuntime runtime{
      .read_memory = +[](void *, uint64_t address, std::span<uint32_t> words) {
        if (words.size() != 1u) return false;
        if (address == 0x2000u) words[0] = 0x11111111u;
        else if (address == 0x2004u) words[0] = 0x22222222u;
        else return false;
        return true;
      }};
  std::vector<uint32_t> flat;
  Check(SrtWalker(plan, runtime).RefreshFlatBuffer(flat) &&
            flat == std::vector<uint32_t>{0x22222222u},
        "Tape truncated a 32-bit memory-info index");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(SrtWalker(program, {}).Evaluate(Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!SrtWalker(program, {}).Evaluate(Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorUsesStrictReaderOnce() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // A host-evaluable branch captures resource reads and needs the writable
  // descriptor's clean provenance for the renderer's disjointness proof.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
                                        {Value(&offset), Value(4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.block_info[0].condition = Value(&condition);
  program.block_info[0].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.block_info[0].terminator.true_block = 1;
  program.block_info[0].terminator.false_block = 2;
  program.block_info[1].id = 1;
  program.block_info[1].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.block_info[2].id = 2;
  program.block_info[2].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  Check(plan.capture_specialization_reads,
        "conditional writable descriptor lost its alias proof");
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; bool clean = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 0x8000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
            reads.ordinary == 0 && reads.strict == 1,
        "GPU-dirty dynamic writable descriptor bypassed strict provenance");
  reads.clean = true;
  reads.strict = 0;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            reads.ordinary == 0 && reads.strict == 1 && snapshot.buffers[0].dwords[0] == 0x8000u &&
            snapshot.specialization_reads ==
                std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}},
        "writable descriptor was evaluated twice or scalar EXEC suppressed its read");
  Libs::Graphics::ResourceMemo memo;
  bool reused = true;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "strict memo did not establish an initial snapshot");
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && reused,
        "unchanged strict descriptor did not reuse its snapshot");
  reads.clean = false;
  Check(!memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "failed strict read reused a stale writable descriptor");
  reads.clean = true;
  Check(memo.Refresh(plan, runtime, snapshot, specialization, true, reused) && !reused,
        "failed strict refresh left a reusable snapshot");
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = ExtractResourcePlan(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
    uint32_t ordinary = 0;
  } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    const auto offset = specialization.images[0].indirect_mapping_offset;
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated scalar reads or bypassed clean provenance");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  TestCfgTapeVerifierRejectsCorruption();
  TestCfgTapeDemandRoots();
  TestCfgTapeCompilationCycleFallsBack();
  TestVectorLoadUsesCleanDependencies(false, true);
  TestVectorLoadUsesCleanDependencies(true, true);
  TestTapeSeparatesCleanAndOrdinaryReadDomains(true);
  TestTapePreservesWideMemoryIndex();
  TestTapeUnsupportedRootFallsBack();
  TestVectorLoadUsesCleanDependencies(false);
  TestVectorLoadUsesCleanDependencies(true);
  TestTapeSeparatesCleanAndOrdinaryReadDomains();
  TestTapeVerifyReplaysReaderDomains();
  TestTapeVerifyMismatchFallbackAndUnmatchedReplay();
  TestCompiledDescriptorGatherMatchesWalker();
  TestFlatDescriptorGatherRequiresOwnCompleteRefresh();
  TestResourceMemoTracksMemoryAndRuntime();
  TestResourceMemoAlternatingObjects();
  TestMappedSrtUsesDirectReaderByDefault();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestFailedRuntimeReadIsMemoizedPerRefresh();
  TestCyclicRuntimeFailureCanRecover();
  TestUniformVectorDescriptorRead();
  TestGuardedVectorReadTapeFallback();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  TestWrittenDescriptorUsesStrictReaderOnce();
  TestFailedMaterializationRejectsStage();
  TestFiniteImageRefreshReusesScalarReads();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
