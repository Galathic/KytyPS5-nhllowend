// KYTY_SRT_COMPILED: the compiled plan executor against the existing materialization path.
//
// Every case runs the same inputs through two independent copies of a plan, one with the flag off
// and one with it on, and demands bit-identical outputs plus an identical sequence of guest dword
// reads. Corpus plans are perturbed (registers, memory, branch conditions) so the compiled program
// is exercised on inputs it has never seen; nothing here replays a frozen evaluation.
#include <unordered_map>
#include <unordered_set>

namespace CompiledSrtTest {
using namespace Libs::Graphics::ShaderRecompiler::IR;

inline void SetCompiled(bool on) {
  const uint64_t bit = uint64_t{1} << Common::RuntimeFlags::kBitSrtCompiled;
  const auto old = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
  Common::RuntimeFlags::g_published.store(on ? (old | bit) : (old & ~bit), std::memory_order_relaxed);
}
inline bool CompiledRequested() {
  return (Common::RuntimeFlags::g_published.load(std::memory_order_relaxed) >> Common::RuntimeFlags::kBitSrtCompiled & 1u) != 0u;
}

// Guest memory: captured words per domain, a deterministic hash for everything else.
struct DiffMemory {
  std::unordered_map<uint64_t, uint32_t> words[2];
  std::unordered_set<uint64_t> invalid[2];
  uint64_t seed = 1;
  bool unknown_fails = false;
  std::vector<std::pair<bool, uint64_t>> log;
  std::vector<uint32_t> observed;  // user-data registers reported through observe_user_data
  uint64_t calls = 0;
  uint32_t Hash(uint64_t address) const {
    uint64_t x = (address >> 2) * 0x9e3779b97f4a7c15ull ^ seed * 0xbf58476d1ce4e5b9ull;
    x ^= x >> 29; x *= 0x94d049bb133111ebull; x ^= x >> 32;
    return static_cast<uint32_t>(x);
  }
  bool Read(bool strict, uint64_t address, std::span<uint32_t> out) {
    ++calls;
    bool ok = true;
    for (size_t i = 0; i < out.size(); ++i) {
      const uint64_t a = address + i * 4;
      log.emplace_back(strict, a);
      if (invalid[strict].count(a)) { ok = false; continue; }
      const auto it = words[strict].find(a);
      if (it != words[strict].end()) out[i] = it->second;
      else if (unknown_fails) ok = false;
      else out[i] = Hash(a);
    }
    return ok;
  }
};

struct Outcome {
  bool ok = false;
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  std::vector<uint8_t> active;
  std::vector<std::pair<bool, uint64_t>> reads;
  std::vector<uint32_t> observed;  // sorted, unique
  uint64_t calls = 0;
};

inline Outcome Run(const ResourcePlan& plan, const ResourcePlanFixture& fx, DiffMemory& memory,
                   const std::vector<uint32_t>& user_data, bool compiled, ResourceDelta* delta) {
  SetCompiled(compiled);
  memory.log.clear();
  memory.observed.clear();
  memory.calls = 0;
  SrtRuntime runtime {.user_data = user_data, .shader_base = fx.shader_base, .userdata = &memory,
                      .workgroup_counts = fx.workgroups};
  runtime.observe_user_data = +[](void* p, uint32_t index) { static_cast<DiffMemory*>(p)->observed.push_back(index); };
  runtime.user_data_observer_userdata = &memory;
  runtime.read_memory = +[](void* p, uint64_t a, std::span<uint32_t> v) {
    return static_cast<DiffMemory*>(p)->Read(false, a, v);
  };
  if (fx.strict_reader)
    runtime.read_specialization_memory = +[](void* p, uint64_t a, std::span<uint32_t> v) {
      return static_cast<DiffMemory*>(p)->Read(true, a, v);
    };
  if (compiled) runtime.delta = delta;
  Outcome result;
  result.ok = MaterializeResources(plan, runtime, result.snapshot, result.specialization);
  result.active = plan.active_sources;
  result.reads = memory.log;
  result.observed = memory.observed;
  std::sort(result.observed.begin(), result.observed.end());
  result.observed.erase(std::unique(result.observed.begin(), result.observed.end()), result.observed.end());
  result.calls = memory.calls;
  return result;
}

inline const char* Differ(const Outcome& a, const Outcome& b) {
  if (a.ok != b.ok) return "success";
  if (!a.ok) return nullptr;  // a declined merged read is retried as single reads, so failed calls may repeat reads
  if (a.reads != b.reads) return "read sequence";
  if (a.observed != b.observed) return "observed user data";
  if (a.snapshot.buffers != b.snapshot.buffers) return "buffers";
  if (a.snapshot.images != b.snapshot.images) return "images";
  if (a.snapshot.samplers != b.snapshot.samplers) return "samplers";
  if (a.snapshot.flattened_srt != b.snapshot.flattened_srt) return "flattened_srt";
  if (a.snapshot.user_data != b.snapshot.user_data) return "user_data";
  if (!(a.snapshot.uniform_fill == b.snapshot.uniform_fill)) return "uniform_fill";
  if (!ResourcePlanFixture::SameRangeCoverage(a.snapshot.specialization_reads, b.snapshot.specialization_reads))
    return "specialization_reads";
  if (!(a.specialization == b.specialization)) return "specialization";
  if (a.active != b.active) return "active_sources";
  return nullptr;
}

struct Tally {
  uint64_t runs = 0, both_ok = 0, both_fail = 0, mismatches = 0;
};

// Runs old and compiled on identical inputs and reports the first difference.
inline bool Compare(const char* what, const ResourcePlan& old_plan, const ResourcePlan& new_plan,
                    const ResourcePlanFixture& fx, DiffMemory& memory,
                    const std::vector<uint32_t>& user_data, ResourceDelta& delta, Tally& tally,
                    Outcome* out_old = nullptr, Outcome* out_new = nullptr) {
  auto a = Run(old_plan, fx, memory, user_data, false, nullptr);
  auto b = Run(new_plan, fx, memory, user_data, true, &delta);
  ++tally.runs;
  (a.ok ? tally.both_ok : tally.both_fail) += (a.ok == b.ok) ? 1 : 0;
  if (const char* difference = Differ(a, b)) {
    ++tally.mismatches;
    std::fprintf(stderr, "compiled differential mismatch: %s: %s (ok old=%d new=%d, reads old=%zu new=%zu)\n",
                 what, difference, a.ok, b.ok, a.reads.size(), b.reads.size());
    return false;
  }
  if (out_old) *out_old = std::move(a);
  if (out_new) *out_new = std::move(b);
  return true;
}

inline DiffMemory MemoryFor(const ResourcePlanFixture& fx) {
  DiffMemory memory;
  for (const auto& read: fx.reads) {
    for (size_t i = 0; i < read.words.size(); ++i) {
      const uint64_t a = read.address + i * 4;
      if (read.valid) memory.words[read.strict][a] = read.words[i];
      else memory.invalid[read.strict].insert(a);
    }
  }
  return memory;
}

// Differential fuzz over one captured plan: register, memory and branch perturbations.
inline int DifferentialOne(const std::filesystem::path& path, uint32_t perturbations, Tally& total) {
  ResourcePlanFixture old_fx, new_fx;
  if (!old_fx.Load(path.string()) || !new_fx.Load(path.string())) {
    std::fprintf(stderr, "compiled differential: load failed %s\n", path.string().c_str());
    return 2;
  }
  auto base_memory = MemoryFor(old_fx);
  ResourceDelta delta;
  Tally tally;
  uint64_t rng = 0x243f6a8885a308d3ull ^ old_fx.plan.shader_hash;
  const auto next = [&] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
  // Warm-up on the captured inputs: records CFG traces, builds tapes and compiled programs.
  for (int i = 0; i < 6; ++i) {
    auto memory = base_memory;
    if (!Compare(path.filename().string().c_str(), old_fx.plan, new_fx.plan, old_fx, memory, old_fx.user_data, delta, tally))
      return 1;
  }
  std::vector<uint64_t> keys[2];
  for (int d = 0; d < 2; ++d) for (const auto& [k, v]: base_memory.words[d]) keys[d].push_back(k);
  for (int d = 0; d < 2; ++d) std::sort(keys[d].begin(), keys[d].end());
  for (uint32_t p = 0; p < perturbations; ++p) {
    auto memory = base_memory;
    auto regs = old_fx.user_data;
    const auto kind = next() % 7;
    std::string what = path.filename().string() + " kind=" + std::to_string(kind) + " #" + std::to_string(p);
    switch (kind) {
      case 0: break;  // unchanged: the steady state
      case 1:         // changed register
        if (!regs.empty()) regs[next() % regs.size()] ^= 1u << (2 + next() % 12);
        break;
      case 2:         // pointer register nudged by whole dwords
        if (!regs.empty()) regs[next() % regs.size()] += 4u * (1 + next() % 8);
        break;
      case 3:         // changed memory word (any domain)
      case 4: {
        const int d = static_cast<int>(next() % 2);
        if (!keys[d].empty()) {
          const auto k = keys[d][next() % keys[d].size()];
          memory.words[d][k] ^= kind == 3 ? (1u << (next() % 20)) : 0xffffffffu;
        }
        break;
      }
      case 5:         // new world: unknown addresses read a different deterministic pattern
        memory.seed = 1 + next() % 5;
        for (int d = 0; d < 2; ++d) if (!keys[d].empty()) memory.words[d][keys[d][next() % keys[d].size()]] = 0;
        break;
      default: {      // reader failure at a captured word
        const int d = static_cast<int>(next() % 2);
        if (!keys[d].empty()) memory.invalid[d].insert(keys[d][next() % keys[d].size()]);
        break;
      }
    }
    if (!Compare(what.c_str(), old_fx.plan, new_fx.plan, old_fx, memory, regs, delta, tally)) return 1;
    if (p % 8 == 7) {  // back to the captured state: a miss must not poison the next hit
      auto again = base_memory;
      if (!Compare((what + " recovery").c_str(), old_fx.plan, new_fx.plan, old_fx, again, old_fx.user_data, delta, tally))
        return 1;
    }
  }
  total.runs += tally.runs; total.both_ok += tally.both_ok; total.both_fail += tally.both_fail;
  return 0;
}

inline void PrintDifferential(size_t plans, const Tally& total, const SrtCompiledStats& before) {
  const auto after = GetSrtCompiledStats();
  std::printf("compiled differential: plans=%zu runs=%llu both_ok=%llu both_fail=%llu mismatches=0 "
              "compiled_calls=%llu (nocfg=%llu cfg=%llu) fallbacks: ineligible=%llu nocfg_unsupported=%llu "
              "cfg_no_trace=%llu cfg_eval=%llu cfg_miss=%llu groups_hit=%llu groups_miss=%llu "
              "desc_reuse=%llu desc_eval=%llu memo_skipped=%llu\n",
              plans, static_cast<unsigned long long>(total.runs),
              static_cast<unsigned long long>(total.both_ok), static_cast<unsigned long long>(total.both_fail),
              static_cast<unsigned long long>(after.compiled - before.compiled),
              static_cast<unsigned long long>(after.nocfg - before.nocfg),
              static_cast<unsigned long long>(after.cfg - before.cfg),
              static_cast<unsigned long long>(after.fallback[0] - before.fallback[0]),
              static_cast<unsigned long long>(after.fallback[1] - before.fallback[1]),
              static_cast<unsigned long long>(after.fallback[2] - before.fallback[2]),
              static_cast<unsigned long long>(after.fallback[3] - before.fallback[3]),
              static_cast<unsigned long long>(after.fallback[4] - before.fallback[4]),
              static_cast<unsigned long long>(after.groups_hit - before.groups_hit),
              static_cast<unsigned long long>(after.groups_miss - before.groups_miss),
              static_cast<unsigned long long>(after.desc_reuse - before.desc_reuse),
              static_cast<unsigned long long>(after.desc_eval - before.desc_eval),
              static_cast<unsigned long long>(after.memo_skipped - before.memo_skipped));
}

inline int DifferentialFile(const char* path, uint32_t perturbations, bool report) {
  const auto before = GetSrtCompiledStats();
  Tally total;
  const int status = DifferentialOne(path, perturbations, total);
  SetCompiled(false);
  if (status == 0 && report) PrintDifferential(1, total, before);
  return status;
}

inline int Differential(const char* directory, uint32_t perturbations) {
  std::error_code ec;
  std::vector<std::filesystem::path> paths;
  for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec))
    if (it->is_regular_file(ec) && it->path().extension() == ".rpf") paths.push_back(it->path());
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) { std::fprintf(stderr, "compiled differential: no .rpf in %s\n", directory); return 2; }
  const auto before = GetSrtCompiledStats();
  Tally total;
  for (const auto& path: paths) {
    if (const int status = DifferentialOne(path, perturbations, total)) { SetCompiled(false); return status; }
  }
  SetCompiled(false);
  PrintDifferential(paths.size(), total, before);
  return 0;
}

// Old path against the compiled executor on the whole corpus: each plan is replayed against a frozen
// reader with the flag off and on, in alternating rounds so neither side owns the warm cache.
// The hashed pass swaps the fixture std::map reader for a hashed index; the map per-word cost
// otherwise dominates what is being measured.
inline int Benchmark(const char* directory, uint64_t iterations) {
  std::error_code ec;
  std::vector<std::filesystem::path> paths;
  for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec))
    if (it->is_regular_file(ec) && it->path().extension() == ".rpf") paths.push_back(it->path());
  std::sort(paths.begin(), paths.end());
  struct Item {
    std::filesystem::path path;
    ResourcePlanFixture old_fx, new_fx;
    ResourcePlanFixture::ReplayBacking old_backing, new_backing;
  };
  for (const bool fast: {false, true}) {
    std::vector<std::unique_ptr<Item>> items;
    for (const auto& path: paths) {
      auto item = std::make_unique<Item>();
      item->path = path;
      if (!item->old_fx.Load(path.string()) || !item->new_fx.Load(path.string())) return 2;
      item->old_backing = ResourcePlanFixture::ReplayBacking(item->old_fx);
      item->new_backing = ResourcePlanFixture::ReplayBacking(item->new_fx);
      if (fast) { item->old_backing.BuildFast(); item->new_backing.BuildFast(); }
      for (int warm = 0; warm < 6; ++warm) {
        SetCompiled(false);
        const auto a = item->old_fx.ReplayPrepared(item->old_backing);
        SetCompiled(true);
        const auto b = item->new_fx.ReplayPrepared(item->new_backing);
        if (!a.matched || !b.matched) {
          std::fprintf(stderr, "compiled benchmark: replay mismatch %s (old=%d new=%d)\n",
                       path.string().c_str(), a.matched, b.matched);
          SetCompiled(false);
          return 1;
        }
      }
      items.push_back(std::move(item));
    }
    const auto before = GetSrtCompiledStats();
    double old_ns = 0, new_ns = 0, old_ns_cfg = 0, new_ns_cfg = 0;
    uint64_t old_calls = 0, new_calls = 0, cfg_plans = 0;
    std::printf("compiled benchmark (%s reader): plans=%zu iterations=%llu\n", fast ? "hashed" : "std::map",
                items.size(), static_cast<unsigned long long>(iterations));
    for (auto& item: items) {
      double plan_old = 0, plan_new = 0;
      uint64_t reads_old = 0, reads_new = 0;
      for (int round = 0; round < 3; ++round) {
        for (int side = 0; side < 2; ++side) {
          const bool compiled = (round & 1) ? side == 0 : side == 1;
          SetCompiled(compiled);
          auto& fx = compiled ? item->new_fx : item->old_fx;
          auto& backing = compiled ? item->new_backing : item->old_backing;
          uint64_t reads = 0;
          const auto start = std::chrono::steady_clock::now();
          for (uint64_t i = 0; i < iterations; ++i) {
            const auto r = fx.ReplayPrepared(backing);
            if (!r.matched) {
              std::fprintf(stderr, "compiled benchmark: iteration mismatch %s compiled=%d\n",
                           item->path.string().c_str(), compiled);
              SetCompiled(false);
              return 1;
            }
            reads += r.ordinary_calls + r.strict_calls;
          }
          const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
          (compiled ? plan_new : plan_old) += ns;
          (compiled ? reads_new : reads_old) += reads;
        }
      }
      const double per_old = plan_old / (3.0 * iterations), per_new = plan_new / (3.0 * iterations);
      const bool cfg = !item->old_fx.plan.control_flow.empty();
      old_ns += per_old; new_ns += per_new;
      old_calls += reads_old / (3 * iterations); new_calls += reads_new / (3 * iterations);
      if (cfg) { old_ns_cfg += per_old; new_ns_cfg += per_new; ++cfg_plans; }
      std::printf("  %s %s old_us=%.2f new_us=%.2f reader_calls old=%llu new=%llu\n",
                  item->path.filename().string().c_str(), cfg ? "cfg  " : "nocfg", per_old / 1000.0, per_new / 1000.0,
                  static_cast<unsigned long long>(reads_old / (3 * iterations)),
                  static_cast<unsigned long long>(reads_new / (3 * iterations)));
    }
    SetCompiled(false);
    const auto after = GetSrtCompiledStats();
    const auto n = static_cast<double>(items.size());
    std::printf("compiled benchmark aggregate (%s reader): old_us_per_plan=%.2f new_us_per_plan=%.2f speedup=%.2fx "
                "reader_calls_per_sweep old=%llu new=%llu | cfg plans(%llu): old=%.2f new=%.2f | nocfg plans(%zu): old=%.2f new=%.2f\n",
                fast ? "hashed" : "std::map", old_ns / n / 1000.0, new_ns / n / 1000.0, old_ns / new_ns,
                static_cast<unsigned long long>(old_calls), static_cast<unsigned long long>(new_calls),
                static_cast<unsigned long long>(cfg_plans),
                cfg_plans ? old_ns_cfg / cfg_plans / 1000.0 : 0.0, cfg_plans ? new_ns_cfg / cfg_plans / 1000.0 : 0.0,
                items.size() - cfg_plans,
                items.size() > cfg_plans ? (old_ns - old_ns_cfg) / (items.size() - cfg_plans) / 1000.0 : 0.0,
                items.size() > cfg_plans ? (new_ns - new_ns_cfg) / (items.size() - cfg_plans) / 1000.0 : 0.0);
    const uint64_t served = after.compiled - before.compiled;
    uint64_t fallbacks = 0;
    for (size_t i = 0; i < after.fallback.size(); ++i) fallbacks += after.fallback[i] - before.fallback[i];
    std::printf("compiled coverage: calls=%llu compiled=%llu (%.1f%%) nocfg=%llu cfg=%llu builds=%llu fallbacks:",
                static_cast<unsigned long long>(after.calls - before.calls), static_cast<unsigned long long>(served),
                100.0 * served / std::max<uint64_t>(1, served + fallbacks),
                static_cast<unsigned long long>(after.nocfg - before.nocfg),
                static_cast<unsigned long long>(after.cfg - before.cfg),
                static_cast<unsigned long long>(after.builds - before.builds));
    for (size_t i = 0; i < after.fallback.size(); ++i)
      std::printf(" %s=%llu", SrtCompiledFallbackName(i), static_cast<unsigned long long>(after.fallback[i] - before.fallback[i]));
    std::printf(" | groups hit=%llu miss=%llu spec_reused=%llu spec_recomputed=%llu\n",
                static_cast<unsigned long long>(after.groups_hit - before.groups_hit),
                static_cast<unsigned long long>(after.groups_miss - before.groups_miss),
                static_cast<unsigned long long>(after.spec_reused - before.spec_reused),
                static_cast<unsigned long long>(after.spec_recomputed - before.spec_recomputed));
  }
  return 0;
}

// --- synthetic plans -------------------------------------------------------------------------

// Pointer chase: user data 0/1 is a pointer to {lo, hi, a, b}; the second window holds a buffer
// descriptor base. Adjacent loads from one pointer exercise the merged read.
// variant 0: the descriptor dwords are the second-window loads themselves; 1: arithmetic over them; 2: arithmetic plus one
// user-data dword (impure, never reusable). With `cfg` the plan gets a branch on user_data[2] (block 1 reads both
// windows and activates the source, block 2 only the first window) and keeps its descriptor gather.
inline ResourcePlan ChasePlan(int variant = 0, bool cfg = false) {
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto& block = AddValueBlock(program);
  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  for (uint32_t i = 0; i < 4; ++i) {
    memory.offset = i * 4;
    program.memory_info.push_back(memory);
  }
  auto& lo = block.AppendNewInst(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  auto& hi = block.AppendNewInst(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(1))});
  auto& first = block.AppendNewInst(ValueOpcode::GetAddressResource, {Value(&lo), Value(&hi)});
  std::array<Inst*, 4> loads {};
  for (uint32_t i = 0; i < 4; ++i) {
    loads[i] = &block.AppendNewInst(ValueOpcode::LoadAddressU32,
                                    {Value(&first), Value(0u), Value(0u), Value(true)});
    loads[i]->SetFlags(MemoryFlags {.index = i, .pc = 0x40 + i});
    program.srt_reads.push_back({Value(loads[i]), i});
  }
  // Second window: pointer taken from the first two loaded dwords, then four loads at +0..+12.
  auto& second = block.AppendNewInst(ValueOpcode::GetAddressResource, {Value(loads[0]), Value(loads[1])});
  for (uint32_t i = 0; i < 4; ++i) {
    auto& load = block.AppendNewInst(ValueOpcode::LoadAddressU32,
                                     {Value(&second), Value(0u), Value(0u), Value(true)});
    load.SetFlags(MemoryFlags {.index = i, .pc = 0x80 + i});
    program.srt_reads.push_back({Value(&load), 4 + i});
  }
  auto& srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  DescriptorSource source;
  for (uint32_t i = 0; i < 4; ++i) {
    auto& flat = block.AppendNewInst(ValueOpcode::ReadConst, {Value(&srt), Value(4u + i)});
    if (variant == 0) {
      source.dwords[i] = Value(&flat);
    } else if (variant == 2 && i == 3) {
      auto& user = block.AppendNewInst(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(2))});
      source.dwords[i] = Value(&user);
    } else {
      auto& sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&flat), Value(3u + i)});
      source.dwords[i] = Value(&sum);
    }
  }
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  auto plan = ExtractResourcePlan(program);
  plan.info.buffers.push_back({.source = 0});
  if (!cfg) {
    plan.descriptor_gathers.clear();
    return plan;
  }
  plan.capture_specialization_reads = true;
  plan.clean_flat_slots = {1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u};
  auto& condition = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
  condition.SetArg(0, Value(static_cast<ScalarReg>(2)));
  plan.control_flow.resize(3);
  plan.control_flow[0].condition = Value(&condition);
  plan.control_flow[0].successors = {1, 2};
  plan.control_flow[1].srt_reads = {0, 1, 2, 3, 4, 5, 6, 7};
  plan.control_flow[1].sources = {0};
  plan.control_flow[2].srt_reads = {0, 1, 2, 3};
  return plan;
}

struct ChaseWorld {
  DiffMemory memory;
  std::vector<uint32_t> regs {0x10000, 0};
  ResourcePlanFixture fx;
  void Build(uint32_t second_lo, uint32_t word2 = 0x40, uint32_t word3 = 0x0) {
    memory = DiffMemory {};
    memory.unknown_fails = true;
    for (int d = 0; d < 2; ++d) {
      const uint64_t base = regs[0] | (uint64_t(regs[1]) << 32);
      memory.words[d][base + 0] = second_lo;
      memory.words[d][base + 4] = 0;
      memory.words[d][base + 8] = 0x77;
      memory.words[d][base + 12] = 0x88;
      memory.words[d][second_lo + 0] = 0x1000;
      memory.words[d][second_lo + 4] = 0x00010000;
      memory.words[d][second_lo + 8] = word2;
      memory.words[d][second_lo + 12] = word3;
    }
  }
};

inline void TestChangedRegisterMemoryAndBranch() {
  // CFG traces exist only with the CFG tape and trace flags; pin them for this case.
  const auto saved_flags = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
  Common::RuntimeFlags::g_published.store(saved_flags | (uint64_t{1} << 2) | (uint64_t{1} << 11), std::memory_order_relaxed);
  Tally tally;
  {
    const auto old_plan = ChasePlan();
    const auto new_plan = ChasePlan();
    ChaseWorld world;
    ResourceDelta delta;
    Outcome a, b;
    const auto before = GetSrtCompiledStats();
    world.Build(0x20000);
    Check(Compare("chase baseline", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) && a.ok,
          "compiled executor matches the tape on a two-window pointer chase");
    // Steady state: the second call is served by the compiled program.
    world.Build(0x20000);
    Check(Compare("chase steady", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) &&
              GetSrtCompiledStats().compiled > before.compiled,
          "compiled program serves the steady-state call");
    Check(GetSrtCompiledStats().groups_hit > before.groups_hit,
          "adjacent loads from one pointer are fetched with a merged read");
    Check(a.snapshot.flattened_srt[4] == 0x1000 && a.snapshot.flattened_srt[6] == 0x40, "chase reads the second window");
    // Changed register: the whole chase moves to another window (new pointer, new memory).
    world.regs[0] = 0x30000;
    world.Build(0x50000, 0x44);
    Check(Compare("chase register", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) &&
              b.snapshot.flattened_srt[6] == 0x44,
          "changed pointer register is re-read and re-followed, never frozen");
    // Changed memory: same registers, one dword behind the pointer differs.
    world.Build(0x50000, 0x48);
    Check(Compare("chase memory", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) &&
              b.snapshot.flattened_srt[6] == 0x48,
          "changed guest memory behind a stable pointer is re-read");
    // The pointer stored in memory moves: the second window address changes without any register change.
    world.Build(0x60000, 0x4c);
    Check(Compare("chase indirect", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) &&
              b.snapshot.flattened_srt[6] == 0x4c,
          "changed in-memory pointer is followed on the next call");
    // Reader failure mid-chain: both paths fail identically and the next call recovers.
    world.Build(0x60000, 0x4c);
    world.memory.invalid[0].insert(0x60000 + 8);
    world.memory.invalid[1].insert(0x60000 + 8);
    Check(Compare("chase failure", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) && !a.ok && !b.ok,
          "failed guest read fails the compiled path exactly like the tape");
    world.Build(0x60000, 0x4c);
    Check(Compare("chase recovery", old_plan, new_plan, world.fx, world.memory, world.regs, delta, tally, &a, &b) && b.ok,
          "compiled path recovers after a failed call");
  }
  {
    // Conditional graph from TestResourcePlanFixture: user_data[2] selects which blocks are read.
    const auto make = [] {
      auto plan = SrtPlan(0x2000);
      plan.shader_hash = 0x5151;
      plan.descriptor_sources[0].dword_count = 4;
      plan.descriptor_sources[0].dwords[2] = Value(16u);
      plan.descriptor_sources[0].dwords[3] = Value(0u);
      plan.descriptor_gathers.clear();
      plan.info.buffers.push_back({.source = 0});
      plan.capture_specialization_reads = true;
      plan.clean_flat_slots = {0u, 1u};
      plan.srt_reads.push_back({plan.srt_reads[0].value, 1u});
      auto& condition = plan.value_storage.emplace_back(ValueOpcode::GetUserData);
      condition.SetArg(0, Value(static_cast<ScalarReg>(2)));
      auto& poison = plan.value_storage.emplace_back(ValueOpcode::LoadAddressU32);
      poison.SetFlags(MemoryFlags {.index = 0});
      auto& address = plan.value_storage.emplace_back(ValueOpcode::GetAddressResource);
      address.SetArg(0, Value(0xdead0000u));
      address.SetArg(1, Value(0u));
      poison.SetArg(0, Value(&address));
      poison.SetArg(1, Value(0u));
      poison.SetArg(2, Value(0u));
      poison.SetArg(3, Value(true));
      plan.srt_reads.push_back({Value(&poison), 2u});
      plan.clean_flat_slots.push_back(0);
      plan.control_flow.resize(3);
      plan.control_flow[0].condition = Value(&condition);
      plan.control_flow[0].successors = {1, 2};
      plan.control_flow[1].srt_reads = {0, 1};
      plan.control_flow[1].sources = {0};
      plan.control_flow[2].srt_reads = {2};
      return plan;
    };
    const auto old_plan = make();
    const auto new_plan = make();
    ResourcePlanFixture fx;
    fx.strict_reader = true;
    ResourceDelta delta;
    DiffMemory memory;
    memory.unknown_fails = true;
    for (int d = 0; d < 2; ++d) {
      memory.words[d][0x2000] = d ? 0x8000 : 0x4000;
      memory.words[d][0xdead0000ull] = 0x1234;
    }
    Tally cfg_tally;
    Outcome a, b;
    const auto before = GetSrtCompiledStats();
    // Three passes on the taken branch: record, then replay compiled.
    for (int i = 0; i < 4; ++i)
      Check(Compare("cfg taken", old_plan, new_plan, fx, memory, std::vector<uint32_t> {0x2000, 0, 17, 19}, delta, cfg_tally, &a, &b),
            "CFG taken branch matches the existing path");
    Check(GetSrtCompiledStats().cfg > before.cfg, "recorded CFG trace is served by the compiled program");
    // Branch flip: condition 0 takes the other successor, whose path was never recorded.
    const auto flip_before = GetSrtCompiledStats();
    Check(Compare("cfg flip", old_plan, new_plan, fx, memory, std::vector<uint32_t> {0x2000, 0, 0, 19}, delta, cfg_tally, &a, &b),
          "CFG branch flip matches the existing path (unrecorded outcome falls back)");
    Check(GetSrtCompiledStats().fallback[4] > flip_before.fallback[4], "unseen branch outcome is counted as a CFG miss");
    for (int i = 0; i < 3; ++i)
      Check(Compare("cfg flipped steady", old_plan, new_plan, fx, memory, std::vector<uint32_t> {0x2000, 0, 0, 19}, delta, cfg_tally, &a, &b),
            "flipped branch matches after being recorded");
    // Alternate branches: both recorded paths replay compiled and never trust the previous branch.
    const auto alt_before = GetSrtCompiledStats();
    for (int i = 0; i < 6; ++i)
      Check(Compare("cfg alternate", old_plan, new_plan, fx, memory,
                    std::vector<uint32_t> {0x2000, 0, (i & 1) ? 17u : 0u, 19}, delta, cfg_tally, &a, &b),
            "alternating CFG branches match the existing path every call");
    Check(GetSrtCompiledStats().cfg >= alt_before.cfg + 6, "both recorded CFG paths are served compiled");
    Check(cfg_tally.mismatches == 0, "no CFG differential mismatch");
  }
  Common::RuntimeFlags::g_published.store(saved_flags & ~(uint64_t{1} << Common::RuntimeFlags::kBitSrtCompiled), std::memory_order_relaxed);
  Check(tally.mismatches == 0, "no pointer-chase differential mismatch");
}


// --- input-keyed descriptor reuse and dead-memo elimination --------------------------------------
// Every step runs the flag-off path and the compiled path on identical inputs and demands identical
// outputs, read sequences and observed user-data registers. The cases perturb exactly what a reused
// descriptor depends on (a leaf word, an unrelated word, a register, a pointer, a failing word) and, for
// the branching plan, flip the branch so the compiled trace misses after steps that skipped their memo
// publication. The checks on the statistics prove the mechanisms actually ran.
inline uint32_t BufferDword(const Outcome& outcome, uint32_t index) {
  return outcome.snapshot.buffers.empty() ? 0xdeadbeefu : outcome.snapshot.buffers[0].dwords[index];
}

inline void SetWindowWord(ChaseWorld& world, uint64_t address, uint32_t value) {
  for (int d = 0; d < 2; ++d) world.memory.words[d][address] = value;
}

inline void TestDescriptorReuseAndDeadMemo() {
  // The A/B switches turn the mechanisms off on purpose; the statistics checks below would then fail.
  if (std::getenv("KYTY_SRT_KEEP_MEMO") != nullptr || std::getenv("KYTY_SRT_NO_DESC_REUSE") != nullptr) return;
  const auto saved_flags = Common::RuntimeFlags::g_published.load(std::memory_order_relaxed);
  Common::RuntimeFlags::g_published.store(saved_flags | (uint64_t{1} << 2) | (uint64_t{1} << 10) | (uint64_t{1} << 11),
                                          std::memory_order_relaxed);
  Tally tally;
  // Straight-line plans: gather-shaped (0), arithmetic (1), arithmetic with a user-data dword (2).
  for (int variant = 0; variant < 3; ++variant) {
    const auto old_plan = ChasePlan(variant);
    const auto new_plan = ChasePlan(variant);
    ChaseWorld world;
    world.regs = {0x10000, 0, 0x55};
    ResourcePlanFixture fx;
    ResourceDelta delta;
    Outcome a, b;
    const uint32_t add2 = variant == 0 ? 0u : 5u;
    const auto expect_word2 = [&](uint32_t base) { return base + add2; };
    for (int i = 0; i < 3; ++i) {
      world.Build(0x20000);
      Check(Compare("reuse warm-up", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) && a.ok,
            "reuse plan: warm-up calls match the existing path");
    }
    Check(BufferDword(b, 2) == expect_word2(0x40), "reuse plan: descriptor carries the memory words");
    auto before = GetSrtCompiledStats();
    world.Build(0x20000);
    Check(Compare("reuse steady", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b),
          "reuse plan: unchanged inputs match the existing path");
    if (variant < 2) {
      Check(GetSrtCompiledStats().desc_reuse > before.desc_reuse, "unchanged descriptor inputs reuse the stored descriptor");
    } else {
      Check(GetSrtCompiledStats().desc_reuse == before.desc_reuse, "a source that reads user data is never reused");
    }
    // A word the descriptor does not depend on changes: still reusable, still identical.
    before = GetSrtCompiledStats();
    world.Build(0x20000);
    SetWindowWord(world, 0x10000 + 8, 0x99);
    Check(Compare("reuse unrelated word", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
              b.snapshot.flattened_srt[2] == 0x99 && BufferDword(b, 2) == expect_word2(0x40),
          "reuse plan: an unrelated flat word changes the flat buffer but not the descriptor");
    if (variant < 2)
      Check(GetSrtCompiledStats().desc_reuse > before.desc_reuse, "an unrelated change keeps the descriptor reusable");
    // A leaf word changes: the stored descriptor must not be used.
    world.Build(0x20000, 0x44);
    Check(Compare("reuse leaf change", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
              BufferDword(b, 2) == expect_word2(0x44),
          "reuse plan: a changed input word produces the new descriptor");
    // Alternate two states: an entry stored for one must never answer the other.
    for (int i = 0; i < 8; ++i) {
      const uint32_t word2 = (i & 1) ? 0x48u : 0x40u;
      world.Build(0x20000, word2);
      Check(Compare("reuse alternate", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
                BufferDword(b, 2) == expect_word2(word2),
            "reuse plan: alternating inputs match the existing path every call");
    }
    // The user-data register behind the impure dword changes.
    world.Build(0x20000);
    world.regs[2] = 0x66;
    Check(Compare("reuse register", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
              (variant != 2 || BufferDword(b, 3) == 0x66u),
          "reuse plan: a changed user-data register is observed");
    // The whole chase moves to another window.
    world.regs[0] = 0x30000;
    world.Build(0x50000, 0x4c);
    Check(Compare("reuse pointer", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
              BufferDword(b, 2) == expect_word2(0x4c),
          "reuse plan: a moved pointer is followed and not answered from the stored descriptor");
    // A failing read: both fail, and the next call recovers.
    world.Build(0x50000, 0x4c);
    world.memory.invalid[0].insert(0x50000 + 8);
    world.memory.invalid[1].insert(0x50000 + 8);
    Check(Compare("reuse failure", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) && !a.ok && !b.ok,
          "reuse plan: a failing read fails both paths");
    world.Build(0x50000, 0x4c);
    Check(Compare("reuse recovery", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) && b.ok &&
              BufferDword(b, 2) == expect_word2(0x4c),
          "reuse plan: the call after a failure is evaluated, not reused");
  }
  // Branching plans: the dead user-data ops skip their publication; a branch flip misses the compiled trace after
  // them and the recording DFS must see the same state as without the elimination.
  for (int variant = 0; variant < 2; ++variant) {
    const auto old_plan = ChasePlan(variant, true);
    const auto new_plan = ChasePlan(variant, true);
    ChaseWorld world;
    world.regs = {0x10000, 0, 1};
    ResourcePlanFixture fx;
    fx.strict_reader = true;
    ResourceDelta delta;
    Outcome a, b;
    const uint32_t add2 = variant == 0 ? 0u : 5u;
    const auto before = GetSrtCompiledStats();
    for (int i = 0; i < 5; ++i) {
      world.Build(0x20000);
      Check(Compare("cfg reuse taken", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) && a.ok,
            "cfg reuse plan: taken branch matches the existing path");
    }
    const auto steady = GetSrtCompiledStats();
    Check(steady.cfg > before.cfg, "cfg reuse plan: recorded trace is served compiled");
    Check(steady.memo_skipped > before.memo_skipped, "cfg reuse plan: dead memo publications are skipped");
    Check(steady.desc_reuse > before.desc_reuse, "cfg reuse plan: descriptors are answered without full evaluation");
    Check(BufferDword(b, 2) == 0x40 + add2, "cfg reuse plan: descriptor carries the second window");
    // Flip: the unrecorded outcome misses after the shared prefix published (or skipped) its steps.
    world.regs[2] = 0;
    world.Build(0x20000);
    const auto miss_before = GetSrtCompiledStats();
    Check(Compare("cfg reuse flip", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b),
          "cfg reuse plan: a branch flip after skipped publications matches the existing path");
    Check(GetSrtCompiledStats().fallback[4] > miss_before.fallback[4], "cfg reuse plan: the flip is a trace miss");
    for (int i = 0; i < 12; ++i) {
      world.regs[2] = (i & 1) ? 1u : 0u;
      world.Build(0x20000, (i & 2) ? 0x44u : 0x40u);
      Check(Compare("cfg reuse alternate", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
                ((i & 1) == 0 || BufferDword(b, 2) == ((i & 2) ? 0x44u : 0x40u) + add2),
            "cfg reuse plan: alternating branches and inputs match the existing path");
    }
    // Memory changes under the taken branch, with a miss forced right after by flipping back.
    for (int i = 0; i < 6; ++i) {
      world.regs[2] = 1;
      world.Build(0x20000, 0x50 + 4 * i);
      SetWindowWord(world, 0x10000 + 8, 0x90 + i);
      Check(Compare("cfg reuse memory", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
                BufferDword(b, 2) == 0x50u + 4u * i + add2,
            "cfg reuse plan: changed window words produce the new descriptor");
      world.regs[2] = 0;
      Check(Compare("cfg reuse memory flip", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b),
            "cfg reuse plan: flipping away after a change matches the existing path");
    }
    // The pointer moves, a read fails, and the call after the failure recovers.
    world.regs[2] = 1;
    world.regs[0] = 0x30000;
    world.Build(0x50000, 0x4c);
    Check(Compare("cfg reuse pointer", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) &&
              BufferDword(b, 2) == 0x4cu + add2,
          "cfg reuse plan: a moved pointer is followed");
    world.memory.invalid[0].insert(0x50000 + 8);
    world.memory.invalid[1].insert(0x50000 + 8);
    Check(Compare("cfg reuse failure", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) && !a.ok && !b.ok,
          "cfg reuse plan: a failing read fails both paths");
    world.Build(0x50000, 0x4c);
    Check(Compare("cfg reuse recovery", old_plan, new_plan, fx, world.memory, world.regs, delta, tally, &a, &b) && b.ok &&
              BufferDword(b, 2) == 0x4cu + add2,
          "cfg reuse plan: recovery after a failed call");
  }
  Common::RuntimeFlags::g_published.store(saved_flags & ~(uint64_t{1} << 22), std::memory_order_relaxed);
  Check(tally.mismatches == 0, "no descriptor reuse / dead memo differential mismatch");
}

} // namespace CompiledSrtTest
