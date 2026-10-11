#pragma once

// Flags are published as one atomic value; readers never retain mutable buffers.
// Refresh only at frame/submit safe points. Hot probes perform one relaxed load.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>

namespace Common::RuntimeFlags {
inline constexpr int64_t kIntervalMs = 250;
inline constexpr unsigned kRevisionShift = 16;
inline constexpr uint64_t kFlagMask = (uint64_t{1} << kRevisionShift) - 1;
inline constexpr uint32_t kUnpinnedBits = uint32_t{1} << kRevisionShift;
static_assert(kRevisionShift < 32, "evaluation bits and unpinned sentinel must fit uint32_t");
struct FlagSet {
 bool resource_dynamic_deps = false;
 bool resource_memo_verify = false;
 bool srt_cfg_tape = false;
 bool srt_cfg_schedule = false;
 bool pm4_event_clean = true;
 bool pm4_write_data_fast = false;
 bool shader_clean_reads = false;
 bool srt_scalar_batch = false;
 bool resource_hash_filter = false;
 bool resource_memo_bypass = false;
 bool srt_cfg_gather = false;
 bool srt_cfg_trace = false;
 bool fault_ahead = false;
 bool resource_spec_cache = false;
 bool fault_parse_gate = false;
 bool idle_trace = false;
};
inline constexpr const char* kSwitchableFlags[] = {
 "KYTY_RESOURCE_DYNAMIC_DEPS", "KYTY_RESOURCE_MEMO_VERIFY", "KYTY_SRT_CFG_TAPE",
 "KYTY_SRT_CFG_SCHEDULE", "KYTY_PM4_EVENT_CLEAN", "KYTY_PM4_WRITE_DATA_FAST", "KYTY_SHADER_CLEAN_READS", "KYTY_SRT_SCALAR_BATCH", "KYTY_RESOURCE_HASH_FILTER", "KYTY_RESOURCE_MEMO_BYPASS", "KYTY_SRT_CFG_GATHER",
 "KYTY_SRT_CFG_TRACE", "KYTY_FAULT_AHEAD", "KYTY_RESOURCE_SPEC_CACHE", "KYTY_FAULT_PARSE_GATE", "KYTY_IDLE_TRACE"
};
inline uint64_t Bits(const FlagSet& f) {
 return uint64_t(f.resource_dynamic_deps) | (uint64_t(f.resource_memo_verify)<<1) |
  (uint64_t(f.srt_cfg_tape)<<2) | (uint64_t(f.srt_cfg_schedule)<<3) |
  (uint64_t(f.pm4_event_clean)<<4) | (uint64_t(f.pm4_write_data_fast)<<5) | (uint64_t(f.shader_clean_reads)<<6) | (uint64_t(f.srt_scalar_batch)<<7) |
  (uint64_t(f.resource_hash_filter)<<8) | (uint64_t(f.resource_memo_bypass)<<9) |
  (uint64_t(f.srt_cfg_gather)<<10) | (uint64_t(f.srt_cfg_trace)<<11) | (uint64_t(f.fault_ahead)<<12) |
  (uint64_t(f.resource_spec_cache)<<13) | (uint64_t(f.fault_parse_gate)<<14) | (uint64_t(f.idle_trace)<<15);
}
// Low kRevisionShift bits flags; remaining bits revision. One atomic publication.
inline std::atomic<uint64_t> g_published{0};
inline std::atomic<int64_t> g_last_poll{0};
inline std::mutex g_refresh_mutex;
inline std::string g_control_path;
inline bool g_path_set = false;
// Shader memory synchronization can submit while materialization is in progress.
// Pin the complete flag set for that evaluation, even if a safe-point poll publishes.
inline thread_local uint32_t g_evaluation_bits = kUnpinnedBits;
class EvaluationScope {
 uint32_t previous;
public:
 EvaluationScope():previous(g_evaluation_bits) {
  if (previous==kUnpinnedBits) g_evaluation_bits=static_cast<uint32_t>(g_published.load(std::memory_order_relaxed)&kFlagMask);
 }
 ~EvaluationScope() { g_evaluation_bits=previous; }
 EvaluationScope(const EvaluationScope&)=delete;
 EvaluationScope& operator=(const EvaluationScope&)=delete;
};
inline FlagSet Current() {
 const auto bits = g_evaluation_bits!=kUnpinnedBits ? g_evaluation_bits : g_published.load(std::memory_order_relaxed);
 return {.resource_dynamic_deps=bool(bits&1),.resource_memo_verify=bool(bits&2),
  .srt_cfg_tape=bool(bits&4),.srt_cfg_schedule=bool(bits&8),.pm4_event_clean=bool(bits&16),
  .pm4_write_data_fast=bool(bits&32),.shader_clean_reads=bool(bits&64),.srt_scalar_batch=bool(bits&128),
  .resource_hash_filter=bool(bits&256),.resource_memo_bypass=bool(bits&512),
  .srt_cfg_gather=bool(bits&1024),.srt_cfg_trace=bool(bits&2048),.fault_ahead=bool(bits&4096),
  .resource_spec_cache=bool(bits&8192),.fault_parse_gate=bool(bits&16384),.idle_trace=bool(bits&32768)};
}
inline uint64_t Revision() { return g_published.load(std::memory_order_relaxed)>>kRevisionShift; }
inline int64_t NowMs() {
 return std::chrono::duration_cast<std::chrono::milliseconds>(
  std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool PollNotDue() {
 const auto last = g_last_poll.load(std::memory_order_relaxed);
 return last && NowMs()-last<kIntervalMs;
}
inline void SetControlFile(const std::string& path) {
 std::lock_guard lock(g_refresh_mutex); g_control_path=path; g_path_set=true;
}
inline bool* FieldFor(FlagSet& f,const char* name) {
 if (!std::strcmp(name,kSwitchableFlags[0])) return &f.resource_dynamic_deps;
 if (!std::strcmp(name,kSwitchableFlags[1])) return &f.resource_memo_verify;
 if (!std::strcmp(name,kSwitchableFlags[2])) return &f.srt_cfg_tape;
 if (!std::strcmp(name,kSwitchableFlags[3])) return &f.srt_cfg_schedule;
 if (!std::strcmp(name,kSwitchableFlags[4])) return &f.pm4_event_clean;
 if (!std::strcmp(name,kSwitchableFlags[5])) return &f.pm4_write_data_fast;
 if (!std::strcmp(name,kSwitchableFlags[6])) return &f.shader_clean_reads;
 if (!std::strcmp(name,kSwitchableFlags[7])) return &f.srt_scalar_batch;
 if (!std::strcmp(name,kSwitchableFlags[8])) return &f.resource_hash_filter;
 if (!std::strcmp(name,kSwitchableFlags[9])) return &f.resource_memo_bypass;
 if (!std::strcmp(name,kSwitchableFlags[10])) return &f.srt_cfg_gather;
 if (!std::strcmp(name,kSwitchableFlags[11])) return &f.srt_cfg_trace;
 if (!std::strcmp(name,kSwitchableFlags[12])) return &f.fault_ahead;
 if (!std::strcmp(name,kSwitchableFlags[13])) return &f.resource_spec_cache;
 if (!std::strcmp(name,kSwitchableFlags[14])) return &f.fault_parse_gate;
 if (!std::strcmp(name,kSwitchableFlags[15])) return &f.idle_trace;
 return nullptr;
}
inline void Trim(std::string& s) {
 const auto first=s.find_first_not_of(" \t\r\n");
 if (first==std::string::npos) { s.clear(); return; }
 const auto last=s.find_last_not_of(" \t\r\n"); s=s.substr(first,last-first+1);
}
inline FlagSet BuildLocked() {
 FlagSet f;
 for (auto* name:kSwitchableFlags) {
  const char* v=std::getenv(name);
  // pm4_event_clean is on by default: it is the correctness-preserving fix for
  // EVENT_WRITE payload reads that false-share a 4 KiB page with GPU-written data.
  // Explicitly set KYTY_PM4_EVENT_CLEAN=0 to force the old faulting read.
  *FieldFor(f,name) = std::strcmp(name,kSwitchableFlags[4])==0 ? !(v && v[0]=='0') : (v && v[0]=='1');
 }
 if (!g_path_set) {
  const char* v=std::getenv("KYTY_FLAG_FILE");
  g_control_path=v && *v ? v : "flags.ctl"; g_path_set=true;
 }
 std::ifstream file(g_control_path,std::ios::binary); std::string line;
 while (std::getline(file,line)) {
  const auto eq=line.find('='); if (eq==std::string::npos) continue;
  auto name=line.substr(0,eq), value=line.substr(eq+1); Trim(name); Trim(value);
  if (value!="0" && value!="1") continue;
  if (auto* field=FieldFor(f,name.c_str())) *field=value=="1";
 }
 return f;
}
inline void Refresh(bool force=false) {
 if (!force && PollNotDue()) return;
 std::lock_guard lock(g_refresh_mutex);
 const auto now=NowMs(), last=g_last_poll.load(std::memory_order_relaxed);
 if (!force && last && now-last<kIntervalMs) return;
 const auto f=BuildLocked(); const auto old=g_published.load(std::memory_order_relaxed);
 g_last_poll.store(now,std::memory_order_relaxed);
 const auto bits=Bits(f);
 if ((old&kFlagMask)==bits && (old>>kRevisionShift)!=0) return;
 const auto revision=(old>>kRevisionShift)+1;
 g_published.store((revision<<kRevisionShift)|bits,std::memory_order_relaxed);
 std::printf("NHL27CFG rev=%llu",static_cast<unsigned long long>(revision));
 auto printable=f;
 for (auto* name:kSwitchableFlags) std::printf(" %s=%d",name,*FieldFor(printable,name));
 std::printf("\n"); std::fflush(stdout);
}
}
