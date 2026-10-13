# NHL 27 performance switches (claude/nhl27-perf)

There are two kinds of switches, and a key name belongs to exactly one of them.

* **Runtime flags** (`src/common/runtimeFlags.h`): read from the environment at startup and from
  `flags.ctl` (`KEY=0|1` per line, `KYTY_FLAG_FILE` names another file) every 250 ms, so they can be
  flipped while the game runs. A key missing from both means the default in `FlagSet`; an explicit
  `0`/`1` overrides it.
* **Env-only startup switches** (`src/common/experiments.cpp`): `KYTY_X=1` enables, `KYTY_X=0`
  disables, unset means the default below. They are read once at startup. For in-match A/B tests,
  `KYTY_EXPERIMENT_FILE` names a file holding a decimal mask (bit numbers below); while the file
  exists its mask replaces the environment values.

## Merged defaults (nhl27 + claude/nhl27-perf)

Where both branches implemented the same feature, only one implementation is left and it is a
runtime flag. The experiment bits for these features are retired (their bit numbers are unused).

### Runtime flags (flags.ctl, live)

| Key | Default | Notes |
|---|---|---|
| `KYTY_GUEST_SYNC` | on | Orders compute work with the guest's own barriers instead of a barrier around every dispatch (+13% with `KYTY_SKIP_REDUNDANT_BINDS`) |
| `KYTY_GUEST_BARRIER_DEDUP` | on | Skips redundant guest global barriers |
| `KYTY_DISPATCH_ARGS_GUARD` | on | See "Always on" below; `0` disables |
| `KYTY_RT_DECODE_MEMO` | on | Memoizes color render-target decoding (this was `KYTY_RENDER_TARGET_MEMO`, +3.4%; that name no longer exists) |
| `KYTY_FAULT_COALESCE` | on | A CPU write fault releases its clean aligned window (`KYTY_FAULT_COALESCE_PAGES`, env only, default 16); +2.0%, write faults -85% |
| `KYTY_BDA_BATCH`, `KYTY_RESOURCE_SPEC_CACHE`, `KYTY_SRT_COMPILED` and the other pengui07 flags | on | See `runtimeFlags.h` |
| `KYTY_LONG_BUFFER_GC_AGE` | off | Keeps idle cache buffers 10x longer; +9.2% in Petr's measurement |
| `KYTY_READBACK_QUEUE` | off | Downloads of a buffer last written by an already submitted command buffer run on a second queue; +1.4% |
| `KYTY_DIRECT_EOP_INTERRUPTS` | off | End-of-pipe interrupts are delivered when the CP reaches them; +0.7% |
| `KYTY_SPARSE_BUFFER_GC` | off | Scans for idle cache buffers every 16th submission; +1.0% |

The four that are off were measured as wins on Petr's tree but are still off in the merged
defaults; turn them on with `KYTY_X=1` or a `flags.ctl` line, or flip the default in `FlagSet`.

### Env-only startup switches (default on, `=0` disables)

| Variable | Mask bit | What it does | Measured |
|---|---|---|---|
| `KYTY_SRT_BATCH_READS` | 128 | Reads contiguous resource-table dwords with one call | +54% in a match |
| `KYTY_SKIP_REDUNDANT_BINDS` | 2 | Skips re-pushing identical descriptors and push constants | (with guest sync) |
| `KYTY_DIRECT_SRT_READS` | 1024 | Evaluates resource-table reads and descriptor copies without the IR interpreter | +22.4% paused match (11.1 -> 13.6 fps) |
| `KYTY_IMAGE_LOOKUP_HINT` | 4096 | Reuses the image a texture last resolved to instead of scanning every page | +2.3% paused match |
| `KYTY_CLEAN_FROM_DIRECT` | 262144 | The strict resource-table evaluator takes words the direct reads already fetched | +3.1% paused match |
| `KYTY_TEXTURE_DESC_CACHE` | 2048 | No consumer in the merged tree (pengui07's draw-prep delta replaces it); accepted for compatibility | was +15.5% |
| `KYTY_ADAPTIVE_RESOURCE_MEMO` | 32768 | No consumer in the merged tree (`KYTY_RESOURCE_MEMO_BYPASS` replaces it); accepted for compatibility | was +2.3% |

Default off (set to `1` to enable): `KYTY_CPU_COPY_FORWARD` (8), `KYTY_FALSE_SHARE_READS` (64),
`KYTY_ASYNC_SUBMIT` (2097152), `KYTY_LAZY_SHARED_WRITES` (4194304), `KYTY_NATIVE_FP16` (8388608),
`KYTY_FAULT_PAGE_READS` (16777216), `KYTY_GPU_COLOR_CLEAR` (67108864), `KYTY_POOLED_BDA_BUFFERS`
(134217728), `KYTY_ASYNC_GUEST_READS` (268435456), `KYTY_BDA_DIRTY_REGIONS` (536870912). All but `KYTY_CPU_COPY_FORWARD`, `KYTY_NATIVE_FP16`
and `KYTY_POOLED_BDA_BUFFERS` are wins for NHL 27 (see the measurement log).
`KYTY_RESOURCE_MULTI_CACHE=1` (16-state resource memo) is read only by the resource memo at startup and has no mask bit.

The bits of the removed switches (1, 4, 256, 512, 8192, 16384, 65536, 131072, 524288, 1048576) are ignored in
an experiment mask; use `flags.ctl` for those features.

Dispatch args guard, `KYTY_PROGRAM_RESOURCE_CACHE=1`, `KYTY_PERF_NO_SYNC_READBACK=1`,
`KYTY_EOP_DIRECT_LABELS=1` and `KYTY_SHADER_PREGEN=0` remain separate environment settings.

## Measurement log (claude/nhl27-perf before the merge)

The table below is the original per-switch log, from when every switch was env-only and default off
(`=1` and the mask bits below describe that build). The merged defaults and key types are the tables
above; read this only for the measurements.

### Recommended for NHL 27 on that build

On top of the fork's usual flags (`KYTY_PROGRAM_RESOURCE_CACHE=1`, `KYTY_PERF_NO_SYNC_READBACK=1`,
`KYTY_EOP_DIRECT_LABELS=1`, `KYTY_SHADER_PREGEN=0`, ...):

| Variable | Mask bit | What it does | Measured |
|---|---|---|---|
| `KYTY_SRT_BATCH_READS=1` | 128 | Reads contiguous resource-table dwords with one call (Codex's switch) | +54% in a match |
| `KYTY_GUEST_SYNC=1` | 1 | Orders compute work with the guest's own barriers instead of a barrier around every dispatch | +13% with the next one |
| `KYTY_SKIP_REDUNDANT_BINDS=1` | 2 | Skips re-pushing identical descriptors and push constants | (with the previous) |
| `KYTY_DIRECT_SRT_READS=1` | 1024 | Evaluates resource-table reads and descriptor copies without the IR interpreter | +22.4% paused match (11.1 -> 13.6 fps) |
| `KYTY_TEXTURE_DESC_CACHE=1` | 2048 | Memoizes texture descriptor decoding | +15.5% in a match |
| `KYTY_IMAGE_LOOKUP_HINT=1` | 4096 | Reuses the image a texture last resolved to instead of scanning every page | +2.3% paused match |
| `KYTY_LONG_BUFFER_GC_AGE=1` | 8192 | Keeps idle cache buffers 10x longer (the collector ages per submission) | +9.2% paused match |
| `KYTY_RENDER_TARGET_MEMO=1` | 16384 | Memoizes color render-target decoding (registers to image description) | +3.4% paused match |
| `KYTY_ADAPTIVE_RESOURCE_MEMO=1` | 32768 | Stops capturing resource-table reads for shaders whose inputs change on every draw (the memo can never hit there) | +2.3% paused match |
| `KYTY_FAULT_COALESCE=1` | 65536 | A CPU write fault on a tracked buffer page releases its whole clean 64 KiB window (`KYTY_FAULT_COALESCE_PAGES`, default 16) | +2.0% paused match, write faults -85% |
| `KYTY_SPARSE_BUFFER_GC=1` | 131072 | Scans for idle cache buffers every 16th guest submission instead of after each one | +1.0% paused match |
| `KYTY_CLEAN_FROM_DIRECT=1` | 262144 | The strict resource-table evaluator (branch conditions, uniform fills, written descriptors) takes words the direct reads already fetched | +3.1% paused match |
| `KYTY_READBACK_QUEUE=1` | 1048576 | GPU-to-CPU downloads of a buffer last written by an already submitted command buffer run on a second queue and wait only for that submission (about 45 us instead of draining everything queued) | +1.4% paused match |
| `KYTY_DIRECT_EOP_INTERRUPTS=1` | 524288 | End-of-pipe interrupts are delivered when the CP reaches them, like the labels (`KYTY_EOP_DIRECT_LABELS`) | +0.7% paused match (+0.8% with the readback queue) |
| `KYTY_FALSE_SHARE_READS=1` | 64 | The GPU thread parses PM4 packets (and serves its own faulting loads) from the backing when the bytes are clean but share a page with GPU-written bytes | +0.5% paused match |
| `KYTY_ASYNC_SUBMIT=1` | 2097152 | `vkQueueSubmit` runs on a worker thread in tick order (about 65 us of GPU-thread time per submit); submit splits default to 48 draws / 0.75 ms with it | +1.5% paused match, +4.5% more with the finer splits |
| `KYTY_LAZY_SHARED_WRITES=1` | 4194304 | A guest thread's store into a page where the GPU wrote at most 256 other bytes no longer waits for that GPU work: the download is queued and merged into guest memory later, keeping bytes the CPU changed meanwhile | +7.3% paused match |
| `KYTY_FAULT_PAGE_READS=1` | 16777216 | Resource-table reads of a page on which the GPU thread took a read fault (the GPU wrote other bytes of it) read clean bytes through the backing instead of faulting and downloading again | +4.8% paused match (busy scene) |
| `KYTY_NATIVE_FP16=1` | 8388608 | f16vec2 math for packed half-float ops instead of 32-bit unpack/pack (now switchable at runtime) | 2-19% less GPU time on several heavy shaders; about +3% gameplay now that the GPU limits part of the time |
| `KYTY_GPU_COLOR_CLEAR=1` | 67108864 | A color target or texture whose fast-clear metadata (DCC/CMASK) the GPU wrote is checked on the GPU; matching slices are cleared through conditional rendering (VK_EXT_conditional_rendering) instead of reading the metadata back and waiting for the GPU | removes a per-frame readback wait (about 5% in scenes that have it); no change where they do not |
| `KYTY_ASYNC_GUEST_READS=1` | 268435456 | A guest thread's read of GPU-written memory no longer stalls the GPU thread: it queues and submits the copy and goes on recording, and the guest thread waits for the copy itself | +5.0% paused match, +4.6% gameplay (PGO build) |
| `KYTY_BDA_DIRTY_REGIONS=1` | 536870912 | With `KYTY_BDA_DIRTY_CACHE=1`, BDA preparation synchronizes only the tracker regions that gained CPU writes instead of every mapped range (a mapping change still scans everything); now an experiment bit, switchable while running | +2.2% paused match, +11% gameplay (8 rounds, noisy) |
| `KYTY_POOLED_BDA_BUFFERS=1` | 134217728 | Cache buffers up to 32 MiB come from VMA's shared blocks instead of one `vkAllocateMemory` each (about 2% of the GPU thread in gameplay) | within noise so far |
| `KYTY_GPU_THREAD_PRIORITY=1` | 256 | The GPU thread runs at highest priority | -8% gameplay (noisy); left off |
| `KYTY_EXEC_SELECTS=1` | 33554432 | Shader pass: VGPR writes under a partial EXEC whose old lanes nobody reads lose their select (part of the shader keys) | experimental: the default rule set removes few selects; `KYTY_DBG_EXECSEL_MODE` enables stronger rules that currently break NHL 27 |

The last five rows were measured after the merge (merged defaults, plus the four runtime flags
that are off by default and `KYTY_PM4_WRITE_DATA_FAST=1`). They are env-only switches, default
off; their mask bits are 64, 2097152, 4194304, 16777216 and 8388608.

### Why a frame waits

NHL 27 submits a whole frame only after the previous frame's end-of-pipe interrupt, then rewrites
command-buffer memory that shares pages with bytes a compute shader wrote into the previous
frame's command buffers. That write faults, and the fault used to download the page after
everything queued on the GPU (11-15 ms per frame). With both switches above the download waits
only for the writer's submission. Recorded with `KYTY_DBG_TIMELINE` (below).

With direct EOP interrupts the CP no longer waits there; instead the game's own render thread
then stores its next command buffer into a page that holds 16 bytes a compute shader
(abce16f02df45052) wrote at the end of the frame, and that store waited for the host GPU to
finish the frame (about 6 ms per frame). `KYTY_LAZY_SHARED_WRITES` lets the store go ahead and
merges the 16 bytes afterwards. The merge writes a GPU dword only where guest memory still holds
the value it had at the fault, so a later CPU store wins, except one that rewrites the very same
value; a CPU read of those bytes before the merge sees their old value.

### Submit split

The merged `submit_split` runtime flag ends a host command buffer early during a long guest
command buffer so the GPU can start on it. Each split is one more `vkQueueSubmit` (about 65 us on
Windows), so without `KYTY_ASYNC_SUBMIT` the default thresholds are now 768 draws or 12 ms (`KYTY_SUBMIT_SPLIT_DRAWS`,
`KYTY_SUBMIT_SPLIT_US`; `split.ctl` overrides them at runtime): +3.9% paused match against
48 draws / 0.75 ms. Turning the split off costs 8%. Deferring the submit across queued guest
submissions (one host command buffer for many of them) lost 16%, because the GPU then idles while
the GPU thread records.

## Profile-guided build

`-DKYTY_PGO=use` (clang-cl) builds the emulator and `common` with the NHL 27 gameplay profile in
`tools/pgo/nhl27.profdata`: 18.6 vs 16.1 fps in gameplay (+15%, three boots of each build).
The profile matches clang 22.1.8; other versions ignore it with a warning. `-DKYTY_PGO=generate`
builds the instrumented emulator that records a new profile, and `-DKYTY_THIN_LTO=ON` adds
ThinLTO; `tools/pgo/README.md` has the steps.

Comparing builds needs separate boots, and scenes differ between boots, so compare the
window-title fps over a few minutes of gameplay in several boots of each. `KYTY_PERF_STATS=1`
must be off for such comparisons: its timers take tens of millions of timestamps per second.

## Always on

`KYTY_DISPATCH_ARGS_GUARD` (a runtime flag, default on; set to `0` in the environment or `flags.ctl` to disable): every indirect dispatch reads its group counts
through a small GPU pass that zeroes counts above the device limit. NHL 27 occasionally issues an
indirect dispatch with 0xffffffff x3 groups during match loads; without the guard the GPU never
finishes it and the device is lost. Zeroed dispatches are reported as `NHL27GUARD` lines.

## Diagnostics

* `KYTY_PERF_STATS=1`: every 5 s, sync waits by cause, per-frame GPU-thread phase times
  (`NHL27PHASE`, `NHL27DRAW`), buffer/image churn and device memory.
* `KYTY_DBG_BREADCRUMBS=1`: GPU breadcrumbs, now with indirect dispatch arguments, printed on
  device loss.
* `KYTY_DBG_IMAGE_CHURN=1`: logs image inserts and frees.
* `KYTY_DBG_SHARED_WRITES=1`: the instruction behind each guest write fault (`NHL27SHW`) and the
  GPU-written bytes a write fault had to download (`NHL27SHD`).
* `KYTY_DBG_PIPELINE_STATS=<hex hash>[,...]|all`: driver statistics of pipelines using those shaders
  (VGPRs, SGPRs, LDS, scratch; `NHL27PSTAT`) and their ISA in `_Shaders` (VK_KHR_pipeline_executable_properties).
* `-DKYTY_LINE_TABLES=ON` (CMake): source lines and inline frames in the PDB, for the sampler.
* `KYTY_DBG_DIRECT_SRT=1`: counts direct and interpreted resource-table refreshes and why plans fall back (`NHL27SRT`).
* `KYTY_PERF_FAULT_RIPS=gpu`: host instruction addresses of the GPU thread's page faults.
* `KYTY_DBG_TIMELINE=<seconds or trigger file>`: records submissions, GPU-thread idle/wake, host
  GPU waits, event-queue waits and triggers, faults and synchronous GPU commands from all threads
  for 1.5 s, then prints them in time order (`NHL27TL`). A non-numeric value names a file whose
  appearance starts the recording.
* `KYTY_PERF_GPU_TIMING=1|2`: per draw/dispatch GPU time by shader (`NHL27GPU`, every 5 s).
  `2` measures each operation from the completion of the previous one, so the times add up to
  GPU busy time; `1` measures from top of pipe (overlapping).
* `tools/profiling/sampler.exe <pid> <seconds> <out> [interval_ms] [tid|rN]`: out-of-process
  stack sampler (about 400 samples/s, dbghelp); `tools/profiling/foldprof.py` summarizes its output.
* Replay benchmarks on a copy of a `_PipelineCache/*.shaders` record (no Vulkan):
  `shader_recompiler_compute_tests --resource-replay-phases <file> [iterations] [direct]`
  (validates the direct path against the interpreter), `--resource-replay-shapes <file>`,
  `--resource-replay-dump <file> <hex hash>`.
