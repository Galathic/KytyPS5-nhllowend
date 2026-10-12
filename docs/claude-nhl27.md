# NHL 27 performance switches (claude/nhl27-perf)

All switches are environment variables and default off, except the dispatch guard. Each one also
has a bit in the runtime experiments mask (`KYTY_EXPERIMENT_FILE` names a file holding a decimal
mask; while the file exists it replaces the environment defaults), which is how the in-match
A/B comparisons below were made.

## Recommended for NHL 27

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

Mask for all of the above: `2096131`, plus `KYTY_SRT_BATCH_READS=1` in the environment.

### Why a frame waits

NHL 27 submits a whole frame only after the previous frame's end-of-pipe interrupt, then rewrites
command-buffer memory that shares pages with bytes a compute shader wrote into the previous
frame's command buffers. That write faults, and the fault used to download the page after
everything queued on the GPU (11-15 ms per frame). With both switches above the download waits
only for the writer's submission. Recorded with `KYTY_DBG_TIMELINE` (below).

### Submit split

The merged `submit_split` runtime flag ends a host command buffer early during a long guest
command buffer so the GPU can start on it. Each split is one more `vkQueueSubmit` (about 65 us on
Windows), so the default thresholds are now 768 draws or 12 ms (`KYTY_SUBMIT_SPLIT_DRAWS`,
`KYTY_SUBMIT_SPLIT_US`; `split.ctl` overrides them at runtime): +3.9% paused match against
48 draws / 0.75 ms. Turning the split off costs 8%. Deferring the submit across queued guest
submissions (one host command buffer for many of them) lost 16%, because the GPU then idles while
the GPU thread records.

## Always on

`KYTY_DISPATCH_ARGS_GUARD` (set to `0` to disable): every indirect dispatch reads its group counts
through a small GPU pass that zeroes counts above the device limit. NHL 27 occasionally issues an
indirect dispatch with 0xffffffff x3 groups during match loads; without the guard the GPU never
finishes it and the device is lost. Zeroed dispatches are reported as `NHL27GUARD` lines.

## Diagnostics

* `KYTY_PERF_STATS=1`: every 5 s, sync waits by cause, per-frame GPU-thread phase times
  (`NHL27PHASE`, `NHL27DRAW`), buffer/image churn and device memory.
* `KYTY_DBG_BREADCRUMBS=1`: GPU breadcrumbs, now with indirect dispatch arguments, printed on
  device loss.
* `KYTY_DBG_IMAGE_CHURN=1`: logs image inserts and frees.
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
