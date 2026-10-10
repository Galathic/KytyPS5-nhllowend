# NHL 27 performance investigation

Work branch: `codex/nhl27-60fps`, based on `petrmolek/KytyPS5-nhl` default
`nhl27` at `6da3cf485f8b481e842dd8f7f09bd3162db527f8` (2026-10-10).
Goal: measured stable 60 FPS during gameplay. This goal has **not been reached**.

## Isolation

The supplied `Documents/KytyPS5-nhl-releases` source snapshot is read-only input.
Codex's clone, build, test binaries, logs, caches and copied saves are inside its
`_codex/nhl60` directory. The only configured remote is the user's
`petrmolek/KytyPS5-nhl` fork; implicit pushes are disabled. Claude's Downloads
checkouts, build tools and run directories are read for comparison and are not
modified. Never stop an emulator by name: test operations use the recorded PID.
Check for another emulator before every launch, including after 16:00 Budapest.
Game files and saved profiles, caches, captures and logs must stay outside Git.

## Sources compared

- User fork: default `nhl27` at `6da3cf48`.
- `pengui07/KytyPS5`: `main` at `307fdcf7` and `nhl27` at `353051a2`
  diverge: `nhl27` has 44 commits absent from `main`, and `main` has 76 absent
  from `nhl27`. The NHL branch contains the targeted renderer/shader fixes;
  neither branch simply supersedes the other. Retain the user fork as the base.
- The connected GitHub account can read the user fork and `pengui07/KytyPS5`,
  but the `pengui07/KytyPS5-nhl` branches endpoint returns 404. No writes to
  either pengui07 repository are permitted.
- Claude's older `Downloads/KytyPS5/nhl27` snapshot at `492781c0`: contains
  AMD D16 packing, initially dirty region tracking, kernel/save and shader fixes.
  Many fork fixes solve overlapping problems differently; do not merge it wholesale.
- Claude's `Downloads/KytyPS5-nhl/perf60` snapshot at `c6a38e00`: three
  commits over the same user-fork base, fixing malformed depth-alias diagnostic
  hunks, adding scripted input and porting the AMD D16 workaround.

## Changes under evaluation

- Retain Claude's AMD D16 sample-packing workaround and renderDraw build fix.
- Adapt scripted input to the SDL event thread, with bounded command/delay
  handling and shutdown key release; avoid a detached thread changing input state.
- `KYTY_BDA_DIRTY_REGIONS=1` with `KYTY_BDA_DIRTY_CACHE=1`: retain the fork's
  generation checks, but synchronize only regions gaining CPU writes or new
  buffers. Mapping changes still require a full scan. Adapted from Claude's
  `da5d0803`; use an intrusive atomic queue instead of allocating from a fault
  handler or taking a separate queue mutex. Clear queue membership before each
  upload so a concurrent write can schedule the region again.
- The dirty-region switch defaults off for controlled A/B measurement.
- Retain resource read buffers and replacement entries from the supplied newest
  snapshot. This avoids repeatedly destroying their allocations on cache misses.
- Correct `WaveReduceU32` SPIR-V emission: the execution scope is a constant ID,
  but `GroupOperationReduce` must be a literal operand. The original emitted
  shader fails validation with "Invalid group operation operand: 24".
  GPU tests cover all six operations, signed values, 1/7/19/32 invocations and
  sparse control flow. An experimental scan lowering was removed once this
  concrete encoding error was found.
- Adapt Claude's `492781c0` compressed-image storage shadows to the NHL
  renderer, preserving its breadcrumb API. Avoid the cache lock when no shadow
  writes are pending. GPU checks preserve untouched blocks, mips and layers
  for both 8-byte and 16-byte compressed blocks and reused storage views.
- Repair shader warm-up: rebound stages need both the compiled-program and
  resource-snapshot pointers. Saving the Vulkan driver cache must retain its
  live handle; previously the first save disabled the cache and all later saves.
- Refresh the single-entry memo directly into its output, avoiding a duplicate
  snapshot and a full copy on each miss. Output-object changes invalidate reuse;
  dependency checks and changing push-data updates remain mandatory.
- `KYTY_PERF_RESOURCE_TIMING=1` reports CPU resource-refresh cost per shader
  source, with calls, hits and plan sizes. It is off by default and must be off
  for final gameplay FPS measurements.
- With resource timing enabled, `KYTY_RESOURCE_CACHE_CONTROL_FILE` can name a
  diagnostic text file containing `1` or `16`. The GPU thread applies capacity
  changes between lookups, invalidating resource memos while retaining compiled
  pipelines. New shader sources inherit the active policy. This allows comparisons
  within one warmed match; exclude transitions from measured intervals.

## Verification and baseline

Hardware: AMD Ryzen 7 5800X3D, Radeon RX 9070 XT, Windows. Isolated test window:
1280x720. Gameplay resolution may be set independently by the guest.

- Release emulator and renderer GPU-test target build successfully with the
  existing portable clang-cl 22.1.8 and VS Build Tools.
- `memory_tracker_tests`: all cases pass, including newly published regions,
  duplicate enqueues, GPU-only ownership, writes during drain, new registrations,
  and concurrent queue producers/consumer.
- Renderer `--bda-dirty-cache-only`: passes with dirty regions both off and on.
- `resource_materialization_tests`: all cases pass with retained allocations.
- Resource memo tests also cover output changes and repeated capacity transitions.
- Renderer `--lds-limit-only` passes with the single-state resource memo enabled.
- `--wave-reduction-only`: SPIR-V validation and GPU results pass for all cases.
- `--flat-d16-only`: all targeted D16/subword GPU cases pass.
- `--block-storage-shadow-only`: both block sizes pass on the GPU.
- `--pipeline-warmup-only`: synthetic shaders with distinct workgroup sizes
  create two pipelines; repeated saves preserve the handle and grow the cache.
- Two runs of Claude's `c6a38e00` binary using the fork's recommended flags
  crashed in `amdvlk64.dll` during the early graphics workload, after logging
  two reductions for pixel shader `1da1fd2871174330`. One used a cold pipeline
  cache and one a copy of the existing cache.
- Holding the remaining settings constant, `KYTY_PARTIAL_WAVE_REDUCTION=0`
  reached the main menu at approximately 20-22 FPS after startup compilation.
  This is an isolation result, not a confirmed shader root cause or gameplay
  improvement. The first transition into team/player rendering causes substantial
  compilation stalls. That run subsequently ended in `ErrorDeviceLost` at
  scheduler tick 9374, frame approximately 7856, before a valid sustained match
  measurement. Later title samples ranged from 0 to 9 FPS while loading/intro
  work continued; these are not a gameplay performance result.
- The candidate with corrected reductions reached a CPU-versus-CPU CAR vs CHI
  match with white ice and advancing players/clock, and completed a period.
  Observed gameplay title samples were generally 6-8 FPS. New shader compilation
  and profiling were enabled; this is not a warmed, sustained benchmark or a
  60 FPS result. The final capture is an intermission screen, so later samples
  must not be included as gameplay. Per-draw GPU timestamp intervals overlap;
  their sum is not a measure of GPU busy time.
- The user permits lower internal resolution at 1080p or above. A guest display
  override experiment registered 1920x1080 but twice faulted at guest PC
  `0x9079189c6`, with shader replay both enabled and disabled. The same build
  reached the menu at 2160p. Read-only guest inspection identifies the selected
  presets as `PS5_BASE` versus `PS5_4K`, with the base configuration lookup
  returning null. NHL's ATTRIBUTE3 is 138412032 (resolution-detection bit 4
  is clear). The unvalidated display override was removed from the working
  source; the isolated experiment patch and logs remain outside Git.
- A separate 2160p control using the compressed-storage and warm-up repairs,
  with all profiling disabled, advanced from frame 9999 to 10520 in 80.6875
  seconds: **6.457 FPS**. The clock advanced from 17:24 to 7:02 with white ice
  and moving players. Process CPU usage averaged 2.634 cores and eight Windows
  3D-engine samples averaged 49.10%. Some shaders were still compiling; this
  is not a fully warmed result. The 1280x720 host window does not establish
  the game's internal render resolution. The 60 FPS objective remains unmet.

User-provided reference: RX 6800 XT / Ryzen 5 5600 / 32 GB DDR4-3200 reports
14-19 FPS in the main menu, 26-29 in franchise menus (30 FPS cap), 7-12 in team
select/versus and 4-6 in a match. The objective here is measured stable **60 FPS
in a match** on the local PC. Hardware differences preclude a direct numerical
comparison without a matching scene and settings.

## Local tools

`tools/codex/build.cmd` builds only in `_Build/codex`, using VS 2022 Build Tools
and the portable LLVM/glslang bundle. `KYTY_TOOLS` overrides its default location
under the user's Downloads directory. It accepts optional target names.
`tools/codex/start-test.ps1` records exact executable SHA256, PID, arguments,
environment and log paths in the isolated run's `test.json`, restores its
calling process environment, and refuses to launch alongside another emulator.
`-Game` overrides the local game path; the default points to the supplied backup.
Profiling switches (`-Stats`, `-GpuTiming`, `-ResourceTiming`) default off.
`capture-test.ps1` captures only a specified emulator process window.
On this machine desktop processes/windows require execution outside the sandbox;
a sandboxed process check alone does not establish that the desktop is idle.

For a valid performance comparison, use the same CPU-versus-CPU CAR vs CHI
match, warmed caches, identical settings, no competing emulator or compiler,
and a sustained gameplay interval with visual confirmation that the clock and
players advance. Record FPS, frame counts, CPU/GPU timings and rendering defects.
Never treat shader skipping, menu FPS or a stopped gameplay clock as success.
