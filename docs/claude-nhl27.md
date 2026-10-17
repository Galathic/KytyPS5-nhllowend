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

## Threaded GPU pipeline (merge of pengui07/nhl27-gta-eadb, 2026-10-14)

pengui07's `nhl27-gta-eadb` (upstream main, TheCruZ's GTA V work and its NHL 27 port) runs the
emulated GPU as a pipeline of threads: command processor, resolve (program selection and SRT
materialization ahead of execution), execution (owns the renderer and caches) and recording
(replays the recorded commands into Vulkan command buffers and submits them).
`KYTY_GPU_PIPELINE=0` (or `--gpu-pipeline 0`) runs everything inline; stage threads are pinned
(`KYTY_GPU_THREAD_CPUS`, default 4,6,8) and run at high priority (`KYTY_GPU_THREAD_PRIORITY=0`
keeps normal priority).

Measured on this merge (cross-boot, 120 s of a match then 60 s paused, play-nhl defaults):

| Build | Match fps | Paused fps |
|---|---|---|
| Merge, pipeline on (PGO profile v4) | 18.0 / 22.9 / 18.4 | 20.7 / 22.7 / 19.5 |
| Merge, pipeline on (stale PGO), 6 boots | 17-22, mean 20.0 | 20.6-26.3, mean 23.6 |
| Merge, `KYTY_GPU_PIPELINE=0` | 14.7 / 14.2 | 16.0 / 18.6 |
| pengui07/nhl27-gta-eadb alone (no PGO) | 15.0 | 15.6 |
| claude/nhl27-perf before the merge, 8 boots the same day | 19-24, mean 20.9 | 19-28, mean 23.2 |

So the pipeline is worth about +40% on the merged code, but the merged code inline is slower than
the pre-merge branch was, and the two cancel out in a match: the merge is about level with the
pre-merge build (the game also switches between 60 and 30 Hz flips on its own, which quantizes the
frame rate). Per-boot spread is several fps; these are not in-match A/Bs.

Changes made in the merge:
- `KYTY_READBACK_QUEUE` with the pipeline lost the device in 3 of 5 boots: a second-queue readback
  waited for a timeline tick the recording thread had not submitted yet. It now waits until the
  tick is submitted (`CommandScheduler::WaitSubmitted`); 3 of 3 boots clean, and a failed
  second-queue submission falls back to the ordinary readback.
- Retired experiment bits: 8 (`KYTY_GPU_THREAD_PRIORITY` as an experiment: the stage threads set
  their own priority, and the env key now belongs to the pipeline), 9 (`KYTY_IMAGE_RECYCLE`:
  image recycling is TheCruZ's, unconditional, 768 MiB kept for NHL 27's aliased targets) and 21
  (`KYTY_ASYNC_SUBMIT` has no consumer: the recording thread submits).
- Bit 0 is now `KYTY_EARLY_SUBMIT_WRITERS` (default off): a draw or dispatch that writes bytes a
  strict resource read had to download ends the host command buffer, so the next such read waits
  only for that work on the second queue. In-match A/B, 4 rounds: 27.0 fps off, 20.8 fps on
  (every round slower), so it stays off: the extra submissions cost more than the shorter waits.
- A dispatch whose arguments the CPU wrote gets them from the stream buffer, which has no device
  address; the dispatch-args guard (and the GPU color-clear check) skip it instead of failing
  (`streamBuffer.cpp:121`, once during a match load).
- The first boot after shader-emitter changes recompiles every shader (minutes at 0 fps while
  loading); later boots use the pipeline cache.

### Readbacks on a compute queue (bit 2, `KYTY_READBACK_COMPUTE`, default on)

`KYTY_DBG_CP_READS=1` prints what the synchronous downloads wait for (`NHL27RBWAIT`, every 2 s:
who read, whether the bytes read were GPU-written themselves, how many submissions back the
buffer's last write is, and whether it had finished) and which command-processor reads of guest
memory are slow (`NHL27CPREAD`: indirect register loads and `WAIT_REG_MEM` values, by time).

In a match, about 300 to 4000 downloads per 2 s copy data whose writer had already finished, yet
each took 0.2 to 0.8 ms (up to a third of the execution thread): the second queue of the graphics
family runs on the same engine as the frame's work and waits behind it. The device also has a
compute-only family (async compute, 8 queues on the RX 9070 XT) and a transfer-only family
(SDMA). Every buffer is now shared with them (concurrent sharing; `KYTY_READBACK_FAMILIES=0`
leaves the device as before), and bit 2 copies on the compute family, bit 8
(`KYTY_READBACK_TRANSFER`) on the transfer family. In-match A/B, 6 rounds of 20 s:

| Readback queue | fps | vs graphics |
|---|---|---|
| Graphics family (before) | 25.3 | |
| Compute family | 28.7 | +13.8% (paired +3.5 +- 1.4, every round faster) |
| Transfer family | 24.7 | -2% (+-7.5: copies of finished data take 0.1 ms, but some phases are much slower) |

With the compute queue, the remaining download waits of the execution thread in play (per 2 s):
- 20 to 40 faults on bytes that only share a page with GPU-written ones, each waiting about 6 ms
  for the previous submission (130 to 230 ms): the plain loads of direct resource-table reads
  (`SrtWalker::DirectReadRun`) and of read-ahead checks (`GuestReadLog::StillValid`). Bit 5
  (`KYTY_FAULT_PAGE_DIRECT`, default on) reads such bytes from the backing once their page has
  faulted on this thread (as `KYTY_FAULT_PAGE_READS` already did for ordinary resource reads);
  with it these waits are gone from `NHL27RBWAIT`.
- About 60 reads of bytes the GPU wrote in the command buffer being recorded (strict resource
  reads): 50 to 300 ms, a real dependency.
- Bit 9 (`KYTY_READBACK_SPIN`, fence polling) stays off: within noise.

The game asks for its flip rate every frame (`VideoOutSetFlipRate`; `KYTY_DBG_QUEUE_TIME=1`
prints changes as `NHL27FLIPRATE`): 60 Hz during play (stretches of 60 to 70 s in a match), 30 Hz
in menus, the pause menu and some stoppages. A paused match is therefore capped at 30 fps and is
no longer a useful A/B scene; in play the execution thread is the limit (`NHL27QUEUES`: q0 busy
990 ms/s, about 110,000 operations/s at 9 us each).

The command processor waits about 23% of its time in `WAIT_REG_MEM`: NHL 27 writes the GPU clock
at end of pipe (`RELEASE_MEM`, 64-bit clock source) and waits for it to be nonzero (`NHL27CPREAD:
wait-eop-clock`, 5 to 35 ms each). Bit 13 (`KYTY_CLOCK_WAIT_AHEAD`, default off) lets such a wait
pass on the command processor when the pending clock write is the same queue's and the test holds
for any later clock value. In-match A/B (8 rounds): -8.4% +- 4.8, within noise, so it stays off; in its phases the game kept running normally.

Render resolution, cross-boot benches of 120 s in a match (`nhl27_scale_0p5.json` sets the
game's ResolutionScale to 0.5; the game registers 3840x2160 display buffers):

| Run | Native | Scale 0.5 |
|---|---|---|
| Pair 1 | 29.1 fps, GPU 3D 72-74% | 29.7 fps |
| Pair 2 | 29.6 fps, GPU 3D 72-74% | 33.8 fps, GPU 3D 52-53% |

So pixels are not what holds the frame rate: the GPU stays busy at half resolution and the gain is
small and uneven. `KYTY_OUTPUT_1080P=1` reports a 1080p output to the game (not measured yet).

The resolve and execution threads share one program lock, held for whole resource
materializations (resolve waited on it 14-17% of its time, execution 4-6%). Bit 14
(`KYTY_COMPUTE_UNLOCKED`) releases it while the execution thread materializes a compute program,
bit 16 (`KYTY_AHEAD_UNLOCKED`) while the resolve thread materializes ahead (a per-entry busy flag
keeps the other thread off that program), and bit 17 (`KYTY_BACKING_READ_MRU`) makes backing
reads remember the thread's last two mappings instead of one. Bit 19 (`KYTY_DISPATCH_AHEAD`)
resolves direct dispatches on the resolve thread as draws are. All default off: in-match A/Bs,
-5% for the dispatch read-ahead (6 rounds, +-7.0) and +0.3% for 14+16+17 together (10 rounds,
+-5.4), within the scene-to-scene noise.

Bit 20 (`KYTY_BACKING_LOOKUP_CACHE`, default off): backing reads of another mapping than the
thread's last one look the mapping up in a per-thread cache of 64 slots (one per 64 KiB block)
before the locked map search; it hits 99.7% of the time. In-match A/B (8 rounds): +6.7% +- 4.3, within noise.

Unconditional execution-thread trims (profile-checked, each 1-2% of the thread before):
- The dispatch-args guard reads its trip counter (in video memory the CPU reads uncached) every
  64 indirect dispatches instead of every one.
- Buffer range clamps remember the last 16 committed-run answers per thread while no page
  changed (a 256 MiB range is a thousand bitmap words).
- The CPU fast-clear check compares color metadata eight bytes at a time.

Bit 15 (`KYTY_WIDE_FAULT_WINDOWS`, default on, was the unused `KYTY_ADAPTIVE_RESOURCE_MEMO`): a
CPU write fault releases `KYTY_FAULT_COALESCE_PAGES` pages (16); off, 4 pages. In play the game
takes about 200 write faults per frame and the 16-page windows upload about 11 MB per frame,
mostly unwritten neighbors. In-match A/B (10 rounds), 4-page windows: +2.1% +- 3.4, and in the
steadiest stretch (rounds 5 to 10) 0.6 to 1 fps slower every round, with write faults 14,500 to
33,000 per 2 s. So the wide windows stay.

In-match fps varies by 5 to 7 fps between adjacent 20 s phases (scene changes), so A/Bs of 6 to 10
rounds only resolve effects of about 10% or more.

### Native min/max (bit 11, `KYTY_NATIVE_MINMAX`, default on)

`V_MIN/V_MAX/V_MIN3/V_MAX3/V_MED3_F32` (and the f32 math of the f16 forms) were emulated exactly:
NaN and signed-zero classification of both operands and four selects, about ten host instructions
each, and the bulk of the `v_cndmask` in NHL 27's heavy shaders. With bit 11 they are GLSL
`NMin`/`NMax`, which an AMD host compiles to the same `v_min`/`v_max` the guest ran (differences
only for signalling NaNs and the sign of equal zeros). The option is folded into the NativeFp16
stage-key word, so keys stay as before while it is off; the first boot with it on translates every
shader again. Bit 11 was the unused `KYTY_TEXTURE_DESC_CACHE`, so the play-nhl default mask is
still 1977916646 (with bit 11 off: 1977914598).

Measured in one boot, paused match, GPU timeline captures A (off), B (on), A again (drift +0.2%):
GPU busy per frame 20.9 -> 18.8 ms (-10%), the top 30 draws/dispatches -13% (CS a36f0b4f -30%,
db7e537e -56%, 4c9b2981 -25%, 9380d724 -13%, PS 9eaaf80c -12%, 136eed67 -7%), 34-38 -> 40 fps.
In-match screenshots look the same.

### Robust buffer loads (bit 8, `KYTY_ROBUST_LOADS`, default on)

Every raw storage-buffer load was wrapped in a branch on EXEC and, per dword, a branch on the
bounds (a `buffer_load_dwordx4` became five branches), so the host compiler could neither batch
loads nor hoist them. The device has robustBufferAccess2 (out-of-range reads return zero, as the
guest's do, against the same bound range the explicit check used), so with bit 8 those loads are
plain loads in every lane. Formatted loads keep their checks (their out-of-range value need not be
zero). Shaders that spill guest VGPRs (LDS lane slots or scratch) keep the branches too: the one
such heavy pixel shader (33037c60) got 16-27% slower without them, as more values stayed live.
`KYTY_ROBUST_LOADS_KEEP_EXEC=1` keeps the EXEC branch and drops only the bounds branches.

A/B/A in a paused match (drift -0.2%): top 30 commands -5.0%, GPU busy per frame 16.9 -> 16.1
ms; PS 136eed67 -13%, 9eaaf80c -10%, 90706b3b -18%, 2b39f281 -18%, f4e33d7e -20%, 33037c60 level;
in-match screenshots unchanged. Bit 8 was `KYTY_READBACK_TRANSFER`, which is now env-only (=1);
likewise `KYTY_READBACK_SPIN` (was bit 9). Bit 9 is now `KYTY_FUSED_MAD` (default off): V_MAD/V_MAC_F32
as one fused multiply-add measured -1% (A/B/A), within noise, so the separately rounded multiply
and add stay. Default mask with play-nhl: 1977916902.

Tools for shader-codegen A/Bs: `gpu-ab.sh <prefix> <maskA> <maskB>` (`PAUSE=1` pauses first)
captures A, switches the mask (new variants compile), captures B, switches back, captures A2, and
compares per-shader GPU time with `gpu-compare.py`; GPU time per draw is far steadier than fps.

### Where the pipeline threads wait (`KYTY_DBG_STAGE_STATS=1`)

`NHL27STALLS` (every 5 s): execution-thread busy, idle and program-lock ms/s, and the resolve
thread's lock and full-queue waits; `ab-stalls.py <console>` groups them by mask. In a match the
execution thread is busy about 900 ms/s, idle 90 to 130 (the command processor waits for its
results), and 33 to 60 ms/s of the busy time is spinning for the program lock while the resolve
thread materializes; the resolve thread waits 90 to 300 ms/s for the lock and about 220 ms/s for
room in the execution queue. With bits 14 and 16 together both lock waits drop to about 0, but
in-match fps (8 rounds) and the paused fps did not move beyond noise, so they stay off.

### How a frame is paced

The game's submission thread waits on an event queue for the end-of-pipe interrupt that the
execution thread raises right after it runs the frame's flip, then submits all of the next frame's
command buffers (about 55) at once. A frame therefore takes as long as the execution thread needs
for it, plus its waits for the GPU: the strict resource reads of bytes the GPU wrote in the command
buffer being recorded submit it and wait for all queued GPU work (`NHL27RBWAIT: inline via=other
from=gpu exact=yes lag=0`, now also by shader with `KYTY_DBG_CP_READS=1`). Nearly all of those came
from one small compute shader (45066550b711ac4b, about 17 dispatches per frame): its first block
reads a GPU-written counter and an index table and exits early when the entry is -1; the resource
walk evaluated that branch, so it downloaded the counter (up to 0.7 s per 2 s while a match loads,
2-10% of the execution thread in play).

Bit 13 (`KYTY_GPU_CONDITIONS_BOTH_PATHS`, default on; was `KYTY_CLOCK_WAIT_AHEAD`, now env-only):
while the direct resource walk evaluates a branch condition, strict reads decline GPU-written bytes
instead of waiting; the condition stays undecided and the walk takes both successors (the union of
their resources; the GPU still takes the branch itself). Those waits are gone; a six-boot A/B was
within the cross-boot noise (33.8 vs 32.2 fps, single boots range 28-39).

Crowd shots: with the stands in view a frame has about 4,050 draws (2,360 in normal play: 556
extra tiny draws of PS 1fa12ecb into an R16 1024x256 target, 250 more depth draws, 140 more of PS
f5a3c1b7), and drops to about 18 fps with the GPU half idle: the execution thread is the limit.
`KYTY_DBG_DRAW_COST=1` (`NHL27DRAWCOST`, every 5 s) gives the execution-thread cost per draw by
shader pair: the common draws take 2 to 2.5 us; the draw that carries the CPU-write
synchronization (BDA sync of the pages the CPU wrote since the last one) takes about 1.6 ms.

Device losses still happen at match load or late in a match in about 1 of 6 boots (2 of 6 in one
run, 1 of 4 in another). With `KYTY_DBG_BREADCRUMBS=1` none occurred in 6 boots, so the markers
change the timing; it is not understood yet. `KYTY_DBG_DUMP_SHADERS=<hash>` now also writes the
guest disassembly and the IR (`_Shaders/NNNN_ir_<stage>_<hash>.txt`); a WAIT_REG_MEM that keeps
failing prints `NHL27STUCKWAIT` (the menu hangs: the game then calls Coredump).

Menu hangs (about 1 boot in 6 since 2026-10-14, the game then calls Coredump): `NHL27STUCKWAIT`
showed the graphics queue waiting on a 64-bit label for "not zero" with no pending write left: the
command processor had waited for the end-of-pipe clock write before it, the execution thread had
written the clock and raised the interrupt, the game woke and reset the label to 0, and only then
did the command processor read memory. The execution thread now records the clock values it
writes, and a wait that had to wait for such a write takes the value written.

Bits 14, 16 and 19 are on by default since 2026-10-15 (lock-free materialization on both threads
and dispatch read-ahead): the execution thread's program-lock waits (40 to 125 ms/s, the most in
crowd shots) are gone, 98.6% of dispatches are resolved on the resolve thread, and its checks no
longer fault (`StillValid(true)`). Four 120 s boots, no hang or device loss; the execution thread
idles 130 to 170 ms/s now (it waits for the command processor more often).

Device losses are GPU timeouts (Windows logs LiveKernelEvent 141, VIDEO_ENGINE_TIMEOUT_DETECTED,
at the same second; such events go back to 2026-10-12). `KYTY_DBG_LOOP_CAP=4000000` did not
prevent them. The draws and dispatches recorded are kept in a CPU-side ring with their command
buffer's tick; on a device loss `NHL27HANG` prints the ones in ticks the GPU had not finished.

`KYTY_FORCE_FLIP_RATE=0` keeps 60 Hz flips where the game asks for 30 (menus reach 40+ fps) and
`KYTY_VBLANK_HZ` overrides the vblank rate; the paused match still ran at about 30 fps with both,
because that is what it costs (GPU busy 70%, the execution thread 93%).

Earlier A/Bs of bits 13, 14, 16 and 17 that reported stalls or -25 to -39% were wrong: their base
mask (1979710566) had the retired bits 13 to 20 set, so adding a bit carried into bits up to 25
(NativeFp16, ExecSelects, FaultPageReads off: shader recompiles at 0 fps). Masks are now built by
OR-ing bits into the default mask the boot prints (`Experiments: mask <default> -> ...`,
1977916646 with play-nhl defaults).

Measured and dropped on the way: per-range write ticks (a download waits for the latest write of
its bytes instead of its buffer's): fewer than 1% of the downloads would wait for an older
submission, and the A/B was -14% +- 7.5.

Slow scenes (about 20 fps, some replays and crowd shots) were readback-bound: the execution
thread spent up to half of each second waiting for GPU-to-CPU downloads (`KYTY_DBG_CP_READS=1`,
`NHL27RBWAIT`). Two kinds dominated:

* Command-processor reads of pages that hold GPU-written bytes (NHL 27 embeds labels the GPU
  writes in its command buffers, and keeps register tables and WAIT_REG_MEM values next to
  compute-written bytes). The page is read-protected, so the read faulted and its download waited
  for the writer, about 6-10 ms each while it still ran, though the bytes read were nearly always
  CPU-written ones. Bit 4 (`KYTY_CP_BACKING_READS`, default on; was `KYTY_SRT_JIT`, now env-only):
  on a read-protected page the command processor reads the backing instead (indirect SH/CX/UC
  register tables, WAIT_REG_MEM/decision values, and PM4 packets), and the execution thread checks
  each such read in order: if an earlier operation wrote those bytes it downloads them and compares
  (`NHL27CPSPEC`; a difference turns the reads off). Reading through the execution thread
  synchronously instead cost 0.24 ms per read and stopped the CP running ahead (exec idle 28%).
  In 1.58 million checked reads over six boots: 0.5-1.6% touched GPU-written bytes, all with the same value,
  no mismatch.
* Guest stores into GPU-written pages: once the CP no longer downloaded those pages early, game
  memcpy fills (vmovntdq/vmovdqa) over whole GPU-written pages faulted once per page, each a
  readback (~90-120 us; up to 1,100/s). Bit 6 (`KYTY_WIDE_WRITE_DOWNLOADS`, default on; was
  `KYTY_FALSE_SHARE_READS`, now env-only): such a store downloads the GPU-written bytes of its
  64 KiB window at once (the window a CPU write fault releases), down to about 50-70/s.

Three in-boot A/B boots (`abwin.sh`, 12 s phases alternating bits 4+6 on/off): readback waits
86/74/71 ms/s on vs 98/133/277 off, execution thread busy 4-5% lower; with the PM4 fix as well no
CP fault remained (normal play ~15 ms/s of readback waits in total, slow windows ~30). The
`NHL27OWNW` diagnostic lists the GPU-written ranges of a page a guest store hits;
`NHL27UNMAP` counts unmaps that drained the GPU and why; `NHL27COREDUMP` prints the guest thread
and return addresses when the game calls Coredump (its `Heartbeat` watchdog does when a boot
stalls, for example while every shader recompiles).

Upload copies (CPU-written buffer pages into the staging buffer before their GPU copy, 3.3% of the
execution thread) now go to the host copy thread like the stream buffer's; the submission waits for
them and guest-visible operations drain them first. In-boot A/B against `AbLegacy` (bit 31, the old
copies): execution-thread time per draw 5.72 -> 5.46 us, per dispatch 4.45 -> 4.26 us.

A GPU A/B of the stronger EXEC-select rules (`KYTY_DBG_EXECSEL_MODE=31`) on the heaviest compute
shader alone (c556aaf3, 3.1 ms per frame in a paused scene of 26.8 ms GPU per frame) removed 901 of
1,506 selects for -8% of its time: not worth the rules' known breakage. It uses 154 VGPRs (wave64).

Device losses at match load (TDR, Windows LiveKernelEvent 141): three of five boots on 2026-10-16
lost the device at the same point of the load (around tick 500,000, the failing submit showing
args=2,1,1,97,0x1140100a00); none did in five boots with `KYTY_READBACK_QUEUE=0`. The second-queue
readback (on the compute family since 2026-10-14, when the losses started) made that queue wait
for the writer's submission on the GPU. Now the CPU waits for the writer first, so the queue's
wait (kept for its memory dependency) is already satisfied when it is submitted: 6 of 6 boots
without a loss.

Two more execution-thread fixes: draw checks of resolved-ahead draws (`GuestReadLog::StillValid`)
compare read-protected pages in the backing instead of taking an exception per plain load (the
16-page fault cache thrashed; 2.7% of the thread), and a pending asynchronous guest read marks
only the GPU-written bytes it carries dirty, not its whole pages (a draw check that hit such a page
waited for that download: another 2.7%).

Indirect dispatches are resolved ahead on the resolve thread too (their program does not depend
on the group counts; with bit 19 `KYTY_DISPATCH_AHEAD`): execution-thread time per indirect dispatch
24.9 -> 11.5 us in an in-boot A/B (about 2.5% of the thread).

Breadcrumbs (`KYTY_DBG_BREADCRUMBS=1`) caught the remaining device loss in a soak: the GPU was
running an indirect dispatch of CS 9ba58a112a994ad8 with 50,589,920 x 1 x 1 groups (the record at
0x11c020a000 grows over a match; CS 8714aa79 had just run with the same record). The counts are
within the device limits, so the guard let them through and the GPU ran for seconds (TDR). The
guard now also zeroes records whose product exceeds 4M groups (NHL 27's normal indirect dispatches
stay below 1,000; a full-screen 4K pass of 8x8 groups is about 130,000). The record itself is
still wrong (not reset where the game expects it); not found yet.

Scalar constant reads (S_BUFFER_LOAD) under robust loads no longer branch on their bounds per
dword (a branch per constant kept each load in a block of its own; the heaviest compute shader had
189): GPU time -6 to -7% per frame in a paused-scene A/B/A (29.0 -> 27.0 ms; d7cfa519 -17%,
others -5 to -15%). Part of the shader keys (code-gen key bit 32).

With the readback waits gone, the pipeline is limited by two threads at once: the resolve thread
(resource walks of the draws and dispatches ahead) is about 88% busy and the execution thread about
80%; the command processor mostly waits for free register snapshots. `NHL27STALLS` now also prints
the resolve thread's idle and busy time. Bits 17 (`KYTY_BACKING_READ_MRU`) and 20
(`KYTY_BACKING_LOOKUP_CACHE`), whose earlier A/Bs were invalid (the mask carry bug), are on by
default: resolve-thread time per operation -7.7% in an in-boot A/B (soak: 6 of 6 boots, 2 min in
the match). Deeper queues between the stages (resolved queue 256 -> 768, rings 320 -> 960, register
snapshots 512 -> 1536) cut execution-thread idle time 181 -> 156 ms/s but not the operations per
second (in-boot A/B): dropped.

Storage-buffer stores (32-bit, and plain 8/16-bit ones) under robust loads drop their bounds branch
too (the EXEC branch stays; a read-modify-write of a containing dword keeps it, so an out-of-range
compare-exchange cannot spin): GPU time -3.1 / -3.7% per frame in a paused A/B/A (pixel shaders
with buffer stores -5 to -9%). Code-gen key bit 64.

The resolve thread read resource-table words one dword per call (its SRT runtime never set
`batch_reads`, which the execution thread's has on by default): every dword went through five
calls down to a memcpy, and each switch between the two threads on one program rebuilt the
program's read runs. It now reads contiguous words in one call as well: resolve-thread busy time
-32% per operation in an in-boot A/B (2.8 -> 1.9 us; 740 -> 475 ms/s), so the execution thread
is the limit again (about 95% busy in heavy play). Small backing reads (4/8/16/32 bytes) are
copied inline. A resolve-side memo of whole materializations (four per program, validated by
re-reading) hit only 10% of the time and cost more than it saved: dropped.

The per-draw proof that no scalar resource read overlaps a write of the same draw (it exits the
emulator when one does; it never fired in NHL 27) is opt-in now, `KYTY_DBG_CHECK_SCALAR_READS=1`:
about 1% of the execution thread.

The low-energy preset patch (`patches/nhl27_lowenergy.json`: resolution scale 0.5, no FSR3)
changes nothing on the CPU side (same draws per second and thread load).

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
| `KYTY_IMAGE_RECYCLE` | 512 | A deleted device image (idle on the GPU by then) is kept and handed out again by the next image with the same creation parameters instead of allocating; at most 32 images / 768 MiB, oldest out first. NHL 27 aliases about seven full-screen targets in one memory range, so the texture cache frees and recreates all of them every frame; each new VMA block cost a `vkAllocateMemory` that waits inside the driver (about 8% of the GPU thread in gameplay, `KYTY_DBG_IMAGE_RECYCLE=1` prints the hit rate) | ~97% of image creations recycled; in-match gameplay A/B +12% fps (+-5.7, 8 rounds) and -13% GPU-thread time per draw (+-9%); paused match unchanged (-1% +-2.4: no churn there) |
| `KYTY_LDS_LANE_SLOTS` | 1073741824 | A vertex or pixel shader whose every LDS access is LaneId * 4 plus constants (the guest compiler spilling VGPRs into per-lane LDS slots) gets a small per-invocation array indexed by the constants instead of a 32 KiB one. Part of the vertex and pixel shader keys, so it switches at runtime (shaders recompile) | NHL 27's heaviest pixel shader (33037c605e42f4ce): scratch 32784 -> 0 bytes; +2.6% paused match (in-match A/B, 6 rounds) |
| `KYTY_TEXTURE_DESC_CACHE` | 2048 | No consumer in the merged tree (pengui07's draw-prep delta replaces it); accepted for compatibility | was +15.5% |
| `KYTY_ADAPTIVE_RESOURCE_MEMO` | 32768 | No consumer in the merged tree (`KYTY_RESOURCE_MEMO_BYPASS` replaces it); accepted for compatibility | was +2.3% |

Default off (set to `1` to enable): `KYTY_CPU_COPY_FORWARD` (8), `KYTY_FALSE_SHARE_READS` (env only since
2026-10-16; bit 64 is now `KYTY_WIDE_WRITE_DOWNLOADS`),
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
| `KYTY_FALSE_SHARE_READS=1` | 64 (env only since 2026-10-16) | The GPU thread parses PM4 packets (and serves its own faulting loads) from the backing when the bytes are clean but share a page with GPU-written bytes | +0.5% paused match |
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

### In-match A/B of GPU-thread time

With `KYTY_DBG_QUEUE_TIME=1` each NHL27QUEUES line also prints the experiment mask, so an abmulti run (`-Env 'KYTY_DBG_QUEUE_TIME=1'`) can be scored by GPU-thread time per draw/dispatch as well as by fps: `python KytyPS5-tools/ab-items.py <console.txt>` averages us/item per mask, dropping the 5 s window that straddles each switch. Gameplay fps per 20 s phase swings by scene (replays, celebrations); paused matches are steadier but do not exercise everything (the image churn above does not happen there).
`KYTY_DBG_IMAGE_CHURN=<seconds>` prints image inserts and frees starting that long after the first image (`=1`: from the start), to catch the match instead of the menus.

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
