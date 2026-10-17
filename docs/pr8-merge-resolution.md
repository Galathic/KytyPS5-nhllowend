# PR 8 merge resolution — 2026-10-17

This merges `nhl27-eadb` at `d88c5920` into the PR head `dba63f1b`.
The conflict resolution retains both branches' rendering switches, memory
diagnostics and compute preparation interfaces.

## Experiment masks

The branches assigned different features to experiment bits 23–26, 29 and 30.
PR 8's existing bits 0–31 retain their meanings. The incoming switches move to
the following **experiment** bits (independent of `runtimeFlags.h` numbering):

| Environment switch | Experiment bit | Default |
|---|---:|---|
| `KYTY_PROTECT_SHARED_LOCK` | 32 | off |
| `KYTY_CLOCK_WAIT_PREDICT` | 33 | off |
| `KYTY_FALSE_SHARE_ANY_THREAD` | 34 | off |
| `KYTY_CP_CLEAN_REG_READS` | 35 | off |
| `KYTY_CP_SKIP_IDLE_SPIN` | 36 | off |
| `KYTY_SPECULATE_DIRTY_READS` | 37 | off |

The experiment mask, decimal file parser and diagnostics now use 64 bits;
initialization reserves bit 63. Existing environment variable names keep their
behavior. Old decimal masks from the base branch must be translated to this
mapping before use. `KYTY_FALSE_SHARE_READS` and `KYTY_SRT_JIT` remain env-only
as in PR 8, because their old experiment bits were already reassigned.

## Compute preparation and memory tracking

`KYTY_DISPATCH_AHEAD` and runtime flag `KYTY_COMPUTE_RESOLVE_AHEAD` enable the
same resolve-ahead implementation. To disable it for an A/B baseline, turn both
off. Snapshots retain their runtime configuration and guest-read validation;
changed configuration or descriptors fall back to execution. Indirect argument
memory remains on the execution path, and indirect thread-dimension dispatches
are not resolved ahead. Each snapshot is released once after its operation.

Both the existing `GetComputeProgramAhead` interface and the base branch's
optional `GetComputeProgram` snapshot argument are supported. Both branches'
shader code-generation switches contribute to shader keys.

The base branch's GPU-fill mutex now protects the PR's chunk filter as well as
the fill list. Shared dirty-byte queries conservatively refuse clean-backing reads while any
asynchronous guest download is pending. An atomic count is published before
downloads remove ordinary dirty ranges and released after backing copies finish. The program-store
lock is initialized before stored programs are loaded, preserving the boot fix.

Execution tracing and stage timing wrap the same operation execution. Pending
guest writes retain their diagnostic kind, guest queue and clock/DMA origin.
Both 1080p output switches retain their respective semantics.

## Validation

The isolated build and test evidence is retained in
`F:\emuf\kyty-play-nhl27\pr8-merge-*.log`.
The new `experiments_tests` exercises independent environment switches,
decimal masks above bit 31, live mask changes and restoration of defaults when
the control file is removed. Compute-resolution, thread-dimension and indirect
resource-table tests cover the overlapping dispatch implementations.

Run the compute resolve-ahead test through CTest, or set `KYTY_PERF_STATS=1`
before starting its executable. The perf branch initializes statistics gates
before `main()`, so setting that variable in the test body is too late for
the counter assertions. CTest supplies the startup environment explicitly.

Release emulator and test builds passed, as did `experiments_tests`,
`gpu_pipeline_tests`, and the compute resolve-ahead, thread-dimension,
indirect-write-table and buffer-cache garbage-collection cases. The broader
`--buffer-cache-range-only` case stopped when the device rejected its two-sample
depth/stencil fixture (`format=129 usage=0x23 samples=2`). The image capability
check and that fixture are unchanged by this merge. Its failed log is retained;
this broader case is not claimed to pass.

Builds and offline tests do not establish an in-game performance gain.
