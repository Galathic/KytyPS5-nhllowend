# Compute resolve-ahead — 2026-10-14

`KYTY_COMPUTE_RESOLVE_AHEAD` is runtime flag **bit 35**, default **off**. With the
resolve GPU stage enabled, existing compute programs and resource snapshots are
materialized on the resolve thread. Execution validates the recorded guest reads and
runtime flag bits, then uses the resolution or falls back to inline materialization.
Binding setup and GPU submission remain ordered on the execution path.

Cold programs, missing permutations, SWAPPC programs and failed reads fall back.
Ordinary indirect dispatches resolve only programs/resources, not their argument
memory. Indirect thread-dimension dispatches stay entirely on execution because their
dimensions are read from guest memory at that point. A 320-slot ring retains each
snapshot until its operation finishes executing.

Bits 33 and 34 retain the pre-existing WRITE_COVER_SKIP and AGED_WRITEBACK registry
declarations to preserve bit numbering. Their uncommitted readback implementations
are **not included** in this change.

## Validation

Release clang-cl builds passed for `kyty_emulator` and
`shader_recompiler_compute_tests`. Targeted tests passed:

- `--compute-resolve-ahead-only`: cold fallback, warmed snapshot identity, accepted
  resolution and actual GPU output, changed descriptor rejection and updated output
  address, ordinary indirect resolution, late thread-dimension and empty skips.
- `--thread-dimensions-only`: 32/64-lane group and thread dimension execution.
- `--indirect-write-tables-only`: existing indirect resource-table behavior.

`KYTY_PERF_STATS=1 KYTY_PERF_STATS_VERBOSE=1` enables `NHL27COMPUTEAHEAD`:
attempts/ready/hits/invalid per frame, parallel resolve ms/frame and execution
validation ms/frame. Validation is included in `NHL27DISPATCHPREP materialize`.

## User free-skate measurement

RX 6800 XT / Ryzen 5 5600; GTA buildgta at **00f8dbbc-dirty**. The tested binary
also included pre-existing uncommitted readback experiments; these numbers are for
the same-process flag difference in that tree, not a benchmark of an isolated clean
commit. User confirmed free skate and operated navigation and flag switches.
Full guards including loop cap 1024; GPU timestamps off, verbose CPU stats on.

Two repeated ~20-second off/on pairs, after 3-second settles, had no compilation:

| Metric | Off | On |
|---|---:|---:|
| FPS from mean frame time | 30.424 | 34.460 |
| Frame time | 32.8691 ms | 29.0189 ms |
| GPU-thread busy | 24.0492 ms | 20.7342 ms |
| Execution compute preparation | 10.5752 ms | 5.8180 ms |
| Validation, included above | 0 | 5.0979 ms |
| Parallel resolve time | 0 | 3.2600 ms |
| Draws/frame | 717.1398 | 718.4380 |
| Dispatches/frame | 357.7026 | 358.9923 |

Measured **+13.27% FPS**, **-3.85 ms/frame**, **-3.32 ms GPU-thread busy**.
Pair frame-time deltas: -3.59 and -4.10 ms; draw/dispatch counts within 0.4%.
Hit rate: 91.56% of ready resolutions. 1,222 off and 1,383 on frames analyzed.
An initial pair also improves (-10.99 ms), but its off baseline is unusually slow;
the final off period drifts slower. Prefer the repeated-pair estimate rather than
the inflated three-pair aggregate. Logged settled flag-on interval peak: 38 FPS.
No device-lost/file-error tags in the snapshot; logs alone do not prove rendering
correctness. No new rendering issue was reported in this exchange.

Local logs, selected windows and reproducible analysis are retained under
`F:\emuf\kyty-play-nhl27\compute-ahead-2026-10-14\`; see `results.md` and
`measured-run.log`. Workspace handoff and FPS records also contain the result.
Default stays off pending match A/B. Next CPU candidate: reduce the 5.10 ms
GuestReadLog validation cost without weakening stale-descriptor safety.
