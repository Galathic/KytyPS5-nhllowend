# NHL 27 gameplay profile for PGO

`nhl27.profdata` is an LLVM IR profile of the emulator playing an NHL 27 (PPSA34063) match:
about 5 minutes of Play Now gameplay, recorded after the boot and menus (counters reset once the
match started), with the NHL 27 recommended `KYTY_*` switches.

Building with it (clang-cl only; the profile matches clang 22.1.8, the version that recorded it):

```
cmake -S . -B _Build/windows -DKYTY_PGO=use
```

`KYTY_PGO_PROFILE` defaults to this file. Another clang version ignores the profile with a
warning (`KYTY_PGO_PROFILE_COMPILER` names the expected version). Functions changed since the
profile was recorded simply get no profile data, so it degrades gradually; record a new one
after larger changes.

Measured on an RX 9070 XT + Ryzen 7 5800X3D: 18.6 vs 16.1 fps in gameplay (+15%, three boots
of each build, same switches).

## Recording a new profile

1. Configure a separate build directory with `-DKYTY_PGO=generate` and build `kyty_emulator`.
   The instrumented emulator writes `LLVM_PROFILE_FILE` every 30 seconds.
2. Run the game with `LLVM_PROFILE_FILE=<path>\nhl27.profraw`, get into a match, then create an
   empty file named `pgo.reset` in the emulator's working directory: the emulator zeroes its
   counters (prints `NHL27PGO: counters reset`), so the profile describes gameplay only.
3. Play for a few minutes and close the emulator.
4. `llvm-profdata merge -o tools/pgo/nhl27.profdata nhl27.profraw` (same LLVM version as the
   compiler), then rebuild with `-DKYTY_PGO=use`.

Do not enable `KYTY_PERF_STATS` while recording: its timers run millions of times per second
and would make their own code look hot.
