# Running on a weak PC

This branch changes how the emulator behaves on small machines and adds a launch profile. Nothing
changes on a machine with 12 or more logical CPUs and 4.5 GiB or more of video memory, except that
the profile sets the switches listed below.

**Expectations.** The upstream numbers (README) are 12-19 fps in a match on a Ryzen 5 5600 and an
RX 6800 XT, and the author says the limit is single-thread CPU speed. A weaker CPU will be slower
than that. The emulator also needs a Vulkan 1.3 driver; a GPU or iGPU without one cannot run it at
all. These changes remove waste on small machines; they do not make the game light.

## What changed in the code

| File | Change |
|---|---|
| `src/common/emulatorConfig.cpp` | The GPU stage threads were pinned to logical CPUs 2, 4, 6, 8, 10 and raised to highest priority, a layout tuned for a 16-thread Ryzen. With fewer than 12 logical CPUs the pins named CPUs that do not exist (Windows ignores them, so the threads float) while five busy highest-priority threads competed with the game's own threads. Below 12 logical CPUs the stages are now unpinned at normal priority. `KYTY_GPU_THREAD_CPUS` and `KYTY_GPU_THREAD_PRIORITY` still override. Windows only: other platforms never pinned or raised priority. |
| `src/graphics/host_gpu/vma.cpp` | The pool of recycled images (up to 768 MiB kept for reuse) is now limited to a sixth of the device-local memory, between 256 and 768 MiB. A 4 GiB GPU gets 682 MiB, 2 GiB gets 341 MiB; 4.5 GiB and up is unchanged. `KYTY_IMAGE_RECYCLE_MB=<n>` sets it directly (`0` turns recycling off). If an allocation still fails, the existing code already frees the pool and retries. |
| `tools/lowend/kyty-lowend.bat`, `.sh` | Launch profile: the stability set the README calls required, plus the performance switches the authors measured on NHL 27 that need no extra memory or special GPU feature. Left out and commented: `KYTY_LONG_BUFFER_GC_AGE` (more memory), `KYTY_NATIVE_FP16` (needs fast FP16), `KYTY_READBACK_QUEUE`. |

The measured gains quoted by the upstream docs were taken on high-end hardware. These changes were
compile-checked but not run against the game, so none of them is measured on a weak PC.

## Things to try on a very small CPU

- 4 logical CPUs or fewer: compare `set KYTY_GPU_PIPELINE=0` (everything on one thread) against the
  default. Upstream measured the pipeline as +40% on a 16-thread CPU; on 4 threads it is a guess.
- Low video memory: `set KYTY_IMAGE_RECYCLE_MB=128` or `0` trades a little speed for headroom.
- The first launch recompiles every shader (minutes at near-0 fps); later launches use the cache.
- `--screen-width/--screen-height` set the window and the emulated video-out size (the profile uses
  1280x720, the default; the README example uses 1920x1080). I did not check whether NHL 27 picks its
  internal render resolution from them, so a lower size is worth trying but is not a promised gain.

## Building without building on the weak PC

The repository has a manual GitHub Actions workflow (`.github/workflows/build.yml`, Windows runner).
Push this branch to your own GitHub fork, open **Actions**, run **Build and Release KytyPS5**, and
download the result. Builds from forks do not replace themselves with official releases.

To build locally instead, follow the README (clang-cl, Ninja, Qt 6). `-DKYTY_PGO=use` needs clang
22.1.8 exactly and gave about +15% upstream. `-DKYTY_THIN_LTO=ON` makes linking slow and memory
hungry, which is a bad trade on a weak build machine.
