# Sports fork integration

Adapted from macery12/KytyPS5-Sports for the NHL fork, 2026-10-10.

## Changes

- `cb5a75b9`: retain up to 16 resource materializations per program instead of only the previous draw's state. Every candidate validates resource-dependent registers, shader base, workgroup counts, memory values and read validity. Non-resource SGPRs still refresh push data. Entries over 4096 recorded words are not retained. Failed materialization invalidates the cache. Existing BDA generation caching remains in use.
- `8775740e` and `33e00566`: record guest shader inputs, raw SRT reads, specialization reads and pipeline descriptions in `_PipelineCache/<title>.shaders`. Replay supplies only recorded memory and skips a program when required reads cannot be reproduced. Vertex, pixel and compute stages are supported; mesh/tessellation recording is excluded. This port serializes startup compilation because our ProgramCache and compiler state are shared. The Sports implementation's parallel worker count is not copied.
- `a7dd7e26`: driver cache identity is GPU vendor/device, driver version and pipeline-cache UUID, rather than emulator revision. Retains our dirty-build opt-in and 30-second save, additionally saving after 128 new pipelines. Loading is capped at 1 GiB. The signature changes once from KytyPC1 to KytyPC2.
- Pixel-quad correctness lead from `23704f12`: use native `OpGroupNonUniformQuadBroadcast` for fragment DPP quad permutations when the GPU supports fragment quad operations. Four constant broadcasts support permutations that differ by destination pixel. Preserve integer bit patterns and inactive-source masking; do not use the Sports float-derivative reconstruction. Compute DPP and other fragment DPP controls retain their existing paths. Shader keys include the new mode.

## Controls

The sole launcher remains `F:\emuf\nhl27.bat`; its always-on section enables:

- `KYTY_RESOURCE_MULTI_CACHE=1` (requires existing `KYTY_PROGRAM_RESOURCE_CACHE=1`); zero restores a single cache entry.
- `KYTY_SHADER_PREGEN=1`; zero disables recording and replay. `--pre-gen false` also disables both.
- `KYTY_PIXEL_QUAD_CORRECT=1`; zero restores the old fragment DPP path. Unsupported GPUs retain the old path automatically.

The first run records previously unseen shaders. Later launches spend time compiling the records before gameplay, reducing compilation hitches during play. This is not a claim of higher sustained FPS. Startup compilation is sequential in this port.

Warm-up record version 3 includes NHL's native indirect shader fields, workgroup/float-mode/LDS fields, front-face encoding, compact colour mask, alpha blending and quad mode. Sports version-2 files are incompatible and start fresh. Complete checksummed records survive a torn trailing append; each append is flushed. Host-dependent quad/subgroup/LDS settings are updated for the current GPU at replay.

## Verification

Focused targets:

- `resource_materialization_tests`: changed memory at the same pointer, strict-read failure, workgroup and base changes, register dependencies, and alternating-object restoration.
- `shader_cfg_tests --sports-only`: validates SPIR-V with the quad mode both enabled and disabled, distinguishes the shader keys, and checks warm-up record recovery and NHL-specific metadata round trips.

These tests do not establish a live FPS improvement or that the pixel-quad change fixes the player-lighting defect. Use the same match scene for performance comparison, and a versus-screen reference for visuals.

Original sources: https://github.com/macery12/KytyPS5-Sports
