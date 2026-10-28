@echo off
rem KytyPS5-nhl launch profile for weak PCs (NHL 27). See docs/low-end.md.
rem
rem Usage:   kyty-lowend.bat "D:\Games\NHL27\eboot.bin" [more kyty_emulator options]
rem Put this file next to kyty_emulator.exe, or set KYTY_EXE to its full path.

if "%~1"=="" (
  echo Usage: %~nx0 "path\to\eboot.bin" [more kyty_emulator options]
  exit /b 1
)
set "GAME=%~1"
shift

if not defined KYTY_EXE set "KYTY_EXE=%~dp0kyty_emulator.exe"
if not exist "%KYTY_EXE%" (
  echo kyty_emulator.exe not found at "%KYTY_EXE%". Copy this file next to it or set KYTY_EXE.
  exit /b 1
)

rem ---- Stability set: the README of this fork says all of these are needed. Do not remove. ----
set KYTY_DBG_LOOP_CAP=1024
set KYTY_DBG_SKIP_BAD_SHADERS=1
set KYTY_PIPELINE_CACHE_DIRTY=1
set KYTY_EOP_DIRECT_LABELS=1
set KYTY_NATIVE_INDIRECT=1
set KYTY_READONLY_BUFFERS=1
set KYTY_CLEAR_REGISTER_WIDE=1
set KYTY_PARTIAL_WAVE_REDUCTION=1
set KYTY_DPP_PRESERVE_INACTIVE=1
set KYTY_WORKGROUP_BUFFER_SYNC=1
set KYTY_BDA_DIRTY_CACHE=1
set KYTY_PROGRAM_RESOURCE_CACHE=1
set KYTY_PERF_NO_SYNC_READBACK=1

rem ---- Performance switches measured by the fork's authors on NHL 27 (high-end hardware). ----
rem These cost no extra video memory and need no special GPU feature
rem (KYTY_GPU_COLOR_CLEAR is skipped by the emulator on GPUs without VK_EXT_conditional_rendering).
rem Skipped when KYTY_LOWEND_SAFE is set (see kyty-lowend-safe.bat).
if not defined KYTY_LOWEND_SAFE (
  set KYTY_DIRECT_EOP_INTERRUPTS=1
  set KYTY_SPARSE_BUFFER_GC=1
  set KYTY_PM4_WRITE_DATA_FAST=1
  set KYTY_FALSE_SHARE_READS=1
  set KYTY_ASYNC_SUBMIT=1
  set KYTY_LAZY_SHARED_WRITES=1
  set KYTY_FAULT_PAGE_READS=1
  set KYTY_GPU_COLOR_CLEAR=1
  set KYTY_ASYNC_GUEST_READS=1
  set KYTY_BDA_DIRTY_REGIONS=1
)

rem Safe mode: no performance switches, and this fork's own changes are switched back to the
rem upstream behavior (768 MiB image pool, highest GPU thread priority, upstream thread pins).
if defined KYTY_LOWEND_SAFE (
  set KYTY_IMAGE_RECYCLE_MB=768
  set KYTY_GPU_THREAD_PRIORITY=1
  set KYTY_GPU_THREAD_CPUS=4,6,8,10,2,12,14
)

rem ---- Left out on purpose; try them one at a time if you have room to spare. ----
rem More video/system memory (keeps idle buffers 10x longer, +9% in the authors' test):
rem set KYTY_LONG_BUFFER_GC_AGE=1
rem Needs fast FP16 on the GPU; may be slower on old/integrated GPUs:
rem set KYTY_NATIVE_FP16=1
rem Second queue for readbacks, +1.4%; falls back to the normal path if the GPU has no second queue:
rem set KYTY_READBACK_QUEUE=1
rem On a CPU with 4 threads or fewer, 6 GPU threads plus the game's own may not fit.
rem Compare with the pipeline off (an untested guess, not a measured win):
rem set KYTY_GPU_PIPELINE=0
rem Cap on video memory kept for recycled images, in MiB (default: 1/6 of VRAM, 256-768; 0 = off):
rem set KYTY_IMAGE_RECYCLE_MB=256
rem Pin or unpin the GPU threads yourself (default: unpinned below 12 logical CPUs):
rem set KYTY_GPU_THREAD_CPUS=off

"%KYTY_EXE%" --game "%GAME%" --screen-width 1280 --screen-height 720 --redzone --readback-linear-images %1 %2 %3 %4 %5 %6 %7 %8 %9
