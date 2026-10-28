#!/usr/bin/env bash
# KytyPS5-nhl launch profile for weak PCs (NHL 27, Linux). See docs/low-end.md.
#
# Usage: kyty-lowend.sh /path/to/game/eboot.bin [more kyty_emulator options]
# Put this file next to kyty_emulator, or set KYTY_EXE to its full path.
# (--redzone exists only in Windows builds, so it is left out here.)

set -eu

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 /path/to/eboot.bin [more kyty_emulator options]" >&2
	exit 1
fi
GAME="$1"
shift

KYTY_EXE="${KYTY_EXE:-$(dirname "$0")/kyty_emulator}"
if [ ! -x "$KYTY_EXE" ]; then
	echo "kyty_emulator not found at $KYTY_EXE. Copy this file next to it or set KYTY_EXE." >&2
	exit 1
fi

# Stability set: the README of this fork says all of these are needed. Do not remove.
export KYTY_DBG_LOOP_CAP=1024
export KYTY_DBG_SKIP_BAD_SHADERS=1
export KYTY_PIPELINE_CACHE_DIRTY=1
export KYTY_EOP_DIRECT_LABELS=1
export KYTY_NATIVE_INDIRECT=1
export KYTY_READONLY_BUFFERS=1
export KYTY_CLEAR_REGISTER_WIDE=1
export KYTY_PARTIAL_WAVE_REDUCTION=1
export KYTY_DPP_PRESERVE_INACTIVE=1
export KYTY_WORKGROUP_BUFFER_SYNC=1
export KYTY_BDA_DIRTY_CACHE=1
export KYTY_PROGRAM_RESOURCE_CACHE=1
export KYTY_PERF_NO_SYNC_READBACK=1

# Performance switches measured by the fork's authors on NHL 27 (high-end hardware); no extra
# video memory, no special GPU feature needed.
# Skipped when KYTY_LOWEND_SAFE is set (kyty-lowend-safe.sh): then this fork's own memory change is
# also switched back to the upstream 768 MiB image pool.
if [ -z "${KYTY_LOWEND_SAFE:-}" ]; then
	export KYTY_DIRECT_EOP_INTERRUPTS=1
	export KYTY_SPARSE_BUFFER_GC=1
	export KYTY_PM4_WRITE_DATA_FAST=1
	export KYTY_FALSE_SHARE_READS=1
	export KYTY_ASYNC_SUBMIT=1
	export KYTY_LAZY_SHARED_WRITES=1
	export KYTY_FAULT_PAGE_READS=1
	export KYTY_GPU_COLOR_CLEAR=1
	export KYTY_ASYNC_GUEST_READS=1
	export KYTY_BDA_DIRTY_REGIONS=1
else
	export KYTY_IMAGE_RECYCLE_MB=768
fi

# Left out on purpose; try one at a time: KYTY_LONG_BUFFER_GC_AGE=1 (more memory),
# KYTY_NATIVE_FP16=1 (needs fast FP16), KYTY_READBACK_QUEUE=1, KYTY_GPU_PIPELINE=0 (untested
# guess for CPUs with 4 threads or fewer), KYTY_IMAGE_RECYCLE_MB=256, KYTY_GPU_THREAD_CPUS=off.

exec "$KYTY_EXE" --game "$GAME" --screen-width 1280 --screen-height 720 \
	--readback-linear-images "$@"
