#!/usr/bin/env bash
# Test profile (Linux): ONLY the required stability set. See kyty-lowend-safe.bat.
# Usage: kyty-lowend-safe.sh /path/to/eboot.bin [more kyty_emulator options]
export KYTY_LOWEND_SAFE=1
exec "$(dirname "$0")/kyty-lowend.sh" "$@"
