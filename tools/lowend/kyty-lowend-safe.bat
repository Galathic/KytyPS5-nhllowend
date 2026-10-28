@echo off
rem Test profile: ONLY the required stability set, with this fork's own changes switched back to
rem the upstream behavior. If the game runs here but freezes with kyty-lowend.bat, a performance
rem switch or one of this fork's changes is the cause. If it freezes here too, it is not those.
rem
rem Usage:   kyty-lowend-safe.bat "D:\Games\NHL27\eboot.bin" [more kyty_emulator options]
set KYTY_LOWEND_SAFE=1
call "%~dp0kyty-lowend.bat" %*
