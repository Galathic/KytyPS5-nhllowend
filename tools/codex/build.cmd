@echo off
setlocal
set "KYTY_SRC=%~dp0..\.."
if not defined KYTY_TOOLS set "KYTY_TOOLS=%USERPROFILE%\Downloads\KytyPS5-tools"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=%KYTY_TOOLS%\LLVM\bin;%KYTY_TOOLS%\glslang\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
cd /d "%KYTY_SRC%" || exit /b 1
if not exist _Build\codex\build.ninja (
  cmake -S . -B _Build/codex -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DKYTY_BUILD_LAUNCHER=OFF -DKYTY_BUILD_ORIGIN=Fork -DKYTY_BUILD_REPOSITORY=petrmolek/KytyPS5-nhl -DKYTY_GLSLANG_VALIDATOR="%KYTY_TOOLS%/glslang/bin/glslang.exe" -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_C_COMPILER_LAUNCHER= || exit /b 1
)
if "%~1"=="" (
  cmake --build _Build/codex --target kyty_emulator -- -j 6 || exit /b 1
) else (
  cmake --build _Build/codex --target %* -- -j 6 || exit /b 1
)
