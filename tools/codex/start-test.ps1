param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string]$RunDirectory,
    [switch]$Stats,
    [switch]$GpuTiming,
    [switch]$ResourceTiming,
    [switch]$DirtyRegions,
    [switch]$MultiResources,
    [switch]$BatchReads,
    [switch]$GpuScalarData,
    [switch]$PreGen,
    [string]$Game = (Join-Path $env:USERPROFILE 'Downloads\NHL27\NHL 27\PPSA34063-app0\eboot.bin'),
    [hashtable]$Overrides = @{}
)
$ErrorActionPreference = 'Stop'
$other = @(Get-Process kyty_emulator -ErrorAction SilentlyContinue)
if ($other.Count) { throw "Another emulator is running (PID $($other.Id -join ', ')); test was not launched." }
$run = [System.IO.Path]::GetFullPath($RunDirectory)
$game = [System.IO.Path]::GetFullPath($Game)
if (-not (Test-Path -LiteralPath $game -PathType Leaf)) { throw "Game executable not found: $game" }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Emulator executable not found: $Executable" }
New-Item -ItemType Directory -Force -Path $run | Out-Null
$settings = @{
    KYTY_DBG_LOOP_CAP='1024'; KYTY_DBG_SKIP_BAD_SHADERS='1'; KYTY_PIPELINE_CACHE_DIRTY='1';
    KYTY_EOP_DIRECT_LABELS='1'; KYTY_NATIVE_INDIRECT='1'; KYTY_READONLY_BUFFERS='1';
    KYTY_CLEAR_REGISTER_WIDE='1'; KYTY_PARTIAL_WAVE_REDUCTION='1'; KYTY_DPP_PRESERVE_INACTIVE='1';
    KYTY_WORKGROUP_BUFFER_SYNC='1'; KYTY_BDA_DIRTY_CACHE='1'; KYTY_PROGRAM_RESOURCE_CACHE='1';
    KYTY_PERF_NO_SYNC_READBACK='1'; KYTY_SHADER_PREGEN=[string][int]$PreGen.IsPresent; KYTY_PIXEL_QUAD_CORRECT='0';
    KYTY_RESOURCE_MULTI_CACHE=[string][int]$MultiResources.IsPresent;
    KYTY_SRT_BATCH_READS=[string][int]$BatchReads.IsPresent;
    KYTY_GPU_SCALAR_DATA=[string][int]$GpuScalarData.IsPresent;
    KYTY_BDA_DIRTY_REGIONS=[string][int]$DirtyRegions.IsPresent;
    KYTY_PERF_STATS=[string][int]$Stats.IsPresent; KYTY_PERF_GPU_TIMING=[string][int]$GpuTiming.IsPresent;
    KYTY_PERF_RESOURCE_TIMING=[string][int]$ResourceTiming.IsPresent;
    KYTY_INPUT_FILE=(Join-Path $run '_input.txt')
}
$saved = @{}
foreach ($name in $Overrides.Keys) { $settings[$name] = [string]$Overrides[$name] }
foreach ($name in $settings.Keys) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
}
try {
    $arguments = @('--game', ('"' + $game + '"'), '--redzone', '--screen-width', '1280',
        '--screen-height', '720', '--pre-gen', $PreGen.IsPresent.ToString().ToLowerInvariant(), '--printf-direction', 'Silent')
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $run "console-$stamp.txt"
    $stderr = Join-Path $run "console-$stamp.err.txt"
    $process = Start-Process -FilePath $Executable -ArgumentList $arguments -WorkingDirectory $run `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $metadata = [ordered]@{
        pid=$process.Id; executable=[System.IO.Path]::GetFullPath($Executable);
        executableSha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash;
        startedUtc=[DateTime]::UtcNow.ToString('o'); arguments=$arguments; environment=$settings;
        stdout=$stdout; stderr=$stderr
    }
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run 'test.json')
    $metadata | ConvertTo-Json -Depth 4
} finally {
    foreach ($name in $settings.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
