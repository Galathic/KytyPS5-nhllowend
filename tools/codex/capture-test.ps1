param([Parameter(Mandatory)][int]$ProcessId, [Parameter(Mandatory)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$process = Get-Process -Id $ProcessId
if ($process.ProcessName -ne 'kyty_emulator') { throw 'The specified process is not the emulator.' }
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class KytyTestWindow {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr window, int command);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window, IntPtr dc, uint flags);
}
'@
$window = $process.MainWindowHandle
if ($window -eq 0) { throw 'The test emulator has no window yet.' }
[KytyTestWindow]::ShowWindow($window, 9) | Out-Null
[KytyTestWindow]::SetForegroundWindow($window) | Out-Null
Start-Sleep -Milliseconds 200
$rect = New-Object KytyTestWindow+Rect
if (-not [KytyTestWindow]::GetWindowRect($window, [ref]$rect)) { throw 'Cannot read emulator window bounds.' }
$bitmap = [System.Drawing.Bitmap]::new($rect.Right - $rect.Left, $rect.Bottom - $rect.Top)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    if ([KytyTestWindow]::GetForegroundWindow() -eq $window) {
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
    } else {
        $dc = $graphics.GetHdc()
        try {
            if (-not [KytyTestWindow]::PrintWindow($window, $dc, 2)) {
                throw 'The emulator is obscured and window capture is unavailable.'
            }
        } finally { $graphics.ReleaseHdc($dc) }
    }
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $process.MainWindowTitle
} finally { $graphics.Dispose(); $bitmap.Dispose() }
