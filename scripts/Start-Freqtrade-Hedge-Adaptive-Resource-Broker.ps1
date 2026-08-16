[CmdletBinding()]
param(
    [string]$UserDataRoot = "",
    [int]$IntervalMilliseconds = 1000,
    [switch]$Worker
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($UserDataRoot)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $UserDataRoot = Join-Path $ProjectRoot "user_data"
}
else {
    $UserDataRoot = [System.IO.Path]::GetFullPath($UserDataRoot)
}

$RuntimeDir = Join-Path $UserDataRoot "runtime"
$SnapshotPath = Join-Path $RuntimeDir "host-resource-snapshot.json"
$PidPath = Join-Path $RuntimeDir "host-resource-broker.pid"
$StopPath = Join-Path $RuntimeDir "host-resource-broker.stop"

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $Encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $Encoding)
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null

if (-not $Worker) {
    if (Test-Path -LiteralPath $PidPath -PathType Leaf) {
        $OldPidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
        $OldPid = 0
        if ([int]::TryParse($OldPidText, [ref]$OldPid)) {
            $Existing = Get-Process -Id $OldPid -ErrorAction SilentlyContinue
            if ($null -ne $Existing) {
                Write-Host ("Adaptive resource broker already running. PID=" + $OldPid)
                exit 0
            }
        }
    }
    if (Test-Path -LiteralPath $StopPath) {
        Remove-Item -LiteralPath $StopPath -Force
    }
    $PowerShell = Join-Path $PSHOME "powershell.exe"
    if (-not (Test-Path -LiteralPath $PowerShell -PathType Leaf)) {
        $PowerShell = "powershell.exe"
    }
    $QuotedScript = '"' + $PSCommandPath + '"'
    $QuotedUserData = '"' + $UserDataRoot + '"'
    $Arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $QuotedScript,
        "-UserDataRoot", $QuotedUserData,
        "-IntervalMilliseconds", $IntervalMilliseconds,
        "-Worker"
    )
    $Process = Start-Process `
        -FilePath $PowerShell `
        -ArgumentList $Arguments `
        -WindowStyle Hidden `
        -PassThru
    Write-Utf8NoBom -Path $PidPath -Text ([string]$Process.Id)
    Write-Host ("Adaptive resource broker started. PID=" + $Process.Id)
    Write-Host ("Snapshot: " + $SnapshotPath)
    exit 0
}

$NativeSource = @"
using System;
using System.Runtime.InteropServices;

public static class HedgeHostResources
{
    [StructLayout(LayoutKind.Sequential)]
    public struct FILETIME
    {
        public uint LowDateTime;
        public uint HighDateTime;
        public ulong Value
        {
            get { return ((ulong)HighDateTime << 32) | LowDateTime; }
        }
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public class MEMORYSTATUSEX
    {
        public uint dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetSystemTimes(
        out FILETIME idleTime,
        out FILETIME kernelTime,
        out FILETIME userTime);

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern bool GlobalMemoryStatusEx([In, Out] MEMORYSTATUSEX buffer);
}
"@

try {
    Add-Type -TypeDefinition $NativeSource -Language CSharp -ErrorAction Stop
}
catch {
    if ($_.Exception.Message -notmatch "already exists") {
        throw
    }
}

function Get-SystemTimesSample {
    $Idle = New-Object HedgeHostResources+FILETIME
    $Kernel = New-Object HedgeHostResources+FILETIME
    $User = New-Object HedgeHostResources+FILETIME
    if (-not [HedgeHostResources]::GetSystemTimes([ref]$Idle, [ref]$Kernel, [ref]$User)) {
        throw "GetSystemTimes failed."
    }
    return [pscustomobject]@{
        Idle = [uint64]$Idle.Value
        Kernel = [uint64]$Kernel.Value
        User = [uint64]$User.Value
    }
}

function Get-MemorySample {
    $Memory = New-Object HedgeHostResources+MEMORYSTATUSEX
    if (-not [HedgeHostResources]::GlobalMemoryStatusEx($Memory)) {
        throw "GlobalMemoryStatusEx failed."
    }
    return [pscustomobject]@{
        Total = [uint64]$Memory.ullTotalPhys
        Available = [uint64]$Memory.ullAvailPhys
    }
}

try {
    Write-Utf8NoBom -Path $PidPath -Text ([string]$PID)

    # Topology is static. Query WMI once; the 1-second hot loop below uses direct
    # kernel APIs only so the broker itself contributes negligible CPU/allocation.
    $Processors = @(Get-CimInstance Win32_Processor)
    $PhysicalCores = [int](($Processors | Measure-Object -Property NumberOfCores -Sum).Sum)
    $LogicalCpus = [int](($Processors | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)
    $Previous = Get-SystemTimesSample

    while (-not (Test-Path -LiteralPath $StopPath)) {
        Start-Sleep -Milliseconds ([math]::Max(250, $IntervalMilliseconds))
        $Current = Get-SystemTimesSample
        $Memory = Get-MemorySample

        $IdleDelta = [double]($Current.Idle - $Previous.Idle)
        $KernelDelta = [double]($Current.Kernel - $Previous.Kernel)
        $UserDelta = [double]($Current.User - $Previous.User)
        $TotalDelta = $KernelDelta + $UserDelta
        if ($TotalDelta -gt 0) {
            $CpuPercent = 100.0 * (1.0 - ($IdleDelta / $TotalDelta))
        }
        else {
            $CpuPercent = 0.0
        }
        $CpuPercent = [math]::Min(100.0, [math]::Max(0.0, $CpuPercent))
        $Previous = $Current

        $Now = [DateTimeOffset]::UtcNow
        $Payload = [ordered]@{
            schema = "freqtrade-hedge-host-resource-v2"
            timestamp_epoch = [double]$Now.ToUnixTimeMilliseconds() / 1000.0
            timestamp_utc = $Now.ToString("o")
            cpu_percent = [math]::Round([double]$CpuPercent, 3)
            memory_available_bytes = [uint64]$Memory.Available
            memory_total_bytes = [uint64]$Memory.Total
            logical_cpus = $LogicalCpus
            physical_cpus = $PhysicalCores
            broker_pid = $PID
            sampler = "GetSystemTimes+GlobalMemoryStatusEx"
        }
        $Temp = $SnapshotPath + ".tmp"
        Write-Utf8NoBom -Path $Temp -Text (($Payload | ConvertTo-Json -Depth 5) + [Environment]::NewLine)
        Move-Item -LiteralPath $Temp -Destination $SnapshotPath -Force
    }
}
finally {
    if (Test-Path -LiteralPath $PidPath) {
        Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $StopPath) {
        Remove-Item -LiteralPath $StopPath -Force -ErrorAction SilentlyContinue
    }
}
