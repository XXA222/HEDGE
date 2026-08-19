[CmdletBinding()]
param(
    [ValidateSet('smoke', 'deep', 'maximum')]
    [string]$Profile = 'maximum',
    [string]$Checklist = 'C:\Users\QX\Downloads\HEDGE_统一功能测试验证主清单_88dded9 (1).md',
    [string]$DataRoot = 'D:\Program Files\HEDGE\artifacts\eth-two-year-deep',
    [int]$RiskTimesteps = 20000,
    [int]$RiskRows = 6000,
    [int]$HprlIterations = 2000,
    [int]$HprlReplayIterations = 10000,
    [int]$PerformanceCycles = 1500,
    [int]$PhaseTimeout = 7200,
    [int]$TrainingTimeout = 21600,
    [string]$OutputDir = '',
    [switch]$RequireData
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "HEDGE virtual-environment Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checklist)) {
    throw "Checklist not found: $Checklist"
}

$arguments = @(
    (Join-Path $ProjectRoot 'tools\run_hedge_deep_validation.py'),
    '--project-root', $ProjectRoot,
    '--checklist', $Checklist,
    '--profile', $Profile,
    '--python', $Python,
    '--data-root', $DataRoot,
    '--risk-timesteps', $RiskTimesteps,
    '--risk-rows', $RiskRows,
    '--hprl-iterations', $HprlIterations,
    '--hprl-replay-iterations', $HprlReplayIterations,
    '--performance-cycles', $PerformanceCycles,
    '--phase-timeout', $PhaseTimeout,
    '--training-timeout', $TrainingTimeout
)
if ($RequireData) { $arguments += '--require-data' }
if ($OutputDir) { $arguments += @('--output-dir', $OutputDir) }

Write-Host "HEDGE deep validation: profile=$Profile"
Write-Host "A failed phase is recorded and does not stop subsequent phases."
& $Python @arguments
exit $LASTEXITCODE
