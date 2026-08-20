[CmdletBinding()]
param(
    [string] $RepoRoot = (Get-Location).Path,
    [ValidateSet("strict-oos", "two-year-split", "integration-full")]
    [string] $ValidationMode = "strict-oos",
    [ValidateSet("fast", "balanced", "deep")]
    [string] $Budget = "balanced",
    [string] $Device = "cpu",
    [string] $StrategyDevice = "cpu",
    [int] $ParallelEnvs = 16,
    [int] $TaskTimeout = 0,
    [switch] $SkipTraining,
    [switch] $ForceTraining
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$SuiteRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { throw "Python not found. Expected .venv\Scripts\python.exe or python on PATH." }
    $Python = $cmd.Source
}

Write-Host "=== HPRL MTF -> real Strategy -> HEDGE/Freqtrade 30 formal backtests ==="
Write-Host ("RepoRoot       : " + $RepoRoot)
Write-Host ("SuiteRoot      : " + $SuiteRoot)
Write-Host ("ValidationMode : " + $ValidationMode)
Write-Host ("Budget         : " + $Budget)
Write-Host ("Device         : " + $Device)
Write-Host ("StrategyDevice : " + $StrategyDevice)
Write-Host ("Python         : " + $Python)

& $Python (Join-Path $SuiteRoot "validate_suite.py")
if ($LASTEXITCODE -ne 0) { throw "Suite source validation failed." }

$argsList = @(
    (Join-Path $SuiteRoot "run_suite.py"),
    "--repo-root", $RepoRoot,
    "--validation-mode", $ValidationMode,
    "--budget", $Budget,
    "--device", $Device,
    "--strategy-device", $StrategyDevice,
    "--parallel-envs", [string]$ParallelEnvs,
    "--task-timeout", [string]$TaskTimeout
)
if ($SkipTraining) { $argsList += "--skip-training" }
if ($ForceTraining) { $argsList += "--force-training" }

& $Python @argsList
$code = $LASTEXITCODE
Write-Host ("Suite exit code: " + $code)
exit $code
