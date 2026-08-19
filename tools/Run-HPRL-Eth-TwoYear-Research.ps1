[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\Program Files\HEDGE\artifacts\eth-two-year-deep',
    [string]$Algorithms = 'fast_dsac,fast_td3,rebrac_v2,simba_sac,xqc',
    [int]$Trials = 12,
    [int]$BaselineSteps = 20000,
    [int]$OptimizationSteps = 30000,
    [int]$FinalSteps = 100000,
    [int]$ParallelEnvs = 16,
    [string]$Device = 'auto',
    [string]$ReplayDevice = 'cpu',
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Script = Join-Path $ProjectRoot 'tools\train_hprl_eth_two_year.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $DataRoot)) { throw "ETH data root not found: $DataRoot" }

$arguments = @(
    $Script,
    '--data-root', $DataRoot,
    '--algorithms', $Algorithms,
    '--trials', $Trials,
    '--baseline-steps', $BaselineSteps,
    '--optimization-steps', $OptimizationSteps,
    '--final-steps', $FinalSteps,
    '--parallel-envs', $ParallelEnvs,
    '--device', $Device,
    '--replay-device', $ReplayDevice,
    '--primary-timeframe', '1h',
    '--compile-mode', 'off',
    '--mixed-precision'
)
if ($OutputDir) { $arguments += @('--output-dir', $OutputDir) }

Write-Host 'HPRL ETH two-year research: all algorithms -> validation optimization -> continued final training -> holdout backtest'
& $Python @arguments
exit $LASTEXITCODE
