[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $DataRoot,

    [string] $ProjectRoot = "",
    [string] $FundingFile = "",
    [string] $ExchangeRiskEvidence = "",
    [ValidateSet("diagnostic", "blind")]
    [string] $HoldoutRole = "diagnostic",
    [ValidateSet("synthetic", "verified")]
    [string] $ExchangeRiskMode = "synthetic",
    [string] $Seeds = "42,43,44",
    [int] $Trials = 18,
    [int] $BaselineSteps = 30000,
    [int] $OptimizationSteps = 50000,
    [int] $OptimizationConfirmSteps = 50000,
    [int] $FinalSteps = 100000,
    [double] $MaxMarketSweeps = 8.0,
    [int] $WalkForwardFolds = 3,
    [int] $WalkForwardSteps = 30000,
    [int] $ParallelEnvs = 16,
    [int] $BatchSize = 512,
    [int] $ReplayCapacity = 200000,
    [switch] $RequireFunding,
    [switch] $RuntimeChecks,
    [switch] $NoMixedPrecision
)

$ErrorActionPreference = "Stop"

function Resolve-HedgeRoot {
    param([string] $Requested)
    $candidates = @()
    if ($Requested) { $candidates += $Requested }
    $candidates += (Get-Location).Path
    $candidates += "D:\Program Files\HEDGE"
    $candidates += "D:\Program Files\freqtradev2026.07-hedge-merge"
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ($candidate -and (Test-Path -LiteralPath (Join-Path $candidate "freqtrade\hedge\hprl"))) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Unable to locate HEDGE project root. Pass -ProjectRoot explicitly."
}

$ProjectRoot = Resolve-HedgeRoot -Requested $ProjectRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Project Python not found: $PythonExe"
}
$Tool = Join-Path $ProjectRoot "tools\train_hprl_eth_two_year.py"
if (-not (Test-Path -LiteralPath $Tool)) {
    throw "Training tool not found: $Tool"
}
if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}

$argsList = @(
    $Tool,
    "--data-root", $DataRoot,
    "--holdout-role", $HoldoutRole,
    "--exchange-risk-mode", $ExchangeRiskMode,
    "--seeds", $Seeds,
    "--trials", [string]$Trials,
    "--baseline-steps", [string]$BaselineSteps,
    "--optimization-steps", [string]$OptimizationSteps,
    "--optimization-confirm-steps", [string]$OptimizationConfirmSteps,
    "--final-steps", [string]$FinalSteps,
    "--max-market-sweeps", [string]$MaxMarketSweeps,
    "--walk-forward-folds", [string]$WalkForwardFolds,
    "--walk-forward-steps", [string]$WalkForwardSteps,
    "--parallel-envs", [string]$ParallelEnvs,
    "--batch-size", [string]$BatchSize,
    "--replay-capacity", [string]$ReplayCapacity
)
if ($ExchangeRiskEvidence) {
    if (-not (Test-Path -LiteralPath $ExchangeRiskEvidence)) { throw "Exchange risk evidence not found: $ExchangeRiskEvidence" }
    $argsList += @("--exchange-risk-evidence", $ExchangeRiskEvidence)
}
if ($FundingFile) {
    if (-not (Test-Path -LiteralPath $FundingFile)) { throw "Funding file not found: $FundingFile" }
    $argsList += @("--funding-file", $FundingFile)
}
if ($RequireFunding) { $argsList += "--require-funding" }
if ($RuntimeChecks) { $argsList += "--runtime-checks" }
if ($NoMixedPrecision) { $argsList += "--no-mixed-precision" }

Write-Host "=== HPRL ETH Learning Integrity Research ==="
Write-Host "Project : $ProjectRoot"
Write-Host "Python  : $PythonExe"
Write-Host "Data    : $DataRoot"
Write-Host "Holdout : $HoldoutRole"
Write-Host "RiskMode: $ExchangeRiskMode"
Write-Host "Seeds   : $Seeds"
Write-Host ""

& $PythonExe @argsList
$code = $LASTEXITCODE
Write-Host ""
Write-Host "Research exit code: $code"
exit $code
