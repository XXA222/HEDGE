[CmdletBinding()]
param(
    [string] $RepoRoot = (Get-Location).Path,
    [string] $DestinationName = "HPRL_FREQTRADE_MTF_V3"
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Destination = Join-Path $RepoRoot $DestinationName

Write-Host "=== Install HPRL Freqtrade MTF V3 source ==="
Write-Host ("RepoRoot    : " + $RepoRoot)
Write-Host ("PackageRoot : " + $PackageRoot)
Write-Host ("Destination : " + $Destination)

$requiredRepo = @(
    "freqtrade",
    "freqtrade\hedge\hprl",
    "freqtrade\hedge\production\hprl_hedge_adapter.py",
    "freqtrade\data\dataprovider.py",
    "freqtrade\strategy\interface.py"
)
foreach ($relative in $requiredRepo) {
    $path = Join-Path $RepoRoot $relative
    if (-not (Test-Path -LiteralPath $path)) {
        throw ("HEDGE repository requirement is missing: " + $path)
    }
}

$requiredPackage = @(
    "artifact_contract.py",
    "features.py",
    "prepare_models.py",
    "run_suite.py",
    "suite_specs.py",
    "strategies\hprl_mtf_v3_base.py",
    "configs\fast_td3.json"
)
foreach ($relative in $requiredPackage) {
    $path = Join-Path $PackageRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw ("Package requirement is missing: " + $path)
    }
}

if (Test-Path -LiteralPath $Destination) {
    $downloads = Join-Path $env:USERPROFILE "Downloads"
    if (-not (Test-Path -LiteralPath $downloads -PathType Container)) {
        New-Item -ItemType Directory -Path $downloads -Force | Out-Null
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backup = Join-Path $downloads ($DestinationName + "-backup-" + $stamp + ".zip")
    Write-Host ("Backing up existing suite to: " + $backup)
    Compress-Archive -LiteralPath $Destination -DestinationPath $backup -CompressionLevel Optimal
    Remove-Item -LiteralPath $Destination -Recurse -Force
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
Get-ChildItem -LiteralPath $PackageRoot -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
}

$verifyFiles = @(
    "artifact_contract.py",
    "features.py",
    "prepare_models.py",
    "run_suite.py",
    "suite_specs.py",
    "strategies\hprl_mtf_v3_base.py"
)
foreach ($relative in $verifyFiles) {
    $source = Join-Path $PackageRoot $relative
    $target = Join-Path $Destination $relative
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash
    $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
    if ($sourceHash -ne $targetHash) {
        throw ("Installed file hash mismatch: " + $relative)
    }
}

$receipt = [ordered]@{
    schema = "hprl-freqtrade-mtf-install-receipt-v1"
    installed_at = (Get-Date).ToUniversalTime().ToString("o")
    repo_root = $RepoRoot
    destination = $Destination
    source_package = $PackageRoot
    tests_executed = $false
    training_executed = $false
    backtests_executed = $false
}
$receiptPath = Join-Path $Destination "SOURCE_INSTALL_RECEIPT.json"
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8

Write-Host ""
Write-Host "INSTALL COMPLETE"
Write-Host ("Installed source: " + $Destination)
Write-Host "No training or backtesting was executed by this installer."
Write-Host "Future full suite entry point:"
Write-Host ('  Set-ExecutionPolicy -Scope Process Bypass -Force; & "' + (Join-Path $Destination "run_all.ps1") + '" -RepoRoot "' + $RepoRoot + '"')
