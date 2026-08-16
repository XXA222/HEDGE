[CmdletBinding()]
param(
    [ValidateSet('Short', 'Long')]
    [string]$Profile = 'Short',
    [string]$ProjectRoot = '',
    [string]$PythonPath = '',
    [string]$OutputRoot = '',
    [int]$ShortBars = 96,
    [int]$LongBars = 10000,
    [int]$ShortTrainingTimesteps = 64,
    [int]$LongTrainingTimesteps = 100000,
    [string]$BacktestCsv = '',
    [switch]$RunBinanceReadOnly,
    [string]$BinanceCredentialPath = ''
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $BinanceCredentialPath) {
    $folderName = ([char]0x542F) + ([char]0x52A8) + 'freqtrade'
    $fileName = 'binance - ' + ([char]0x526F) + ([char]0x672C) + '.txt'
    $BinanceCredentialPath = Join-Path (Join-Path $env:USERPROFILE 'Desktop') (Join-Path $folderName (Join-Path 'api' $fileName))
}
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $scriptRoot }
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root does not exist: $ProjectRoot"
}

if (-not $PythonPath) { $PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Project Python does not exist: $PythonPath"
}

if (-not $OutputRoot) { $OutputRoot = Join-Path $ProjectRoot 'artifacts\hedge-validation' }
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$outputDirectory = Join-Path $OutputRoot ($Profile.ToLowerInvariant() + '-' + $runId)
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$results = [System.Collections.Generic.List[object]]::new()

function Invoke-PythonCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $safeName = $Name -replace '[^A-Za-z0-9_.-]', '_'
    $logPath = Join-Path $outputDirectory ($safeName + '.log')
    $started = [Diagnostics.Stopwatch]::StartNew()
    $exitCode = 1
    Push-Location $ProjectRoot
    try {
        & $PythonPath @Arguments 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $started.Stop()
    }
    $result = [pscustomobject]@{
        name = $Name
        status = if ($exitCode -eq 0) { 'PASS' } else { 'FAIL' }
        exit_code = $exitCode
        elapsed_seconds = [Math]::Round($started.Elapsed.TotalSeconds, 3)
        log = $logPath
        arguments = $Arguments
    }
    $results.Add($result)
    if ($exitCode -ne 0) { throw "Validation '$Name' failed with exit code $exitCode. See $logPath" }
}

function Invoke-BinanceReadOnlyCheck {
    $checker = Join-Path $scriptRoot 'Test-Freqtrade-Hedge-BinanceReadOnly-PS51.ps1'
    if (-not (Test-Path -LiteralPath $checker -PathType Leaf)) { throw "Missing Binance checker: $checker" }
    $safeName = 'binance-readonly'
    $logPath = Join-Path $outputDirectory ($safeName + '.log')
    $reportPath = Join-Path $outputDirectory ($safeName + '.json')
    $started = [Diagnostics.Stopwatch]::StartNew()
    $exitCode = 1
    Push-Location $ProjectRoot
    try {
        & $checker `
            -CredentialPath $BinanceCredentialPath `
            -OutputPath $reportPath 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $started.Stop()
    }
    $result = [pscustomobject]@{
        name = 'binance-readonly'
        status = if ($exitCode -eq 0) { 'PASS' } else { 'FAIL' }
        exit_code = $exitCode
        elapsed_seconds = [Math]::Round($started.Elapsed.TotalSeconds, 3)
        report = $reportPath
        log = $logPath
    }
    $results.Add($result)
    if ($exitCode -ne 0) { throw "Binance read-only validation failed. See $reportPath and $logPath" }
}

$failure = $null
try {
    if ($ShortBars -lt 40 -or $LongBars -lt 40) { throw 'Backtest bars must be at least 40.' }
    if ($ShortTrainingTimesteps -lt 1 -or $LongTrainingTimesteps -lt 1) {
        throw 'Training timesteps must be positive.'
    }

    $backtestOutput = Join-Path $outputDirectory 'backtest'
    $backtestArguments = @(
        'tools\run_simple_hedge_backtest.py',
        '--output-dir', $backtestOutput
    )
    if ($BacktestCsv) {
        $backtestArguments += @('--csv', ([IO.Path]::GetFullPath($BacktestCsv)))
    }
    else {
        $bars = if ($Profile -eq 'Long') { $LongBars } else { $ShortBars }
        $backtestArguments += @('--bars', [string]$bars)
    }
    Invoke-PythonCheck -Name 'synthetic-or-csv-backtest' -Arguments $backtestArguments

    $trainingTimesteps = if ($Profile -eq 'Long') { $LongTrainingTimesteps } else { $ShortTrainingTimesteps }
    Invoke-PythonCheck -Name 'risklevel-training' -Arguments @(
        'tools\run_hedge_risklevel_training.py',
        '--timesteps', [string]$trainingTimesteps,
        '--output', (Join-Path $outputDirectory 'risklevel-model')
    )

    Invoke-PythonCheck -Name 'backtesting-core-self-test' -Arguments @(
        '-m', 'freqtrade.hedge.backtesting.cli', 'self-test'
    )
    Invoke-PythonCheck -Name 'paper-integrated-smoke' -Arguments @(
        'tools\hedge_integrated_smoke.py'
    )
    Invoke-PythonCheck -Name 'mlrl-contract-matrix' -Arguments @(
        'tools\run_hedge_mlrl_validation.py',
        '--json-out', (Join-Path $outputDirectory 'mlrl-contract-matrix.json')
    )

    if ($Profile -eq 'Long') {
        Invoke-PythonCheck -Name 'full-hedge-test-suite' -Arguments @(
            '-m', 'pytest', 'tests/hedge', '-q'
        )
        Invoke-PythonCheck -Name 'sb3-training-smoke' -Arguments @(
            'tools\validate_hedge_mlrl_sb3.py',
            '--require-sb3',
            '--json-out', (Join-Path $outputDirectory 'sb3-smoke.json')
        )
    }
    if ($RunBinanceReadOnly) { Invoke-BinanceReadOnlyCheck }
}
catch {
    $failure = $_.Exception.Message
    Write-Error $failure
}
finally {
    $report = [ordered]@{
        schema = 'freqtrade-hedge-validation-run-v1'
        generated_at = [DateTimeOffset]::UtcNow.ToString('o')
        profile = $Profile
        project_root = $ProjectRoot
        offline_training_and_backtest = $true
        exchange_writes = 'NOT_INVOKED'
        status = if ($null -eq $failure -and @($results | Where-Object { $_.status -ne 'PASS' }).Count -eq 0) { 'PASS' } else { 'FAIL' }
        failure = $failure
        results = $results
    }
    $reportPath = Join-Path $outputDirectory 'validation-report.json'
    $report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    Write-Output ('validation_report=' + $reportPath)
    Write-Output ('validation_status=' + $report.status)
}
if ($null -ne $failure) { exit 1 }
