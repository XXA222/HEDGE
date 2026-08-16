#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$Config = "",

    [string]$Strategy = "",

    [string]$Timerange = "20240815-20260815",

    [int]$ParityDays = 3,

    [double]$MemoryCeilingGiB = 12.0,

    [int]$MaxElapsedSeconds = 7200,

    [int]$SampleSeconds = 2,

    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

function Fail([string]$Message) {
    Write-Host ("FINAL TWO-YEAR R4.2 ACCEPTANCE: FAIL - " + $Message)
    exit 2
}

function Quote-WindowsArgument([string]$Value) {
    if ($Value.Length -eq 0) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]92) {
            $backslashes += 1
            continue
        }
        if ($character -eq [char]34) {
            if ($backslashes -gt 0) {
                [void]$builder.Append(('\' * ($backslashes * 2)))
                $backslashes = 0
            }
            [void]$builder.Append('\"')
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function New-HedgeArguments(
    [string]$ConfigPath,
    [string]$Range,
    [string]$ResultPath,
    [bool]$ExportEvents
) {
    $values = @(
        "-m", "freqtrade",
        "hedge-backtesting",
        "--config", $ConfigPath,
        "--timerange", $Range,
        "--hedge-export-filename", $ResultPath
    )
    if (-not [string]::IsNullOrWhiteSpace($Strategy)) {
        $values += @("--strategy", $Strategy)
    }
    if ($ExportEvents) {
        $values += "--hedge-export-events"
    }
    return $values
}

function Invoke-SynchronousHedge(
    [string[]]$Arguments,
    [string]$StdoutPath,
    [string]$StderrPath
) {
    & $Python @Arguments 1> $StdoutPath 2> $StderrPath
    return [int]$LASTEXITCODE
}

function Resolve-HedgeConfigPath([string]$Root, [string]$Requested) {
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        $requestedPaths = @()
        if ([System.IO.Path]::IsPathRooted($Requested)) {
            $requestedPaths += [System.IO.Path]::GetFullPath($Requested)
        }
        else {
            $requestedPaths += [System.IO.Path]::GetFullPath((Join-Path $Root $Requested))
            $requestedPaths += [System.IO.Path]::GetFullPath($Requested)
        }
        foreach ($requestedFull in ($requestedPaths | Select-Object -Unique)) {
            if (Test-Path -LiteralPath $requestedFull -PathType Leaf) {
                return $requestedFull
            }
        }
        Write-Host ("Requested config not found; auto-discovering instead: " + $Requested)
    }

    $searchSpecs = @()
    $userData = Join-Path $Root "user_data"
    if (Test-Path -LiteralPath $userData -PathType Container) {
        $searchSpecs += New-Object PSObject -Property @{ Path = $userData; Recurse = $true; Weight = 30 }
    }
    $configsDir = Join-Path $Root "configs"
    if (Test-Path -LiteralPath $configsDir -PathType Container) {
        $searchSpecs += New-Object PSObject -Property @{ Path = $configsDir; Recurse = $true; Weight = 20 }
    }
    $searchSpecs += New-Object PSObject -Property @{ Path = $Root; Recurse = $false; Weight = 10 }

    $seen = @{}
    $candidates = @()
    foreach ($spec in $searchSpecs) {
        if ($spec.Recurse) {
            $files = @(Get-ChildItem -LiteralPath $spec.Path -File -Recurse -Filter "*.json" -ErrorAction SilentlyContinue)
        }
        else {
            $files = @(Get-ChildItem -LiteralPath $spec.Path -File -Filter "*.json" -ErrorAction SilentlyContinue)
        }
        foreach ($file in $files) {
            if ($seen.ContainsKey($file.FullName)) {
                continue
            }
            $seen[$file.FullName] = $true
            if (
                $file.Name -notmatch '(?i)^config.*\.json$' -and
                $file.DirectoryName -notmatch '(?i)[\\/]configs?([\\/]|$)'
            ) {
                continue
            }
            try {
                $raw = Get-Content -Raw -LiteralPath $file.FullName
            }
            catch {
                continue
            }
            if ($raw -notmatch '(?i)"exchange"\s*:') {
                continue
            }
            $score = [int]$spec.Weight
            if ($file.Name -ieq "config.json") { $score += 100 }
            elseif ($file.Name -match '(?i)^config') { $score += 40 }
            if ($raw -match '(?i)"hedge_mode_enabled"\s*:') { $score += 30 }
            if ($raw -match '(?i)"hedge"\s*:') { $score += 30 }
            if ($raw -match '(?i)"freqai"\s*:') { $score += 10 }
            if ($raw -match '(?i)"trading_mode"\s*:') { $score += 5 }
            $candidates += New-Object PSObject -Property @{ Path = $file.FullName; Score = $score }
        }
    }

    if ($candidates.Count -eq 0) {
        throw (
            "No Freqtrade config could be auto-detected. Searched project root, user_data, and configs. " +
            "Pass -Config with the real local config path."
        )
    }
    $ordered = @($candidates | Sort-Object -Property @{Expression='Score';Descending=$true}, @{Expression='Path';Descending=$false})
    $bestScore = [int]$ordered[0].Score
    $best = @($ordered | Where-Object { [int]$_.Score -eq $bestScore })
    if ($best.Count -ne 1) {
        Write-Host "Multiple equally ranked config candidates were found:"
        foreach ($item in $best) {
            Write-Host ("  " + $item.Path)
        }
        throw "Config auto-discovery is ambiguous. Re-run with -Config <exact-path>."
    }
    Write-Host ("Auto-detected config: " + $best[0].Path)
    return [string]$best[0].Path
}

function Parse-Timerange([string]$Range) {
    if ($Range -notmatch '^(\d{8})-(\d{8})$') {
        throw "Timerange must use closed YYYYMMDD-YYYYMMDD form for R4.2 acceptance."
    }
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $start = [datetime]::ParseExact($Matches[1], "yyyyMMdd", $culture)
    $end = [datetime]::ParseExact($Matches[2], "yyyyMMdd", $culture)
    if ($end -le $start) {
        throw "Timerange end must be after start."
    }
    return @($start, $end)
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Fail "ProjectRoot not found."
}
try {
    $Config = Resolve-HedgeConfigPath $ProjectRoot $Config
}
catch {
    Fail $_.Exception.Message
}
if ($MemoryCeilingGiB -le 0) {
    Fail "MemoryCeilingGiB must be positive."
}
if ($MaxElapsedSeconds -lt 1) {
    Fail "MaxElapsedSeconds must be positive."
}
if ($SampleSeconds -lt 1) {
    Fail "SampleSeconds must be positive."
}
if ($ParityDays -lt 1) {
    Fail "ParityDays must be positive."
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Fail "Project-local .venv Python not found."
}

$VersionPath = Join-Path $ProjectRoot "CLEAN-MAINLINE-VERSION.txt"
if (-not (Test-Path -LiteralPath $VersionPath -PathType Leaf)) {
    Fail "CLEAN-MAINLINE-VERSION.txt not found."
}
$Version = (Get-Content -Raw -LiteralPath $VersionPath).Trim()
if ($Version -notlike "freqtrade-hedge-clean-mainline-v1.7-final-closure-risklevel-wf-2y-r4.2-*") {
    Fail ("Final Closure R4.2 source is not installed. Current version: " + $Version)
}

try {
    $parsedRange = Parse-Timerange $Timerange
}
catch {
    Fail $_.Exception.Message
}
$fullStart = $parsedRange[0]
$fullEnd = $parsedRange[1]
$requestedDays = ($fullEnd - $fullStart).TotalDays
if ($requestedDays -lt 700) {
    Fail "R4.2 final acceptance requires at least 700 days of requested history."
}
$parityStart = $fullEnd.AddDays(-1 * $ParityDays)
if ($parityStart -lt $fullStart) {
    $parityStart = $fullStart
}
$ParityTimerange = $parityStart.ToString("yyyyMMdd") + "-" + $fullEnd.ToString("yyyyMMdd")

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDirectory = Join-Path $env:USERPROFILE ("Downloads\Hedge-Final-TwoYear-R41-" + $stamp)
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$ParityCompact = Join-Path $OutputDirectory "parity-compact.json"
$ParityDetailed = Join-Path $OutputDirectory "parity-detailed.json"
$ParityCompactStdout = Join-Path $OutputDirectory "parity-compact.stdout.log"
$ParityCompactStderr = Join-Path $OutputDirectory "parity-compact.stderr.log"
$ParityDetailedStdout = Join-Path $OutputDirectory "parity-detailed.stdout.log"
$ParityDetailedStderr = Join-Path $OutputDirectory "parity-detailed.stderr.log"
$ParityReport = Join-Path $OutputDirectory "parity-consistency.json"
$ParityValidatorStdout = Join-Path $OutputDirectory "parity-validator.stdout.log"
$ParityValidatorStderr = Join-Path $OutputDirectory "parity-validator.stderr.log"

$ResultJson = Join-Path $OutputDirectory "two-year-hedge-result.json"
$StdoutLog = Join-Path $OutputDirectory "two-year-stdout.log"
$StderrLog = Join-Path $OutputDirectory "two-year-stderr.log"
$SamplesCsv = Join-Path $OutputDirectory "resource-samples.csv"
$SummaryJson = Join-Path $OutputDirectory "acceptance-summary.json"

Set-Location -LiteralPath $ProjectRoot

# Adaptive phase-boundary release only. Never force full GC in the per-bar loop.
$env:HEDGE_MEMORY_RELEASE_MODE = "adaptive"
$env:HEDGE_MEMORY_GC_RSS_MIB = "768"
$env:HEDGE_MEMORY_GC_PRESSURE_RATIO = "0.55"
$env:HEDGE_MEMORY_GC_HARD_PRESSURE_RATIO = "0.80"
$env:HEDGE_MEMORY_GC_COOLDOWN_SECONDS = "2"
$env:HEDGE_MEMORY_TRIM = "1"

Write-Host ""
Write-Host "=== R4.2 real-data compact/detailed parity preflight ==="
Write-Host ("Parity timerange: " + $ParityTimerange)

$compactArgs = New-HedgeArguments $Config $ParityTimerange $ParityCompact $false
$compactExit = Invoke-SynchronousHedge $compactArgs $ParityCompactStdout $ParityCompactStderr
if ($compactExit -ne 0) {
    Fail ("Compact parity backtest failed with ExitCode=" + $compactExit + ". See " + $ParityCompactStderr)
}
if (-not (Test-Path -LiteralPath $ParityCompact -PathType Leaf)) {
    Fail "Compact parity result was not written."
}

$detailedArgs = New-HedgeArguments $Config $ParityTimerange $ParityDetailed $true
$detailedExit = Invoke-SynchronousHedge $detailedArgs $ParityDetailedStdout $ParityDetailedStderr
if ($detailedExit -ne 0) {
    Fail ("Detailed parity backtest failed with ExitCode=" + $detailedExit + ". See " + $ParityDetailedStderr)
}
if (-not (Test-Path -LiteralPath $ParityDetailed -PathType Leaf)) {
    Fail "Detailed parity result was not written."
}

$validatorPath = Join-Path $ProjectRoot "tools\validate_hedge_backtest_consistency.py"
if (-not (Test-Path -LiteralPath $validatorPath -PathType Leaf)) {
    Fail "R4 compact/detailed consistency validator is missing."
}
$validatorArgs = @(
    $validatorPath,
    "--compact", $ParityCompact,
    "--detailed", $ParityDetailed,
    "--output", $ParityReport
)
$validatorExit = Invoke-SynchronousHedge $validatorArgs $ParityValidatorStdout $ParityValidatorStderr
if ($validatorExit -ne 0) {
    Fail ("Compact/detailed consistency gate failed. See " + $ParityReport)
}

$parityPayload = Get-Content -Raw -LiteralPath $ParityReport | ConvertFrom-Json
if ([string]$parityPayload.status -ne "PASS") {
    Fail "Compact/detailed consistency report did not pass."
}

Write-Host "Parity gate: PASS"
Write-Host ""
Write-Host "=== R4.2 full two-year 1m compact acceptance ==="

$fullArgs = New-HedgeArguments $Config $Timerange $ResultJson $false
$argumentLine = (($fullArgs | ForEach-Object { Quote-WindowsArgument ([string]$_) }) -join " ")

"timestamp,elapsed_seconds,working_set_mib,private_mib,total_cpu_seconds" |
    Set-Content -LiteralPath $SamplesCsv -Encoding ascii

$started = Get-Date
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList $argumentLine `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

$peakWorkingSet = 0L
$peakPrivate = 0L
$guardFailure = $null
$lastCpuSeconds = 0.0

while (-not $process.HasExited) {
    Start-Sleep -Seconds $SampleSeconds
    try {
        $process.Refresh()
        if ($process.HasExited) {
            break
        }
        $working = [int64]$process.WorkingSet64
        $private = [int64]$process.PrivateMemorySize64
        if ($working -gt $peakWorkingSet) { $peakWorkingSet = $working }
        if ($private -gt $peakPrivate) { $peakPrivate = $private }
        $lastCpuSeconds = $process.TotalProcessorTime.TotalSeconds
        $line = "{0},{1:F1},{2:F1},{3:F1},{4:F3}" -f `
            (Get-Date -Format "o"), `
            $stopwatch.Elapsed.TotalSeconds, `
            ($working / 1MB), `
            ($private / 1MB), `
            $lastCpuSeconds
        Add-Content -LiteralPath $SamplesCsv -Value $line -Encoding ascii

        if (($working / 1GB) -gt $MemoryCeilingGiB) {
            $guardFailure = "Working-set memory ceiling exceeded."
            try { $process.Kill() } catch {}
            break
        }
        if ($stopwatch.Elapsed.TotalSeconds -gt $MaxElapsedSeconds) {
            $guardFailure = "Wall-clock ceiling exceeded."
            try { $process.Kill() } catch {}
            break
        }
    }
    catch {
        if (-not $process.HasExited) {
            throw
        }
    }
}

try { $process.WaitForExit() } catch {}
$stopwatch.Stop()
$exitCode = $process.ExitCode
try {
    $process.Refresh()
    $lastCpuSeconds = [Math]::Max($lastCpuSeconds, $process.TotalProcessorTime.TotalSeconds)
}
catch {}

$resultExists = Test-Path -LiteralPath $ResultJson -PathType Leaf
$compactReplay = $false
$indexedRows = $false
$cachedChronology = $false
$bitmaskSlots = $false
$flatIdleBypass = $false
$flatIdleBypassCount = 0L
$processedBars = 0L
$retainedSnapshots = 0L
$returnCount = -1L
$riskMetricSource = $null
$riskPeriodsPerYear = $null
$resultParseError = $null
$pnlReconciliation = $null
$resultTimeframe = $null

if ($resultExists) {
    try {
        $result = Get-Content -Raw -LiteralPath $ResultJson | ConvertFrom-Json
        $resultTimeframe = [string]$result.timeframe
        $report = $result.report
        if ($null -ne $report) {
            $compactReplay = ([string]$report.replay_mode -eq "COMPACT_ORDERED_STREAM_V2")
            $indexedRows = ([string]$report.stream_row_mode -eq "INDEXED_ARRAY_VIEW_V2")
            $cachedChronology = ([string]$report.chronology_mode -eq "CACHED_TIMEFRAME_SECONDS_V2")
            $bitmaskSlots = ([string]$report.slot_validation_mode -eq "BITMASK_SLOT_VALIDATION_V1")
            $flatIdleBypass = ([string]$report.matcher_mode -eq "FLAT_IDLE_BYPASS_V1")
            $flatIdleBypassCount = [int64]$report.flat_idle_matcher_bypass_count
            $processedBars = [int64]$report.processed_bar_count
            $retainedSnapshots = [int64]$report.retained_snapshot_count
            $returnCount = [int64]$report.equity_return_count
            $pnlReconciliation = $report.pnl_reconciliation_error
        }
        if ($null -ne $result.hedge_native -and $null -ne $result.hedge_native.metadata) {
            $riskMetricSource = $result.hedge_native.metadata.risk_metric_source
            $riskPeriodsPerYear = $result.hedge_native.metadata.risk_periods_per_year
        }
    }
    catch {
        $resultParseError = $_.Exception.Message
    }
}

$momentCountExact = ($processedBars -gt 0 -and $returnCount -eq ($processedBars - 1))
$riskMetricsExact = ([string]$riskMetricSource -eq "BAR_RETURN_MOMENTS")
$retentionBounded = ($retainedSnapshots -ge 2 -and $retainedSnapshots -le 2049)
$timeframeOneMinute = ($resultTimeframe -eq "1m")
$fullTwoYearBarCoverage = ($processedBars -ge 1000000)
$elapsedSeconds = $stopwatch.Elapsed.TotalSeconds
$barsPerSecond = $(if ($elapsedSeconds -gt 0 -and $processedBars -gt 0) { $processedBars / $elapsedSeconds } else { 0.0 })
$cpuCoreEquivalents = $(if ($elapsedSeconds -gt 0) { $lastCpuSeconds / $elapsedSeconds } else { 0.0 })

$passed = (
    ($null -eq $guardFailure) -and
    ($exitCode -eq 0) -and
    $resultExists -and
    $compactReplay -and
    $indexedRows -and
    $cachedChronology -and
    $bitmaskSlots -and
    $flatIdleBypass -and
    $riskMetricsExact -and
    $momentCountExact -and
    $retentionBounded -and
    $timeframeOneMinute -and
    $fullTwoYearBarCoverage -and
    ($null -eq $resultParseError)
)

$summary = [ordered]@{
    schema = "freqtrade-hedge-two-year-runtime-acceptance-r4-v1"
    status = $(if ($passed) { "PASS" } else { "FAIL" })
    source_version = $Version
    project_root = $ProjectRoot
    config = $Config
    strategy = $Strategy
    timerange = $Timerange
    parity_timerange = $ParityTimerange
    parity_status = [string]$parityPayload.status
    parity_report = $ParityReport
    started_at = $started.ToString("o")
    elapsed_seconds = [Math]::Round($elapsedSeconds, 3)
    exit_code = $exitCode
    peak_working_set_mib = [Math]::Round($peakWorkingSet / 1MB, 1)
    peak_private_mib = [Math]::Round($peakPrivate / 1MB, 1)
    memory_ceiling_gib = $MemoryCeilingGiB
    wall_clock_ceiling_seconds = $MaxElapsedSeconds
    total_cpu_seconds = [Math]::Round($lastCpuSeconds, 3)
    average_cpu_core_equivalents = [Math]::Round($cpuCoreEquivalents, 3)
    processed_bar_count = $processedBars
    full_two_year_bar_coverage = $fullTwoYearBarCoverage
    result_timeframe = $resultTimeframe
    one_minute_timeframe_verified = $timeframeOneMinute
    bars_per_second = [Math]::Round($barsPerSecond, 3)
    retained_snapshot_count = $retainedSnapshots
    compact_replay_verified = $compactReplay
    indexed_array_row_view_verified = $indexedRows
    cached_timeframe_chronology_verified = $cachedChronology
    bitmask_slot_validation_verified = $bitmaskSlots
    flat_idle_matcher_bypass_verified = $flatIdleBypass
    flat_idle_matcher_bypass_count = $flatIdleBypassCount
    exact_bar_risk_metrics_verified = $riskMetricsExact
    bar_return_moment_count_exact = $momentCountExact
    risk_metric_source = $riskMetricSource
    risk_periods_per_year = $riskPeriodsPerYear
    equity_return_count = $returnCount
    pnl_reconciliation_error = $pnlReconciliation
    snapshot_retention_bounded = $retentionBounded
    guard_failure = $guardFailure
    result_exists = $resultExists
    result_parse_error = $resultParseError
    result_json = $ResultJson
    samples_csv = $SamplesCsv
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
}

$summary | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $SummaryJson -Encoding utf8

Write-Host ""
Write-Host "============================================================"
Write-Host (" HEDGE FINAL TWO-YEAR R4.2 ACCEPTANCE: " + $summary.status)
Write-Host "============================================================"
Write-Host ("Parity          : " + $summary.parity_status)
Write-Host ("ExitCode        : " + $exitCode)
Write-Host ("ElapsedSeconds  : " + $summary.elapsed_seconds)
Write-Host ("Peak RSS MiB    : " + $summary.peak_working_set_mib)
Write-Host ("Peak Private    : " + $summary.peak_private_mib)
Write-Host ("Bars            : " + $processedBars)
Write-Host ("Bars/sec        : " + $summary.bars_per_second)
Write-Host ("Timeframe       : " + $resultTimeframe)
Write-Host ("2Y bar coverage : " + $fullTwoYearBarCoverage)
Write-Host ("CPU core-equiv  : " + $summary.average_cpu_core_equivalents)
Write-Host ("Compact Replay  : " + $compactReplay)
Write-Host ("Indexed Rows    : " + $indexedRows)
Write-Host ("Cached Chrono   : " + $cachedChronology)
Write-Host ("Bitmask Slots   : " + $bitmaskSlots)
Write-Host ("Flat Bypass     : " + $flatIdleBypass + " count=" + $flatIdleBypassCount)
Write-Host ("Risk Metrics    : " + $riskMetricSource)
Write-Host ("Return Moments  : " + $returnCount)
Write-Host ("Result          : " + $ResultJson)
Write-Host ("Summary         : " + $SummaryJson)
Write-Host ("Samples         : " + $SamplesCsv)

if (-not $passed) {
    exit 2
}
exit 0
