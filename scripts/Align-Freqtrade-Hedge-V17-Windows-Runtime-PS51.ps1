[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$Proxy = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}
else {
    $ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-freqai-rl.txt"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw ("Project-local Python is missing: " + $Python)
}
if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    throw ("requirements-freqai-rl.txt is missing: " + $Requirements)
}

$Text = Get-Content -LiteralPath $Requirements -Raw
$Match = [regex]::Match($Text, '(?m)^tqdm==([^;\s]+)')
if (-not $Match.Success) {
    throw "Unable to resolve the V1.7 tqdm pin from requirements-freqai-rl.txt."
}
$RequiredTqdm = $Match.Groups[1].Value

function Get-TqdmVersion {
    & $Python -c "import importlib.metadata as m; print(m.version('tqdm'))"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query tqdm from the project-local venv."
    }
}

$Before = (Get-TqdmVersion | Out-String).Trim()
Write-Host ("tqdm before : " + $Before)
Write-Host ("tqdm target : " + $RequiredTqdm)

if ($Before -ne $RequiredTqdm) {
    $Args = @(
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--upgrade-strategy", "only-if-needed"
    )
    if (-not [string]::IsNullOrWhiteSpace($Proxy)) {
        $Args += @("--proxy", $Proxy)
    }
    $Args += ("tqdm==" + $RequiredTqdm)
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to align the V1.7 tqdm pin."
    }
}

$After = (Get-TqdmVersion | Out-String).Trim()
if ($After -ne $RequiredTqdm) {
    throw ("tqdm alignment failed: expected=" + $RequiredTqdm + "; actual=" + $After)
}

& $Python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check failed after tqdm alignment."
}

Write-Host "WINDOWS_V17_RUNTIME_ALIGNMENT: PASS" -ForegroundColor Green
