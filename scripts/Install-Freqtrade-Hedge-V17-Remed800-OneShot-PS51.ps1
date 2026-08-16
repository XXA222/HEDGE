param(
    [string]$ProjectRoot = "",
    [switch]$SkipRuntimeInstall,
    [switch]$SkipSmokeTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process Bypass -Force

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host ("=== " + $Label + " ===")
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw ($Label + " failed with exit code " + $LASTEXITCODE)
    }
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ProjectRoot = Split-Path -Parent $ScriptDir
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "pyproject.toml"))) {
    throw ("Project root is invalid: " + $ProjectRoot)
}
Set-Location -LiteralPath $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        Invoke-Checked "Create Python 3.12 venv" {
            & $PyLauncher.Source -3.12 -m venv $VenvDir
        }
    }
    else {
        $SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -eq $SystemPython) {
            throw "Python 3.12 is required. Neither py.exe nor python.exe was found."
        }
        $Version = & $SystemPython.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne "3.12") {
            throw ("Python 3.12 is required; detected " + $Version)
        }
        Invoke-Checked "Create Python 3.12 venv" {
            & $SystemPython.Source -m venv $VenvDir
        }
    }
}

$Version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne "3.12") {
    throw ("Project venv must use Python 3.12; detected " + $Version)
}

if (-not $SkipRuntimeInstall) {
    Invoke-Checked "Upgrade build tooling" {
        & $Python -m pip install --disable-pip-version-check --no-cache-dir --upgrade pip "setuptools==83.0.0" wheel
    }

    $TorchState = & $Python -c "import importlib.util; s=importlib.util.find_spec('torch'); print('missing' if s is None else __import__('torch').__version__ + '|' + str(__import__('torch').version.cuda))"
    if ($LASTEXITCODE -ne 0) {
        $TorchState = "missing"
    }
    if (($TorchState -notmatch '^2\.13\.0(\+cpu)?\|None$')) {
        Invoke-Checked "Install CPU-only PyTorch 2.13.0" {
            & $Python -m pip install --disable-pip-version-check --no-cache-dir --upgrade --force-reinstall `
                --index-url "https://download.pytorch.org/whl/cpu" "torch==2.13.0"
        }
    }

    Invoke-Checked "Install Freqtrade and Hedge ML/RL dependencies" {
        & $Python -m pip install --disable-pip-version-check --no-cache-dir `
            -r "requirements-freqai-rl.txt" -r "requirements-hedge-mlrl.txt"
    }

    # Reassert CPU-only Torch after dependency resolution.
    $CudaState = & $Python -c "import torch; print('cpu' if torch.version.cuda is None else 'cuda')"
    if ($LASTEXITCODE -ne 0 -or $CudaState.Trim() -ne "cpu") {
        Invoke-Checked "Reassert CPU-only PyTorch" {
            & $Python -m pip install --disable-pip-version-check --no-cache-dir --upgrade --force-reinstall `
                --index-url "https://download.pytorch.org/whl/cpu" "torch==2.13.0"
        }
    }

    Invoke-Checked "Install clean mainline package" {
        & $Python -m pip install --disable-pip-version-check --no-cache-dir --no-deps .
    }

    Invoke-Checked "Dependency consistency" {
        & $Python -m pip check
    }

    Invoke-Checked "Register project-local source" {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
            (Join-Path $ProjectRoot "scripts\Configure-Freqtrade-Hedge-LocalSource.ps1") `
            -ProjectRoot $ProjectRoot
    }
}

$ArtifactDir = Join-Path $ProjectRoot "artifacts\remediation800"
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

Invoke-Checked "Compile source" {
    & $Python -m compileall -q "freqtrade\hedge" "freqtrade\freqai\hedge_rl" "tools"
}

Invoke-Checked "Bounded runtime collection gate" {
    & $Python "tools\validate_hedge_bounded_runtime_collections.py"
}

$MatrixPath = Join-Path $ArtifactDir "remediation800-validation.json"
Invoke-Checked "800-point audit remediation matrix" {
    & $Python "tools\validate_hedge_audit_remediation_800.py" `
        --project-root $ProjectRoot --output $MatrixPath
}

$BenchmarkPath = Join-Path $ArtifactDir "h01-h02-1500-regression.json"
Invoke-Checked "1500-cycle H01/H02 regression benchmark" {
    $env:PYTHONPATH = $ProjectRoot
    & $Python "tools\benchmark_hedge_audit_h01_1500.py" --output $BenchmarkPath
}

if (-not $SkipSmokeTests) {
    Invoke-Checked "Install focused test runner" {
        & $Python -m pip install --disable-pip-version-check --no-cache-dir "pytest==9.1.1"
    }
    Invoke-Checked "Audit remediation smoke tests" {
        $env:PYTHONPATH = $ProjectRoot
        & $Python -m pytest -q -o "addopts=" --confcutdir=tests/hedge `
            "tests/hedge/execution/test_audit_remediation_core.py"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " FREQTRADE-HEDGE V1.7 REMEDIATION800: PASS"
Write-Host "============================================================"
Write-Host ("Project  : " + $ProjectRoot)
Write-Host ("Python   : " + $Python)
Write-Host ("Matrix   : " + $MatrixPath)
Write-Host ("Benchmark: " + $BenchmarkPath)
Write-Host "Training : CPU-only"
