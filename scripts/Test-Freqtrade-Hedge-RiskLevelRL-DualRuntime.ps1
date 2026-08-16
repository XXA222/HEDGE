[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$Image = "freqtrade-hedge:1.7-risklevel-rl-cpu",
    [switch]$SkipWindowsValidation,
    [switch]$SkipDockerValidation,
    [switch]$SkipDockerBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}
else {
    $ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw ("ProjectRoot does not exist: " + $ProjectRoot)
}

$Downloads = Join-Path $env:USERPROFILE "Downloads"
if (-not (Test-Path -LiteralPath $Downloads -PathType Container)) {
    $Downloads = $ProjectRoot
}
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$AuditRoot = Join-Path $Downloads ("Freqtrade-Hedge-RiskLevelRL-V1.7-DualRuntime-" + $Timestamp)
New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
$Results = New-Object System.Collections.ArrayList

function Add-Result {
    param([string]$Name, [string]$Target, [int]$ExitCode, [string]$Detail)
    [void]$Results.Add([ordered]@{
        Name = $Name
        Target = $Target
        ExitCode = $ExitCode
        Status = $(if ($ExitCode -eq 0) { "PASS" } else { "FAIL" })
        Detail = $Detail
    })
}

function Invoke-NativeStep {
    param(
        [string]$Name,
        [string]$Target,
        [string]$Command,
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )
    Write-Host ""
    Write-Host ("=== " + $Target + " :: " + $Name + " ===") -ForegroundColor Cyan
    $Pushed = $false
    try {
        if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
            Push-Location -LiteralPath $WorkingDirectory
            $Pushed = $true
        }
        & $Command @Arguments
        $Code = $LASTEXITCODE
        if ($null -eq $Code) { $Code = 0 }
    }
    catch {
        $Code = 1
        Write-Warning $_.Exception.Message
    }
    finally {
        if ($Pushed) { Pop-Location }
    }
    Add-Result -Name $Name -Target $Target -ExitCode $Code -Detail (($Arguments -join " "))
}

if (-not $SkipWindowsValidation) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        Add-Result -Name "project-local-python" -Target "Windows" -ExitCode 1 -Detail $Python
    }
    else {
        Invoke-NativeStep -Name "source-authority" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "-c",
            "import pathlib,freqtrade; root=pathlib.Path.cwd().resolve(); module=pathlib.Path(freqtrade.__file__).resolve(); assert root in module.parents,(root,module); print('WINDOWS_SOURCE_AUTHORITY: PASS')"
        )
        Invoke-NativeStep -Name "risklevel-dependencies" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "-c",
            "import importlib.metadata as m,torch,gymnasium,stable_baselines3,sb3_contrib; assert m.version('tqdm') == '4.69.0'; print('torch='+str(torch.__version__)); print('cuda_available='+str(torch.cuda.is_available())); print('WINDOWS_RISKLEVEL_DEPS: PASS')"
        )
        Invoke-NativeStep -Name "dual-runtime-400" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_risklevel_rl_dual_runtime_400.py",
            "--output", (Join-Path $AuditRoot "windows-dual-runtime-400.json")
        )
        Invoke-NativeStep -Name "action-reward-200" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_risk_action_reward_200.py",
            "--output", (Join-Path $AuditRoot "windows-action-reward-200.json")
        )
        Invoke-NativeStep -Name "risklevel-400" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_risk_level_rl_v1_400.py",
            "--report", (Join-Path $AuditRoot "windows-risklevel-400.json")
        )
        Invoke-NativeStep -Name "risk-memory-400" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_risk_level_rl_v2_memory_400.py",
            "--output", (Join-Path $AuditRoot "windows-risk-memory-400.json")
        )
        Invoke-NativeStep -Name "integration-400" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_risklevel_rl_integration_400.py",
            "--output", (Join-Path $AuditRoot "windows-integration-400.json")
        )
        Invoke-NativeStep -Name "adaptive-cpu-400" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_cpu_adaptive_400.py",
            "--output", (Join-Path $AuditRoot "windows-adaptive-cpu-400.json")
        )
        Invoke-NativeStep -Name "real-sb3-cpu-smoke" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "tools\validate_hedge_risklevel_sb3_cpu_smoke.py"
        )
        Invoke-NativeStep -Name "focused-pytest" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @(
            "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
            "--confcutdir=tests\hedge\mlrl",
            "tests\hedge\mlrl\test_risk_action_reward_contract.py",
            "tests\hedge\mlrl\test_risk_level_rl.py",
            "tests\hedge\mlrl\test_risk_level_rl_memory.py",
            "tests\hedge\mlrl\test_risk_level_rl_mainline_integration.py"
        )
        Invoke-NativeStep -Name "pip-check" -Target "Windows" -Command $Python -WorkingDirectory $ProjectRoot -Arguments @("-m", "pip", "check")
    }

    Write-Host ""
    Write-Host "=== Windows :: PowerShell 5.1 AST ===" -ForegroundColor Cyan
    $AstFailures = @()
    Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Filter "*.ps1" |
        Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\user_data\*" } |
        ForEach-Object {
            $Tokens = $null
            $Errors = $null
            [void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$Tokens, [ref]$Errors)
            if ($Errors.Count -gt 0) {
                $AstFailures += $_.FullName
                $Errors | ForEach-Object { Write-Warning $_.Message }
            }
        }
    Add-Result -Name "powershell-5.1-ast" -Target "Windows" -ExitCode $(if ($AstFailures.Count -eq 0) { 0 } else { 1 }) -Detail ("failures=" + $AstFailures.Count)
}

if (-not $SkipDockerValidation) {
    $Docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -eq $Docker) {
        Add-Result -Name "docker-cli" -Target "Docker" -ExitCode 1 -Detail "docker command not found"
    }
    else {
        Invoke-NativeStep -Name "docker-engine" -Target "Docker" -Command "docker" -Arguments @("version", "--format", "{{.Server.Version}}")
        if (-not $SkipDockerBuild) {
            Invoke-NativeStep -Name "build-current-source" -Target "Docker" -Command "docker" -WorkingDirectory $ProjectRoot -Arguments @(
                "build", "--build-arg", "INSTALL_HEDGE_RISKLEVEL_RL=true", "--tag", $Image, "."
            )
        }
        Invoke-NativeStep -Name "compose-source-authority" -Target "Docker" -Command "docker" -WorkingDirectory $ProjectRoot -Arguments @("compose", "config", "--quiet")
        Invoke-NativeStep -Name "image-runtime-gate" -Target "Docker" -Command "docker" -Arguments @(
            "run", "--rm", "--entrypoint", "/opt/hedge-venv/bin/python", $Image,
            "-c", "import pathlib,torch,gymnasium,stable_baselines3,sb3_contrib,freqtrade; root=pathlib.Path('/opt/freqtrade-hedge').resolve(); module=pathlib.Path(freqtrade.__file__).resolve(); assert root in module.parents,(root,module); print('DOCKER_RISKLEVEL_RUNTIME: PASS')"
        )
        foreach ($Spec in @(
            @("real-sb3-cpu-smoke", "tools/validate_hedge_risklevel_sb3_cpu_smoke.py"),
            @("dual-runtime-400", "tools/validate_hedge_risklevel_rl_dual_runtime_400.py"),
            @("action-reward-200", "tools/validate_hedge_risk_action_reward_200.py"),
            @("risk-memory-400", "tools/validate_hedge_risk_level_rl_v2_memory_400.py"),
            @("integration-400", "tools/validate_hedge_risklevel_rl_integration_400.py"),
            @("adaptive-cpu-400", "tools/validate_hedge_cpu_adaptive_400.py")
        )) {
            Invoke-NativeStep -Name $Spec[0] -Target "Docker" -Command "docker" -Arguments @(
                "run", "--rm", "--entrypoint", "/opt/hedge-venv/bin/python", $Image, $Spec[1]
            )
        }
    }
}

$Failures = @($Results | Where-Object { $_.ExitCode -ne 0 })
$Summary = [ordered]@{
    Schema = "freqtrade-hedge-risklevel-rl-v1.7-dual-runtime-windows-docker"
    ProjectRoot = $ProjectRoot
    Image = $Image
    Timestamp = $Timestamp
    Status = $(if ($Failures.Count -eq 0) { "PASS" } else { "FAIL" })
    FailedSteps = $Failures.Count
    Results = $Results
}
$SummaryPath = Join-Path $AuditRoot "SUMMARY.json"
$Encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($SummaryPath, (($Summary | ConvertTo-Json -Depth 8) + [Environment]::NewLine), $Encoding)
Write-Host ""
Write-Host ("Dual-runtime summary: " + $SummaryPath) -ForegroundColor Cyan
if ($Failures.Count -eq 0) {
    Write-Host "FREQTRADE-HEDGE RISK-LEVEL RL DUAL-RUNTIME GATE: PASS" -ForegroundColor Green
    exit 0
}
Write-Host "FREQTRADE-HEDGE RISK-LEVEL RL DUAL-RUNTIME GATE: FAIL" -ForegroundColor Red
exit 1
