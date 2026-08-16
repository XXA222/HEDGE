[CmdletBinding()]
param(
    [string]$Container = "freqtrade-hedge-clean-v121-dryrun",
    [string]$Proxy = "http://host.docker.internal:7897",
    [string]$PytestTarget = "tests/hedge",
    [switch]$KeepTemporaryVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RuntimePython = "/opt/hedge-venv/bin/python"
$RuntimeSite = "/opt/hedge-venv/lib/python3.12/site-packages"
$TestVenv = "/tmp/freqtrade-hedge-pytest-venv"
$TestPython = $TestVenv + "/bin/python"
$TestSite = $TestVenv + "/lib/python3.12/site-packages"
$Pth = $TestSite + "/freqtrade-hedge-runtime.pth"

$Required = @(
    "pytest==9.1.1",
    "pytest-asyncio==1.4.0",
    "pytest-cov==7.1.0",
    "pytest-mock==3.15.1",
    "pytest-random-order==1.2.0",
    "pytest-timeout==2.4.0",
    "pytest-xdist==3.8.0"
)

$Running = (& docker inspect --format "{{.State.Running}}" $Container).Trim()
if ($Running -ne "true") {
    throw "The target container must already be running in maintenance/test mode."
}

$Before = (& docker exec $Container $RuntimePython -m pip freeze --all | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to snapshot /opt/hedge-venv before isolated pytest."
}

$PytestExitCode = $null
$PrimaryError = $null
try {
    & docker exec --user 0 $Container rm -rf $TestVenv
    if ($LASTEXITCODE -ne 0) { throw "Unable to clear temporary pytest venv." }

    & docker exec $Container $RuntimePython -m venv $TestVenv
    if ($LASTEXITCODE -ne 0) { throw "Unable to create temporary pytest venv." }

    $PthText = $RuntimeSite + "`n/opt/freqtrade-hedge`n"
    $PthText | & docker exec -i --user 0 $Container sh -c ("cat > " + $Pth)
    if ($LASTEXITCODE -ne 0) { throw "Unable to link runtime site-packages into test venv." }

    $InstallArgs = @(
        "exec", $Container, $TestPython, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-cache-dir",
        "--upgrade-strategy", "only-if-needed"
    )
    if (-not [string]::IsNullOrWhiteSpace($Proxy)) {
        $InstallArgs += @("--proxy", $Proxy)
    }
    $InstallArgs += $Required
    & docker @InstallArgs
    if ($LASTEXITCODE -ne 0) { throw "Unable to install isolated pytest tooling." }

    & docker exec `
        -e "PYTHONDONTWRITEBYTECODE=1" `
        -e "PYTHONHASHSEED=0" `
        --workdir "/opt/freqtrade-hedge" `
        $Container `
        $TestPython -m pytest -q -ra -p no:cacheprovider -o "addopts=" --tb=short $PytestTarget
    $PytestExitCode = $LASTEXITCODE
}
catch {
    $PrimaryError = $_
}
finally {
    # Runtime venv immutability gate must run even when pytest itself fails.
    try {
        $After = (& docker exec $Container $RuntimePython -m pip freeze --all | Out-String)
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to snapshot /opt/hedge-venv after isolated pytest."
        }
        if ($Before -ne $After) {
            throw "/opt/hedge-venv changed during isolated pytest; runtime venv contamination detected."
        }
        Write-Host "DOCKER_RUNTIME_VENV_IMMUTABLE: PASS" -ForegroundColor Green
    }
    catch {
        if ($null -eq $PrimaryError) {
            $PrimaryError = $_
        }
        else {
            Write-Error $_
        }
    }

    if (-not $KeepTemporaryVenv) {
        & docker exec --user 0 $Container rm -rf $TestVenv 2>$null | Out-Null
    }
}

if ($null -ne $PrimaryError) {
    throw $PrimaryError
}
if ($null -eq $PytestExitCode -or $PytestExitCode -ne 0) {
    throw ("Docker isolated pytest failed with ExitCode=" + $PytestExitCode)
}

Write-Host "DOCKER_ISOLATED_PYTEST: PASS" -ForegroundColor Green
