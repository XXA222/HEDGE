[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ConfigPath = "user_data\config_hedge.json",
    [string]$Strategy = "",
    [string]$Image = "freqtrade-hedge:1.7-risklevel-rl-cpu",
    [string]$ContainerName = "freqtrade-hedge-risklevel-rl",
    [switch]$RebuildImage,
    [switch]$SkipImageValidation,
    [switch]$SkipHostResourceBroker
)

$ErrorActionPreference = "Stop"

$DockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if (-not $DockerCommand) {
    throw "Docker CLI was not found. Install/start Docker Desktop first."
}
& docker version --format "{{.Server.Version}}" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop exists, but the Docker Engine is not running."
}

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$UserData = Join-Path $Root "user_data"
if (-not (Test-Path -LiteralPath $UserData -PathType Container)) {
    New-Item -ItemType Directory -Path $UserData -Force | Out-Null
}
$UserData = (Resolve-Path -LiteralPath $UserData).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $Root $ConfigPath)).Path

$UserDataPrefix = $UserData.TrimEnd('\') + '\'
if (-not $Config.StartsWith($UserDataPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The runtime config must be located under $UserData"
}

$Data = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($Data.dry_run -ne $true) {
    throw "Docker runtime is locked to dry_run=true in this release."
}
if (-not $Data.hedge -or $Data.hedge.read_only -ne $true) {
    throw "Docker runtime requires hedge.read_only=true."
}
if ($Data.hedge.live_trading_enabled -ne $false) {
    throw "Docker runtime requires hedge.live_trading_enabled=false."
}
if ($Data.hedge.operation_mode -notin @("paper", "readonly", "read_only")) {
    throw "Docker runtime requires paper/readonly operation_mode."
}

& docker image inspect $Image *> $null
$ImageExists = ($LASTEXITCODE -eq 0)
if ($RebuildImage -or -not $ImageExists) {
    Write-Host "Building Hedge Risk-Level RL image with CPU training policy..." -ForegroundColor Cyan
    & docker build `
        --build-arg "INSTALL_HEDGE_RISKLEVEL_RL=true" `
        --tag $Image `
        $Root
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build Docker runtime image: $Image"
    }
}
if (-not $SkipImageValidation) {
    & docker run --rm --entrypoint /opt/hedge-venv/bin/python $Image -c "import pathlib,torch,gymnasium,stable_baselines3,sb3_contrib,freqtrade; root=pathlib.Path('/opt/freqtrade-hedge').resolve(); module=pathlib.Path(freqtrade.__file__).resolve(); assert root in module.parents,(root,module); print('DOCKER_RISKLEVEL_RUNTIME_GATE: PASS')"
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Risk-Level RL image validation failed: $Image"
    }
}

$Existing = & docker ps -a --filter "name=^/$ContainerName$" --format "{{.ID}}"
if ($Existing) {
    & docker rm -f $ContainerName *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to remove the previous container: $ContainerName"
    }
}

$RelativeConfig = $Config.Substring($UserDataPrefix.Length).Replace('\', '/')
$ContainerConfig = "/opt/freqtrade-hedge/user_data/$RelativeConfig"
if (-not $SkipHostResourceBroker) {
    $Broker = Join-Path $Root "scripts\Start-Freqtrade-Hedge-Adaptive-Resource-Broker.ps1"
    if (Test-Path -LiteralPath $Broker -PathType Leaf) {
        $PowerShellExe = Join-Path $PSHOME "powershell.exe"
        if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) {
            $PowerShellExe = "powershell.exe"
        }
        & $PowerShellExe -NoProfile -ExecutionPolicy Bypass -File $Broker -UserDataRoot $UserData
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start the Windows host resource broker."
        }
    }
}
$Args = @(
    "run", "--detach", "--init",
    "--name", $ContainerName,
    "--security-opt", "no-new-privileges:true",
    "--env", "FT_APP_ENV=docker",
    "--env", "PYTHONUNBUFFERED=1",
    "--publish", "127.0.0.1:8080:8080",
    "--volume", "${UserData}:/opt/freqtrade-hedge/user_data",
    $Image,
    "freqtrade", "trade", "--config", $ContainerConfig
)
if ($Strategy) {
    $Args += @("--strategy", $Strategy)
}

& docker @Args
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start Docker runtime."
}

Write-Host "Hedge runtime started inside Docker: $ContainerName" -ForegroundColor Green
Write-Host "Image: $Image" -ForegroundColor Cyan
Write-Host "Config: $ContainerConfig" -ForegroundColor Cyan
Write-Host "Logs: docker logs -f $ContainerName" -ForegroundColor Cyan
