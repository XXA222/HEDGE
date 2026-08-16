[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $HprlArguments
)

$ErrorActionPreference = "Stop"

$containerName = "HEDGE-HPRL-GPU"
$imageName = "freqtrade-hedge:hprl-gpu-20260812-222754"
$containerProjectRoot = "/opt/freqtrade-hedge"
$containerPython = "/opt/hedge-venv/bin/python"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is unavailable. Start Docker Desktop and retry."
}

& docker info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is unavailable. Start Docker Desktop and retry."
}

$existingId = (@(& docker container ls --all --quiet --filter "name=^/$containerName$") -join "").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Docker containers."
}

if (-not $existingId) {
    & docker image inspect $imageName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Required mature HPRL GPU image is missing: $imageName"
    }

    $mount = "${projectRoot}:${containerProjectRoot}"
    & docker run --detach --name $containerName --gpus all `
        --volume $mount `
        --workdir $containerProjectRoot `
        --entrypoint /bin/sh `
        $imageName `
        -c "trap 'exit 0' TERM INT; while true; do sleep 3600; done" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create $containerName from $imageName."
    }
} else {
    $actualImage = (@(& docker inspect --format "{{.Config.Image}}" $containerName) -join "").Trim()
    if ($LASTEXITCODE -ne 0 -or $actualImage -ne $imageName) {
        throw "Container $containerName does not use the required image ($imageName)."
    }

    $mountJson = (@(& docker inspect --format "{{json .Mounts}}" $containerName) -join "").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $mountJson) {
        throw "Unable to inspect $containerName mounts."
    }
    $projectMount = @($mountJson | ConvertFrom-Json | Where-Object Destination -eq $containerProjectRoot)
    if ($projectMount.Count -ne 1 -or -not $projectMount[0].Source) {
        throw "Container $containerName does not mount $containerProjectRoot."
    }
    $actualSource = [string] $projectMount[0].Source
    if ([System.IO.Path]::GetFullPath($actualSource).TrimEnd('\') -ne $projectRoot.TrimEnd('\')) {
        throw "Container $containerName is mounted from '$actualSource', expected '$projectRoot'."
    }

    $running = (@(& docker inspect --format "{{.State.Running}}" $containerName) -join "").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect $containerName state."
    }
    if ($running -ne "true") {
        & docker start $containerName | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to start $containerName."
        }
    }
}

if (-not $HprlArguments -or $HprlArguments.Count -eq 0) {
    $HprlArguments = @("device", "--device", "cuda")
}

& docker exec --workdir $containerProjectRoot $containerName `
    $containerPython -m freqtrade.hedge.hprl @HprlArguments
if ($LASTEXITCODE -ne 0) {
    throw "HPRL GPU command failed with exit code $LASTEXITCODE."
}
