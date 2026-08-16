[CmdletBinding()]
param(
    [ValidateRange(1, 16)]
    [int] $Segments = 8
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$wheelName = "torch-2.13.0+cu130-cp312-cp312-win_amd64.whl"
$wheelUrl = "https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-win_amd64.whl"
$wheelLength = [int64] 1915519202
$wheelSha256 = "2efab1e83604ca628c6d85b9e188c153690980498d1297081a9dad704919303c"
$cacheRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "HEDGE\wheel-cache"))
$expectedCacheRoot = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "HEDGE\wheel-cache"))
$partsRoot = Join-Path $cacheRoot "$wheelName.parts"
$wheelPath = Join-Path $cacheRoot $wheelName

if ($cacheRoot -ne $expectedCacheRoot -or -not $cacheRoot.StartsWith([System.IO.Path]::GetFullPath($env:LOCALAPPDATA), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unexpected wheel cache path: $cacheRoot"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python"
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "Windows curl.exe is unavailable."
}

New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
New-Item -ItemType Directory -Path $partsRoot -Force | Out-Null

$wheelIsValid = $false
if (Test-Path -LiteralPath $wheelPath -PathType Leaf) {
    $existing = Get-Item -LiteralPath $wheelPath
    if ($existing.Length -eq $wheelLength) {
        $existingHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $wheelIsValid = $existingHash -eq $wheelSha256
    }
}

if (-not $wheelIsValid) {
    $ranges = for ($index = 0; $index -lt $Segments; $index++) {
        $start = [int64] [Math]::Floor($wheelLength * $index / $Segments)
        $end = [int64] [Math]::Floor($wheelLength * ($index + 1) / $Segments) - 1
        [pscustomobject]@{
            Index = $index
            Start = $start
            End = $end
            Length = $end - $start + 1
            Path = Join-Path $partsRoot ("part-{0:D2}.bin" -f $index)
        }
    }

    $jobs = foreach ($range in $ranges) {
        $existingPartLength = if (Test-Path -LiteralPath $range.Path -PathType Leaf) {
            (Get-Item -LiteralPath $range.Path).Length
        } else {
            -1
        }
        if ($existingPartLength -eq $range.Length) {
            Write-Host "Reusing completed segment $($range.Index + 1)/$Segments"
            continue
        }

        Start-Job -ArgumentList $wheelUrl, $range.Start, $range.End, $range.Path, $range.Index, $Segments -ScriptBlock {
            param($url, $start, $end, $path, $index, $count)
            & curl.exe --fail --location --silent --show-error `
                --retry 12 --retry-delay 2 --retry-all-errors --connect-timeout 30 `
                --range "$start-$end" --output $path $url
            if ($LASTEXITCODE -ne 0) {
                throw "Download failed for segment $($index + 1)/$count with exit code $LASTEXITCODE"
            }
            "Downloaded segment $($index + 1)/$count"
        }
    }

    if ($jobs) {
        $jobs | Receive-Job -Wait -AutoRemoveJob
    }

    foreach ($range in $ranges) {
        if (-not (Test-Path -LiteralPath $range.Path -PathType Leaf)) {
            throw "Downloaded segment is missing: $($range.Path)"
        }
        $actualLength = (Get-Item -LiteralPath $range.Path).Length
        if ($actualLength -ne $range.Length) {
            throw "Segment $($range.Index + 1) has $actualLength bytes; expected $($range.Length)."
        }
    }

    $output = [System.IO.File]::Open($wheelPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        foreach ($range in $ranges) {
            $input = [System.IO.File]::OpenRead($range.Path)
            try {
                $input.CopyTo($output)
            } finally {
                $input.Dispose()
            }
        }
        $output.Flush($true)
    } finally {
        $output.Dispose()
    }

    $actualWheelLength = (Get-Item -LiteralPath $wheelPath).Length
    if ($actualWheelLength -ne $wheelLength) {
        throw "Merged wheel has $actualWheelLength bytes; expected $wheelLength."
    }
    $actualHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $wheelSha256) {
        throw "Wheel SHA-256 mismatch: $actualHash"
    }
    Write-Output "PYTORCH_WHEEL_SHA256=PASS"
}

& $python -m pip install --no-deps --upgrade --force-reinstall $wheelPath
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch CUDA wheel installation failed with exit code $LASTEXITCODE."
}

& $python -c "import torch; assert torch.__version__ == '2.13.0+cu130', torch.__version__; assert torch.cuda.is_available(); assert 'sm_120' in torch.cuda.get_arch_list(); print('WINDOWS_TORCH_CUDA_INSTALL=PASS'); print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
if ($LASTEXITCODE -ne 0) {
    throw "Installed Windows PyTorch CUDA runtime failed verification."
}
