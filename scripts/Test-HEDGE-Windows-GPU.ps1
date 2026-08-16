[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python"
}

Push-Location $projectRoot
try {
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Python dependency verification failed" }

    & $python -c "import torch; assert torch.__version__ == '2.13.0+cu130', torch.__version__; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0) == (12, 0); assert 'sm_120' in torch.cuda.get_arch_list(); x=torch.arange(1048576, device='cuda', dtype=torch.float32); x.square().mean(); torch.cuda.synchronize(); print('WINDOWS_CUDA_TENSOR=PASS')"
    if ($LASTEXITCODE -ne 0) { throw "Windows CUDA tensor verification failed" }

    & $python -m freqtrade.hedge.hprl device --device cuda
    if ($LASTEXITCODE -ne 0) { throw "HPRL CUDA device verification failed" }

    & $python -m freqtrade.hedge.hprl smoke --device cuda
    if ($LASTEXITCODE -ne 0) { throw "HPRL CUDA smoke failed" }

    & $python -m freqtrade.hedge.hprl train-smoke `
        --device cuda `
        --algorithm fast_td3 `
        --mixed-precision `
        --expected-updates 5 `
        --hardware-profile rtx5070_laptop
    if ($LASTEXITCODE -ne 0) { throw "HPRL Windows GPU train-smoke failed" }
} finally {
    Pop-Location
}

Write-Output "HEDGE_HPRL_WINDOWS_GPU_VERIFY=PASS"

