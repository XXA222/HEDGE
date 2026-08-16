[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python is missing: $python"
}

Push-Location $projectRoot
try {
    & $python -c "import pathlib, freqtrade, freqtrade.hedge.hprl as h; root=pathlib.Path.cwd().resolve(); ft=pathlib.Path(freqtrade.__file__).resolve(); hp=pathlib.Path(h.__file__).resolve(); assert root in ft.parents, ft; assert root in hp.parents, hp; print(f'freqtrade={ft}'); print(f'hprl={hp}'); print(f'hprl_api={h.HPRL_API_VERSION}')"
    if ($LASTEXITCODE -ne 0) { throw "Source authority verification failed" }

    & $python -m freqtrade.hedge.hprl compat --project-root $projectRoot
    if ($LASTEXITCODE -ne 0) { throw "HPRL compatibility verification failed" }

    & $python -m freqtrade.hedge.hprl inspect
    if ($LASTEXITCODE -ne 0) { throw "HPRL inspection failed" }

    & $python -m freqtrade --version
    if ($LASTEXITCODE -ne 0) { throw "Freqtrade startup verification failed" }
} finally {
    Pop-Location
}

Write-Output "HEDGE_HPRL_LOCAL_VERIFY=PASS"
