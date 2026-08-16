[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "Start-HEDGE-HPRL-GPU.ps1"

& $launcher device --device cuda
& $launcher smoke --device cuda
& $launcher train-smoke `
    --device cuda `
    --algorithm fast_td3 `
    --mixed-precision `
    --expected-updates 3 `
    --hardware-profile rtx5070_laptop

Write-Output "HEDGE_HPRL_GPU_VERIFY=PASS"

