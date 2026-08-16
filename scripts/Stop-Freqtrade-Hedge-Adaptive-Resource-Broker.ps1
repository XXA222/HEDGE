[CmdletBinding()]
param(
    [string]$UserDataRoot = "D:\Program Files\freqtradev2026.07-hedge-merge-docker\dry-run\research\mtf-2y\user_data"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RuntimeDir = Join-Path $UserDataRoot "runtime"
$PidPath = Join-Path $RuntimeDir "host-resource-broker.pid"
$StopPath = Join-Path $RuntimeDir "host-resource-broker.stop"
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
Set-Content -LiteralPath $StopPath -Value "stop" -Encoding ASCII

if (Test-Path -LiteralPath $PidPath -PathType Leaf) {
    $PidText = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    $BrokerPid = 0
    if ([int]::TryParse($PidText, [ref]$BrokerPid)) {
        for ($Index = 0; $Index -lt 20; $Index++) {
            if ($null -eq (Get-Process -Id $BrokerPid -ErrorAction SilentlyContinue)) {
                break
            }
            Start-Sleep -Milliseconds 250
        }
        if ($null -ne (Get-Process -Id $BrokerPid -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $BrokerPid -Force
        }
    }
}
Write-Host "Adaptive resource broker stopped."
