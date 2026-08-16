[CmdletBinding()]
param(
    [string]$CredentialPath = '',
    [string]$OutputPath = '',
    [ValidateRange(5, 120)]
    [int]$TimeoutSeconds = 15
)

$ErrorActionPreference = 'Stop'

if (-not $CredentialPath) {
    $folderName = ([char]0x542F) + ([char]0x52A8) + 'freqtrade'
    $fileName = 'binance - ' + ([char]0x526F) + ([char]0x672C) + '.txt'
    $CredentialPath = Join-Path (Join-Path $env:USERPROFILE 'Desktop') (Join-Path $folderName (Join-Path 'api' $fileName))
}

function Get-HmacSha256Hex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Query,
        [Parameter(Mandatory = $true)]
        [string]$Secret
    )

    $hmac = [System.Security.Cryptography.HMACSHA256]::new(
        [Text.Encoding]::UTF8.GetBytes($Secret)
    )
    try {
        return ([BitConverter]::ToString(
                $hmac.ComputeHash([Text.Encoding]::UTF8.GetBytes($Query))
            ) -replace '-', '').ToLowerInvariant()
    }
    finally {
        $hmac.Dispose()
    }
}

function Get-ErrorSummary {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)

    $status = $null
    $code = 'unknown'
    if ($null -ne $ErrorRecord.Exception.Response) {
        try { $status = [int]$ErrorRecord.Exception.Response.StatusCode } catch {}
        try {
            $reader = [IO.StreamReader]::new($ErrorRecord.Exception.Response.GetResponseStream())
            try {
                $body = $reader.ReadToEnd()
                $parsed = $body | ConvertFrom-Json
                if ($null -ne $parsed.code) { $code = [string]$parsed.code }
            }
            finally { $reader.Dispose() }
        }
        catch {}
    }
    return [pscustomobject]@{ status = $status; code = $code }
}

function Invoke-PublicRead {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Uri
    )

    $started = [Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-WebRequest -Uri $Uri -Method Get -TimeoutSec $TimeoutSeconds -UseBasicParsing
        $started.Stop()
        $data = $response.Content | ConvertFrom-Json
        return [pscustomobject]@{
            name = $Name
            kind = 'public'
            status = 'PASS'
            http_status = [int]$response.StatusCode
            elapsed_ms = $started.ElapsedMilliseconds
            server_time_ms = [int64]$data.serverTime
        }
    }
    catch {
        $started.Stop()
        $summary = Get-ErrorSummary $_
        return [pscustomobject]@{
            name = $Name
            kind = 'public'
            status = 'FAIL'
            http_status = $summary.status
            error_code = $summary.code
            elapsed_ms = $started.ElapsedMilliseconds
        }
    }
}

function Invoke-SignedRead {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$BaseUri,
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$ApiKey,
        [Parameter(Mandatory = $true)]
        [string]$ApiSecret
    )

    $started = [Diagnostics.Stopwatch]::StartNew()
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $query = 'timestamp=' + $timestamp + '&recvWindow=10000'
    $signature = Get-HmacSha256Hex -Query $query -Secret $ApiSecret
    $uri = $BaseUri + $Path + '?' + $query + '&signature=' + $signature
    try {
        $response = Invoke-WebRequest `
            -Uri $uri `
            -Method Get `
            -Headers @{ 'X-MBX-APIKEY' = $ApiKey } `
            -TimeoutSec $TimeoutSeconds `
            -UseBasicParsing
        $started.Stop()
        $data = $response.Content | ConvertFrom-Json
        $extra = [ordered]@{}
        if ($Name -eq 'api_restrictions') {
            $flags = @(
                'enableReading',
                'enableFutures',
                'enableWithdrawals',
                'enableInternalTransfer',
                'permitsUniversalTransfer',
                'enableSpotAndMarginTrading'
            )
            $extra.flags_present = @($flags | Where-Object { $null -ne $data.$_ })
            $permissionFlags = [ordered]@{}
            foreach ($flag in $flags) {
                $permissionFlags[$flag] = if ($null -eq $data.$flag) { $null } else { [bool]$data.$flag }
            }
            $extra.permission_flags = [pscustomobject]$permissionFlags
        }
        if ($Name -eq 'futures_account') {
            $extra.asset_count = @($data.assets).Count
            $extra.position_count = @($data.positions).Count
        }
        return [pscustomobject]@{
            name = $Name
            kind = 'signed_read_only'
            status = 'PASS'
            http_status = [int]$response.StatusCode
            elapsed_ms = $started.ElapsedMilliseconds
            details = [pscustomobject]$extra
        }
    }
    catch {
        $started.Stop()
        $summary = Get-ErrorSummary $_
        return [pscustomobject]@{
            name = $Name
            kind = 'signed_read_only'
            status = 'FAIL'
            http_status = $summary.status
            error_code = $summary.code
            elapsed_ms = $started.ElapsedMilliseconds
        }
    }
}

$credentialFile = Get-Item -LiteralPath $CredentialPath -ErrorAction Stop
$credentialLines = @(
    Get-Content -LiteralPath $credentialFile.FullName |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($credentialLines.Count -ne 2 -or $credentialLines[0].Length -ne 64 -or $credentialLines[1].Length -ne 64) {
    throw 'Binance credential file must contain exactly two non-empty 64-character lines.'
}

$apiKey = $credentialLines[0]
$apiSecret = $credentialLines[1]
$checks = @(
    (Invoke-PublicRead -Name 'spot_time' -Uri 'https://api.binance.com/api/v3/time'),
    (Invoke-PublicRead -Name 'futures_time' -Uri 'https://fapi.binance.com/fapi/v1/time'),
    (Invoke-PublicRead -Name 'futures_testnet_time' -Uri 'https://testnet.binancefuture.com/fapi/v1/time'),
    (Invoke-SignedRead -Name 'api_restrictions' -BaseUri 'https://api.binance.com' -Path '/sapi/v1/account/apiRestrictions' -ApiKey $apiKey -ApiSecret $apiSecret),
    (Invoke-SignedRead -Name 'futures_account' -BaseUri 'https://fapi.binance.com' -Path '/fapi/v2/account' -ApiKey $apiKey -ApiSecret $apiSecret)
)
$restrictionCheck = $checks | Where-Object { $_.name -eq 'api_restrictions' } | Select-Object -First 1
$accountCheck = $checks | Where-Object { $_.name -eq 'futures_account' } | Select-Object -First 1
$permissionFlags = if ($restrictionCheck -and $restrictionCheck.details) {
    $restrictionCheck.details.permission_flags
}
else { $null }
$readOnlyPermissionPass = (
    $restrictionCheck.status -eq 'PASS' -and
    $accountCheck.status -eq 'PASS' -and
    $permissionFlags.enableReading -eq $true -and
    $permissionFlags.enableWithdrawals -eq $false -and
    $permissionFlags.enableInternalTransfer -eq $false -and
    $permissionFlags.permitsUniversalTransfer -eq $false -and
    $permissionFlags.enableSpotAndMarginTrading -eq $false
)

$report = [ordered]@{
    schema = 'freqtrade-hedge-binance-readonly-network-v1'
    generated_at = [DateTimeOffset]::UtcNow.ToString('o')
    status = if (@($checks | Where-Object { $_.status -ne 'PASS' }).Count -eq 0 -and $readOnlyPermissionPass) { 'PASS' } else { 'FAIL' }
    exchange_writes = 'LOCKED_BY_SCOPE_AND_ENDPOINTS'
    read_only_permission_policy = if ($readOnlyPermissionPass) { 'PASS' } else { 'FAIL' }
    credential_path = $credentialFile.FullName
    checks = $checks
}
$rendered = $report | ConvertTo-Json -Depth 8
Write-Output $rendered
if ($OutputPath) {
    $destination = [IO.Path]::GetFullPath($OutputPath)
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Set-Content -LiteralPath $destination -Value ($rendered + [Environment]::NewLine) -Encoding UTF8
}
if ($report.status -ne 'PASS') { exit 1 }
