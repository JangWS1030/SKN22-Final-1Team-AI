param(
    [string]$Name = "run",
    [string]$LogDir = "",
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [string[]]$Arguments = @()
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $projectRoot "output\command_logs"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeName = ($Name -replace "[^a-zA-Z0-9._-]", "_")
$logPath = Join-Path $LogDir "${timestamp}_${safeName}.log"
$metaPath = Join-Path $LogDir "${timestamp}_${safeName}.json"
$latestLogPath = Join-Path $LogDir "latest.log"
$latestMetaPath = Join-Path $LogDir "latest.json"

$meta = [ordered]@{
    timestamp = (Get-Date).ToString("s")
    cwd = (Get-Location).Path
    executable = $Executable
    arguments = $Arguments
    name = $safeName
}
$meta | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $metaPath

Write-Host "[log] writing to $logPath"
Write-Host "[cmd] $Executable $($Arguments -join ' ')"

function Quote-Argument([string]$Value) {
    if ([string]::IsNullOrEmpty($Value)) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

$quotedExecutable = Quote-Argument $Executable
$quotedArguments = ($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
$cmdLine = if ([string]::IsNullOrWhiteSpace($quotedArguments)) {
    "$quotedExecutable 2>&1"
} else {
    "$quotedExecutable $quotedArguments 2>&1"
}

try {
    cmd /d /c $cmdLine | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
finally {
    if (-not (Test-Path $logPath)) {
        New-Item -ItemType File -Path $logPath | Out-Null
    }
}

Copy-Item -Force $logPath $latestLogPath

$latestMeta = [ordered]@{
    timestamp = (Get-Date).ToString("s")
    cwd = (Get-Location).Path
    executable = $Executable
    arguments = $Arguments
    name = $safeName
    exit_code = $exitCode
    log_path = $logPath
}
$latestMeta | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $latestMetaPath

Write-Host "[exit] $exitCode"
Write-Host "[latest] $latestLogPath"
exit $exitCode
