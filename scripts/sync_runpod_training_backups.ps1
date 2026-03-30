param(
    [string]$Remote = "runpod-hair",
    [string]$RemoteWorkRoot = "/workspace/hair_swap_generation",
    [string]$LocalBackupRoot = "",
    [int]$IntervalSeconds = 1200,
    [int]$FailureRetrySeconds = 120
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($LocalBackupRoot)) {
    $LocalBackupRoot = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "output\\runpod_training_backups"
}

$targets = @(
    @{ Name = "logs_longtail"; RemotePath = "$RemoteWorkRoot/logs_longtail" },
    @{ Name = "state_longtail"; RemotePath = "$RemoteWorkRoot/state_longtail" },
    @{ Name = "generation_lora_stage3_longtail_4090"; RemotePath = "$RemoteWorkRoot/output/training/generation_lora_stage3_longtail_4090" },
    @{ Name = "generation_lora_stage4_garment_reveal_4090"; RemotePath = "$RemoteWorkRoot/output/training/generation_lora_stage4_garment_reveal_4090" },
    @{ Name = "benchmarks"; RemotePath = "$RemoteWorkRoot/output/benchmarks" },
    @{ Name = "docs_longtail"; RemotePath = "$RemoteWorkRoot/output/docs_longtail" }
)

New-Item -ItemType Directory -Force -Path $LocalBackupRoot | Out-Null

function Test-RemotePathExists {
    param(
        [string]$RemoteName,
        [string]$RemotePath
    )

    try {
        $result = ssh $RemoteName "test -e '$RemotePath' && echo exists || true" 2>$null
        return $result -match "exists"
    } catch {
        return $false
    }
}

function Copy-RemoteTree {
    param(
        [string]$RemoteName,
        [string]$RemotePath,
        [string]$LocalDestination
    )

    $remoteSpec = "${RemoteName}:$RemotePath"
    scp -r $remoteSpec $LocalDestination | Out-Null
}

while ($true) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $latestRoot = Join-Path $LocalBackupRoot "latest"
    $stagingRoot = Join-Path $LocalBackupRoot "_staging_$timestamp"
    $success = $true

    try {
        New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null

        foreach ($target in $targets) {
            if (-not (Test-RemotePathExists -RemoteName $Remote -RemotePath $target.RemotePath)) {
                continue
            }

            Copy-RemoteTree -RemoteName $Remote -RemotePath $target.RemotePath -LocalDestination $stagingRoot
        }

        if (Test-Path $latestRoot) {
            Remove-Item -LiteralPath $latestRoot -Recurse -Force
        }
        Move-Item -LiteralPath $stagingRoot -Destination $latestRoot

        $latestMarker = Join-Path $LocalBackupRoot "latest_snapshot.txt"
        Set-Content -LiteralPath $latestMarker -Value $latestRoot -Encoding UTF8
    } catch {
        $success = $false
        $errorLog = Join-Path $LocalBackupRoot "backup_errors.log"
        "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $_.Exception.Message | Add-Content -LiteralPath $errorLog -Encoding UTF8
        if (Test-Path $stagingRoot) {
            Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if ($success) {
        Start-Sleep -Seconds $IntervalSeconds
    } else {
        Start-Sleep -Seconds $FailureRetrySeconds
    }
}
