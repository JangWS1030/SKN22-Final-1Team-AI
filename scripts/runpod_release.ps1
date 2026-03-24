param(
    [string]$Image = "",
    [string]$ImageTag = "",
    [string]$ImageRepo = "",
    [int]$Timeout = 1200,
    [int]$PollInterval = 5,
    [switch]$SkipHealthCheck,
    [switch]$SkipVersionWait,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw ".env file not found: $Path"
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $parts = $trimmed -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Import-DotEnv -Path (Join-Path $repoRoot ".env")

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw "python or py executable not found."
}

$args = @("scripts/runpod_release.py", "--timeout", $Timeout.ToString(), "--poll-interval", $PollInterval.ToString())

if ($Image) {
    $args += @("--image", $Image)
}
if ($ImageTag) {
    $args += @("--image-tag", $ImageTag)
}
if ($ImageRepo) {
    $args += @("--image-repo", $ImageRepo)
}
if ($SkipHealthCheck) {
    $args += "--skip-health-check"
}
if ($SkipVersionWait) {
    $args += "--skip-version-wait"
}
if ($Force) {
    $args += "--force"
}
if ($DryRun) {
    $args += "--dry-run"
}

Write-Host ("[runpod_release] python command: {0} {1}" -f $python.Name, ($args -join " "))
& $python.Source @args
exit $LASTEXITCODE
