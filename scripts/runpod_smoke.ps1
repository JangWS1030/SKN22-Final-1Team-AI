param(
    [switch]$HealthCheck,
    [string]$Image = ".\images\1234.jpg",
    [string]$Hairstyle = "wolf cut, layered bangs",
    [string]$Color = "ash brown",
    [int]$TopK = 1,
    [string]$OutputDir = ".\output\runpod_debug",
    [switch]$NoIntermediates
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

$args = @("tests/test_runpod.py")

if ($HealthCheck) {
    $args += "--health-check"
} else {
    $args += @(
        "--image", $Image,
        "--hairstyle", $Hairstyle,
        "--color", $Color,
        "--top-k", $TopK.ToString()
    )

    if (-not $NoIntermediates) {
        $args += @("--return-intermediates", "--output-dir", $OutputDir)
    }
}

Write-Host ("[runpod_smoke] python command: {0} {1}" -f $python.Name, ($args -join " "))
& $python.Source @args
exit $LASTEXITCODE
