param(
    [switch]$RuntimeOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    python -m venv .venv
}

if ($RuntimeOnly) {
    & $Python -m pip install -e .
} else {
    & $Python -m pip install -e ".[dev]"
}

Write-Host "Environment ready."
Write-Host "Start the app with: .\scripts\start.ps1"
