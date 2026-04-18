param(
    [string]$Scan = "data/scans/test_wide_001",
    [string]$LensCalibration = "config/lens_calibration.local.json",
    [ValidateSet("average", "max_weight")]
    [string]$Strategy = "max_weight",
    [ValidateSet("angular", "sphere")]
    [string]$Projection = "sphere",
    [switch]$NoLensCalibration
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}

$ArgsList = @(
    "backend\app\ptz_pano\tools\build_panorama.py",
    "--scan", $Scan,
    "--strategy", $Strategy,
    "--projection", $Projection
)

if (-not $NoLensCalibration -and (Test-Path (Join-Path $RepoRoot $LensCalibration))) {
    $ArgsList += @("--lens-calibration", $LensCalibration)
}

& $Python @ArgsList

Write-Host "Refresh the viewer to load the rebuilt panorama."
