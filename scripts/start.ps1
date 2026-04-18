param(
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8000,
    [string]$Config = "config/camera.local.json",
    [string]$TargetHfovScale = "0.45",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $RepoRoot $Config

Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}

if (-not (Test-Path $ConfigPath)) {
    throw "Missing $Config. Copy config/camera.example.json to config/camera.local.json and edit it."
}

$env:PYTHONPATH = Join-Path $RepoRoot "backend\app"
$env:PTZ_PANO_CAMERA_CONFIG = $Config
$env:PTZ_PANO_TARGET_HFOV_SCALE = $TargetHfovScale

$ArgsList = @(
    "-m", "uvicorn",
    "ptz_pano.api.main:app",
    "--host", $HostName,
    "--port", "$Port"
)

if ($Reload) {
    $ArgsList += @("--reload", "--reload-dir", "backend/app")
}

Write-Host "PTZ Pano backend starting..."
Write-Host "Local:   http://localhost:$Port/"
Write-Host "Network: http://10.1.1.80:$Port/ or http://10.1.1.13:$Port/ if those interfaces are active"
Write-Host "Config:  $Config"

& $Python @ArgsList
