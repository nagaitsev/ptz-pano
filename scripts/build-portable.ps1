param(
    [string]$Name = "PTZ-Pano",
    [switch]$IncludeLocalConfig,
    [switch]$IncludeScans,
    [switch]$KeepBuildWork,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$DistRoot = Join-Path $RepoRoot "dist"
$BundlePath = Join-Path $DistRoot $Name
$BuildWorkPath = Join-Path $RepoRoot "build"
$PyInstallerDistRoot = Join-Path $env:TEMP "ptz-pano-pyinstaller-dist"
$PyInstallerBundlePath = Join-Path $PyInstallerDistRoot $Name
$OldBundleBackupPath = $null

Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}

if (-not $SkipInstall) {
    & $Python -m pip install -e ".[dev,build]"
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed with exit code $LASTEXITCODE."
    }
}

$separator = [IO.Path]::PathSeparator
$dataArgs = @(
    "--add-data", "backend/app/ptz_pano/api/viewer.html${separator}ptz_pano/api",
    "--add-data", "backend/app/ptz_pano/api/calibrate_lens.html${separator}ptz_pano/api"
)

if (Test-Path $PyInstallerBundlePath) {
    Remove-Item -Path $PyInstallerBundlePath -Recurse -Force
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name $Name `
    --distpath $PyInstallerDistRoot `
    --workpath $BuildWorkPath `
    --paths "backend/app" `
    --collect-submodules "ptz_pano" `
    --hidden-import "uvicorn.logging" `
    --hidden-import "uvicorn.loops.auto" `
    --hidden-import "uvicorn.protocols.http.auto" `
    --hidden-import "uvicorn.protocols.websockets.auto" `
    --hidden-import "uvicorn.lifespan.on" `
    @dataArgs `
    "backend/app/ptz_pano/tools/portable_server.py"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

if (Test-Path $BundlePath) {
    $backupSuffix = Get-Date -Format "yyyyMMdd_HHmmss"
    $OldBundleBackupPath = Join-Path $DistRoot "${Name}-previous-${backupSuffix}"
    Move-Item -Path $BundlePath -Destination $OldBundleBackupPath -Force
}
New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null
Copy-Item -Path $PyInstallerBundlePath -Destination $BundlePath -Recurse -Force

if (Test-Path (Join-Path $BundlePath "config")) {
    Remove-Item -Path (Join-Path $BundlePath "config") -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $BundlePath "config") | Out-Null
Copy-Item -Path "config/camera.example.json" -Destination (Join-Path $BundlePath "config/camera.example.json")
if ($IncludeLocalConfig -and (Test-Path "config/camera.local.json")) {
    Copy-Item -Path "config/camera.local.json" -Destination (Join-Path $BundlePath "config/camera.local.json")
}
if (Test-Path "config/lens_calibration.local.json") {
    Copy-Item -Path "config/lens_calibration.local.json" -Destination (Join-Path $BundlePath "config/lens_calibration.local.json")
}

if (Test-Path (Join-Path $BundlePath "data")) {
    Remove-Item -Path (Join-Path $BundlePath "data") -Recurse -Force
}
$calibrationPath = Join-Path $BundlePath "data/calibration"
$projectsPath = Join-Path $BundlePath "data/projects"
$scansPath = Join-Path $BundlePath "data/scans"
New-Item -ItemType Directory -Path $calibrationPath -Force | Out-Null
New-Item -ItemType Directory -Path $projectsPath -Force | Out-Null
New-Item -ItemType Directory -Path $scansPath -Force | Out-Null
Copy-Item -Path "data/calibration/*.json" -Destination $calibrationPath -ErrorAction SilentlyContinue
if ($IncludeScans) {
    Copy-Item -Path "data/scans/*" -Destination $scansPath -Recurse -Force -ErrorAction SilentlyContinue
}

$mediaMtxSourcePath = Join-Path $RepoRoot "tools\mediamtx\bin"
$mediaMtxBundlePath = Join-Path $BundlePath "tools\mediamtx"
if (Test-Path $mediaMtxBundlePath) {
    Remove-Item -Path $mediaMtxBundlePath -Recurse -Force
}
if (Test-Path $mediaMtxSourcePath) {
    New-Item -ItemType Directory -Path $mediaMtxBundlePath -Force | Out-Null
    Copy-Item -Path (Join-Path $mediaMtxSourcePath "*") -Destination $mediaMtxBundlePath -Recurse -Force
} else {
    Write-Warning "MediaMTX binaries were not found under tools\\mediamtx\\bin. Portable preview will fall back to JPEG."
}

$readmePath = Join-Path $BundlePath "README-portable.txt"
@"
PTZ Pano Portable

Start:
  $Name.exe

Then open:
  http://127.0.0.1:8000/

Camera settings:
  Edit the camera IP in the web UI settings panel, or edit config\camera.local.json.
  If config\camera.local.json is missing, the app creates it from config\camera.example.json.
  On startup the portable build rewrites RTSP host to the configured camera IP and starts a local MediaMTX gateway for WebRTC preview.

Command-line options:
  $Name.exe --port 8010
  $Name.exe --host 127.0.0.1 --no-browser
  $Name.exe --config config\camera.local.json

Runtime data:
  New scans are saved under data\scans.
  Calibration files are stored under data\calibration.

Build notes:
  Rebuild with -IncludeScans if you want this folder to include existing saved panoramas.
"@ | Set-Content -Path $readmePath -Encoding UTF8

if (-not $KeepBuildWork -and (Test-Path $BuildWorkPath)) {
    Remove-Item -Path $BuildWorkPath -Recurse -Force
}

if ($OldBundleBackupPath -and (Test-Path $OldBundleBackupPath)) {
    try {
        Remove-Item -Path $OldBundleBackupPath -Recurse -Force
    } catch {
        Write-Warning "Previous portable bundle was moved to $OldBundleBackupPath and could not be removed automatically."
    }
}

Write-Host "Portable bundle ready:"
Write-Host $BundlePath
Write-Host "Copy this whole folder to the target computer and run $Name.exe."
