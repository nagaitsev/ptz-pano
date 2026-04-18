# PTZ Pano

Modular application scaffold for scanning with a PTZ camera, building a spherical
panorama, and binding panorama hotspots back to camera pan/tilt/zoom positions.

The first target profile is PTZOptics over VISCA TCP. Sony VISCA over IP can be
added later as another camera profile without changing scan, capture, stitching,
or hotspot logic.

The package is deliberately split into independently debuggable modules:

- `camera`: PTZ command interface and camera-specific profiles.
- `capture`: RTSP and HTTP snapshot frame capture.
- `calibration`: zoom-to-FOV tables, lens calibration, and interpolation.
- `scan`: scan grid planning and scan execution.
- `stitching`: panorama assembly from saved scans; safe to iterate without camera access.
- `panorama`: compatibility import path for older code.
- `hotspots`: storage for panorama hotspots mapped to PTZ poses.
- `storage`: scan/project filesystem repositories.
- `api`: FastAPI boundary for the UI.
- `tools`: command-line debug entrypoints for each subsystem.

More detail:

- [Architecture](docs/architecture.md)
- [Modules](docs/modules.md)
- [Processing logic](docs/pipeline.md)

## Quick Start

Run once:

```powershell
.\scripts\setup.ps1
```

Then start the whole app:

```powershell
.\scripts\start.ps1
```

Open:

- `http://localhost:8000/`
- `http://10.1.1.80:8000/` from another device on the same network, when that interface is active
- `http://10.1.1.13:8000/` from another device on the same network, when that interface is active

Copy `config/camera.example.json` to `config/camera.local.json` and edit the
camera host, RTSP URL, and VISCA port if the local config does not exist yet.

## Common Tasks

Run checks:

```powershell
.\scripts\check.ps1
```

Rebuild the current saved panorama without moving the camera:

```powershell
.\scripts\build-panorama.ps1
```

Start with code reload while editing:

```powershell
.\scripts\start.ps1 -Reload
```

## Module Map

The working modules live under `backend/app/ptz_pano/`:

- `api/`: web app, viewer, calibration pages, HTTP endpoints.
- `camera/`: PTZ/VISCA control only.
- `capture/`: frame capture only.
- `calibration/`: FOV and lens calibration.
- `scan/`: scan planning and execution.
- `stitching/`: panorama assembly from saved scans.
- `panorama/`: compatibility path that re-exports stitching.
- `hotspots/`: saved panorama points mapped to PTZ poses.
- `storage/`: scan folder filesystem contract.
- `tools/`: command-line module debug tools.

Each module folder has its own `README.md`.

## Debug Commands

```powershell
python -m ptz_pano.tools.camera_ping --config config/camera.local.json
python -m ptz_pano.tools.plan_scan --config config/camera.local.json
python -m ptz_pano.tools.camera_raw --config config/camera.local.json --hex "81 01 06 01 18 14 03 03 FF"
python -m ptz_pano.tools.camera_status --config config/camera.local.json
python -m ptz_pano.tools.camera_move --config config/camera.local.json --home
python -m ptz_pano.tools.capture_frame --config config/camera.local.json --out data/test-frame.jpg
python -m ptz_pano.tools.run_scan --config config/camera.local.json --scan-id scan_001
python -m ptz_pano.tools.build_panorama --scan data/scans/scan_001 --lens-calibration config/lens_calibration.local.json --projection sphere --strategy max_weight
```

## Viewer

The simple startup path is:

```powershell
.\scripts\start.ps1
```

Open `http://localhost:8000/`. Use mouse wheel or pinch to zoom, drag to pan, and
press `Навести камеру` to move the camera to the center reticle. The viewer uses
the latest scan with a `preview.jpg`, or a specific scan with
`http://localhost:8000/?scan=<scan-id>`. The `k` slider temporarily scales the
target field of view before choosing camera zoom; lower values zoom in more.
The yaw/pitch inputs temporarily offset the reticle target in degrees.
The `Настройки` button opens scan/stitch controls for rebuilding the current
panorama or capturing a new scan and stitching it. In scan settings, optional
horizontal and vertical angle fields define the capture window in degrees around
the camera's current position; leave them empty to use `config/camera.local.json`.

## Current Scope

The scan and stitching paths now operate from saved scan folders. The stitching
module can rebuild `panorama.jpg`, `preview.jpg`, and `panorama_manifest.json`
without talking to the camera, and the viewer picks up those artifacts after a
page refresh.

## Repository Purpose

This repository is intended to be the source project for the PTZ panorama
application. Keep hardware-specific code behind camera/capture profiles and keep
saved scan folders as the contract between acquisition, panorama building, and
the future UI.
