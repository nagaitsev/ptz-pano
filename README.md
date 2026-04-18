# PTZ Pano

Modular application scaffold for scanning with a PTZ camera, building a spherical
panorama, and binding panorama hotspots back to camera pan/tilt/zoom positions.

The first target profile is PTZOptics over VISCA TCP. Sony VISCA over IP can be
added later as another camera profile without changing scan, capture, panorama,
or hotspot logic.

The package is deliberately split into independently debuggable modules:

- `camera`: PTZ command interface and camera-specific profiles.
- `capture`: RTSP and HTTP snapshot frame capture.
- `calibration`: zoom-to-FOV tables and interpolation.
- `scan`: scan grid planning and scan execution.
- `panorama`: panorama build pipeline entrypoint.
- `hotspots`: storage for panorama hotspots mapped to PTZ poses.
- `storage`: scan/project filesystem repositories.
- `api`: FastAPI boundary for the UI.
- `tools`: command-line debug entrypoints for each subsystem.

More detail:

- [Architecture](docs/architecture.md)
- [Modules](docs/modules.md)
- [Processing logic](docs/pipeline.md)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Debug Commands

Copy `config/camera.example.json` to `config/camera.local.json` and edit the
camera host, RTSP URL, and VISCA port.

```powershell
python -m ptz_pano.tools.camera_ping --config config/camera.local.json
python -m ptz_pano.tools.plan_scan --config config/camera.local.json
python -m ptz_pano.tools.camera_raw --config config/camera.local.json --hex "81 01 06 01 18 14 03 03 FF"
python -m ptz_pano.tools.camera_status --config config/camera.local.json
python -m ptz_pano.tools.camera_move --config config/camera.local.json --home
python -m ptz_pano.tools.capture_frame --config config/camera.local.json --out data/test-frame.jpg
python -m ptz_pano.tools.run_scan --config config/camera.local.json --scan-id scan_001
python -m ptz_pano.tools.build_panorama --scan data/scans/scan_001
```

## Viewer

After building a panorama, start the backend and open the viewer:

```powershell
$env:PTZ_PANO_CAMERA_CONFIG='config/camera.local.json'
$env:PTZ_PANO_TARGET_HFOV_SCALE='0.75'
uvicorn ptz_pano.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`. Use mouse wheel or pinch to zoom, drag to pan, and
press `Навести камеру` to move the camera to the center reticle. The viewer uses
the latest scan with a `preview.jpg`, or a specific scan with
`http://localhost:8000/?scan=<scan-id>`. The `k` slider temporarily scales the
target field of view before choosing camera zoom; lower values zoom in more.

## Current Scope

This is the initial backend skeleton. The VISCA TCP module can send basic raw
VISCA commands, while panorama composition is intentionally a replaceable
pipeline stub that writes a manifest. The next implementation step is to build
the spherical OpenCV compositor against saved scan folders.

## Repository Purpose

This repository is intended to be the source project for the PTZ panorama
application. Keep hardware-specific code behind camera/capture profiles and keep
saved scan folders as the contract between acquisition, panorama building, and
the future UI.
