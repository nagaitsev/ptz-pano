# Modules

This project is split into small modules that can be debugged independently.
Each module owns one concern and communicates through explicit Python interfaces
or saved scan artifacts.

For normal operation, use the root scripts:

```powershell
.\scripts\setup.ps1
.\scripts\start.ps1
.\scripts\build-panorama.ps1
.\scripts\check.ps1
```

For module work, open the module folder under `backend/app/ptz_pano/`; each
folder has a short `README.md`.

## `camera`

Owns PTZ control only. It does not capture frames, plan scans, or build
panoramas.

Primary interface:

```python
class CameraController:
    def home(self) -> None: ...
    def stop(self) -> None: ...
    def move_absolute(self, pose: CameraPose) -> None: ...
    def set_zoom(self, zoom: int) -> None: ...
    def close(self) -> None: ...
```

Current implementation:

- `PtzOpticsViscaTcpController`: PTZOptics VISCA over TCP, default port `5678`.

Debug tools:

- `python -m ptz_pano.tools.camera_ping`
- `python -m ptz_pano.tools.camera_move`
- `python -m ptz_pano.tools.camera_raw`
- `python -m ptz_pano.tools.camera_status`

## `capture`

Owns frame acquisition only. It does not move the camera.

Primary interface:

```python
class FrameCapture:
    def grab_frame(self, output_path: Path) -> Path: ...
```

Current implementations:

- `RtspCapture`: reads one frame from RTSP through OpenCV.
- `SnapshotCapture`: downloads one JPEG from an HTTP snapshot URL.

Debug tool:

- `python -m ptz_pano.tools.capture_frame`

## `calibration`

Owns the mapping from camera zoom units to optical field of view.

The panorama builder needs this mapping to place each captured frame on a
spherical panorama without relying only on feature matching.

When a scan config contains `calibration.fov_table`, `run_scan` stores
interpolated `hfov_deg` and `vfov_deg` in every frame entry.

When `calibration.pan_units_per_degree` and
`calibration.tilt_units_per_degree` are configured, `run_scan` also stores
derived `yaw_deg` and `pitch_deg` in every frame pose. These values are used by
the panorama compositor instead of guessing the conversion from VISCA units.

Current model:

```json
{
  "samples": [
    { "zoom": 0, "hfov_deg": 60.0, "vfov_deg": 35.0 },
    { "zoom": 1000, "hfov_deg": 40.0, "vfov_deg": 23.0 }
  ]
}
```

Debug tool:

- `python -m ptz_pano.tools.calibrate_fov`

## `scan`

Owns scan planning and execution. It coordinates a camera controller and a frame
capture source, but it does not know VISCA packet details or panorama math.

The planner creates a serpentine grid so adjacent frames are captured with less
unnecessary camera travel.

Debug tools:

- `python -m ptz_pano.tools.plan_scan`
- `python -m ptz_pano.tools.run_scan`

## `storage`

Owns filesystem layout for scan artifacts.

Every scan is saved as a self-contained folder:

```text
data/scans/<scan-id>/
  scan.json
  frames/
    frame_0001.jpg
  panorama/
    panorama_manifest.json
  hotspots.json
```

This lets the panorama builder, UI, and tests work from saved data without
touching the real camera.

## `stitching`

Owns panorama generation from a saved scan folder. It never moves the camera and
can be debugged using only `scan.json` plus saved frame images.

Current implementation:

1. Load `scan.json`.
2. Optionally undistort frames from lens calibration samples.
3. Estimate feature alignment between neighboring frames.
4. Project frame pixels onto the panorama sphere.
5. Remap into an equirectangular image.
6. Blend overlaps with feathered weights.
7. Save `panorama.jpg`, `preview.jpg`, and `panorama_manifest.json`.

Debug tool:

- `python -m ptz_pano.tools.build_panorama`

The historical `panorama` package remains as a compatibility import path and
re-exports the current stitching classes.

## `hotspots`

Owns panorama points that map back to camera commands.

Each hotspot stores both panorama coordinates and native camera pose:

```json
{
  "id": "entrance",
  "title": "Entrance",
  "panorama_yaw_deg": 42.5,
  "panorama_pitch_deg": -3.0,
  "pose": {
    "pan": 1860,
    "tilt": -120,
    "zoom": 2400
  }
}
```

The UI can render the hotspot on the sphere, while the backend can move the
camera to the exact saved PTZ pose.

## `api`

Owns HTTP/WebSocket adapters for the future frontend. Business logic should stay
in the modules above, not inside route handlers.

Current endpoint:

- `GET /health`
- `GET /scans/{scan_id}`
