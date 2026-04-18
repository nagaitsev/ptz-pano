# Architecture

The backend is organized around filesystem artifacts instead of process state.
Every scan is a self-contained folder that can be rescanned, rebuilt, archived,
or replayed without talking to the camera again.

```text
data/scans/<scan-id>/
  scan.json
  frames/
    frame_0001.jpg
  panorama/
    panorama_manifest.json
  hotspots.json
```

## Module Boundaries

`camera` owns PTZ commands only. It exposes `CameraController` and hides VISCA
transport details.

`capture` owns image acquisition only. It exposes `FrameCapture` and does not
know how the camera moves.

`scan` coordinates a controller and a capture source. It writes frame files and
scan metadata, but does not stitch images.

`calibration` maps camera zoom units to optical field of view. The panorama
builder consumes this data, but scan execution can run without it.

`panorama` consumes a completed scan folder. It never moves the camera.

`hotspots` maps panorama coordinates to camera-native PTZ commands.

`api` adapts those modules to HTTP/WebSocket for a future UI. Business logic
stays outside FastAPI route handlers.

## Development Order

1. Verify PTZOptics VISCA TCP commands with `camera_move`.
2. Verify image capture with `capture_frame`.
3. Run a small scan and inspect the saved `scan.json`.
4. Implement and iterate on `panorama` using saved scans only.
5. Add the frontend viewer and hotspot editor against stable scan artifacts.

