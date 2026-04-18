# Processing Logic

The application has two main runtime modes: scanning and panorama operation.

## Scan Mode

Scan mode captures source material for a panorama.

```text
camera profile + capture source + scan settings
  -> scan planner
  -> camera move_absolute()
  -> settle
  -> read actual camera position
  -> capture frame
  -> save frame metadata
  -> completed scan folder
```

The scan runner writes progress into `scan.json` after each frame. Each frame
stores the actual position read back from the camera after settling. If a scan is
interrupted, the saved folder still contains useful partial data for inspection
and debugging.

The first supported scan path is:

```text
PTZOptics VISCA TCP 5678
  + RTSP stream 1
  + serpentine pan/tilt grid
```

## Panorama Build Mode

Panorama build mode consumes a completed scan. It should never move the camera.

```text
scan.json + frames + FOV table
  -> spherical projection
  -> equirectangular panorama
  -> overlap blending
  -> panorama artifacts
```

The core idea is to use PTZ metadata as geometry. Since each frame was captured
at a known `pan`, `tilt`, and `zoom`, the compositor can place images on the
sphere by angle rather than depending entirely on feature matching.

Feature matching can still be added later as a local refinement step, but it is
not the primary source of truth.

## Hotspot Operation Mode

After a panorama exists, the user can add points of interest.

```text
click on panorama point
  -> hotspot lookup
  -> camera pose
  -> move_absolute(pan, tilt, zoom)
```

A hotspot stores both visual position and camera-native command data. This keeps
the UI flexible while making camera movement deterministic.

## Calibration Logic

The FOV table maps zoom units to horizontal and vertical field of view.

```text
zoom
  -> interpolate nearest FOV samples
  -> hfov/vfov
  -> projection footprint for frame
```

The initial calibration path is manual:

1. Set a zoom value.
2. Center a visible target.
3. Move the target between known horizontal frame positions.
4. Measure the required pan delta.
5. Repeat vertically for tilt.
6. Save `zoom -> hfov/vfov`.

This can later be replaced or supplemented by OpenCV calibration.

## Debugging Rules

Each subsystem should remain testable without the rest of the app:

- Camera problems: use `camera_ping` and `camera_move`.
- Capture problems: use `capture_frame`.
- Scan route problems: use `plan_scan` before `run_scan`.
- Panorama problems: rebuild from an existing scan folder.
- Hotspot problems: inspect `hotspots.json` and issue a direct camera move.

This separation is intentional. It makes it possible to return to one subsystem
later without disturbing the rest of the project.
