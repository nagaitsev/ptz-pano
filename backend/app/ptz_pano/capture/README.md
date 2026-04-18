# Capture

Frame acquisition layer.

Owns:

- RTSP single-frame capture
- HTTP snapshot capture

Does not move the camera or decide where to scan.

Debug command:

```powershell
python -m ptz_pano.tools.capture_frame --config config/camera.local.json --out data/test-frame.jpg
```
