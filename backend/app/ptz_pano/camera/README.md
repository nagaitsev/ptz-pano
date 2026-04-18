# Camera

PTZ command layer.

Owns:

- VISCA/IP transport
- camera profiles
- absolute movement, zoom, stop, home, status inquiries

Does not capture images or build panoramas.

Debug commands:

```powershell
python -m ptz_pano.tools.camera_status --config config/camera.local.json
python -m ptz_pano.tools.camera_move --config config/camera.local.json --home
```
