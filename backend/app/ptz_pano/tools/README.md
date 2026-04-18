# Tools

Command-line entrypoints for debugging modules independently.

Use scripts from the repository root for common tasks:

```powershell
.\scripts\start.ps1
.\scripts\build-panorama.ps1
.\scripts\check.ps1
```

Use individual Python modules when debugging a specific subsystem:

```powershell
python -m ptz_pano.tools.camera_status --config config/camera.local.json
python -m ptz_pano.tools.build_panorama --scan data/scans/test_wide_001
```
