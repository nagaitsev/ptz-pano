# Tools

Command-line entrypoints for debugging modules independently.

Use scripts from the repository root for common tasks:

```powershell
.\scripts\start.ps1
.\scripts\build-panorama.ps1
.\scripts\build-portable.ps1 -Name PTZ-Pano-current -IncludeLocalConfig
.\scripts\check.ps1
```

Use individual Python modules when debugging a specific subsystem:

```powershell
python -m ptz_pano.tools.camera_status --config config/camera.local.json
python -m ptz_pano.tools.build_panorama --scan data/scans/test_wide_001
```

Portable bundles run through `ptz_pano.tools.portable_server`. The launcher is
responsible for creating the runtime layout, starting the local `MediaMTX`
gateway for WebRTC preview, and writing portable logs under `logs\`.
