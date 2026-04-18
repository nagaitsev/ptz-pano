# Scan

Scan planning and execution.

Owns:

- pan/tilt scan grid planning
- camera move/capture coordination
- writing `scan.json` and frame files

Does not stitch images. Stitching reads the completed scan folder later.

Debug commands:

```powershell
python -m ptz_pano.tools.plan_scan --config config/camera.local.json
python -m ptz_pano.tools.run_scan --config config/camera.local.json --scan-id scan_001
```
