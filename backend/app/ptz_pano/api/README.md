# API

FastAPI boundary for the web viewer and calibration pages.

Owns:

- HTTP routes
- viewer HTML
- camera targeting endpoints
- lens-calibration capture endpoints

Does not own camera protocol details, scan planning, capture, or stitching math.

Run the full app from the repository root:

```powershell
.\scripts\start.ps1
```
