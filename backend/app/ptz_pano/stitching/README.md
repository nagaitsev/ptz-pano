# Stitching Module

This package owns panorama assembly only. It reads an existing scan folder and
writes artifacts back into that scan, so the API and viewer can pick them up
without rerunning camera capture or scan planning.

## Inputs

- `data/scans/<scan-id>/scan.json`
- `data/scans/<scan-id>/frames/*.jpg`
- optional lens calibration JSON, for example `config/lens_calibration.local.json`

## Outputs

- `data/scans/<scan-id>/panorama/panorama.jpg`
- `data/scans/<scan-id>/panorama/preview.jpg`
- `data/scans/<scan-id>/panorama/panorama_manifest.json`

## Run Only Stitching

```powershell
.\.venv\Scripts\python.exe backend\app\ptz_pano\tools\build_panorama.py `
  --scan data\scans\test_wide_001 `
  --lens-calibration config\lens_calibration.local.json
```

Omit `--lens-calibration` to rebuild with the raw frame images.

The main viewer loads the latest `preview.jpg` and manifest from the scan
folder, so refreshing `http://localhost:8000/` is enough after rebuilding.
