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
  --lens-calibration config\lens_calibration.local.json `
  --stitch-quality quality `
  --projection sphere `
  --strategy max_weight
```

Omit `--lens-calibration` to rebuild with the raw frame images.

The default compositor uses OpenCV's high-level `Stitcher` in `PANORAMA` mode.
It runs OpenCV's own matching, camera estimation, seam, exposure, and blending
pipeline. The manifest records the selected OpenCV parameters and component
frames.

For large scans, the builder skips the high-level OpenCV stitcher and uses the
fast telemetry-based spherical compositor with weighted overlap handling. This
also acts as a fallback when OpenCV cannot estimate a panorama. The fallback
defaults to the fast weighted path. Use `--stitch-quality quality` to run local
feature alignment before the telemetry compositor and switch the fallback to
GraphCut seams, exposure compensation, and MultiBand blending. The manifest
records fallback timing diagnostics under `stitching.timings`.

The main viewer loads the latest `preview.jpg` and manifest from the scan
folder, so refreshing `http://localhost:8000/` is enough after rebuilding.
