# Calibration

Calibration data and interpolation.

Owns:

- zoom-to-FOV tables
- lens distortion calibration loading
- conversion helpers used by scan and stitching

Typical local files:

- `data/calibration/*.local.fov.json`
- `config/lens_calibration.local.json`

Lens calibration is optional. Stitching can run without it.
