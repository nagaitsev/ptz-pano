# Storage

Filesystem contract for saved scans and project artifacts.

Owns:

- scan folder lookup
- loading and saving scan documents
- stable paths used by API, scan, stitching, and viewer

Main layout:

```text
data/scans/<scan-id>/
  scan.json
  frames/
  panorama/
```
