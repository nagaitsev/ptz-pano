from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptz_pano.jsonio import write_json
from ptz_pano.storage.scan_repository import ScanRepository


@dataclass(frozen=True)
class PanoramaBuilder:
    repository: ScanRepository

    def build_manifest(self, scan_id: str) -> Path:
        document = self.repository.load_document(scan_id)
        missing_geometry = [
            frame.file
            for frame in document.frames
            if frame.hfov_deg is None or frame.vfov_deg is None
        ]
        status = "ready_for_compositor" if not missing_geometry else "missing_fov"
        output_path = self.repository.scan_path(scan_id) / "panorama" / "panorama_manifest.json"
        write_json(
            output_path,
            {
                "scan_id": document.id,
                "status": status,
                "frames": document.frames,
                "missing_geometry": missing_geometry,
                "next_step": "Implement OpenCV spherical remap and blending here.",
            },
        )
        return output_path
