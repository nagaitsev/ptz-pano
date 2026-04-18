from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptz_pano.jsonio import write_json
from ptz_pano.panorama.simple_compositor import SimpleCompositor
from ptz_pano.storage.scan_repository import ScanRepository


@dataclass(frozen=True)
class PanoramaBuilder:
    repository: ScanRepository
    compositor: SimpleCompositor = SimpleCompositor()

    def build_manifest(self, scan_id: str) -> Path:
        document = self.repository.load_document(scan_id)
        missing_geometry = [
            frame.file
            for frame in document.frames
            if frame.hfov_deg is None or frame.vfov_deg is None
        ]
        status = "ready_for_compositor" if not missing_geometry else "missing_fov"
        output_path = self.repository.scan_path(scan_id) / "panorama" / "panorama_manifest.json"
        panorama_file = None
        if not missing_geometry:
            panorama_file = self.compositor.build(
                self.repository.scan_path(scan_id),
                document.frames,
                self.repository.scan_path(scan_id) / "panorama" / "panorama.jpg",
            )
        write_json(
            output_path,
            {
                "scan_id": document.id,
                "status": status,
                "panorama_file": None if panorama_file is None else str(panorama_file.name),
                "frames": document.frames,
                "missing_geometry": missing_geometry,
                "next_step": "Replace simple placement with spherical remap and multiband blending.",
            },
        )
        return output_path
