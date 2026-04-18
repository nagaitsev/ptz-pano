from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptz_pano.jsonio import write_json
from ptz_pano.models import to_jsonable
from ptz_pano.panorama.alignment import FeatureAligner
from ptz_pano.panorama.simple_compositor import SimpleCompositor
from ptz_pano.storage.scan_repository import ScanRepository


@dataclass(frozen=True)
class PanoramaBuilder:
    repository: ScanRepository
    compositor: SimpleCompositor = SimpleCompositor()
    aligner: FeatureAligner = FeatureAligner()

    def build_manifest(self, scan_id: str) -> Path:
        document = self.repository.load_document(scan_id)
        missing_geometry = [
            frame.file
            for frame in document.frames
            if frame.hfov_deg is None or frame.vfov_deg is None
        ]
        status = "ready_for_compositor" if not missing_geometry else "missing_fov"
        output_path = self.repository.scan_path(scan_id) / "panorama" / "panorama_manifest.json"
        panorama_result = None
        alignment_result = None
        compositor_frames = document.frames
        if not missing_geometry:
            alignment_result = self.aligner.align(
                self.repository.scan_path(scan_id),
                document.frames,
            )
            compositor_frames = alignment_result.frames
            panorama_result = self.compositor.build(
                self.repository.scan_path(scan_id),
                compositor_frames,
                self.repository.scan_path(scan_id) / "panorama" / "panorama.jpg",
            )
        write_json(
            output_path,
            {
                "scan_id": document.id,
                "status": status,
                "panorama_file": None
                if panorama_result is None
                else str(panorama_result.panorama_path.name),
                "preview_file": None
                if panorama_result is None or panorama_result.preview_path is None
                else str(panorama_result.preview_path.name),
                "coverage_percent": None
                if panorama_result is None
                else panorama_result.coverage_percent,
                "content_bbox": None if panorama_result is None else panorama_result.content_bbox,
                "alignment": None
                if alignment_result is None
                else {
                    "applied": alignment_result.applied,
                    "horizontal_pairs": to_jsonable(alignment_result.horizontal_pairs),
                    "vertical_pairs": to_jsonable(alignment_result.vertical_pairs),
                },
                "frames": compositor_frames,
                "missing_geometry": missing_geometry,
                "next_step": "Replace simple placement with spherical remap and multiband blending.",
            },
        )
        return output_path
