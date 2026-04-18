from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptz_pano.jsonio import read_json, write_json
from ptz_pano.models import (
    CameraConfig,
    CameraPose,
    CaptureConfig,
    FrameMetadata,
    ScanDocument,
)


@dataclass(frozen=True)
class ScanRepository:
    root: Path

    def create_scan(self, scan_id: str) -> Path:
        path = self.scan_path(scan_id)
        (path / "frames").mkdir(parents=True, exist_ok=True)
        (path / "panorama").mkdir(parents=True, exist_ok=True)
        return path

    def scan_path(self, scan_id: str) -> Path:
        return self.root / scan_id

    def save_document(self, document: ScanDocument) -> None:
        write_json(self.scan_path(document.id) / "scan.json", document)

    def load_document(self, scan_id: str) -> ScanDocument:
        data = read_json(self.scan_path(scan_id) / "scan.json")
        camera = CameraConfig(**data["camera"])
        capture_data = data["capture"]
        if capture_data.get("resolution") is not None:
            capture_data = dict(capture_data)
            capture_data["resolution"] = tuple(capture_data["resolution"])
        capture = CaptureConfig(**capture_data)
        frames = [
            FrameMetadata(
                index=item["index"],
                file=item["file"],
                pose=CameraPose(**item["pose"]),
                hfov_deg=item.get("hfov_deg"),
                vfov_deg=item.get("vfov_deg"),
            )
            for item in data.get("frames", [])
        ]
        return ScanDocument(id=data["id"], camera=camera, capture=capture, frames=frames)

