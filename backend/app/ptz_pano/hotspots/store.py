from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptz_pano.jsonio import read_json, write_json
from ptz_pano.models import CameraPose, Hotspot


@dataclass(frozen=True)
class HotspotStore:
    path: Path

    def list(self) -> list[Hotspot]:
        if not self.path.exists():
            return []
        data = read_json(self.path)
        return [
            Hotspot(
                id=item["id"],
                title=item["title"],
                panorama_yaw_deg=item["panorama_yaw_deg"],
                panorama_pitch_deg=item["panorama_pitch_deg"],
                pose=CameraPose(**item["pose"]),
            )
            for item in data.get("hotspots", [])
        ]

    def save_all(self, hotspots: list[Hotspot]) -> None:
        write_json(self.path, {"hotspots": hotspots})

