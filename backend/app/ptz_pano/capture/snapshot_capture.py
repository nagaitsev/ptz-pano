from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class SnapshotCapture:
    source: str
    timeout_sec: float = 5.0

    def grab_frame(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(self.source, timeout=self.timeout_sec) as response:
            output_path.write_bytes(response.read())
        return output_path

