from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RtspCapture:
    source: str

    def grab_frame(self, output_path: Path) -> Path:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(self.source)
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"failed to read frame from RTSP source: {self.source}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"failed to write frame: {output_path}")
        return output_path

