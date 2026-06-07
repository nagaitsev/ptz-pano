from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ptz_pano.models import CameraPose


class CameraController(Protocol):
    def home(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def move_absolute(self, pose: CameraPose) -> None:
        ...

    def move_direction(self, pan_speed: int, tilt_speed: int, pan_dir: int, tilt_dir: int) -> None:
        ...

    def get_position(self) -> CameraPose:
        ...

    def get_zoom(self) -> int:
        ...

    def set_zoom(self, zoom: int) -> None:
        ...

    def close(self) -> None:
        ...


class FrameCapture(Protocol):
    def grab_frame(self, output_path: Path) -> Path:
        ...
