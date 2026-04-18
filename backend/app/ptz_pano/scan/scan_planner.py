from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ptz_pano.models import CameraPose


@dataclass(frozen=True)
class ScanPlanConfig:
    pan_min: int
    pan_max: int
    pan_step: int
    tilt_min: int
    tilt_max: int
    tilt_step: int
    zoom: int


class ScanPlanner:
    def __init__(self, config: ScanPlanConfig) -> None:
        if config.pan_step <= 0 or config.tilt_step <= 0:
            raise ValueError("scan steps must be positive")
        self.config = config

    def poses(self) -> Iterator[CameraPose]:
        tilts = list(_inclusive_range(self.config.tilt_min, self.config.tilt_max, self.config.tilt_step))
        pans = list(_inclusive_range(self.config.pan_min, self.config.pan_max, self.config.pan_step))
        for row, tilt in enumerate(tilts):
            row_pans = pans if row % 2 == 0 else list(reversed(pans))
            for pan in row_pans:
                yield CameraPose(pan=pan, tilt=tilt, zoom=self.config.zoom)


def _inclusive_range(start: int, stop: int, step: int) -> Iterator[int]:
    value = start
    while value <= stop:
        yield value
        value += step

