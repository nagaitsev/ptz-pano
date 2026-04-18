from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Literal

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
    order: Literal["row_snake", "column_snake"] = "row_snake"


class ScanPlanner:
    def __init__(self, config: ScanPlanConfig) -> None:
        if config.pan_step <= 0 or config.tilt_step <= 0:
            raise ValueError("scan steps must be positive")
        self.config = config

    def poses(self) -> Iterator[CameraPose]:
        tilts = list(_inclusive_range(self.config.tilt_min, self.config.tilt_max, self.config.tilt_step))
        pans = list(_inclusive_range(self.config.pan_min, self.config.pan_max, self.config.pan_step))
        if self.config.order == "column_snake":
            for column, pan in enumerate(pans):
                column_tilts = tilts if column % 2 == 0 else list(reversed(tilts))
                for tilt in column_tilts:
                    yield CameraPose(pan=pan, tilt=tilt, zoom=self.config.zoom)
            return

        for row, tilt in enumerate(tilts):
            row_pans = pans if row % 2 == 0 else list(reversed(pans))
            for pan in row_pans:
                yield CameraPose(pan=pan, tilt=tilt, zoom=self.config.zoom)


def apply_scan_angle_window(
    config: ScanPlanConfig,
    center: CameraPose,
    horizontal_deg: float | None = None,
    vertical_deg: float | None = None,
    pan_units_per_degree: float | None = None,
    tilt_units_per_degree: float | None = None,
) -> ScanPlanConfig:
    updates = {}
    if horizontal_deg is not None:
        if pan_units_per_degree is None:
            raise ValueError("pan_units_per_degree is required for horizontal scan angle")
        half_range = round(horizontal_deg * pan_units_per_degree / 2)
        updates["pan_min"] = center.pan - half_range
        updates["pan_max"] = center.pan + half_range
    if vertical_deg is not None:
        if tilt_units_per_degree is None:
            raise ValueError("tilt_units_per_degree is required for vertical scan angle")
        half_range = round(vertical_deg * tilt_units_per_degree / 2)
        updates["tilt_min"] = center.tilt - half_range
        updates["tilt_max"] = center.tilt + half_range
    if not updates:
        return config
    return replace(config, **updates)


def _inclusive_range(start: int, stop: int, step: int) -> Iterator[int]:
    value = start
    while value <= stop:
        yield value
        value += step
