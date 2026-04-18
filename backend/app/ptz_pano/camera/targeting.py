from __future__ import annotations

from dataclasses import dataclass

from ptz_pano.calibration import FovTable
from ptz_pano.models import CameraPose


@dataclass(frozen=True)
class TargetingConfig:
    pan_units_per_degree: float
    tilt_units_per_degree: float
    fov_table: FovTable | None = None


@dataclass(frozen=True)
class CameraTarget:
    yaw_deg: float
    pitch_deg: float
    target_hfov_deg: float


def target_to_pose(target: CameraTarget, config: TargetingConfig) -> CameraPose:
    pan = round(target.yaw_deg * config.pan_units_per_degree)
    tilt = round(target.pitch_deg * config.tilt_units_per_degree)
    zoom = 0
    if config.fov_table is not None:
        zoom = config.fov_table.zoom_for_hfov(target.target_hfov_deg)
    return CameraPose(
        pan=pan,
        tilt=tilt,
        zoom=zoom,
        yaw_deg=target.yaw_deg,
        pitch_deg=target.pitch_deg,
    )

