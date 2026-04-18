from __future__ import annotations

import pytest

from ptz_pano.models import CameraPose
from ptz_pano.scan import ScanPlanConfig, apply_scan_angle_window


def test_apply_scan_angle_window_centers_ranges_on_pose() -> None:
    config = ScanPlanConfig(
        pan_min=-1000,
        pan_max=1000,
        pan_step=256,
        tilt_min=-500,
        tilt_max=500,
        tilt_step=128,
        zoom=0,
    )

    result = apply_scan_angle_window(
        config=config,
        center=CameraPose(pan=144, tilt=-72, zoom=0),
        horizontal_deg=90,
        vertical_deg=30,
        pan_units_per_degree=14.4,
        tilt_units_per_degree=14.4,
    )

    assert result.pan_min == -504
    assert result.pan_max == 792
    assert result.tilt_min == -288
    assert result.tilt_max == 144
    assert result.pan_step == config.pan_step
    assert result.tilt_step == config.tilt_step
    assert result.zoom == config.zoom


def test_apply_scan_angle_window_requires_units_for_degrees() -> None:
    config = ScanPlanConfig(
        pan_min=-1000,
        pan_max=1000,
        pan_step=256,
        tilt_min=-500,
        tilt_max=500,
        tilt_step=128,
        zoom=0,
    )

    with pytest.raises(ValueError, match="pan_units_per_degree"):
        apply_scan_angle_window(
            config=config,
            center=CameraPose(pan=0, tilt=0, zoom=0),
            horizontal_deg=90,
        )
