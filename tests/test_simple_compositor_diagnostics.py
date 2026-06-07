from __future__ import annotations

import cv2
import numpy as np

from ptz_pano.models import CameraPose, FrameMetadata
from ptz_pano.stitching.simple_compositor import SimpleCompositor


def test_weighted_compositor_reports_diagnostics(tmp_path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    image_path = frame_dir / "frame_0001.jpg"
    image = np.full((8, 12, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    frame = FrameMetadata(
        index=1,
        file="frames/frame_0001.jpg",
        pose=CameraPose(pan=0, tilt=0, zoom=0, yaw_deg=0.0, pitch_deg=0.0),
        hfov_deg=60.0,
        vfov_deg=35.0,
    )
    compositor = SimpleCompositor(width=64, height=32, strategy="max_weight", projection="sphere")

    result = compositor.build(tmp_path, [frame], tmp_path / "panorama" / "panorama.jpg")

    assert result.details is not None
    assert result.details["algorithm"] == "telemetry_compositor"
    assert result.details["strategy"] == "max_weight"
    assert result.details["projection"] == "sphere"
    assert result.details["blender"] == "weighted"
    assert "total_sec" in result.details["timings"]
