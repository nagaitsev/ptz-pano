from __future__ import annotations

from ptz_pano.models import CameraPose, FrameMetadata
from ptz_pano.stitching.alignment import FeatureAligner


def test_feature_aligner_converts_ptz_units_to_degrees() -> None:
    frame = FrameMetadata(
        index=1,
        file="frames/frame_0001.jpg",
        pose=CameraPose(pan=144, tilt=-72, zoom=0),
    )
    aligner = FeatureAligner(pan_units_per_degree=14.4, tilt_units_per_degree=14.4)

    assert aligner._frame_yaw(frame) == 10.0
    assert aligner._frame_pitch(frame) == -5.0


def test_feature_aligner_prefers_explicit_pose_angles() -> None:
    frame = FrameMetadata(
        index=1,
        file="frames/frame_0001.jpg",
        pose=CameraPose(pan=144, tilt=-72, zoom=0, yaw_deg=12.5, pitch_deg=-3.25),
    )
    aligner = FeatureAligner(pan_units_per_degree=14.4, tilt_units_per_degree=14.4)

    assert aligner._frame_yaw(frame) == 12.5
    assert aligner._frame_pitch(frame) == -3.25
