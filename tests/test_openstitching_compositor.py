from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np

from ptz_pano.models import CameraPose, FrameMetadata
from ptz_pano.stitching.openstitching_compositor import (
    OpenStitchingCompositor,
    _frame_point_to_yaw_pitch,
)


def _frame(index: int) -> FrameMetadata:
    return FrameMetadata(
        index=index,
        file=f"frames/frame_{index:04d}.jpg",
        pose=CameraPose(pan=0, tilt=0, zoom=0, yaw_deg=0.0, pitch_deg=0.0),
        hfov_deg=60.0,
        vfov_deg=35.0,
    )


def test_openstitching_compositor_passes_all_frame_files_and_records_subset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    received_settings: list[dict[str, object]] = []
    received_images: list[list[str]] = []

    class FakeImages:
        names = ["frames/frame_0001.jpg", "frames/frame_0003.jpg"]

    class FakeStitcher:
        geometry_trace = {
            "cameras": [{"focal": 100.0, "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}],
            "final_corners": [(0, 0)],
            "final_sizes": [(24, 12)],
            "final_source_sizes": [(24, 12)],
            "pre_crop_corners": [(0, 0)],
            "pre_crop_sizes": [(24, 12)],
            "final_crop_intersections": [(0, 0, 24, 12)],
            "warper_type": "spherical",
            "warper_scale": 100.0,
            "final_camera_aspect": 1.0,
        }

        def __init__(self, **settings: object) -> None:
            received_settings.append(settings)
            self.images = FakeImages()

        def stitch(self, images: list[str]) -> np.ndarray:
            received_images.append(images)
            return np.zeros((12, 24, 3), dtype=np.uint8)

    module = types.SimpleNamespace(Stitcher=FakeStitcher)
    monkeypatch.setitem(sys.modules, "stitching", module)
    monkeypatch.setattr(
        "ptz_pano.stitching.openstitching_compositor._create_tracing_stitcher",
        lambda stitcher_type, settings: FakeStitcher(**settings),
    )
    monkeypatch.setattr(
        "ptz_pano.stitching.openstitching_compositor.cv2.imwrite",
        lambda path, image: True,
    )
    class FakeWarper:
        def __init__(self, warper_type: str, scale: float) -> None:
            pass

        def warpPoint(self, point, k, r):
            return point

    monkeypatch.setattr(
        "ptz_pano.stitching.openstitching_compositor.cv2.PyRotationWarper",
        FakeWarper,
    )

    scan_path = tmp_path / "scan"
    frames_path = scan_path / "frames"
    frames_path.mkdir(parents=True)
    for index in range(1, 4):
        (frames_path / f"frame_{index:04d}.jpg").write_bytes(b"jpeg")

    result = OpenStitchingCompositor(settings={"confidence_threshold": 0.35}).build(
        scan_path,
        [_frame(1), _frame(2), _frame(3)],
        scan_path / "panorama" / "panorama.jpg",
    )

    assert received_images == [
        [
            str(frames_path / "frame_0001.jpg"),
            str(frames_path / "frame_0002.jpg"),
            str(frames_path / "frame_0003.jpg"),
        ]
    ]
    assert received_settings[0]["detector"] == "sift"
    assert received_settings[0]["confidence_threshold"] == 0.35
    assert result.preview_path == scan_path / "panorama" / "preview.jpg"
    assert result.coverage_percent == 100.0
    assert result.content_bbox == (0, 0, 23, 11)
    assert result.details is not None
    assert result.details["algorithm"] == "openstitching"
    assert result.details["input_frame_count"] == 3
    assert result.details["used_frame_count"] == 2
    assert result.details["used_frame_files"] == ["frames/frame_0001.jpg", "frames/frame_0003.jpg"]
    assert result.details["ptz_mapping"]["status"] == "control_points"
    assert result.details["openstitching_geometry"]["final_corners"] == [[0, 0]]


def test_frame_point_to_yaw_pitch_uses_angular_fov() -> None:
    frame = FrameMetadata(
        index=1,
        file="frames/frame_0001.jpg",
        pose=CameraPose(pan=0, tilt=0, zoom=0, yaw_deg=10.0, pitch_deg=5.0),
        hfov_deg=60.0,
        vfov_deg=40.0,
    )

    center = _frame_point_to_yaw_pitch(frame, 0.5, 0.5)
    right = _frame_point_to_yaw_pitch(frame, 1.0, 0.5)
    top = _frame_point_to_yaw_pitch(frame, 0.5, 0.0)

    assert center == {"yaw_deg": 10.0, "pitch_deg": 5.0}
    assert right["yaw_deg"] > center["yaw_deg"]
    assert round(right["yaw_deg"], 6) == 40.0
    assert top["pitch_deg"] > center["pitch_deg"]
    assert round(top["pitch_deg"], 6) == 25.0


def test_openstitching_falls_back_to_opencv_compositor_on_flann_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeStitcher:
        def __init__(self, **settings: object) -> None:
            self.images = types.SimpleNamespace(names=None)

        def stitch(self, images: list[str]) -> np.ndarray:
            raise cv2.error(
                "OpenCV(4.13.0) ... miniflann.cpp:521: error: (-215:Assertion failed) (size_t)knn <= index_->size() in function 'cv::flann::runKnnSearch_'"
            )

    class FakeFallbackCompositor:
        def __init__(self, *, lens_calibration=None, fallback_quality="fast") -> None:
            self.lens_calibration = lens_calibration
            self.fallback_quality = fallback_quality

        def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path):
            return types.SimpleNamespace(
                panorama_path=output_path,
                preview_path=output_path.with_name("preview.jpg"),
                coverage_percent=77.7,
                content_bbox=(1, 2, 3, 4),
                details={"algorithm": "telemetry_quality_fallback"},
            )

    module = types.SimpleNamespace(Stitcher=FakeStitcher)
    monkeypatch.setitem(sys.modules, "stitching", module)
    monkeypatch.setattr(
        "ptz_pano.stitching.openstitching_compositor._create_tracing_stitcher",
        lambda stitcher_type, settings: FakeStitcher(**settings),
    )
    monkeypatch.setattr(
        "ptz_pano.stitching.openstitching_compositor.OpenCvStitcherCompositor",
        FakeFallbackCompositor,
    )

    scan_path = tmp_path / "scan"
    frames_path = scan_path / "frames"
    frames_path.mkdir(parents=True)
    for index in range(1, 3):
        (frames_path / f"frame_{index:04d}.jpg").write_bytes(b"jpeg")

    result = OpenStitchingCompositor(fallback_quality="quality").build(
        scan_path,
        [_frame(1), _frame(2)],
        scan_path / "panorama" / "panorama.jpg",
    )

    assert result.coverage_percent == 77.7
    assert result.content_bbox == (1, 2, 3, 4)
    assert result.details["algorithm"] == "telemetry_quality_fallback"
    assert result.details["openstitching_error_type"] == "error"
    assert "cv::flann::runKnnSearch_" in result.details["openstitching_error"]
    assert result.details["openstitching_failed"] is True
