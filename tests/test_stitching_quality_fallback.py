from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ptz_pano.models import CameraPose, FrameMetadata
from ptz_pano.stitching.alignment import AlignmentResult
from ptz_pano.stitching.simple_compositor import CompositorResult


def _frame(index: int, yaw_deg: float = 0.0) -> FrameMetadata:
    return FrameMetadata(
        index=index,
        file=f"frames/frame_{index:04d}.jpg",
        pose=CameraPose(pan=0, tilt=0, zoom=0, yaw_deg=yaw_deg, pitch_deg=0.0),
        hfov_deg=60.0,
        vfov_deg=35.0,
    )


class _RecordingAligner:
    def __init__(self) -> None:
        self.received_frames: list[FrameMetadata] | None = None
        self.adjusted_frames: list[FrameMetadata] = []

    def align(self, scan_path: Path, frames: list[FrameMetadata]) -> AlignmentResult:
        self.received_frames = frames
        self.adjusted_frames = [
            replace(frame, pose=replace(frame.pose, yaw_deg=frame.pose.yaw_deg + 1.5))
            for frame in frames
        ]
        return AlignmentResult(
            frames=self.adjusted_frames,
            horizontal_pairs=[],
            vertical_pairs=[],
            applied=True,
        )


def test_quality_fallback_aligns_frames_and_uses_detail_compositor(monkeypatch) -> None:
    from ptz_pano.stitching import opencv_stitcher
    from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor

    constructed: list[dict[str, object]] = []
    built_with_frames: list[FrameMetadata] = []

    class RecordingCompositor:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def build(
            self,
            scan_path: Path,
            frames: list[FrameMetadata],
            output_path: Path,
        ) -> CompositorResult:
            built_with_frames.extend(frames)
            return CompositorResult(
                panorama_path=output_path,
                preview_path=output_path.with_name("preview.jpg"),
                coverage_percent=75.0,
                content_bbox=(0, 0, 99, 49),
                details={"blender": "multi_band", "timings": {"compose_sec": 0.25}},
            )

    monkeypatch.setattr(opencv_stitcher, "SimpleCompositor", RecordingCompositor)
    aligner = _RecordingAligner()
    frames = [_frame(1, yaw_deg=0.0), _frame(2, yaw_deg=30.0)]

    compositor = OpenCvStitcherCompositor(
        max_opencv_frames=1,
        fallback_quality="quality",
        fallback_aligner=aligner,
    )
    result = compositor.build(Path("scan"), frames, Path("panorama.jpg"))

    assert aligner.received_frames == frames
    assert built_with_frames == aligner.adjusted_frames
    assert constructed[0]["blender"] == "multi_band"
    assert constructed[0]["seam_finder"] == "graph_cut"
    assert constructed[0]["exposure_compensator"] == "gain_blocks"
    assert result.details is not None
    assert result.details["algorithm"] == "telemetry_quality_fallback"
    assert result.details["fallback_quality"] == "quality"
    assert result.details["fallback_alignment"]["applied"] is True
    assert result.details["fallback_compositor"]["blender"] == "multi_band"
    assert "fallback_total_sec" in result.details["timings"]


def test_fast_fallback_keeps_weighted_compositor_without_alignment(monkeypatch) -> None:
    from ptz_pano.stitching import opencv_stitcher
    from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor

    constructed: list[dict[str, object]] = []

    class RecordingCompositor:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def build(
            self,
            scan_path: Path,
            frames: list[FrameMetadata],
            output_path: Path,
        ) -> CompositorResult:
            return CompositorResult(
                panorama_path=output_path,
                preview_path=None,
                coverage_percent=50.0,
                content_bbox=None,
                details={"blender": "weighted", "timings": {"compose_sec": 0.1}},
            )

    monkeypatch.setattr(opencv_stitcher, "SimpleCompositor", RecordingCompositor)
    aligner = _RecordingAligner()

    compositor = OpenCvStitcherCompositor(
        max_opencv_frames=1,
        fallback_quality="fast",
        fallback_aligner=aligner,
    )
    result = compositor.build(Path("scan"), [_frame(1), _frame(2)], Path("panorama.jpg"))

    assert aligner.received_frames is None
    assert constructed[0]["blender"] == "weighted"
    assert result.details is not None
    assert result.details["algorithm"] == "telemetry_weighted_fallback"
    assert result.details["fallback_quality"] == "fast"


def test_quality_fallback_uses_weighted_compositor_when_scan_is_large(monkeypatch) -> None:
    from ptz_pano.stitching import opencv_stitcher
    from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor

    constructed: list[dict[str, object]] = []

    class RecordingCompositor:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        def build(
            self,
            scan_path: Path,
            frames: list[FrameMetadata],
            output_path: Path,
        ) -> CompositorResult:
            return CompositorResult(
                panorama_path=output_path,
                preview_path=None,
                coverage_percent=50.0,
                content_bbox=None,
                details={"blender": constructed[-1]["blender"], "timings": {"compose_sec": 0.1}},
            )

    monkeypatch.setattr(opencv_stitcher, "SimpleCompositor", RecordingCompositor)

    compositor = OpenCvStitcherCompositor(
        max_opencv_frames=1,
        fallback_quality="quality",
        max_detail_blend_frames=2,
        fallback_aligner=_RecordingAligner(),
    )
    result = compositor.build(Path("scan"), [_frame(1), _frame(2), _frame(3)], Path("panorama.jpg"))

    assert constructed[0]["blender"] == "weighted"
    assert result.details is not None
    assert result.details["fallback_compositor"]["blender"] == "weighted"
    assert result.details["detail_blend_skipped"] == "too_many_frames_for_detail_blend"


def test_partial_opencv_component_falls_back_to_all_frames(monkeypatch) -> None:
    from ptz_pano.stitching import opencv_stitcher
    from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor

    built_with_frames: list[FrameMetadata] = []

    class PartialStitcher:
        def setPanoConfidenceThresh(self, value: float) -> None:
            pass

        def setRegistrationResol(self, value: float) -> None:
            pass

        def setSeamEstimationResol(self, value: float) -> None:
            pass

        def setCompositingResol(self, value: float) -> None:
            pass

        def setWaveCorrection(self, value: bool) -> None:
            pass

        def stitch(self, images: list[np.ndarray]) -> tuple[int, np.ndarray]:
            return opencv_stitcher.cv2.Stitcher_OK, np.zeros((8, 8, 3), dtype=np.uint8)

        def component(self) -> list[int]:
            return [1, 2]

    class RecordingCompositor:
        def __init__(self, **kwargs: object) -> None:
            pass

        def build(
            self,
            scan_path: Path,
            frames: list[FrameMetadata],
            output_path: Path,
        ) -> CompositorResult:
            built_with_frames.extend(frames)
            return CompositorResult(
                panorama_path=output_path,
                preview_path=output_path.with_name("preview.jpg"),
                coverage_percent=66.0,
                content_bbox=(0, 0, 10, 10),
                details={"blender": "weighted", "timings": {"compose_sec": 0.1}},
            )

    monkeypatch.setattr(opencv_stitcher.cv2, "Stitcher_create", lambda mode: PartialStitcher())
    monkeypatch.setattr(
        opencv_stitcher.cv2,
        "imread",
        lambda path: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(opencv_stitcher.cv2, "imwrite", lambda path, image: True)
    monkeypatch.setattr(opencv_stitcher, "SimpleCompositor", RecordingCompositor)

    frames = [_frame(1), _frame(2), _frame(3), _frame(4)]
    result = OpenCvStitcherCompositor(fallback_quality="fast").build(
        Path("scan"),
        frames,
        Path("panorama.jpg"),
    )

    assert built_with_frames == frames
    assert result.details is not None
    assert result.details["algorithm"] == "telemetry_weighted_fallback"
    assert result.details["opencv_partial_component"] is True
    assert result.details["opencv_component"] == [1, 2]
    assert result.details["opencv_frame_count"] == 4
