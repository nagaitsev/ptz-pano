from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import cv2

from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.models import FrameMetadata, to_jsonable
from ptz_pano.stitching.alignment import AlignmentResult, FeatureAligner
from ptz_pano.stitching.simple_compositor import CompositorResult, SimpleCompositor


_STITCHER_STATUS_NAMES = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


@dataclass(frozen=True)
class OpenCvStitcherCompositor:
    lens_calibration: LensCalibration | None = None
    mode: Literal["panorama", "scans"] = "panorama"
    fallback_quality: Literal["fast", "quality"] = "fast"
    fallback_aligner: FeatureAligner = field(default_factory=FeatureAligner)
    max_detail_blend_frames: int = 16
    pano_confidence_thresh: float = 1.0
    registration_resol: float = 0.4
    seam_estimation_resol: float = 0.4
    compositing_resol: float = -1
    wave_correction: bool = True
    fallback_to_telemetry: bool = True
    max_opencv_frames: int = 16
    strategy: str = "opencv_stitcher"
    projection: str = "opencv_panorama"
    uses_external_alignment: bool = False

    def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path) -> CompositorResult:
        total_start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(frames) < 2:
            raise ValueError("OpenCV stitcher needs at least two frames")
        if len(frames) > self.max_opencv_frames and self.fallback_to_telemetry:
            return self._build_telemetry_fallback(
                scan_path,
                frames,
                output_path,
                {
                    "algorithm": self._fallback_algorithm(),
                    "opencv_skipped": True,
                    "skip_reason": "too_many_frames_for_high_level_stitcher",
                    "frame_count": len(frames),
                    "max_opencv_frames": self.max_opencv_frames,
                },
            )

        load_start = time.perf_counter()
        images = []
        for frame in sorted(frames, key=lambda item: item.index):
            image = cv2.imread(str(scan_path / frame.file))
            if image is None:
                raise RuntimeError(f"failed to read frame image: {scan_path / frame.file}")
            if self.lens_calibration is not None:
                image = self.lens_calibration.undistort(image, frame.pose.zoom)
            images.append(image)
        timings = {"load_images_sec": _elapsed(load_start)}

        attempts = _stitcher_attempts(
            self.pano_confidence_thresh,
            self.registration_resol,
            self.seam_estimation_resol,
            self.wave_correction,
        )
        stitch_start = time.perf_counter()
        attempt_results: list[dict[str, object]] = []
        stitcher = None
        panorama = None
        status = cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL
        selected_attempt = None
        for attempt in attempts:
            stitcher = cv2.Stitcher_create(_opencv_stitcher_mode(self.mode))
            stitcher.setPanoConfidenceThresh(float(attempt["pano_confidence_thresh"]))
            stitcher.setRegistrationResol(float(attempt["registration_resol"]))
            stitcher.setSeamEstimationResol(float(attempt["seam_estimation_resol"]))
            stitcher.setCompositingResol(self.compositing_resol)
            stitcher.setWaveCorrection(bool(attempt["wave_correction"]))

            try:
                status, panorama = stitcher.stitch(images)
                status_name = _STITCHER_STATUS_NAMES.get(status, f"UNKNOWN_STATUS_{status}")
                attempt_results.append({**attempt, "status": status_name})
            except cv2.error as error:
                status = cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL
                panorama = None
                attempt_results.append(
                    {
                        **attempt,
                        "status": "OPENCV_ERROR",
                        "error": str(error).splitlines()[0],
                    }
                )
            if status == cv2.Stitcher_OK and panorama is not None:
                selected_attempt = attempt
                break
        timings["opencv_stitch_sec"] = _elapsed(stitch_start)

        if status != cv2.Stitcher_OK or panorama is None or stitcher is None:
            if self.fallback_to_telemetry:
                return self._build_telemetry_fallback(
                    scan_path,
                    frames,
                    output_path,
                    {
                        "algorithm": self._fallback_algorithm(),
                        "opencv_skipped": False,
                        "opencv_attempts": attempt_results,
                        "opencv_timings": timings,
                    },
                )
            status_name = _STITCHER_STATUS_NAMES.get(status, f"UNKNOWN_STATUS_{status}")
            raise RuntimeError(f"OpenCV stitcher failed: {status_name}")

        component = [int(index) for index in stitcher.component()]
        expected_component = set(range(len(frames)))
        actual_component = set(component)
        if actual_component != expected_component:
            missing_component = sorted(expected_component - actual_component)
            if self.fallback_to_telemetry:
                return self._build_telemetry_fallback(
                    scan_path,
                    frames,
                    output_path,
                    {
                        "algorithm": self._fallback_algorithm(),
                        "opencv_skipped": False,
                        "opencv_rejected": "partial_component",
                        "opencv_partial_component": True,
                        "opencv_status": _STITCHER_STATUS_NAMES[status],
                        "opencv_component": component,
                        "opencv_missing_component": missing_component,
                        "opencv_frame_count": len(frames),
                        "opencv_attempts": attempt_results,
                        "opencv_timings": timings,
                    },
                )
            raise RuntimeError(
                f"OpenCV stitcher used only {len(actual_component)}/{len(frames)} frames"
            )

        if not cv2.imwrite(str(output_path), panorama):
            raise RuntimeError(f"failed to write panorama: {output_path}")

        preview_path = output_path.with_name("preview.jpg")
        if not cv2.imwrite(str(preview_path), panorama):
            raise RuntimeError(f"failed to write panorama preview: {preview_path}")

        height, width = panorama.shape[:2]
        return CompositorResult(
            panorama_path=output_path,
            preview_path=preview_path,
            coverage_percent=100.0,
            content_bbox=(0, 0, width - 1, height - 1),
            details={
                "algorithm": "opencv_stitcher",
                "mode": self.mode,
                "status": _STITCHER_STATUS_NAMES[status],
                "component": component,
                "pano_confidence_thresh": selected_attempt["pano_confidence_thresh"],
                "registration_resol": selected_attempt["registration_resol"],
                "seam_estimation_resol": selected_attempt["seam_estimation_resol"],
                "compositing_resol": self.compositing_resol,
                "wave_correction": selected_attempt["wave_correction"],
                "attempts": attempt_results,
                "timings": {**timings, "total_sec": _elapsed(total_start)},
            },
        )

    def _build_telemetry_fallback(
        self,
        scan_path: Path,
        frames: list[FrameMetadata],
        output_path: Path,
        details: dict[str, object],
    ) -> CompositorResult:
        fallback_start = time.perf_counter()
        timings: dict[str, float] = {}
        alignment_result: AlignmentResult | None = None
        compositor_frames = frames
        if self.fallback_quality == "quality":
            align_start = time.perf_counter()
            alignment_result = self.fallback_aligner.align(scan_path, frames)
            compositor_frames = alignment_result.frames
            timings["alignment_sec"] = _elapsed(align_start)

        compose_start = time.perf_counter()
        use_detail_blend = (
            self.fallback_quality == "quality"
            and len(compositor_frames) <= self.max_detail_blend_frames
        )
        if use_detail_blend:
            fallback = SimpleCompositor(
                lens_calibration=self.lens_calibration,
                strategy="max_weight",
                projection="sphere",
                seam_finder="graph_cut",
                exposure_compensator="gain_blocks",
                blender="multi_band",
            )
        else:
            fallback = SimpleCompositor(
                lens_calibration=self.lens_calibration,
                strategy="max_weight",
                projection="sphere",
                blender="weighted",
            )
        fallback_result = fallback.build(scan_path, compositor_frames, output_path)
        timings["compose_sec"] = _elapsed(compose_start)
        timings["fallback_total_sec"] = _elapsed(fallback_start)

        merged_details = {
            **details,
            "fallback_quality": self.fallback_quality,
            "fallback_alignment": _alignment_details(alignment_result),
            "fallback_compositor": fallback_result.details or {},
            "timings": timings,
        }
        if self.fallback_quality == "quality" and not use_detail_blend:
            merged_details["detail_blend_skipped"] = "too_many_frames_for_detail_blend"
            merged_details["max_detail_blend_frames"] = self.max_detail_blend_frames
        return replace(fallback_result, details=merged_details)

    def _fallback_algorithm(self) -> str:
        if self.fallback_quality == "quality":
            return "telemetry_quality_fallback"
        return "telemetry_weighted_fallback"


def _alignment_details(alignment_result: AlignmentResult | None) -> dict[str, object] | None:
    if alignment_result is None:
        return None
    return {
        "applied": alignment_result.applied,
        "horizontal_pairs": to_jsonable(alignment_result.horizontal_pairs),
        "vertical_pairs": to_jsonable(alignment_result.vertical_pairs),
    }


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def _opencv_stitcher_mode(mode: str) -> int:
    if mode == "scans":
        return cv2.Stitcher_SCANS
    return cv2.Stitcher_PANORAMA


def _stitcher_attempts(
    pano_confidence_thresh: float,
    registration_resol: float,
    seam_estimation_resol: float,
    wave_correction: bool,
) -> list[dict[str, object]]:
    attempts = [
        {
            "pano_confidence_thresh": pano_confidence_thresh,
            "registration_resol": registration_resol,
            "seam_estimation_resol": seam_estimation_resol,
            "wave_correction": wave_correction,
        }
    ]
    for candidate in [
        (1.0, 0.4, 0.4, True),
        (1.0, 0.4, 0.4, False),
        (1.0, 0.6, 0.4, True),
        (0.6, 0.4, 0.4, True),
    ]:
        attempt = {
            "pano_confidence_thresh": candidate[0],
            "registration_resol": candidate[1],
            "seam_estimation_resol": candidate[2],
            "wave_correction": candidate[3],
        }
        if attempt not in attempts:
            attempts.append(attempt)
    return attempts
