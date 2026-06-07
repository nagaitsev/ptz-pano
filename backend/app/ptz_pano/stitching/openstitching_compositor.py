from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.models import FrameMetadata
from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor
from ptz_pano.stitching.simple_compositor import CompositorResult


@dataclass(frozen=True)
class OpenStitchingCompositor:
    lens_calibration: LensCalibration | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    fallback_quality: str = "fast"
    strategy: str = "openstitching"
    projection: str = "opencv_spherical"
    uses_external_alignment: bool = False

    def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path) -> CompositorResult:
        try:
            from stitching import Stitcher
        except ImportError as error:
            raise RuntimeError(
                "OpenStitching engine requires the 'stitching-headless' package"
            ) from error

        if self.lens_calibration is not None:
            raise RuntimeError("OpenStitching engine does not support lens calibration yet")

        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_frames = sorted(frames, key=lambda item: item.index)
        frame_paths = [scan_path / frame.file for frame in sorted_frames]
        missing = [str(path) for path in frame_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"OpenStitching input frame is missing: {missing[0]}")

        settings = {
            "detector": "sift",
            "confidence_threshold": 0.2,
            **self.settings,
        }
        try:
            panorama, stitcher, timings = self._stitch_paths(
                frame_paths=frame_paths,
                settings=settings,
                stitcher_type=Stitcher,
                total_start=start,
            )
        except cv2.error as error:
            return self._build_opencv_fallback(
                scan_path,
                sorted_frames,
                output_path,
                settings,
                error,
            )

        if panorama is None:
            raise RuntimeError("OpenStitching returned an empty panorama")
        if not cv2.imwrite(str(output_path), panorama):
            raise RuntimeError(f"failed to write panorama: {output_path}")

        preview_path = output_path.with_name("preview.jpg")
        if not cv2.imwrite(str(preview_path), panorama):
            raise RuntimeError(f"failed to write panorama preview: {preview_path}")

        height, width = panorama.shape[:2]
        return self._compose_openstitching_result(
            frames=sorted_frames,
            frame_paths=frame_paths,
            output_path=output_path,
            preview_path=preview_path,
            panorama_size=(width, height),
            stitcher=stitcher,
            settings=settings,
            timings=timings,
        )

    def _build_opencv_fallback(
        self,
        scan_path: Path,
        frames: list[FrameMetadata],
        output_path: Path,
        settings: dict[str, Any],
        error: cv2.error,
    ) -> CompositorResult:
        fallback = OpenCvStitcherCompositor(
            lens_calibration=self.lens_calibration,
            fallback_quality="quality" if self.fallback_quality == "quality" else "fast",
        )
        result = fallback.build(scan_path, frames, output_path)
        details = dict(result.details or {})
        details.update(
            {
                "openstitching_failed": True,
                "openstitching_error_type": type(error).__name__,
                "openstitching_error": str(error).strip(),
                "openstitching_settings": settings,
            }
        )
        return CompositorResult(
            panorama_path=result.panorama_path,
            preview_path=result.preview_path,
            coverage_percent=result.coverage_percent,
            content_bbox=result.content_bbox,
            details=details,
        )

    def _stitch_paths(
        self,
        *,
        frame_paths: list[Path],
        settings: dict[str, Any],
        stitcher_type,
        total_start: float,
    ) -> tuple[np.ndarray | None, Any, dict[str, float]]:
        stitcher = _create_tracing_stitcher(stitcher_type, settings)
        stitch_start = time.perf_counter()
        panorama = stitcher.stitch([str(path) for path in frame_paths])
        timings = {
            "openstitching_stitch_sec": _elapsed(stitch_start),
            "total_sec": _elapsed(total_start),
        }
        return panorama, stitcher, timings

    def _compose_openstitching_result(
        self,
        *,
        frames: list[FrameMetadata],
        frame_paths: list[Path],
        output_path: Path,
        preview_path: Path,
        panorama_size: tuple[int, int],
        stitcher,
        settings: dict[str, Any],
        timings: dict[str, float],
    ) -> CompositorResult:
        used_names = _stitcher_image_names(stitcher)
        geometry_trace = _jsonable_geometry_trace(getattr(stitcher, "geometry_trace", None))
        ptz_mapping = _build_ptz_mapping(
            frames=frames,
            frame_paths=frame_paths,
            used_names=used_names,
            geometry_trace=getattr(stitcher, "geometry_trace", None),
            panorama_size=panorama_size,
        )
        width, height = panorama_size
        return CompositorResult(
            panorama_path=output_path,
            preview_path=preview_path,
            coverage_percent=100.0,
            content_bbox=(0, 0, width - 1, height - 1),
            details={
                "algorithm": "openstitching",
                "input_frame_count": len(frame_paths),
                "used_frame_count": None if used_names is None else len(used_names),
                "used_frame_files": used_names,
                "openstitching_geometry": geometry_trace,
                "settings": settings,
                "ptz_mapping": ptz_mapping,
                "timings": timings,
            },
        )


def _create_tracing_stitcher(stitcher_type, settings: dict[str, Any]):
    class TracingStitcher(stitcher_type):
        def stitch(self, images, feature_masks=[]):
            from stitching.images import Images

            self.images = Images.of(
                images,
                self.medium_megapix,
                self.low_megapix,
                self.final_megapix,
            )

            imgs = self.resize_medium_resolution()
            features = self.find_features(imgs, feature_masks)
            matches = self.match_features(features)
            imgs, features, matches = self.subset(imgs, features, matches)
            cameras = self.estimate_camera_parameters(features, matches)
            cameras = self.refine_camera_parameters(features, matches, cameras)
            cameras = self.perform_wave_correction(cameras)
            self.estimate_scale(cameras)

            imgs = self.resize_low_resolution(imgs)
            imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras)
            self.prepare_cropper(imgs, masks, corners, sizes)
            imgs, masks, corners, sizes = self.crop_low_resolution(
                imgs,
                masks,
                corners,
                sizes,
            )
            self.estimate_exposure_errors(corners, imgs, masks)
            seam_masks = self.find_seam_masks(imgs, corners, masks)

            final_source_sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
            final_camera_aspect = self.images.get_ratio(
                Images.Resolution.MEDIUM,
                Images.Resolution.FINAL,
            )
            imgs = self.resize_final_resolution()
            imgs, masks, pre_crop_corners, pre_crop_sizes = self.warp(
                imgs,
                cameras,
                final_source_sizes,
                final_camera_aspect,
            )
            imgs, masks, corners, sizes = self.crop_final_resolution(
                imgs,
                masks,
                pre_crop_corners,
                pre_crop_sizes,
            )
            self.geometry_trace = {
                "cameras": [_camera_details(camera) for camera in cameras],
                "final_corners": corners,
                "final_sizes": sizes,
                "final_source_sizes": final_source_sizes,
                "pre_crop_corners": pre_crop_corners,
                "pre_crop_sizes": pre_crop_sizes,
                "final_crop_intersections": _final_crop_intersections(
                    self.cropper,
                    self.images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL),
                ),
                "warper_type": self.warper.warper_type,
                "warper_scale": self.warper.scale,
                "final_camera_aspect": final_camera_aspect,
            }
            self.set_masks(masks)
            imgs = self.compensate_exposure_errors(corners, imgs)
            seam_masks = self.resize_seam_masks(seam_masks)

            self.initialize_composition(corners, sizes)
            self.blend_images(imgs, seam_masks, corners)
            return self.create_final_panorama()

    return TracingStitcher(**settings)


def _stitcher_image_names(stitcher) -> list[str] | None:
    images = getattr(stitcher, "images", None)
    names = getattr(images, "names", None)
    if names is None:
        return None
    return [str(name) for name in names]


def _build_ptz_mapping(
    frames: list[FrameMetadata],
    frame_paths: list[Path],
    used_names: list[str] | None,
    geometry_trace: dict[str, Any] | None,
    panorama_size: tuple[int, int],
    grid_size: int = 5,
) -> dict[str, Any]:
    if used_names is None or geometry_trace is None:
        return {"status": "missing_geometry", "control_points": []}
    path_to_frame: dict[str, FrameMetadata] = {}
    for path, frame in zip(frame_paths, frames):
        path_to_frame[str(path)] = frame
        path_to_frame[str(path.resolve())] = frame
        path_to_frame[frame.file] = frame
    control_points: list[dict[str, Any]] = []
    for used_index, used_name in enumerate(used_names):
        frame = path_to_frame.get(used_name) or path_to_frame.get(str(Path(used_name)))
        if frame is None:
            frame = path_to_frame.get(str(Path(used_name).resolve()))
        if frame is None or not _frame_has_mapping(frame):
            continue
        control_points.extend(
            _frame_control_points(
                used_index,
                used_name,
                frame,
                geometry_trace,
                panorama_size,
                grid_size,
            )
        )
    return {
        "status": "control_points" if control_points else "missing_frame_geometry",
        "method": "openstitching_warped_control_grid",
        "grid_size": grid_size,
        "control_point_count": len(control_points),
        "control_points": control_points,
    }


def _frame_control_points(
    used_index: int,
    used_name: str,
    frame: FrameMetadata,
    geometry_trace: dict[str, Any],
    panorama_size: tuple[int, int],
    grid_size: int,
) -> list[dict[str, Any]]:
    cameras = geometry_trace["cameras"]
    final_source_sizes = geometry_trace["final_source_sizes"]
    pre_crop_corners = geometry_trace["pre_crop_corners"]
    final_corners = geometry_trace["final_corners"]
    final_crop_intersections = geometry_trace["final_crop_intersections"] or [
        (0, 0, width, height) for width, height in geometry_trace["pre_crop_sizes"]
    ]

    if used_index >= len(cameras):
        return []
    if used_index >= len(final_source_sizes):
        return []
    if used_index >= len(pre_crop_corners) or used_index >= len(final_corners):
        return []
    if used_index >= len(final_crop_intersections):
        return []

    camera = cameras[used_index]
    source_w, source_h = final_source_sizes[used_index]
    pre_x, pre_y = pre_crop_corners[used_index]
    final_x, final_y = final_corners[used_index]
    crop_x, crop_y, crop_w, crop_h = final_crop_intersections[used_index]
    warper = cv2.PyRotationWarper(
        geometry_trace["warper_type"],
        float(geometry_trace["warper_scale"]) * float(geometry_trace["final_camera_aspect"]),
    )
    k = _camera_k(camera, float(geometry_trace["final_camera_aspect"]))
    r = np.asarray(camera["R"], dtype=np.float32)
    points: list[dict[str, Any]] = []
    for row in range(grid_size):
        v = row / (grid_size - 1) if grid_size > 1 else 0.5
        for col in range(grid_size):
            u = col / (grid_size - 1) if grid_size > 1 else 0.5
            source_x = u * (source_w - 1)
            source_y = v * (source_h - 1)
            warped_x, warped_y = warper.warpPoint((float(source_x), float(source_y)), k, r)
            local_x = warped_x - pre_x
            local_y = warped_y - pre_y
            if local_x < crop_x or local_y < crop_y:
                continue
            if local_x > crop_x + crop_w - 1 or local_y > crop_y + crop_h - 1:
                continue
            pano_x = final_x + (local_x - crop_x)
            pano_y = final_y + (local_y - crop_y)
            if pano_x < 0 or pano_y < 0 or pano_x >= panorama_size[0] or pano_y >= panorama_size[1]:
                continue
            pose = _frame_point_to_yaw_pitch(frame, u, v)
            if pose is None:
                continue
            points.append(
                {
                    "frame_index": frame.index,
                    "frame_file": frame.file,
                    "used_file": used_name,
                    "source_uv": [round(u, 6), round(v, 6)],
                    "pano_xy": [round(float(pano_x), 3), round(float(pano_y), 3)],
                    **pose,
                }
            )
    return points


def _frame_point_to_yaw_pitch(frame: FrameMetadata, u: float, v: float) -> dict[str, float] | None:
    if not _frame_has_mapping(frame):
        return None
    assert frame.pose.yaw_deg is not None
    assert frame.pose.pitch_deg is not None
    assert frame.hfov_deg is not None
    assert frame.vfov_deg is not None
    x_norm = (u * 2.0) - 1.0
    y_norm = 1.0 - (v * 2.0)
    yaw_delta = np.rad2deg(np.arctan(x_norm * np.tan(np.deg2rad(frame.hfov_deg / 2.0))))
    pitch_delta = np.rad2deg(np.arctan(y_norm * np.tan(np.deg2rad(frame.vfov_deg / 2.0))))
    return {
        "yaw_deg": round(float(frame.pose.yaw_deg + yaw_delta), 6),
        "pitch_deg": round(float(frame.pose.pitch_deg + pitch_delta), 6),
    }


def _frame_has_mapping(frame: FrameMetadata) -> bool:
    return (
        frame.pose.yaw_deg is not None
        and frame.pose.pitch_deg is not None
        and frame.hfov_deg is not None
        and frame.vfov_deg is not None
    )


def _camera_details(camera) -> dict[str, Any]:
    return {
        "focal": float(camera.focal),
        "aspect": float(getattr(camera, "aspect", 1.0)),
        "ppx": float(getattr(camera, "ppx", 0.0)),
        "ppy": float(getattr(camera, "ppy", 0.0)),
        "R": np.asarray(camera.R, dtype=float).tolist(),
    }


def _camera_k(camera: dict[str, Any], aspect: float) -> np.ndarray:
    k = np.eye(3, dtype=np.float32)
    k[0, 0] = float(camera["focal"]) * aspect
    k[1, 1] = float(camera["focal"]) * float(camera.get("aspect", 1.0)) * aspect
    k[0, 2] = float(camera.get("ppx", 0.0)) * aspect
    k[1, 2] = float(camera.get("ppy", 0.0)) * aspect
    return k


def _final_crop_intersections(cropper, aspect: float) -> list[tuple[int, int, int, int]]:
    if not getattr(cropper, "do_crop", False):
        return []
    return [tuple(rect.times(aspect)) for rect in cropper.intersection_rectangles]


def _jsonable_geometry_trace(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    result = dict(trace)
    result["final_corners"] = [list(item) for item in result["final_corners"]]
    result["final_sizes"] = [list(item) for item in result["final_sizes"]]
    result["final_source_sizes"] = [list(item) for item in result["final_source_sizes"]]
    result["pre_crop_corners"] = [list(item) for item in result["pre_crop_corners"]]
    result["pre_crop_sizes"] = [list(item) for item in result["pre_crop_sizes"]]
    result["final_crop_intersections"] = [
        list(item) for item in result["final_crop_intersections"]
    ]
    return result


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 6)
