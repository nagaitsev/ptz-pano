from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.models import FrameMetadata


@dataclass(frozen=True)
class CompositorResult:
    panorama_path: Path
    preview_path: Path | None
    coverage_percent: float
    content_bbox: tuple[int, int, int, int] | None
    details: dict[str, object] | None = None


@dataclass(frozen=True)
class SimpleCompositor:
    width: int = 4096
    height: int = 2048
    pan_units_per_degree: float = 2448 / 170
    tilt_units_per_degree: float = 14.4
    lens_calibration: LensCalibration | None = None
    strategy: Literal["average", "max_weight"] = "average"
    projection: Literal["angular", "sphere"] = "angular"
    seam_finder: Literal["graph_cut", "none"] = "graph_cut"
    seam_scale: float = 0.4
    exposure_compensator: Literal["gain_blocks", "none"] = "gain_blocks"
    blender: Literal["multi_band", "weighted"] = "weighted"

    def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path) -> CompositorResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.blender == "multi_band":
            return self._build_detail(scan_path, frames, output_path)

        return self._build_weighted(scan_path, frames, output_path)

    def _build_weighted(
        self,
        scan_path: Path,
        frames: list[FrameMetadata],
        output_path: Path,
    ) -> CompositorResult:
        total_start = time.perf_counter()
        canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        footprint = np.zeros((self.height, self.width), dtype=bool)

        if self.strategy == "average":
            weights = np.zeros((self.height, self.width, 1), dtype=np.float32)
        else:
            # For max_weight, we store the best weight seen so far for each pixel
            best_weights = np.zeros((self.height, self.width), dtype=np.float32)

        for frame in frames:
            if frame.hfov_deg is None or frame.vfov_deg is None:
                raise ValueError(f"frame is missing FOV metadata: {frame.file}")
            image = cv2.imread(str(scan_path / frame.file))
            if image is None:
                raise RuntimeError(f"failed to read frame image: {scan_path / frame.file}")
            if self.lens_calibration is not None:
                image = self.lens_calibration.undistort(image, frame.pose.zoom)

            # Use pyramidal mask for max_weight to ensure unique winner in overlaps
            warped, mask, valid_footprint, x0, y0 = self._warp_frame(
                image,
                frame,
                pyramidal=(self.strategy == "max_weight"),
            )
            h, w = warped.shape[:2]
            footprint[y0 : y0 + h, x0 : x0 + w] |= valid_footprint

            if self.strategy == "average":
                canvas[y0 : y0 + h, x0 : x0 + w] += warped.astype(np.float32) * mask
                weights[y0 : y0 + h, x0 : x0 + w] += mask
            else:
                # max_weight strategy: only update pixel if current weight is better
                flat_mask = mask[:, :, 0]
                current_best = best_weights[y0 : y0 + h, x0 : x0 + w]
                update_mask = flat_mask > current_best

                # Apply update to canvas and best_weights
                patch = canvas[y0 : y0 + h, x0 : x0 + w]
                patch[update_mask] = warped[update_mask].astype(np.float32)
                current_best[update_mask] = flat_mask[update_mask]

        if self.strategy == "average":
            np.divide(canvas, weights, out=canvas, where=weights > 0)
            populated = footprint
        else:
            populated = footprint

        result = np.clip(canvas, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"failed to write panorama: {output_path}")

        preview_path, content_bbox = _write_preview(result, populated, output_path)
        coverage_percent = float(populated.mean() * 100)
        return CompositorResult(
            panorama_path=output_path,
            preview_path=preview_path,
            coverage_percent=coverage_percent,
            content_bbox=content_bbox,
            details=self._diagnostics(total_start, len(frames)),
        )

    def _build_detail(
        self,
        scan_path: Path,
        frames: list[FrameMetadata],
        output_path: Path,
    ) -> CompositorResult:
        total_start = time.perf_counter()
        warped_images: list[np.ndarray] = []
        seam_masks: list[np.ndarray] = []
        exposure_masks: list[np.ndarray] = []
        corners: list[tuple[int, int]] = []

        for frame in frames:
            if frame.hfov_deg is None or frame.vfov_deg is None:
                raise ValueError(f"frame is missing FOV metadata: {frame.file}")
            image = cv2.imread(str(scan_path / frame.file))
            if image is None:
                raise RuntimeError(f"failed to read frame image: {scan_path / frame.file}")
            if self.lens_calibration is not None:
                image = self.lens_calibration.undistort(image, frame.pose.zoom)

            warped, _mask, valid_footprint, x0, y0 = self._warp_frame(
                image,
                frame,
                pyramidal=(self.strategy == "max_weight"),
            )
            valid_mask = valid_footprint.astype(np.uint8) * 255
            warped_images.append(warped)
            seam_masks.append(valid_mask.copy())
            exposure_masks.append(valid_mask)
            corners.append((x0, y0))

        if not warped_images:
            raise ValueError("cannot build panorama without frames")

        if self.exposure_compensator == "gain_blocks" and len(warped_images) > 1:
            compensator = cv2.detail.ExposureCompensator_createDefault(
                cv2.detail.ExposureCompensator_GAIN_BLOCKS
            )
            compensator.feed(corners, warped_images, exposure_masks)
        else:
            compensator = None

        if self.seam_finder == "graph_cut" and len(warped_images) > 1:
            seam_masks = _find_graph_cut_seams(
                warped_images,
                seam_masks,
                corners,
                self.seam_scale,
            )

        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_MULTI_BAND, False)
        if hasattr(blender, "setNumBands"):
            blender.setNumBands(_num_blend_bands(warped_images))
        roi = cv2.detail.resultRoi(corners, [image.shape[:2][::-1] for image in warped_images])
        blender.prepare(roi)

        for index, (image, mask, corner) in enumerate(zip(warped_images, seam_masks, corners)):
            compensated = image.copy()
            if compensator is not None:
                compensator.apply(index, corner, compensated, exposure_masks[index])
            blender.feed(compensated.astype(np.int16), mask, corner)

        blended, blended_mask = blender.blend(None, None)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        x0, y0, width, height = roi
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        populated = np.zeros((self.height, self.width), dtype=bool)
        x1 = _clamp(x0 + width, 0, self.width)
        y1 = _clamp(y0 + height, 0, self.height)
        canvas[y0:y1, x0:x1] = blended[: y1 - y0, : x1 - x0]
        populated[y0:y1, x0:x1] = blended_mask[: y1 - y0, : x1 - x0] > 0

        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"failed to write panorama: {output_path}")

        preview_path, content_bbox = _write_preview(canvas, populated, output_path)
        coverage_percent = float(populated.mean() * 100)
        return CompositorResult(
            panorama_path=output_path,
            preview_path=preview_path,
            coverage_percent=coverage_percent,
            content_bbox=content_bbox,
            details=self._diagnostics(total_start, len(frames)),
        )

    def _warp_frame(
        self,
        image: np.ndarray,
        frame: FrameMetadata,
        pyramidal: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        assert frame.hfov_deg is not None
        assert frame.vfov_deg is not None

        yaw_deg = frame.pose.yaw_deg
        if yaw_deg is None:
            yaw_deg = frame.pose.pan / self.pan_units_per_degree
        pitch_deg = frame.pose.pitch_deg
        if pitch_deg is None:
            pitch_deg = frame.pose.tilt / self.tilt_units_per_degree

        x_center = _yaw_to_x(yaw_deg, self.width)
        y_center = _pitch_to_y(pitch_deg, self.height)
        output_w = max(1, int(np.ceil(self.width * frame.hfov_deg / 360)))
        output_h = max(1, int(np.ceil(self.height * frame.vfov_deg / 180)))
        x0 = _clamp(round(x_center - output_w / 2), 0, self.width - output_w)
        y0 = _clamp(round(y_center - output_h / 2), 0, self.height - output_h)

        xs = np.arange(x0, x0 + output_w, dtype=np.float32)
        ys = np.arange(y0, y0 + output_h, dtype=np.float32)
        panorama_x, panorama_y = np.meshgrid(xs, ys)
        sample_yaw = panorama_x / self.width * 360 - 180
        sample_pitch = 90 - panorama_y / self.height * 180

        if self.projection == "sphere":
            x_norm, y_norm, valid = _project_sphere_to_frame(
                sample_yaw,
                sample_pitch,
                yaw_deg,
                pitch_deg,
                frame.hfov_deg,
                frame.vfov_deg,
            )
        else:
            x_norm, y_norm, valid = _project_angular_to_frame(
                sample_yaw,
                sample_pitch,
                yaw_deg,
                pitch_deg,
                frame.hfov_deg,
                frame.vfov_deg,
            )

        source_x = ((x_norm + 1) * 0.5 * (image.shape[1] - 1)).astype(np.float32)
        source_y = ((1 - y_norm) * 0.5 * (image.shape[0] - 1)).astype(np.float32)
        source_x[~valid] = -1
        source_y[~valid] = -1
        warped = cv2.remap(
            image,
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask = _feather_mask_from_norm(x_norm, y_norm, valid, pyramidal=pyramidal)
        return warped, mask, valid, x0, y0

    def _diagnostics(self, total_start: float, frame_count: int) -> dict[str, object]:
        return {
            "algorithm": "telemetry_compositor",
            "strategy": self.strategy,
            "projection": self.projection,
            "blender": self.blender,
            "seam_finder": self.seam_finder,
            "seam_scale": self.seam_scale,
            "exposure_compensator": self.exposure_compensator,
            "frame_count": frame_count,
            "output_size": [self.width, self.height],
            "timings": {"total_sec": round(time.perf_counter() - total_start, 6)},
        }


def _feather_mask(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    edge_x = np.minimum(x, 1 - x)
    edge_y = np.minimum(y, 1 - y)
    feather_x = np.clip(edge_x * 12, 0.05, 1)
    feather_y = np.clip(edge_y * 12, 0.05, 1)
    return (feather_y[:, None] * feather_x[None, :])[:, :, None]


def _num_blend_bands(images: list[np.ndarray]) -> int:
    max_side = max(max(image.shape[:2]) for image in images)
    return max(1, min(8, int(np.ceil(np.log2(max_side))) - 4))


def _find_graph_cut_seams(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    scale: float,
) -> list[np.ndarray]:
    if scale >= 1:
        seam_images = [image.astype(np.float32) for image in images]
        seam_masks = [mask.copy() for mask in masks]
        seam_corners = corners
    else:
        scale = max(scale, 0.1)
        seam_images = []
        seam_masks = []
        seam_corners = []
        for image, mask, (x0, y0) in zip(images, masks, corners):
            width = max(1, round(image.shape[1] * scale))
            height = max(1, round(image.shape[0] * scale))
            seam_images.append(
                cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
            )
            seam_masks.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
            seam_corners.append((round(x0 * scale), round(y0 * scale)))

    seam_finder = cv2.detail.GraphCutSeamFinder("COST_COLOR")
    seam_finder.find(seam_images, seam_corners, seam_masks)

    if scale >= 1:
        return seam_masks

    return [
        cv2.resize(mask, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        for mask, image in zip(seam_masks, images)
    ]


def _feather_mask_from_norm(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    valid: np.ndarray,
    pyramidal: bool = False,
) -> np.ndarray:
    edge_x = 1 - np.abs(x_norm)
    edge_y = 1 - np.abs(y_norm)

    if pyramidal:
        # Sharp falloff from the very center, ideal for max_weight
        feather = edge_x * edge_y
    else:
        # Top-hat style: 1.0 in the center 75%, linear ramp to 0 in outer 12.5%
        feather = np.clip(edge_x * 8, 0.0, 1.0) * np.clip(edge_y * 8, 0.0, 1.0)

    feather[~valid] = 0
    return feather[:, :, None].astype(np.float32)


def _project_angular_to_frame(
    sample_yaw: np.ndarray,
    sample_pitch: np.ndarray,
    frame_yaw: float,
    frame_pitch: float,
    hfov_deg: float,
    vfov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_yaw = _normalize_degrees(sample_yaw - frame_yaw)
    delta_pitch = sample_pitch - frame_pitch
    x_norm = np.tan(np.deg2rad(delta_yaw)) / np.tan(np.deg2rad(hfov_deg / 2))
    y_norm = np.tan(np.deg2rad(delta_pitch)) / np.tan(np.deg2rad(vfov_deg / 2))
    valid = (np.abs(x_norm) <= 1) & (np.abs(y_norm) <= 1)
    return x_norm, y_norm, valid


def _project_sphere_to_frame(
    sample_yaw: np.ndarray,
    sample_pitch: np.ndarray,
    frame_yaw: float,
    frame_pitch: float,
    hfov_deg: float,
    vfov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = _spherical_directions(sample_yaw, sample_pitch)
    right, up, forward = _camera_basis(frame_yaw, frame_pitch)

    local_x = directions[:, :, 0] * right[0] + directions[:, :, 1] * right[1] + directions[:, :, 2] * right[2]
    local_y = directions[:, :, 0] * up[0] + directions[:, :, 1] * up[1] + directions[:, :, 2] * up[2]
    local_z = (
        directions[:, :, 0] * forward[0]
        + directions[:, :, 1] * forward[1]
        + directions[:, :, 2] * forward[2]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        x_norm = local_x / (local_z * np.tan(np.deg2rad(hfov_deg / 2)))
        y_norm = local_y / (local_z * np.tan(np.deg2rad(vfov_deg / 2)))
    valid = (local_z > 0) & (np.abs(x_norm) <= 1) & (np.abs(y_norm) <= 1)
    x_norm[~np.isfinite(x_norm)] = 0
    y_norm[~np.isfinite(y_norm)] = 0
    return x_norm, y_norm, valid


def _spherical_directions(yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cos_pitch = np.cos(pitch)
    return np.stack(
        (
            cos_pitch * np.sin(yaw),
            np.sin(pitch),
            cos_pitch * np.cos(yaw),
        ),
        axis=2,
    ).astype(np.float32)


def _camera_basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _spherical_directions(
        np.array([[yaw_deg]], dtype=np.float32),
        np.array([[pitch_deg]], dtype=np.float32),
    )[0, 0]
    world_up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0], dtype=np.float32)
    else:
        right = right / right_norm
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def _yaw_to_x(yaw_deg: float, width: int) -> float:
    return (_normalize_degrees(yaw_deg) + 180) / 360 * width


def _pitch_to_y(pitch_deg: float, height: int) -> float:
    return (90 - pitch_deg) / 180 * height


def _normalize_degrees(value: np.ndarray | float) -> np.ndarray | float:
    return (value + 180) % 360 - 180


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _write_preview(
    panorama: np.ndarray,
    populated: np.ndarray,
    output_path: Path,
    margin: int = 32,
) -> tuple[Path | None, tuple[int, int, int, int] | None]:
    ys, xs = np.where(populated)
    if len(xs) == 0:
        return None, None

    x0 = _clamp(int(xs.min()) - margin, 0, panorama.shape[1] - 1)
    y0 = _clamp(int(ys.min()) - margin, 0, panorama.shape[0] - 1)
    x1 = _clamp(int(xs.max()) + margin, 0, panorama.shape[1] - 1)
    y1 = _clamp(int(ys.max()) + margin, 0, panorama.shape[0] - 1)
    preview = panorama[y0 : y1 + 1, x0 : x1 + 1]
    preview_path = output_path.with_name("preview.jpg")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"failed to write panorama preview: {preview_path}")
    return preview_path, (x0, y0, x1, y1)
