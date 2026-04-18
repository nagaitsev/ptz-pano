from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class SimpleCompositor:
    width: int = 4096
    height: int = 2048
    pan_units_per_degree: float = 2448 / 170
    tilt_units_per_degree: float = 14.4
    lens_calibration: LensCalibration | None = None

    def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path) -> CompositorResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        weights = np.zeros((self.height, self.width, 1), dtype=np.float32)

        for frame in frames:
            if frame.hfov_deg is None or frame.vfov_deg is None:
                raise ValueError(f"frame is missing FOV metadata: {frame.file}")
            image = cv2.imread(str(scan_path / frame.file))
            if image is None:
                raise RuntimeError(f"failed to read frame image: {scan_path / frame.file}")
            if self.lens_calibration is not None:
                image = self.lens_calibration.undistort(image, frame.pose.zoom)

            warped, mask, x0, y0 = self._warp_frame(image, frame)
            h, w = warped.shape[:2]
            canvas[y0 : y0 + h, x0 : x0 + w] += warped.astype(np.float32) * mask
            weights[y0 : y0 + h, x0 : x0 + w] += mask

        result = np.zeros_like(canvas, dtype=np.uint8)
        np.divide(canvas, weights, out=canvas, where=weights > 0)
        populated = weights[:, :, 0] > 0
        result[populated] = np.clip(canvas[populated], 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"failed to write panorama: {output_path}")

        preview_path, content_bbox = _write_preview(result, populated, output_path)
        coverage_percent = float(populated.mean() * 100)
        return CompositorResult(
            panorama_path=output_path,
            preview_path=preview_path,
            coverage_percent=coverage_percent,
            content_bbox=content_bbox,
        )

    def _warp_frame(
        self,
        image: np.ndarray,
        frame: FrameMetadata,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
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

        delta_yaw = _normalize_degrees(sample_yaw - yaw_deg)
        delta_pitch = sample_pitch - pitch_deg
        x_norm = np.tan(np.deg2rad(delta_yaw)) / np.tan(np.deg2rad(frame.hfov_deg / 2))
        y_norm = np.tan(np.deg2rad(delta_pitch)) / np.tan(np.deg2rad(frame.vfov_deg / 2))
        valid = (np.abs(x_norm) <= 1) & (np.abs(y_norm) <= 1)

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

        mask = _feather_mask_from_norm(x_norm, y_norm, valid)
        return warped, mask, x0, y0


def _feather_mask(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    edge_x = np.minimum(x, 1 - x)
    edge_y = np.minimum(y, 1 - y)
    feather_x = np.clip(edge_x * 12, 0.05, 1)
    feather_y = np.clip(edge_y * 12, 0.05, 1)
    return (feather_y[:, None] * feather_x[None, :])[:, :, None]


def _feather_mask_from_norm(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    edge_x = 1 - np.abs(x_norm)
    edge_y = 1 - np.abs(y_norm)
    feather = np.clip(edge_x * 8, 0.05, 1) * np.clip(edge_y * 8, 0.05, 1)
    feather[~valid] = 0
    return feather[:, :, None].astype(np.float32)


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
