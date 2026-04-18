from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ptz_pano.models import FrameMetadata


@dataclass(frozen=True)
class SimpleCompositor:
    width: int = 4096
    height: int = 2048
    pan_units_per_degree: float = 512 / 10
    tilt_units_per_degree: float = 512 / 10

    def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        weights = np.zeros((self.height, self.width, 1), dtype=np.float32)

        for frame in frames:
            if frame.hfov_deg is None or frame.vfov_deg is None:
                raise ValueError(f"frame is missing FOV metadata: {frame.file}")
            image = cv2.imread(str(scan_path / frame.file))
            if image is None:
                raise RuntimeError(f"failed to read frame image: {scan_path / frame.file}")

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
        return output_path

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

        output_w = max(1, round(self.width * frame.hfov_deg / 360))
        output_h = max(1, round(self.height * frame.vfov_deg / 180))
        resized = cv2.resize(image, (output_w, output_h), interpolation=cv2.INTER_AREA)

        x_center = round((yaw_deg + 180) / 360 * self.width)
        y_center = round((90 - pitch_deg) / 180 * self.height)
        x0 = _clamp(x_center - output_w // 2, 0, self.width - output_w)
        y0 = _clamp(y_center - output_h // 2, 0, self.height - output_h)

        mask = _feather_mask(output_w, output_h)
        return resized, mask, x0, y0


def _feather_mask(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    edge_x = np.minimum(x, 1 - x)
    edge_y = np.minimum(y, 1 - y)
    feather_x = np.clip(edge_x * 12, 0.05, 1)
    feather_y = np.clip(edge_y * 12, 0.05, 1)
    return (feather_y[:, None] * feather_x[None, :])[:, :, None]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

