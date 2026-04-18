from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ptz_pano.jsonio import read_json


@dataclass(frozen=True)
class LensSample:
    zoom: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    resolution: tuple[int, int]
    rms_error: float | None = None
    samples_used: int | None = None


@dataclass(frozen=True)
class LensCalibration:
    samples: tuple[LensSample, ...]

    @classmethod
    def from_file(cls, path: Path) -> LensCalibration:
        data = read_json(path)
        samples = []
        for zoom_text, sample_data in data.get("zooms", {}).items():
            samples.append(_sample_from_dict(int(zoom_text), sample_data))
        if not samples:
            raise ValueError(f"lens calibration contains no zoom samples: {path}")
        return cls(samples=tuple(sorted(samples, key=lambda sample: sample.zoom)))

    def nearest_sample(self, zoom: int) -> LensSample:
        return min(self.samples, key=lambda sample: abs(sample.zoom - zoom))

    def undistort(self, image: np.ndarray, zoom: int) -> np.ndarray:
        sample = self.nearest_sample(zoom)
        height, width = image.shape[:2]
        if sample.resolution != (width, height):
            raise ValueError(
                "lens calibration resolution mismatch: "
                f"image={(width, height)} calibration={sample.resolution}"
            )
        return cv2.undistort(image, sample.camera_matrix, sample.dist_coeffs, None, sample.camera_matrix)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "zoom": sample.zoom,
                "resolution": sample.resolution,
                "rms_error": sample.rms_error,
                "samples_used": sample.samples_used,
            }
            for sample in self.samples
        ]


def _sample_from_dict(zoom: int, data: dict[str, Any]) -> LensSample:
    resolution = data["resolution"]
    return LensSample(
        zoom=zoom,
        camera_matrix=np.array(data["matrix"], dtype=np.float64),
        dist_coeffs=np.array(data["dist"], dtype=np.float64),
        resolution=(int(resolution[0]), int(resolution[1])),
        rms_error=data.get("rms_error"),
        samples_used=data.get("samples_used"),
    )
