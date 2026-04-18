from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptz_pano.jsonio import read_json, write_json
from ptz_pano.models import FovSample


@dataclass(frozen=True)
class FovTable:
    samples: tuple[FovSample, ...]

    @classmethod
    def load(cls, path: Path) -> "FovTable":
        data = read_json(path)
        return cls(tuple(FovSample(**item) for item in data["samples"]))

    def save(self, path: Path) -> None:
        write_json(path, {"samples": list(self.samples)})

    def fov_for_zoom(self, zoom: int) -> tuple[float, float]:
        if not self.samples:
            raise ValueError("FOV table is empty")

        samples = sorted(self.samples, key=lambda item: item.zoom)
        if zoom <= samples[0].zoom:
            return samples[0].hfov_deg, samples[0].vfov_deg
        if zoom >= samples[-1].zoom:
            return samples[-1].hfov_deg, samples[-1].vfov_deg

        for left, right in zip(samples, samples[1:]):
            if left.zoom <= zoom <= right.zoom:
                span = right.zoom - left.zoom
                ratio = (zoom - left.zoom) / span
                hfov = left.hfov_deg + (right.hfov_deg - left.hfov_deg) * ratio
                vfov = left.vfov_deg + (right.vfov_deg - left.vfov_deg) * ratio
                return hfov, vfov

        raise RuntimeError("unreachable FOV interpolation state")

