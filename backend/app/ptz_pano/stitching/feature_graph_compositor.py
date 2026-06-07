from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.models import FrameMetadata, to_jsonable
from ptz_pano.stitching.alignment import (
    _create_feature_tools,
    _ratio_test_matches,
    _scaled_image,
)
from ptz_pano.stitching.simple_compositor import CompositorResult


@dataclass(frozen=True)
class FeatureEdge:
    source_index: int
    target_index: int
    transform_target_to_source: np.ndarray
    matches: int
    inliers: int
    mean_error_px: float


@dataclass(frozen=True)
class FeatureGraphCompositor:
    lens_calibration: LensCalibration | None = None
    feature_finder: str = "sift"
    registration_scale: float = 0.35
    min_matches: int = 24
    min_inliers: int = 12
    ransac_reproj_threshold: float = 4.0
    max_canvas_side: int = 12000
    blend_mode: Literal["average", "max_weight"] = "max_weight"
    strategy: str = "feature_graph"
    projection: str = "feature_mosaic"
    uses_external_alignment: bool = False

    def build(self, scan_path: Path, frames: list[FrameMetadata], output_path: Path) -> CompositorResult:
        total_start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(frames) < 2:
            raise ValueError("feature graph compositor needs at least two frames")

        sorted_frames = sorted(frames, key=lambda item: item.index)
        load_start = time.perf_counter()
        images = self._load_images(scan_path, sorted_frames)
        timings = {"load_images_sec": _elapsed(load_start)}

        match_start = time.perf_counter()
        edges = self._match_edges(images, sorted_frames)
        timings["match_edges_sec"] = _elapsed(match_start)
        if not edges:
            raise RuntimeError("feature graph compositor could not find usable frame matches")

        place_start = time.perf_counter()
        transforms = place_by_strongest_tree(len(sorted_frames), edges)
        timings["place_graph_sec"] = _elapsed(place_start)
        placed = [(index, transform) for index, transform in enumerate(transforms) if transform is not None]
        if len(placed) < 2:
            raise RuntimeError("feature graph compositor could not place enough frames")

        compose_start = time.perf_counter()
        panorama, populated, content_bbox = self._compose(images, placed)
        timings["compose_sec"] = _elapsed(compose_start)

        if not cv2.imwrite(str(output_path), panorama):
            raise RuntimeError(f"failed to write panorama: {output_path}")
        preview_path = output_path.with_name("preview.jpg")
        if content_bbox is None:
            preview = panorama
        else:
            x0, y0, x1, y1 = content_bbox
            preview = panorama[y0 : y1 + 1, x0 : x1 + 1]
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"failed to write panorama preview: {preview_path}")

        coverage_percent = float(populated.mean() * 100)
        timings["total_sec"] = _elapsed(total_start)
        return CompositorResult(
            panorama_path=output_path,
            preview_path=preview_path,
            coverage_percent=coverage_percent,
            content_bbox=content_bbox,
            details={
                "algorithm": "feature_graph",
                "projection": self.projection,
                "strategy": self.strategy,
                "feature_finder": self.feature_finder,
                "registration_scale": self.registration_scale,
                "blend_mode": self.blend_mode,
                "placed_frames": len(placed),
                "frame_count": len(sorted_frames),
                "edges": [_edge_details(edge, sorted_frames) for edge in edges],
                "timings": timings,
            },
        )

    def _load_images(self, scan_path: Path, frames: list[FrameMetadata]) -> list[np.ndarray]:
        images = []
        for frame in frames:
            image = cv2.imread(str(scan_path / frame.file))
            if image is None:
                raise RuntimeError(f"failed to read frame image: {scan_path / frame.file}")
            if self.lens_calibration is not None:
                image = self.lens_calibration.undistort(image, frame.pose.zoom)
            images.append(image)
        return images

    def _match_edges(self, images: list[np.ndarray], frames: list[FrameMetadata]) -> list[FeatureEdge]:
        detector, matcher = _create_feature_tools(self.feature_finder)
        keypoints = []
        descriptors = []
        scales = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scaled, scale = _scaled_image(gray, self.registration_scale)
            kp, desc = detector.detectAndCompute(scaled, None)
            keypoints.append(kp)
            descriptors.append(desc)
            scales.append(scale)

        edges: list[FeatureEdge] = []
        for source_index, source_frame in enumerate(frames):
            for target_index in range(source_index + 1, len(frames)):
                target_frame = frames[target_index]
                if not _metadata_may_overlap(source_frame, target_frame):
                    continue
                edge = self._match_pair(
                    source_index,
                    target_index,
                    keypoints[source_index],
                    descriptors[source_index],
                    keypoints[target_index],
                    descriptors[target_index],
                    matcher,
                    scales[source_index],
                )
                if edge is not None:
                    edges.append(edge)
        return edges

    def _match_pair(
        self,
        source_index: int,
        target_index: int,
        source_keypoints: tuple[cv2.KeyPoint, ...],
        source_descriptors: np.ndarray | None,
        target_keypoints: tuple[cv2.KeyPoint, ...],
        target_descriptors: np.ndarray | None,
        matcher: cv2.DescriptorMatcher,
        scale: float,
    ) -> FeatureEdge | None:
        if source_descriptors is None or target_descriptors is None:
            return None

        matches = matcher.knnMatch(target_descriptors, source_descriptors, k=2)
        good_matches = _ratio_test_matches(matches)
        if len(good_matches) < self.min_matches:
            return None

        target_points = np.float32([target_keypoints[match.queryIdx].pt for match in good_matches])
        source_points = np.float32([source_keypoints[match.trainIdx].pt for match in good_matches])
        transform, inliers = cv2.estimateAffinePartial2D(
            target_points,
            source_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )
        if transform is None or inliers is None:
            return None

        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        if inlier_count < self.min_inliers:
            return None

        projected = cv2.transform(target_points[inlier_mask][None, :, :], transform)[0]
        errors = np.linalg.norm(projected - source_points[inlier_mask], axis=1)
        full_transform = transform.astype(np.float64)
        full_transform[:, 2] /= scale
        return FeatureEdge(
            source_index=source_index,
            target_index=target_index,
            transform_target_to_source=full_transform,
            matches=len(good_matches),
            inliers=inlier_count,
            mean_error_px=float(errors.mean() / scale),
        )

    def _compose(
        self,
        images: list[np.ndarray],
        placed: list[tuple[int, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
        bounds = []
        for index, transform in placed:
            h, w = images[index].shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            warped = cv2.transform(corners[None, :, :], transform)[0]
            bounds.append(warped)

        all_corners = np.vstack(bounds)
        min_x = float(np.floor(all_corners[:, 0].min()))
        min_y = float(np.floor(all_corners[:, 1].min()))
        max_x = float(np.ceil(all_corners[:, 0].max()))
        max_y = float(np.ceil(all_corners[:, 1].max()))
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        if width <= 0 or height <= 0:
            raise RuntimeError("feature graph compositor produced an empty canvas")
        if max(width, height) > self.max_canvas_side:
            raise RuntimeError(
                f"feature graph canvas is too large: {width}x{height}, max side {self.max_canvas_side}"
            )

        offset = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y]], dtype=np.float64)
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        weights = np.zeros((height, width, 1), dtype=np.float32)
        best_weights = np.zeros((height, width), dtype=np.float32)
        populated = np.zeros((height, width), dtype=bool)

        for index, transform in placed:
            image = images[index]
            full_transform = _compose_affine(offset, transform)
            warped = cv2.warpAffine(
                image,
                full_transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask = _feather_mask(image.shape[1], image.shape[0])
            warped_mask = cv2.warpAffine(
                mask,
                full_transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[:, :, None]
            if self.blend_mode == "max_weight":
                flat_mask = warped_mask[:, :, 0]
                update_mask = flat_mask > best_weights
                canvas[update_mask] = warped[update_mask].astype(np.float32)
                best_weights[update_mask] = flat_mask[update_mask]
                weights[update_mask, 0] = flat_mask[update_mask]
            else:
                canvas += warped.astype(np.float32) * warped_mask
                weights += warped_mask
            populated |= warped_mask[:, :, 0] > 0.001

        if self.blend_mode == "average":
            np.divide(canvas, weights, out=canvas, where=weights > 0)
        result = np.clip(canvas, 0, 255).astype(np.uint8)
        return result, populated, _content_bbox(populated)


def place_by_strongest_tree(
    frame_count: int,
    edges: list[FeatureEdge],
) -> list[np.ndarray | None]:
    transforms: list[np.ndarray | None] = [None] * frame_count
    transforms[0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    sorted_edges = sorted(edges, key=_edge_sort_key)

    made_progress = True
    while made_progress:
        made_progress = False
        for edge in sorted_edges:
            source_transform = transforms[edge.source_index]
            target_transform = transforms[edge.target_index]
            if source_transform is not None and target_transform is None:
                transforms[edge.target_index] = _compose_affine(
                    source_transform,
                    edge.transform_target_to_source,
                )
                made_progress = True
            elif target_transform is not None and source_transform is None:
                transforms[edge.source_index] = _compose_affine(
                    target_transform,
                    _invert_affine(edge.transform_target_to_source),
                )
                made_progress = True
    return transforms


def _metadata_may_overlap(left: FrameMetadata, right: FrameMetadata) -> bool:
    if left.hfov_deg is None or left.vfov_deg is None or right.hfov_deg is None or right.vfov_deg is None:
        return True
    left_yaw = left.pose.yaw_deg
    right_yaw = right.pose.yaw_deg
    left_pitch = left.pose.pitch_deg
    right_pitch = right.pose.pitch_deg
    if left_yaw is None or right_yaw is None or left_pitch is None or right_pitch is None:
        return True
    yaw_limit = (left.hfov_deg + right.hfov_deg) * 0.55
    pitch_limit = (left.vfov_deg + right.vfov_deg) * 0.6
    return (
        abs(_normalize_degrees(left_yaw - right_yaw)) <= yaw_limit
        and abs(left_pitch - right_pitch) <= pitch_limit
    )


def _edge_sort_key(edge: FeatureEdge) -> tuple[int, int, float]:
    return (-edge.inliers, -edge.matches, edge.mean_error_px)


def _compose_affine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    composed = _to_homogeneous(first) @ _to_homogeneous(second)
    return composed[:2, :]


def _invert_affine(transform: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_to_homogeneous(transform))[:2, :]


def _to_homogeneous(transform: np.ndarray) -> np.ndarray:
    return np.vstack([transform, np.array([0.0, 0.0, 1.0], dtype=np.float64)])


def _feather_mask(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    edge_x = np.minimum(x, 1 - x)
    edge_y = np.minimum(y, 1 - y)
    feather = np.clip(edge_x * 12, 0, 1)[None, :] * np.clip(edge_y * 12, 0, 1)[:, None]
    return feather.astype(np.float32)


def _content_bbox(populated: np.ndarray, margin: int = 24) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(populated)
    if len(xs) == 0:
        return None
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, populated.shape[1] - 1)
    y1 = min(int(ys.max()) + margin, populated.shape[0] - 1)
    return x0, y0, x1, y1


def _edge_details(edge: FeatureEdge, frames: list[FrameMetadata]) -> dict[str, object]:
    return {
        "source_frame": frames[edge.source_index].file,
        "target_frame": frames[edge.target_index].file,
        "matches": edge.matches,
        "inliers": edge.inliers,
        "mean_error_px": edge.mean_error_px,
        "transform_target_to_source": to_jsonable(edge.transform_target_to_source.tolist()),
    }


def _normalize_degrees(value: float) -> float:
    return (value + 180) % 360 - 180


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 6)
