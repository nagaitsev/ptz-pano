from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from ptz_pano.models import CameraPose, FrameMetadata


@dataclass(frozen=True)
class HorizontalAlignment:
    left_frame: str
    right_frame: str
    nominal_delta_yaw_deg: float
    observed_delta_yaw_deg: float
    dx_px: float
    dy_px: float
    matches: int
    inliers: int


@dataclass(frozen=True)
class VerticalAlignment:
    lower_frame: str
    upper_frame: str
    nominal_delta_pitch_deg: float
    observed_delta_pitch_deg: float
    dx_px: float
    dy_px: float
    matches: int
    inliers: int


@dataclass(frozen=True)
class AlignmentResult:
    frames: list[FrameMetadata]
    horizontal_pairs: list[HorizontalAlignment]
    vertical_pairs: list[VerticalAlignment]
    applied: bool


@dataclass(frozen=True)
class FeatureAligner:
    min_matches: int = 24
    min_inliers: int = 12
    max_pitch_delta_deg: float = 5.0
    max_yaw_delta_deg: float = 5.0
    pan_units_per_degree: float = 2448 / 170
    tilt_units_per_degree: float = 14.4
    feature_finder: str = "sift"
    registration_scale: float = 0.4

    def align(self, scan_path: Path, frames: list[FrameMetadata]) -> AlignmentResult:
        if len(frames) < 2:
            return AlignmentResult(frames=frames, horizontal_pairs=[], vertical_pairs=[], applied=False)

        rows = _group_rows(frames, self.tilt_units_per_degree)
        adjusted_by_file: dict[str, FrameMetadata] = {}
        horizontal_pairs: list[HorizontalAlignment] = []

        for _row_pitch, row_frames in rows:
            aligned_row, row_pairs = self._align_row(scan_path, row_frames)
            horizontal_pairs.extend(row_pairs)
            adjusted_by_file.update({frame.file: frame for frame in aligned_row})

        yaw_adjusted = [adjusted_by_file.get(frame.file, frame) for frame in frames]
        pitch_adjusted, vertical_pairs = self._align_rows(scan_path, rows, yaw_adjusted)

        return AlignmentResult(
            frames=pitch_adjusted,
            horizontal_pairs=horizontal_pairs,
            vertical_pairs=vertical_pairs,
            applied=bool(horizontal_pairs or vertical_pairs),
        )

    def _align_row(
        self,
        scan_path: Path,
        row_frames: list[FrameMetadata],
    ) -> tuple[list[FrameMetadata], list[HorizontalAlignment]]:
        sorted_frames = sorted(row_frames, key=self._frame_yaw)
        aligned_yaws = [self._frame_yaw(frame) for frame in sorted_frames]
        pairs: list[HorizontalAlignment] = []

        for index, (left, right) in enumerate(zip(sorted_frames, sorted_frames[1:]), start=1):
            pair = self._align_horizontal_pair(scan_path, left, right)
            if pair is None:
                continue
            pairs.append(pair)
            aligned_yaws[index] = aligned_yaws[index - 1] + pair.observed_delta_yaw_deg

        if not pairs:
            return row_frames, []

        corrections = [
            aligned_yaw - self._frame_yaw(frame)
            for aligned_yaw, frame in zip(aligned_yaws, sorted_frames)
        ]
        offset = float(np.median(corrections))
        adjusted = [
            _with_angles(frame, yaw_deg=aligned_yaw - offset, pitch_deg=self._frame_pitch(frame))
            for frame, aligned_yaw in zip(sorted_frames, aligned_yaws)
        ]
        return adjusted, pairs

    def _align_rows(
        self,
        scan_path: Path,
        rows: list[tuple[float, list[FrameMetadata]]],
        yaw_adjusted: list[FrameMetadata],
    ) -> tuple[list[FrameMetadata], list[VerticalAlignment]]:
        adjusted_lookup = {frame.file: frame for frame in yaw_adjusted}
        row_frames = [
            [adjusted_lookup.get(frame.file, frame) for frame in frames]
            for _pitch, frames in rows
        ]
        aligned_pitches = [
            float(np.median([self._frame_pitch(frame) for frame in frames]))
            for frames in row_frames
        ]
        pairs: list[VerticalAlignment] = []

        for row_index, (lower_row, upper_row) in enumerate(zip(row_frames, row_frames[1:]), start=1):
            observed: list[float] = []
            for lower, upper in _pair_columns(lower_row, upper_row):
                if abs(self._frame_yaw(lower) - self._frame_yaw(upper)) > self.max_yaw_delta_deg:
                    continue
                pair = self._align_vertical_pair(scan_path, lower, upper)
                if pair is None:
                    continue
                pairs.append(pair)
                observed.append(pair.observed_delta_pitch_deg)
            if observed:
                aligned_pitches[row_index] = aligned_pitches[row_index - 1] + float(np.median(observed))

        if not pairs:
            return yaw_adjusted, []

        nominal_rows = [pitch for pitch, _frames in rows]
        corrections = [
            aligned_pitch - nominal_pitch
            for aligned_pitch, nominal_pitch in zip(aligned_pitches, nominal_rows)
        ]
        offset = float(np.median(corrections))
        pitch_by_file = {}
        for row_pitch, frames in zip(aligned_pitches, row_frames):
            adjusted_pitch = row_pitch - offset
            for frame in frames:
                pitch_by_file[frame.file] = adjusted_pitch

        adjusted = [
            _with_angles(
                frame,
                yaw_deg=self._frame_yaw(frame),
                pitch_deg=pitch_by_file.get(frame.file, self._frame_pitch(frame)),
            )
            for frame in yaw_adjusted
        ]
        return adjusted, pairs

    def _align_horizontal_pair(
        self,
        scan_path: Path,
        left: FrameMetadata,
        right: FrameMetadata,
    ) -> HorizontalAlignment | None:
        if left.hfov_deg is None or right.hfov_deg is None:
            return None

        left_full = cv2.imread(str(scan_path / left.file), cv2.IMREAD_GRAYSCALE)
        right_full = cv2.imread(str(scan_path / right.file), cv2.IMREAD_GRAYSCALE)
        if left_full is None or right_full is None:
            return None

        left_image, scale = _scaled_image(left_full, self.registration_scale)
        right_image, _ = _scaled_image(right_full, self.registration_scale)
        detector, matcher = _create_feature_tools(self.feature_finder)
        left_keypoints, left_descriptors = detector.detectAndCompute(left_image, None)
        right_keypoints, right_descriptors = detector.detectAndCompute(right_image, None)
        if left_descriptors is None or right_descriptors is None:
            return None

        matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        good_matches = _ratio_test_matches(matches)
        if len(good_matches) < self.min_matches:
            return None

        left_points = np.float32([left_keypoints[match.queryIdx].pt for match in good_matches])
        right_points = np.float32([right_keypoints[match.trainIdx].pt for match in good_matches])
        transform, inliers = cv2.estimateAffinePartial2D(
            right_points,
            left_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4,
        )
        if transform is None or inliers is None:
            return None

        inlier_count = int(inliers.sum())
        if inlier_count < self.min_inliers:
            return None

        dx_px = float(transform[0, 2] / scale)
        dy_px = float(transform[1, 2] / scale)
        hfov_deg = (left.hfov_deg + right.hfov_deg) / 2
        observed_delta_yaw_deg = dx_px / left_full.shape[1] * hfov_deg
        return HorizontalAlignment(
            left_frame=left.file,
            right_frame=right.file,
            nominal_delta_yaw_deg=self._frame_yaw(right) - self._frame_yaw(left),
            observed_delta_yaw_deg=observed_delta_yaw_deg,
            dx_px=dx_px,
            dy_px=dy_px,
            matches=len(good_matches),
            inliers=inlier_count,
        )

    def _align_vertical_pair(
        self,
        scan_path: Path,
        lower: FrameMetadata,
        upper: FrameMetadata,
    ) -> VerticalAlignment | None:
        if lower.vfov_deg is None or upper.vfov_deg is None:
            return None

        lower_full = cv2.imread(str(scan_path / lower.file), cv2.IMREAD_GRAYSCALE)
        upper_full = cv2.imread(str(scan_path / upper.file), cv2.IMREAD_GRAYSCALE)
        if lower_full is None or upper_full is None:
            return None

        lower_image, scale = _scaled_image(lower_full, self.registration_scale)
        upper_image, _ = _scaled_image(upper_full, self.registration_scale)
        detector, matcher = _create_feature_tools(self.feature_finder)
        lower_keypoints, lower_descriptors = detector.detectAndCompute(lower_image, None)
        upper_keypoints, upper_descriptors = detector.detectAndCompute(upper_image, None)
        if lower_descriptors is None or upper_descriptors is None:
            return None

        matches = matcher.knnMatch(lower_descriptors, upper_descriptors, k=2)
        good_matches = _ratio_test_matches(matches)
        if len(good_matches) < self.min_matches:
            return None

        lower_points = np.float32([lower_keypoints[match.queryIdx].pt for match in good_matches])
        upper_points = np.float32([upper_keypoints[match.trainIdx].pt for match in good_matches])
        transform, inliers = cv2.estimateAffinePartial2D(
            upper_points,
            lower_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4,
        )
        if transform is None or inliers is None:
            return None

        inlier_count = int(inliers.sum())
        if inlier_count < self.min_inliers:
            return None

        dx_px = float(transform[0, 2] / scale)
        dy_px = float(transform[1, 2] / scale)
        vfov_deg = (lower.vfov_deg + upper.vfov_deg) / 2
        observed_delta_pitch_deg = -dy_px / lower_full.shape[0] * vfov_deg
        return VerticalAlignment(
            lower_frame=lower.file,
            upper_frame=upper.file,
            nominal_delta_pitch_deg=self._frame_pitch(upper) - self._frame_pitch(lower),
            observed_delta_pitch_deg=observed_delta_pitch_deg,
            dx_px=dx_px,
            dy_px=dy_px,
            matches=len(good_matches),
            inliers=inlier_count,
        )

    def _frame_yaw(self, frame: FrameMetadata) -> float:
        if frame.pose.yaw_deg is not None:
            return frame.pose.yaw_deg
        return frame.pose.pan / self.pan_units_per_degree

    def _frame_pitch(self, frame: FrameMetadata) -> float:
        if frame.pose.pitch_deg is not None:
            return frame.pose.pitch_deg
        return frame.pose.tilt / self.tilt_units_per_degree


def _with_angles(frame: FrameMetadata, yaw_deg: float, pitch_deg: float) -> FrameMetadata:
    pose = CameraPose(
        pan=frame.pose.pan,
        tilt=frame.pose.tilt,
        zoom=frame.pose.zoom,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )
    return replace(frame, pose=pose)


def _create_feature_tools(feature_finder: str) -> tuple[cv2.Feature2D, cv2.DescriptorMatcher]:
    if feature_finder == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), cv2.BFMatcher(cv2.NORM_L2)
    return cv2.ORB_create(nfeatures=5000), cv2.BFMatcher(cv2.NORM_HAMMING)


def _scaled_image(image: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    if scale >= 1:
        return image, 1.0
    scale = max(scale, 0.1)
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), scale


def _ratio_test_matches(matches: tuple[tuple[cv2.DMatch, ...], ...]) -> list[cv2.DMatch]:
    good_matches = []
    for candidates in matches:
        if len(candidates) < 2:
            continue
        nearest, second = candidates[:2]
        if nearest.distance < 0.75 * second.distance:
            good_matches.append(nearest)
    return good_matches

def _group_rows(
    frames: list[FrameMetadata],
    tilt_units_per_degree: float,
) -> list[tuple[float, list[FrameMetadata]]]:
    rows: dict[float, list[FrameMetadata]] = {}
    for frame in frames:
        pitch = frame.pose.pitch_deg
        if pitch is None:
            pitch = frame.pose.tilt / tilt_units_per_degree
        pitch = round(pitch, 6)
        rows.setdefault(pitch, []).append(frame)
    return [
        (
            pitch,
            sorted(
                row_frames,
                key=lambda frame: (
                    frame.pose.yaw_deg
                    if frame.pose.yaw_deg is not None
                    else frame.pose.pan
                ),
            ),
        )
        for pitch, row_frames in sorted(rows.items(), key=lambda item: item[0])
    ]


def _pair_columns(
    lower_row: list[FrameMetadata],
    upper_row: list[FrameMetadata],
) -> list[tuple[FrameMetadata, FrameMetadata]]:
    pairs = []
    available = list(upper_row)
    for lower in sorted(
        lower_row,
        key=lambda frame: frame.pose.yaw_deg if frame.pose.yaw_deg is not None else frame.pose.pan,
    ):
        if not available:
            break
        upper = min(
            available,
            key=lambda frame: abs(
                (frame.pose.yaw_deg if frame.pose.yaw_deg is not None else frame.pose.pan)
                - (lower.pose.yaw_deg if lower.pose.yaw_deg is not None else lower.pose.pan)
            ),
        )
        available.remove(upper)
        pairs.append((lower, upper))
    return pairs
