from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from ptz_pano.models import CameraPose, FrameMetadata


@dataclass(frozen=True)
class PairAlignment:
    left_frame: str
    right_frame: str
    nominal_delta_yaw_deg: float
    observed_delta_yaw_deg: float
    dx_px: float
    dy_px: float
    matches: int
    inliers: int


@dataclass(frozen=True)
class AlignmentResult:
    frames: list[FrameMetadata]
    pairs: list[PairAlignment]
    applied: bool


@dataclass(frozen=True)
class FeatureAligner:
    min_matches: int = 24
    min_inliers: int = 12
    max_pitch_delta_deg: float = 5.0

    def align(self, scan_path: Path, frames: list[FrameMetadata]) -> AlignmentResult:
        if len(frames) < 2:
            return AlignmentResult(frames=frames, pairs=[], applied=False)

        sorted_frames = sorted(frames, key=lambda frame: (_frame_pitch(frame), _frame_yaw(frame)))
        aligned_yaws = [_frame_yaw(frame) for frame in sorted_frames]
        pairs: list[PairAlignment] = []

        for index, (left, right) in enumerate(zip(sorted_frames, sorted_frames[1:]), start=1):
            if abs(_frame_pitch(left) - _frame_pitch(right)) > self.max_pitch_delta_deg:
                continue
            pair = self._align_pair(scan_path, left, right)
            if pair is None:
                continue
            pairs.append(pair)
            aligned_yaws[index] = aligned_yaws[index - 1] + pair.observed_delta_yaw_deg

        if not pairs:
            return AlignmentResult(frames=frames, pairs=[], applied=False)

        corrections = [
            aligned_yaw - _frame_yaw(frame)
            for aligned_yaw, frame in zip(aligned_yaws, sorted_frames)
        ]
        offset = float(np.median(corrections))
        adjusted_by_file = {
            frame.file: _with_yaw(frame, aligned_yaw - offset)
            for frame, aligned_yaw in zip(sorted_frames, aligned_yaws)
        }
        adjusted = [adjusted_by_file.get(frame.file, frame) for frame in frames]
        return AlignmentResult(frames=adjusted, pairs=pairs, applied=True)

    def _align_pair(
        self,
        scan_path: Path,
        left: FrameMetadata,
        right: FrameMetadata,
    ) -> PairAlignment | None:
        if left.hfov_deg is None or right.hfov_deg is None:
            return None

        left_image = cv2.imread(str(scan_path / left.file), cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(str(scan_path / right.file), cv2.IMREAD_GRAYSCALE)
        if left_image is None or right_image is None:
            return None

        orb = cv2.ORB_create(nfeatures=5000)
        left_keypoints, left_descriptors = orb.detectAndCompute(left_image, None)
        right_keypoints, right_descriptors = orb.detectAndCompute(right_image, None)
        if left_descriptors is None or right_descriptors is None:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        good_matches = []
        for nearest, second in matches:
            if nearest.distance < 0.75 * second.distance:
                good_matches.append(nearest)
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

        dx_px = float(transform[0, 2])
        dy_px = float(transform[1, 2])
        hfov_deg = (left.hfov_deg + right.hfov_deg) / 2
        observed_delta_yaw_deg = dx_px / left_image.shape[1] * hfov_deg
        return PairAlignment(
            left_frame=left.file,
            right_frame=right.file,
            nominal_delta_yaw_deg=_frame_yaw(right) - _frame_yaw(left),
            observed_delta_yaw_deg=observed_delta_yaw_deg,
            dx_px=dx_px,
            dy_px=dy_px,
            matches=len(good_matches),
            inliers=inlier_count,
        )


def _with_yaw(frame: FrameMetadata, yaw_deg: float) -> FrameMetadata:
    pose = CameraPose(
        pan=frame.pose.pan,
        tilt=frame.pose.tilt,
        zoom=frame.pose.zoom,
        yaw_deg=yaw_deg,
        pitch_deg=frame.pose.pitch_deg,
    )
    return replace(frame, pose=pose)


def _frame_yaw(frame: FrameMetadata) -> float:
    if frame.pose.yaw_deg is not None:
        return frame.pose.yaw_deg
    return float(frame.pose.pan)


def _frame_pitch(frame: FrameMetadata) -> float:
    if frame.pose.pitch_deg is not None:
        return frame.pose.pitch_deg
    return float(frame.pose.tilt)

