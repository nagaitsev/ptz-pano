from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ptz_pano.camera.interfaces import CameraController, FrameCapture
from ptz_pano.calibration import FovTable
from ptz_pano.models import CameraPose, FrameMetadata, ScanDocument
from ptz_pano.scan.scan_planner import ScanPlanner
from ptz_pano.storage.scan_repository import ScanRepository


@dataclass
class ScanRunner:
    camera: CameraController
    capture: FrameCapture
    repository: ScanRepository
    settle_sec: float = 1.0
    fov_table: FovTable | None = None
    pan_units_per_degree: float | None = None
    tilt_units_per_degree: float | None = None

    def run(self, document: ScanDocument, planner: ScanPlanner) -> Path:
        scan_path = self.repository.create_scan(document.id)
        frames_path = scan_path / "frames"

        for index, pose in enumerate(planner.poses(), start=1):
            self.camera.move_absolute(pose)
            time.sleep(self.settle_sec)
            actual_pose = self._pose_with_angles(self.camera.get_position())
            hfov_deg, vfov_deg = self._fov_for_zoom(actual_pose.zoom)
            frame_name = f"frame_{index:04d}.jpg"
            self.capture.grab_frame(frames_path / frame_name)
            document.frames.append(
                FrameMetadata(
                    index=index,
                    file=f"frames/{frame_name}",
                    pose=actual_pose,
                    hfov_deg=hfov_deg,
                    vfov_deg=vfov_deg,
                )
            )
            self.repository.save_document(document)

        return scan_path

    def _fov_for_zoom(self, zoom: int) -> tuple[float | None, float | None]:
        if self.fov_table is None:
            return None, None
        return self.fov_table.fov_for_zoom(zoom)

    def _pose_with_angles(self, pose: CameraPose) -> CameraPose:
        yaw_deg = pose.yaw_deg
        pitch_deg = pose.pitch_deg
        if yaw_deg is None and self.pan_units_per_degree:
            yaw_deg = pose.pan / self.pan_units_per_degree
        if pitch_deg is None and self.tilt_units_per_degree:
            pitch_deg = pose.tilt / self.tilt_units_per_degree
        return CameraPose(
            pan=pose.pan,
            tilt=pose.tilt,
            zoom=pose.zoom,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
        )
