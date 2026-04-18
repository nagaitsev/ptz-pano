from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ptz_pano.camera.interfaces import CameraController, FrameCapture
from ptz_pano.calibration import FovTable
from ptz_pano.models import FrameMetadata, ScanDocument
from ptz_pano.scan.scan_planner import ScanPlanner
from ptz_pano.storage.scan_repository import ScanRepository


@dataclass
class ScanRunner:
    camera: CameraController
    capture: FrameCapture
    repository: ScanRepository
    settle_sec: float = 1.0
    fov_table: FovTable | None = None

    def run(self, document: ScanDocument, planner: ScanPlanner) -> Path:
        scan_path = self.repository.create_scan(document.id)
        frames_path = scan_path / "frames"

        for index, pose in enumerate(planner.poses(), start=1):
            self.camera.move_absolute(pose)
            time.sleep(self.settle_sec)
            actual_pose = self.camera.get_position()
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
