from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ptz_pano.camera.interfaces import CameraController, FrameCapture
from ptz_pano.models import FrameMetadata, ScanDocument
from ptz_pano.scan.scan_planner import ScanPlanner
from ptz_pano.storage.scan_repository import ScanRepository


@dataclass
class ScanRunner:
    camera: CameraController
    capture: FrameCapture
    repository: ScanRepository
    settle_sec: float = 1.0

    def run(self, document: ScanDocument, planner: ScanPlanner) -> Path:
        scan_path = self.repository.create_scan(document.id)
        frames_path = scan_path / "frames"

        for index, pose in enumerate(planner.poses(), start=1):
            self.camera.move_absolute(pose)
            time.sleep(self.settle_sec)
            actual_pose = self.camera.get_position()
            frame_name = f"frame_{index:04d}.jpg"
            self.capture.grab_frame(frames_path / frame_name)
            document.frames.append(FrameMetadata(index=index, file=f"frames/{frame_name}", pose=actual_pose))
            self.repository.save_document(document)

        return scan_path
