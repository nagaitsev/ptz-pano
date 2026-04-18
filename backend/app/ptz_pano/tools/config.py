from __future__ import annotations

from pathlib import Path
from typing import Any

from ptz_pano.capture import RtspCapture, SnapshotCapture
from ptz_pano.camera.ptzoptics_visca_tcp import PtzOpticsViscaTcpController
from ptz_pano.jsonio import read_json
from ptz_pano.models import CameraConfig, CaptureConfig


def load_app_config(path: Path) -> dict[str, Any]:
    return read_json(path)


def load_camera_config(path: Path) -> CameraConfig:
    data = load_app_config(path)
    return CameraConfig(**data["camera"])


def load_capture_config(path: Path) -> CaptureConfig:
    data = load_app_config(path)
    capture = dict(data["capture"])
    if capture.get("resolution") is not None:
        capture["resolution"] = tuple(capture["resolution"])
    return CaptureConfig(**capture)


def build_camera(path: Path) -> PtzOpticsViscaTcpController:
    config = load_camera_config(path)
    if config.profile != "ptzoptics-visca-tcp":
        raise ValueError(f"unsupported camera profile: {config.profile}")
    return PtzOpticsViscaTcpController(config)


def build_capture(path: Path):
    config = load_capture_config(path)
    if config.kind == "rtsp":
        return RtspCapture(config.source)
    if config.kind == "snapshot":
        return SnapshotCapture(config.source)
    raise ValueError(f"unsupported capture kind: {config.kind}")

