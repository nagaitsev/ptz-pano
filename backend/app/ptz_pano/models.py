from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class CameraConfig:
    vendor: str
    host: str
    port: int
    transport: Literal["tcp", "udp"] = "tcp"
    profile: str = "ptzoptics-visca-tcp"
    timeout_sec: float = 2.0


@dataclass(frozen=True)
class CaptureConfig:
    kind: Literal["rtsp", "snapshot"]
    source: str
    resolution: tuple[int, int] | None = None


@dataclass(frozen=True)
class CameraPose:
    pan: int
    tilt: int
    zoom: int
    yaw_deg: float | None = None
    pitch_deg: float | None = None


@dataclass(frozen=True)
class FovSample:
    zoom: int
    hfov_deg: float
    vfov_deg: float


@dataclass(frozen=True)
class FrameMetadata:
    index: int
    file: str
    pose: CameraPose
    hfov_deg: float | None = None
    vfov_deg: float | None = None


@dataclass
class ScanDocument:
    id: str
    camera: CameraConfig
    capture: CaptureConfig
    frames: list[FrameMetadata] = field(default_factory=list)


@dataclass(frozen=True)
class Hotspot:
    id: str
    title: str
    panorama_yaw_deg: float
    panorama_pitch_deg: float
    pose: CameraPose


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def camera_config_from_dict(data: dict[str, Any]) -> CameraConfig:
    return CameraConfig(**data)


def capture_config_from_dict(data: dict[str, Any]) -> CaptureConfig:
    resolution = data.get("resolution")
    normalized = dict(data)
    if resolution is not None:
        normalized["resolution"] = tuple(resolution)
    return CaptureConfig(**normalized)


def pose_from_dict(data: dict[str, Any]) -> CameraPose:
    return CameraPose(**data)

