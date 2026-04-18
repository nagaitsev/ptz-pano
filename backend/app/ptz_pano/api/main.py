from __future__ import annotations

import os
from pathlib import Path
from time import sleep

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ptz_pano.camera.targeting import CameraTarget, target_to_pose
from ptz_pano.models import to_jsonable
from ptz_pano.storage.scan_repository import ScanRepository
from ptz_pano.tools.config import build_camera, load_targeting_config

app = FastAPI(title="PTZ Pano")
repository = ScanRepository(Path("data/scans"))
CAMERA_CONFIG_PATH = Path(os.environ.get("PTZ_PANO_CAMERA_CONFIG", "config/camera.local.json"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scans/{scan_id}")
def get_scan(scan_id: str) -> dict:
    document = repository.load_document(scan_id)
    return {
        "id": document.id,
        "camera": document.camera,
        "capture": document.capture,
        "frames": document.frames,
    }


@app.get("/", response_class=HTMLResponse)
def viewer_page() -> str:
    return (Path(__file__).resolve().parent / "viewer.html").read_text(encoding="utf-8")


@app.get("/api/latest-scan")
def latest_scan() -> dict[str, str]:
    scan_id = _latest_scan_id()
    if scan_id is None:
        raise HTTPException(status_code=404, detail="no scans with panorama preview found")
    return {"id": scan_id}


@app.get("/scans/{scan_id}/panorama/{filename}")
def get_panorama_file(scan_id: str, filename: str) -> FileResponse:
    if filename not in {"panorama.jpg", "preview.jpg", "panorama_manifest.json"}:
        raise HTTPException(status_code=404, detail="unsupported panorama artifact")
    path = repository.scan_path(scan_id) / "panorama" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="panorama artifact not found")
    return FileResponse(path)


@app.get("/scans/{scan_id}/panorama-info")
def panorama_info(scan_id: str) -> dict:
    manifest_path = repository.scan_path(scan_id) / "panorama" / "panorama_manifest.json"
    preview_path = repository.scan_path(scan_id) / "panorama" / "preview.jpg"
    panorama_path = repository.scan_path(scan_id) / "panorama" / "panorama.jpg"
    if not manifest_path.exists() or not preview_path.exists() or not panorama_path.exists():
        raise HTTPException(status_code=404, detail="panorama artifacts are incomplete")

    from ptz_pano.jsonio import read_json

    manifest = read_json(manifest_path)
    preview = cv2.imread(str(preview_path))
    panorama = cv2.imread(str(panorama_path))
    if preview is None or panorama is None:
        raise HTTPException(status_code=500, detail="failed to read panorama images")

    return {
        "scan_id": scan_id,
        "preview_url": f"/scans/{scan_id}/panorama/preview.jpg",
        "panorama_url": f"/scans/{scan_id}/panorama/panorama.jpg",
        "preview_size": [preview.shape[1], preview.shape[0]],
        "panorama_size": [panorama.shape[1], panorama.shape[0]],
        "content_bbox": manifest.get("content_bbox"),
        "coverage_percent": manifest.get("coverage_percent"),
        "alignment": manifest.get("alignment"),
    }


class TargetRequest(BaseModel):
    yaw_deg: float = Field(ge=-180, le=180)
    pitch_deg: float = Field(ge=-90, le=90)
    target_hfov_deg: float = Field(gt=0, le=180)
    execute: bool = True


@app.post("/camera/target")
def move_camera_to_target(request: TargetRequest) -> dict:
    targeting_config = load_targeting_config(CAMERA_CONFIG_PATH)
    target = CameraTarget(
        yaw_deg=request.yaw_deg,
        pitch_deg=request.pitch_deg,
        target_hfov_deg=request.target_hfov_deg,
    )
    pose = target_to_pose(target, targeting_config)

    actual_pose = None
    if request.execute:
        camera = build_camera(CAMERA_CONFIG_PATH)
        try:
            camera.move_absolute(pose)
            sleep(0.3)
            actual_pose = camera.get_position()
        finally:
            camera.close()

    return {
        "target": request.model_dump(),
        "command_pose": to_jsonable(pose),
        "actual_pose": None if actual_pose is None else to_jsonable(actual_pose),
    }


def _latest_scan_id() -> str | None:
    candidates = []
    if not repository.root.exists():
        return None
    for path in repository.root.iterdir():
        if path.is_dir() and (path / "panorama" / "preview.jpg").exists():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).name
