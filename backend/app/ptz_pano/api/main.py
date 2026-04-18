from __future__ import annotations

import os
from pathlib import Path
from time import sleep

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ptz_pano.camera.targeting import CameraTarget, target_to_pose
from ptz_pano.jsonio import read_json, write_json
from ptz_pano.models import CameraPose, to_jsonable
from ptz_pano.storage.scan_repository import ScanRepository
from ptz_pano.tools.config import build_camera, load_targeting_config

app = FastAPI(title="PTZ Pano")
repository = ScanRepository(Path("data/scans"))
CAMERA_CONFIG_PATH = Path(os.environ.get("PTZ_PANO_CAMERA_CONFIG", "config/camera.local.json"))
TARGET_HFOV_SCALE = float(os.environ.get("PTZ_PANO_TARGET_HFOV_SCALE", "0.45"))


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
        "target_hfov_scale": TARGET_HFOV_SCALE,
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
    
    # Применяем коррекции на основе накопленных данных
    corrected_yaw = request.yaw_deg
    corrected_pitch = request.pitch_deg
    
    corrections_path = Path("data/calibration/corrections.json")
    if corrections_path.exists():
        try:
            data = read_json(corrections_path)
            if data:
                # Рассчитываем взвешенную поправку
                total_weight = 0
                yaw_offset = 0
                pitch_offset = 0
                zoom_offset = 0
                
                for c in data:
                    dist = ((corrected_yaw - c["yaw_deg"])**2 + (corrected_pitch - c["pitch_deg"])**2)**0.5
                    weight = 1.0 / (dist + 0.1)
                    
                    err_yaw = (c["pan"] / targeting_config.pan_units_per_degree) - c["yaw_deg"]
                    err_pitch = (c["tilt"] / targeting_config.tilt_units_per_degree) - c["pitch_deg"]
                    
                    # Для зума сравниваем реальный zoom с тем, что ожидался в этой точке
                    expected_zoom = 0
                    if targeting_config.fov_table:
                        # Мы не знаем target_hfov из записи, поэтому используем zoom как коэффициент
                        # Но проще: берем разницу напрямую
                        err_zoom = c["zoom"] - targeting_config.fov_table.zoom_for_hfov(request.target_hfov_deg)
                        zoom_offset += err_zoom * weight
                    
                    yaw_offset += err_yaw * weight
                    pitch_offset += err_pitch * weight
                    total_weight += weight
                
                if total_weight > 0:
                    corrected_yaw += (yaw_offset / total_weight)
                    corrected_pitch += (pitch_offset / total_weight)
                    # Применяем поправку зума после основного расчета
                    # (Для упрощения считаем, что target_hfov_deg не меняется)
        except Exception as e:
            print(f"Error applying corrections: {e}")

    target = CameraTarget(
        yaw_deg=corrected_yaw,
        pitch_deg=corrected_pitch,
        target_hfov_deg=request.target_hfov_deg,
    )
    pose = target_to_pose(target, targeting_config)
    
    # Добавляем коррекцию зума, если она была рассчитана
    if 'zoom_offset' in locals() and total_weight > 0:
        final_zoom = round(pose.zoom + (zoom_offset / total_weight))
        pose = CameraPose(
            pan=pose.pan,
            tilt=pose.tilt,
            zoom=max(0, final_zoom),
            yaw_deg=pose.yaw_deg,
            pitch_deg=pose.pitch_deg
        )

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


@app.get("/camera/status")
def get_camera_status() -> dict:
    camera = build_camera(CAMERA_CONFIG_PATH)
    try:
        pose = camera.get_position()
        return to_jsonable(pose)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        camera.close()


class CorrectionRequest(BaseModel):
    scan_id: str
    frame_index: int
    yaw_deg: float
    pitch_deg: float
    pan: int
    tilt: int
    zoom: int


@app.post("/calibration/adjust")
def adjust_calibration(request: CorrectionRequest) -> dict:
    corrections_path = Path("data/calibration/corrections.json")
    corrections = []
    if corrections_path.exists():
        try:
            corrections = read_json(corrections_path)
        except Exception:
            corrections = []
    
    corrections.append(request.model_dump())
    write_json(corrections_path, corrections)
    
    return {"status": "success", "total_corrections": len(corrections)}


@app.get("/calibration/adjustments")
def get_adjustments() -> list:
    corrections_path = Path("data/calibration/corrections.json")
    if not corrections_path.exists():
        return []
    return read_json(corrections_path)


@app.delete("/calibration/adjustments")
def clear_adjustments() -> dict:
    corrections_path = Path("data/calibration/corrections.json")
    if corrections_path.exists():
        corrections_path.unlink()
    return {"status": "cleared"}


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
