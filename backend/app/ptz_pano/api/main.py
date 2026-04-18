from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from time import sleep
from typing import Callable, Literal
from uuid import uuid4

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ptz_pano.calibration import FovTable
from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.camera.targeting import CameraTarget, target_to_pose
from ptz_pano.jsonio import read_json, write_json
from ptz_pano.models import CameraPose, ScanDocument, to_jsonable
from ptz_pano.scan import ScanPlanConfig, ScanPlanner, ScanRunner
from ptz_pano.storage.scan_repository import ScanRepository
from ptz_pano.stitching import PanoramaBuilder
from ptz_pano.stitching.simple_compositor import SimpleCompositor
from ptz_pano.tools.config import (
    build_camera,
    build_capture,
    load_app_config,
    load_camera_config,
    load_capture_config,
    load_targeting_config,
)

app = FastAPI(title="PTZ Pano")
repository = ScanRepository(Path("data/scans"))
CAMERA_CONFIG_PATH = Path(os.environ.get("PTZ_PANO_CAMERA_CONFIG", "config/camera.local.json"))
TARGET_HFOV_SCALE = float(os.environ.get("PTZ_PANO_TARGET_HFOV_SCALE", "0.45"))
DEFAULT_LENS_CALIBRATION_PATH = Path(
    os.environ.get("PTZ_PANO_LENS_CALIBRATION", "config/lens_calibration.local.json")
)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


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


class StitchRequest(BaseModel):
    scan_id: str
    strategy: Literal["average", "max_weight"] = "max_weight"
    projection: Literal["angular", "sphere"] = "sphere"
    use_lens_calibration: bool = True
    lens_calibration_path: str | None = None


class ScanAndStitchRequest(BaseModel):
    scan_id: str | None = None
    stitch_after: bool = True
    strategy: Literal["average", "max_weight"] = "max_weight"
    projection: Literal["angular", "sphere"] = "sphere"
    use_lens_calibration: bool = True
    lens_calibration_path: str | None = None


@app.post("/api/stitch")
def start_stitch_job(request: StitchRequest) -> dict:
    if not repository.scan_path(request.scan_id).exists():
        raise HTTPException(status_code=404, detail="scan not found")
    job_id = _start_job(
        "stitch",
        lambda: _build_panorama(
            scan_id=request.scan_id,
            strategy=request.strategy,
            projection=request.projection,
            use_lens_calibration=request.use_lens_calibration,
            lens_calibration_path=request.lens_calibration_path,
        ),
    )
    return {"job_id": job_id}


@app.post("/api/scan-and-stitch")
def start_scan_and_stitch_job(request: ScanAndStitchRequest) -> dict:
    scan_id = request.scan_id or time.strftime("scan_%Y%m%d_%H%M%S")
    if repository.scan_path(scan_id).exists():
        raise HTTPException(status_code=409, detail="scan already exists")

    def run() -> dict:
        scan_path = _run_scan(scan_id)
        result: dict = {"scan_id": scan_id, "scan_path": str(scan_path)}
        if request.stitch_after:
            result["panorama"] = _build_panorama(
                scan_id=scan_id,
                strategy=request.strategy,
                projection=request.projection,
                use_lens_calibration=request.use_lens_calibration,
                lens_calibration_path=request.lens_calibration_path,
            )
        return result

    job_id = _start_job("scan_and_stitch", run)
    return {"job_id": job_id, "scan_id": scan_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(job)


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


@app.get("/chessboard", response_class=HTMLResponse)
def chessboard_page() -> str:
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <title>Chessboard</title>
        <style>
            :root { --cell-size: 80px; }
            * { box-sizing: border-box; }
            body, html {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: #777;
            }
            body {
                display: grid;
                place-items: center;
                width: 100vw;
                height: 100vh;
                height: 100dvh;
                padding:
                    max(12px, env(safe-area-inset-top))
                    max(12px, env(safe-area-inset-right))
                    max(12px, env(safe-area-inset-bottom))
                    max(12px, env(safe-area-inset-left));
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(10, var(--cell-size));
                grid-template-rows: repeat(7, var(--cell-size));
                width: calc(var(--cell-size) * 10);
                height: calc(var(--cell-size) * 7);
                box-shadow: 0 0 0 3px #fff;
            }
            .square { width: 100%; height: 100%; }
            .black { background: black; }
            .white { background: white; }
        </style>
    </head>
    <body>
        <div class="grid">
            <!-- Генерируем 70 квадратов (10x7) -->
            <script>
                const grid = document.querySelector('.grid');
                const columns = 10;
                const rows = 7;
                const margin = 24;

                function fitBoard() {
                    const viewport = window.visualViewport;
                    const width = viewport ? viewport.width : window.innerWidth;
                    const height = viewport ? viewport.height : window.innerHeight;
                    const cell = Math.floor(Math.min((width - margin * 2) / columns, (height - margin * 2) / rows));
                    document.documentElement.style.setProperty('--cell-size', `${Math.max(12, cell)}px`);
                }

                for (let r = 0; r < 7; r++) {
                    for (let c = 0; c < 10; c++) {
                        const div = document.createElement('div');
                        div.className = 'square ' + ((r + c) % 2 === 0 ? 'white' : 'black');
                        grid.appendChild(div);
                    }
                }

                fitBoard();
                window.addEventListener('resize', fitBoard);
                if (window.visualViewport) {
                    window.visualViewport.addEventListener('resize', fitBoard);
                }
            </script>
        </div>
    </body>
    </html>
    """


@app.get("/calibrate-lens", response_class=HTMLResponse)
def calibrate_lens_page() -> str:
    return (Path(__file__).resolve().parent / "calibrate_lens.html").read_text(encoding="utf-8")


@app.post("/calibration/lens/capture")
def capture_lens_sample() -> dict:
    camera = build_camera(CAMERA_CONFIG_PATH)
    capture = build_capture(CAMERA_CONFIG_PATH)
    try:
        pose = camera.get_position()
        zoom = pose.zoom
        
        # Создаем папку для образцов
        sample_dir = Path(f"data/calibration/lens_samples/{zoom}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{timestamp}.jpg"
        filepath = sample_dir / filename
        
        capture.grab_frame(filepath)
        
        img = cv2.imread(str(filepath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners, detector = _find_chessboard_corners(gray, (9, 6))
        
        preview_filename = f"preview_{timestamp}.jpg"
        preview_path = Path(f"data/calibration/lens_samples/previews/{preview_filename}")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        
        if found:
            cv2.drawChessboardCorners(img, (9, 6), corners, found)
        
        cv2.imwrite(str(preview_path), img)
        
        return {
            "status": "ok",
            "zoom": zoom,
            "found": found,
            "detector": detector,
            "preview_url": f"/calibration/lens/preview/{preview_filename}"
        }
    finally:
        camera.close()


@app.get("/calibration/lens/preview/{filename}")
def get_lens_preview(filename: str) -> FileResponse:
    path = Path(f"data/calibration/lens_samples/previews/{filename}")
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path)


def _find_chessboard_corners(
    gray: cv2.typing.MatLike,
    pattern_size: tuple[int, int],
) -> tuple[bool, cv2.typing.MatLike | None, str | None]:
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            return True, corners, "findChessboardCornersSB"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined, "findChessboardCorners"


def _start_job(kind: str, target: Callable[[], dict]) -> str:
    job_id = uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def run_job() -> None:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()
        try:
            result = target()
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["error"] = str(exc)
        else:
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["result"] = result

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    return job_id


def _run_scan(scan_id: str) -> Path:
    raw_config = load_app_config(CAMERA_CONFIG_PATH)
    fov_table = None
    pan_units_per_degree = None
    tilt_units_per_degree = None
    calibration_config = raw_config.get("calibration")
    if calibration_config:
        if calibration_config.get("fov_table"):
            fov_table = FovTable.load(Path(calibration_config["fov_table"]))
        pan_units_per_degree = calibration_config.get("pan_units_per_degree")
        tilt_units_per_degree = calibration_config.get("tilt_units_per_degree")

    raw_scan_config = dict(raw_config["scan"])
    settle_sec = raw_scan_config.pop("settle_sec", 1.0)
    scan_config = ScanPlanConfig(**raw_scan_config)
    document = ScanDocument(
        id=scan_id,
        camera=load_camera_config(CAMERA_CONFIG_PATH),
        capture=load_capture_config(CAMERA_CONFIG_PATH),
    )

    camera = build_camera(CAMERA_CONFIG_PATH)
    capture = build_capture(CAMERA_CONFIG_PATH)
    runner = ScanRunner(
        camera=camera,
        capture=capture,
        repository=repository,
        settle_sec=settle_sec,
        fov_table=fov_table,
        pan_units_per_degree=pan_units_per_degree,
        tilt_units_per_degree=tilt_units_per_degree,
    )
    try:
        return runner.run(document, ScanPlanner(scan_config))
    finally:
        camera.close()


def _build_panorama(
    scan_id: str,
    strategy: Literal["average", "max_weight"],
    projection: Literal["angular", "sphere"],
    use_lens_calibration: bool,
    lens_calibration_path: str | None,
) -> dict:
    lens_calibration = None
    resolved_lens_path = None
    if use_lens_calibration:
        candidate = Path(lens_calibration_path) if lens_calibration_path else DEFAULT_LENS_CALIBRATION_PATH
        if candidate.exists():
            lens_calibration = LensCalibration.from_file(candidate)
            resolved_lens_path = str(candidate)

    compositor = SimpleCompositor(
        lens_calibration=lens_calibration,
        strategy=strategy,
        projection=projection,
    )
    manifest_path = PanoramaBuilder(repository, compositor=compositor).build_manifest(scan_id)
    return {
        "scan_id": scan_id,
        "manifest_path": str(manifest_path),
        "preview_url": f"/scans/{scan_id}/panorama/preview.jpg",
        "panorama_url": f"/scans/{scan_id}/panorama/panorama.jpg",
        "strategy": strategy,
        "projection": projection,
        "lens_calibration_path": resolved_lens_path,
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
