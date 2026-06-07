from __future__ import annotations

import logging
import os
import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from time import sleep
from typing import Callable, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ptz_pano.calibration import FovTable
from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.camera.targeting import CameraTarget, target_to_pose
from ptz_pano.jsonio import read_json, write_json
from ptz_pano.logging_utils import setup_logging
from ptz_pano.models import CameraPose, ScanDocument, to_jsonable
from ptz_pano.scan import ScanPlanConfig, ScanPlanner, ScanRunner, apply_scan_angle_window
from ptz_pano.storage.scan_repository import ScanRepository
from ptz_pano.stitching import PanoramaBuilder
from ptz_pano.stitching.openstitching_compositor import OpenStitchingCompositor
from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor
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
_preview_workers: dict[str, "RtspPreviewWorker"] = {}
_preview_workers_lock = threading.Lock()
LOCAL_CORRECTIONS_PATH = Path("data/calibration/local_corrections.json")
LOCAL_CORRECTIONS_PER_AREA = 5
LOCAL_CORRECTION_AREA_RADIUS_DEG = 20.0
LOCAL_CORRECTION_REPLACE_RADIUS_DEG = 3.0
APP_LOG_PATH = Path(os.environ.get("PTZ_PANO_LOG_PATH", "data/logs/ptz-pano.log"))
setup_logging(APP_LOG_PATH)
logger = logging.getLogger("ptz_pano.api")


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


@app.get("/camera/video.mjpg")
def camera_video() -> StreamingResponse:
    capture_config = load_capture_config(CAMERA_CONFIG_PATH)
    if capture_config.kind != "rtsp":
        raise HTTPException(status_code=400, detail="live preview requires RTSP capture")
    worker = _preview_worker(capture_config.source)
    return StreamingResponse(
        _latest_frame_mjpeg_stream(worker.latest_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/preview.jpg")
def camera_preview_frame() -> Response:
    capture_config = load_capture_config(CAMERA_CONFIG_PATH)
    if capture_config.kind != "rtsp":
        raise HTTPException(status_code=400, detail="preview requires RTSP capture")
    try:
        content = _jpeg_preview_frame(capture_config.source)
    except RuntimeError as error:
        logger.warning("Camera preview frame failed for source %s: %s", capture_config.source, error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


class StitchRequest(BaseModel):
    scan_id: str
    stitch_engine: Literal["opencv", "openstitching"] = "opencv"
    strategy: Literal["average", "max_weight"] = "max_weight"
    projection: Literal["angular", "sphere"] = "sphere"
    stitch_quality: Literal["fast", "quality"] = "fast"
    use_lens_calibration: bool = True
    lens_calibration_path: str | None = None


class ScanAndStitchRequest(BaseModel):
    scan_id: str | None = None
    stitch_after: bool = True
    horizontal_angle_deg: float | None = Field(default=None, gt=0, le=360)
    vertical_angle_deg: float | None = Field(default=None, gt=0, le=180)
    scan_order: Literal["row_snake", "column_snake"] | None = None
    stitch_engine: Literal["opencv", "openstitching"] = "opencv"
    strategy: Literal["average", "max_weight"] = "max_weight"
    projection: Literal["angular", "sphere"] = "sphere"
    stitch_quality: Literal["fast", "quality"] = "fast"
    use_lens_calibration: bool = True
    lens_calibration_path: str | None = None


class CameraSettingsRequest(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    whep_url: str | None = None


class CameraJogRequest(BaseModel):
    x: float = Field(ge=-1, le=1)
    y: float = Field(ge=-1, le=1)


class CameraZoomStepRequest(BaseModel):
    direction: int = Field(ge=-1, le=1)
    step: int = Field(default=400, ge=1, le=4096)


@app.get("/api/camera-settings")
def get_camera_settings() -> dict:
    raw_config = load_app_config(CAMERA_CONFIG_PATH)
    camera_config = raw_config.get("camera", {})
    capture_config = raw_config.get("capture", {})
    return {
        "host": camera_config.get("host", ""),
        "camera_port": camera_config.get("port"),
        "capture_source": capture_config.get("source", ""),
        "whep_url": capture_config.get("whep_url", ""),
    }


@app.post("/api/camera-settings")
def update_camera_settings(request: CameraSettingsRequest) -> dict:
    host = _normalize_camera_host(request.host)
    raw_config = load_app_config(CAMERA_CONFIG_PATH)
    raw_config.setdefault("camera", {})["host"] = host

    capture_config = raw_config.get("capture")
    if isinstance(capture_config, dict) and isinstance(capture_config.get("source"), str):
        capture_config["source"] = _replace_url_host(capture_config["source"], host)
        if request.whep_url is not None:
            capture_config["whep_url"] = request.whep_url.strip()

    write_json(CAMERA_CONFIG_PATH, raw_config)
    return get_camera_settings()


@app.post("/api/stitch")
def start_stitch_job(request: StitchRequest) -> dict:
    if not repository.scan_path(request.scan_id).exists():
        raise HTTPException(status_code=404, detail="scan not found")
    job_id = _start_job(
        "stitch",
        lambda: _build_panorama(
            scan_id=request.scan_id,
            stitch_engine=request.stitch_engine,
            strategy=request.strategy,
            projection=request.projection,
            stitch_quality=request.stitch_quality,
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
        scan_path = _run_scan(
            scan_id,
            horizontal_angle_deg=request.horizontal_angle_deg,
            vertical_angle_deg=request.vertical_angle_deg,
            scan_order=request.scan_order,
        )
        result: dict = {"scan_id": scan_id, "scan_path": str(scan_path)}
        if request.stitch_after:
            result["panorama"] = _build_panorama(
                scan_id=scan_id,
                stitch_engine=request.stitch_engine,
                strategy=request.strategy,
                projection=request.projection,
                stitch_quality=request.stitch_quality,
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
        "stitching": manifest.get("stitching"),
    }


class TargetRequest(BaseModel):
    scan_id: str | None = None
    yaw_deg: float = Field(ge=-180, le=180)
    pitch_deg: float = Field(ge=-90, le=90)
    target_hfov_deg: float = Field(gt=0, le=180)
    execute: bool = True


@app.post("/camera/target")
def move_camera_to_target(request: TargetRequest) -> dict:
    targeting_config = load_targeting_config(CAMERA_CONFIG_PATH)
    local_corrections = _read_local_corrections()
    corrected = _apply_local_target_corrections(
        request.yaw_deg,
        request.pitch_deg,
        local_corrections,
        scan_id=request.scan_id,
    )

    target = CameraTarget(
        yaw_deg=corrected["yaw_deg"],
        pitch_deg=corrected["pitch_deg"],
        target_hfov_deg=request.target_hfov_deg,
    )
    pose = target_to_pose(target, targeting_config)
    logger.info(
        "Camera target request yaw=%.3f pitch=%.3f corrected_yaw=%.3f corrected_pitch=%.3f execute=%s",
        request.yaw_deg,
        request.pitch_deg,
        corrected["yaw_deg"],
        corrected["pitch_deg"],
        request.execute,
    )

    actual_pose = None
    try:
        if request.execute:
            camera = build_camera(CAMERA_CONFIG_PATH)
            try:
                camera.move_absolute(pose)
                sleep(0.3)
                actual_pose = camera.get_position()
            finally:
                camera.close()
    except Exception:
        logger.exception("Camera target execution failed")
        raise

    return {
        "target": request.model_dump(),
        "corrected_target": {
            "yaw_deg": corrected["yaw_deg"],
            "pitch_deg": corrected["pitch_deg"],
            "applied": corrected["applied"],
            "correction_count": corrected["correction_count"],
        },
        "command_pose": to_jsonable(pose),
        "actual_pose": None if actual_pose is None else to_jsonable(actual_pose),
    }


@app.get("/camera/status")
def get_camera_status() -> dict:
    try:
        camera = build_camera(CAMERA_CONFIG_PATH)
        try:
            pose = camera.get_position()
        finally:
            camera.close()
        logger.debug("Camera status read pan=%s tilt=%s zoom=%s", pose.pan, pose.tilt, pose.zoom)
        return to_jsonable(pose)
    except Exception as e:
        logger.exception("Camera status request failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camera/jog")
def jog_camera(request: CameraJogRequest) -> dict:
    pan_dir, pan_speed, tilt_dir, tilt_speed = _jog_command_from_vector(request.x, request.y)
    camera = build_camera(CAMERA_CONFIG_PATH)
    try:
        if pan_dir == 0 and tilt_dir == 0:
            camera.stop()
            logger.info("Camera jog stop")
            return {"status": "stopped", "x": request.x, "y": request.y}
        camera.move_direction(
            pan_speed=pan_speed,
            tilt_speed=tilt_speed,
            pan_dir=pan_dir,
            tilt_dir=tilt_dir,
        )
        logger.info(
            "Camera jog x=%.3f y=%.3f pan_dir=%s pan_speed=%s tilt_dir=%s tilt_speed=%s",
            request.x,
            request.y,
            pan_dir,
            pan_speed,
            tilt_dir,
            tilt_speed,
        )
        return {
            "status": "moving",
            "x": request.x,
            "y": request.y,
            "pan_dir": pan_dir,
            "pan_speed": pan_speed,
            "tilt_dir": tilt_dir,
            "tilt_speed": tilt_speed,
        }
    except Exception:
        logger.exception("Camera jog failed for x=%.3f y=%.3f", request.x, request.y)
        raise
    finally:
        camera.close()


@app.post("/camera/zoom-step")
def camera_zoom_step(request: CameraZoomStepRequest) -> dict:
    if request.direction == 0:
        return {"status": "idle"}
    camera = build_camera(CAMERA_CONFIG_PATH)
    try:
        pose = camera.get_position()
        next_zoom = _next_zoom_value(pose.zoom, request.direction, request.step)
        camera.set_zoom(next_zoom)
        logger.info(
            "Camera zoom step direction=%s step=%s from=%s to=%s",
            request.direction,
            request.step,
            pose.zoom,
            next_zoom,
        )
        return {
            "status": "ok",
            "direction": request.direction,
            "step": request.step,
            "zoom": next_zoom,
        }
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


class LocalCorrectionRequest(BaseModel):
    scan_id: str
    desired_yaw_deg: float = Field(ge=-180, le=180)
    desired_pitch_deg: float = Field(ge=-90, le=90)
    observed_yaw_deg: float = Field(ge=-180, le=180)
    observed_pitch_deg: float = Field(ge=-90, le=90)
    target_hfov_deg: float | None = Field(default=None, gt=0, le=180)


@app.post("/calibration/local-adjust")
def add_local_adjustment(request: LocalCorrectionRequest) -> dict:
    corrections = _read_local_corrections()
    correction = request.model_dump()
    correction["delta_yaw_deg"] = _normalize_delta(
        request.desired_yaw_deg - request.observed_yaw_deg
    )
    correction["delta_pitch_deg"] = request.desired_pitch_deg - request.observed_pitch_deg
    correction["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    corrections = _remember_local_correction(corrections, correction)
    LOCAL_CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(LOCAL_CORRECTIONS_PATH, corrections)
    return {"status": "success", "total_corrections": len(corrections), "correction": correction}


@app.get("/calibration/local-adjustments")
def get_local_adjustments() -> list:
    return _read_local_corrections()


@app.delete("/calibration/local-adjustments")
def clear_local_adjustments() -> dict:
    if LOCAL_CORRECTIONS_PATH.exists():
        LOCAL_CORRECTIONS_PATH.unlink()
    return {"status": "success"}


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


def _run_scan(
    scan_id: str,
    horizontal_angle_deg: float | None = None,
    vertical_angle_deg: float | None = None,
    scan_order: Literal["row_snake", "column_snake"] | None = None,
) -> Path:
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
    if scan_order is not None:
        scan_config = replace(scan_config, order=scan_order)
    document = ScanDocument(
        id=scan_id,
        camera=load_camera_config(CAMERA_CONFIG_PATH),
        capture=load_capture_config(CAMERA_CONFIG_PATH),
    )

    camera = build_camera(CAMERA_CONFIG_PATH)
    capture = build_capture(CAMERA_CONFIG_PATH)
    try:
        if horizontal_angle_deg is not None or vertical_angle_deg is not None:
            scan_config = apply_scan_angle_window(
                config=scan_config,
                center=camera.get_position(),
                horizontal_deg=horizontal_angle_deg,
                vertical_deg=vertical_angle_deg,
                pan_units_per_degree=pan_units_per_degree,
                tilt_units_per_degree=tilt_units_per_degree,
            )
        runner = ScanRunner(
            camera=camera,
            capture=capture,
            repository=repository,
            settle_sec=settle_sec,
            fov_table=fov_table,
            pan_units_per_degree=pan_units_per_degree,
            tilt_units_per_degree=tilt_units_per_degree,
        )
        return runner.run(document, ScanPlanner(scan_config))
    finally:
        camera.close()


def _build_panorama(
    scan_id: str,
    stitch_engine: Literal["opencv", "openstitching"],
    strategy: Literal["average", "max_weight"],
    projection: Literal["angular", "sphere"],
    stitch_quality: Literal["fast", "quality"],
    use_lens_calibration: bool,
    lens_calibration_path: str | None,
) -> dict:
    lens_calibration = None
    resolved_lens_path = None
    if use_lens_calibration and stitch_engine != "openstitching":
        candidate = Path(lens_calibration_path) if lens_calibration_path else DEFAULT_LENS_CALIBRATION_PATH
        if candidate.exists():
            lens_calibration = LensCalibration.from_file(candidate)
            resolved_lens_path = str(candidate)

    if stitch_engine == "openstitching":
        compositor = OpenStitchingCompositor(
            lens_calibration=lens_calibration,
            fallback_quality=stitch_quality,
        )
    else:
        compositor = OpenCvStitcherCompositor(
            lens_calibration=lens_calibration,
            fallback_quality=stitch_quality,
        )
    manifest_path = PanoramaBuilder(repository, compositor=compositor).build_manifest(scan_id)
    return {
        "scan_id": scan_id,
        "manifest_path": str(manifest_path),
        "preview_url": f"/scans/{scan_id}/panorama/preview.jpg",
        "panorama_url": f"/scans/{scan_id}/panorama/panorama.jpg",
        "strategy": compositor.strategy,
        "projection": compositor.projection,
        "stitch_engine": stitch_engine,
        "stitch_quality": getattr(compositor, "fallback_quality", None),
        "lens_calibration_path": resolved_lens_path,
    }


def _mjpeg_stream(
    source: str,
    capture_factory=None,
    jpeg_encoder=None,
    frame_delay_sec: float = 0.12,
    max_frames: int | None = None,
):
    capture_factory = capture_factory or cv2.VideoCapture
    jpeg_encoder = jpeg_encoder or _encode_jpeg
    capture = capture_factory(source)
    sent = 0
    try:
        while max_frames is None or sent < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                sleep(min(frame_delay_sec, 0.5))
                continue
            encoded_ok, encoded = jpeg_encoder(frame)
            if not encoded_ok:
                sleep(frame_delay_sec)
                continue
            yield _mjpeg_part(encoded)
            sent += 1
            sleep(frame_delay_sec)
    finally:
        capture.release()


def _jpeg_preview_frame(
    source: str,
    capture_factory=None,
    jpeg_encoder=None,
) -> bytes:
    capture_factory = capture_factory or cv2.VideoCapture
    jpeg_encoder = jpeg_encoder or _encode_jpeg
    capture = capture_factory(source)
    try:
        if hasattr(capture, "set"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read preview frame from RTSP source: {source}")
        encoded_ok, encoded = jpeg_encoder(frame)
        if not encoded_ok:
            raise RuntimeError("failed to encode preview frame")
        return encoded.tobytes()
    finally:
        capture.release()


class RtspPreviewWorker:
    def __init__(self, source: str) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._frame = None
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _run(self) -> None:
        while True:
            logger.info("Opening RTSP preview stream %s", self.source)
            capture = cv2.VideoCapture(_low_latency_rtsp_source(self.source))
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while True:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        logger.warning("RTSP preview read failed for %s; reconnecting", self.source)
                        break
                    with self._lock:
                        self._frame = frame
            finally:
                capture.release()
            sleep(0.5)


def _preview_worker(source: str) -> RtspPreviewWorker:
    with _preview_workers_lock:
        worker = _preview_workers.get(source)
        if worker is None:
            worker = RtspPreviewWorker(source)
            _preview_workers[source] = worker
        worker.start()
        return worker


def _latest_frame_mjpeg_stream(
    latest_frame,
    jpeg_encoder=None,
    frame_delay_sec: float = 0.08,
    max_frames: int | None = None,
):
    jpeg_encoder = jpeg_encoder or _encode_jpeg
    sent = 0
    while max_frames is None or sent < max_frames:
        frame = latest_frame()
        if frame is None:
            sleep(min(frame_delay_sec, 0.2))
            continue
        encoded_ok, encoded = jpeg_encoder(frame)
        if not encoded_ok:
            sleep(frame_delay_sec)
            continue
        yield _mjpeg_part(encoded)
        sent += 1
        sleep(frame_delay_sec)


def _encode_jpeg(frame):
    return cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])


def _mjpeg_part(encoded) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Cache-Control: no-store\r\n\r\n"
        + encoded.tobytes()
        + b"\r\n"
    )


def _low_latency_rtsp_source(source: str) -> str:
    if not source.startswith("rtsp://") or "?" in source:
        return source
    return f"{source}?rtsp_transport=tcp&fflags=nobuffer&flags=low_delay"



def _normalize_camera_host(value: str) -> str:
    host = value.strip()
    if "://" in host:
        parsed = urlsplit(host)
        host = parsed.hostname or ""
    if not host or any(char.isspace() for char in host):
        raise HTTPException(status_code=400, detail="invalid camera host")
    if "/" in host or ":" in host:
        raise HTTPException(status_code=400, detail="enter host or IP without port/path")
    return host


def _replace_url_host(source: str, host: str) -> str:
    parsed = urlsplit(source)
    if not parsed.scheme or not parsed.netloc:
        return source

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    netloc = f"{auth}{host}"
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _read_local_corrections() -> list[dict]:
    if not LOCAL_CORRECTIONS_PATH.exists():
        return []
    try:
        data = read_json(LOCAL_CORRECTIONS_PATH)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _apply_local_target_corrections(
    yaw_deg: float,
    pitch_deg: float,
    corrections: list[dict],
    scan_id: str | None = None,
    max_distance_deg: float = 45.0,
    sigma_deg: float = 18.0,
) -> dict[str, float | bool | int]:
    yaw_offset = 0.0
    pitch_offset = 0.0
    weight_total = 0.0
    used = 0
    for correction in corrections:
        try:
            correction_scan_id = correction.get("scan_id")
            anchor_yaw = float(correction["desired_yaw_deg"])
            anchor_pitch = float(correction["desired_pitch_deg"])
            delta_yaw = float(correction["delta_yaw_deg"])
            delta_pitch = float(correction["delta_pitch_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if scan_id and correction_scan_id and correction_scan_id != scan_id:
            continue
        yaw_distance = _normalize_delta(yaw_deg - anchor_yaw)
        pitch_distance = pitch_deg - anchor_pitch
        distance = math.hypot(yaw_distance, pitch_distance)
        if distance > max_distance_deg:
            continue
        weight = math.exp(-(distance * distance) / (2 * sigma_deg * sigma_deg))
        yaw_offset += delta_yaw * weight
        pitch_offset += delta_pitch * weight
        weight_total += weight
        used += 1
    if weight_total == 0:
        return {
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "applied": False,
            "correction_count": 0,
        }
    return {
        "yaw_deg": _clamp(_normalize_degrees(yaw_deg + yaw_offset / weight_total), -180, 180),
        "pitch_deg": _clamp(pitch_deg + pitch_offset / weight_total, -90, 90),
        "applied": True,
        "correction_count": used,
    }


def _remember_local_correction(
    existing: list[dict],
    correction: dict,
    max_per_area: int = LOCAL_CORRECTIONS_PER_AREA,
    area_radius_deg: float = LOCAL_CORRECTION_AREA_RADIUS_DEG,
    replace_radius_deg: float = LOCAL_CORRECTION_REPLACE_RADIUS_DEG,
) -> list[dict]:
    nearby: list[dict] = []
    distant: list[dict] = []
    try:
        correction_scan_id = correction.get("scan_id")
        anchor_yaw = float(correction["desired_yaw_deg"])
        anchor_pitch = float(correction["desired_pitch_deg"])
    except (KeyError, TypeError, ValueError):
        return [*existing, correction]

    for item in existing:
        try:
            item_scan_id = item.get("scan_id")
            item_yaw = float(item["desired_yaw_deg"])
            item_pitch = float(item["desired_pitch_deg"])
        except (KeyError, TypeError, ValueError):
            distant.append(item)
            continue
        yaw_distance = _normalize_delta(item_yaw - anchor_yaw)
        pitch_distance = item_pitch - anchor_pitch
        distance = math.hypot(yaw_distance, pitch_distance)
        if correction_scan_id and item_scan_id and item_scan_id != correction_scan_id:
            distant.append(item)
        elif distance <= replace_radius_deg:
            continue
        elif distance <= area_radius_deg:
            nearby.append(item)
        else:
            distant.append(item)

    kept_nearby = nearby[-max(0, max_per_area - 1) :]
    return [*distant, *kept_nearby, correction]


def _normalize_delta(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def _normalize_degrees(value: float) -> float:
    normalized = _normalize_delta(value)
    if normalized == -180.0 and value > 0:
        return 180.0
    return normalized


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _next_zoom_value(current_zoom: int, direction: int, step: int, max_zoom: int = 0x4000) -> int:
    return int(_clamp(current_zoom + direction * step, 0, max_zoom))


def _jog_command_from_vector(
    x: float,
    y: float,
    *,
    deadzone: float = 0.14,
    max_pan_speed: int = 0x18,
    max_tilt_speed: int = 0x14,
) -> tuple[int, int, int, int]:
    x = _clamp(float(x), -1.0, 1.0)
    y = _clamp(float(y), -1.0, 1.0)

    def component(value: float, max_speed: int) -> tuple[int, int]:
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0, 0
        scaled = (magnitude - deadzone) / (1.0 - deadzone)
        speed = max(1, min(max_speed, int(round(1 + scaled * (max_speed - 1)))))
        direction = -1 if value < 0 else 1
        return direction, speed

    pan_dir, pan_speed = component(x, max_pan_speed)
    tilt_dir, tilt_speed = component(y, max_tilt_speed)
    return pan_dir, pan_speed, tilt_dir, tilt_speed


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
