from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import uvicorn

LOCAL_WHEP_PORT = 8889
LOCAL_WHEP_PATH = "camera/whep"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PTZ Pano from a portable bundle.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="config/camera.local.json")
    parser.add_argument("--target-hfov-scale", default="0.45")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    base_dir = _runtime_dir()
    os.chdir(base_dir)
    _ensure_runtime_layout(base_dir)
    _append_runtime_log(base_dir, "portable startup")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    _ensure_camera_config(config_path)
    gateway_process, whep_url = _start_local_webrtc_gateway(base_dir, config_path)

    os.environ["PTZ_PANO_CAMERA_CONFIG"] = str(config_path)
    os.environ["PTZ_PANO_TARGET_HFOV_SCALE"] = args.target_hfov_scale
    if whep_url:
        os.environ["PTZ_PANO_DEFAULT_WHEP_URL"] = whep_url

    url = f"http://127.0.0.1:{args.port}/"
    print("PTZ Pano portable server starting...")
    print(f"Local:  {url}")
    print(f"Config: {config_path}")
    print(f"Logs:   {base_dir / 'logs'}")
    if whep_url:
        print(f"WebRTC: {whep_url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(
            "ptz_pano.api.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    finally:
        _stop_process(gateway_process)
        _append_runtime_log(base_dir, "portable shutdown")


def _runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[4]


def _ensure_runtime_layout(base_dir: Path) -> None:
    for path in [
        base_dir / "config",
        base_dir / "data" / "calibration",
        base_dir / "data" / "projects",
        base_dir / "data" / "scans",
        base_dir / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _ensure_camera_config(config_path: Path) -> None:
    if config_path.exists():
        return
    example_path = config_path.with_name("camera.example.json")
    if not example_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. Create it or place camera.example.json next to it."
        )
    shutil.copy2(example_path, config_path)


def _default_whep_url() -> str:
    return f"http://127.0.0.1:{LOCAL_WHEP_PORT}/{LOCAL_WHEP_PATH}"


def _ensure_portable_capture_defaults(config_path: Path, whep_url: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    camera = config.setdefault("camera", {})
    capture = config.setdefault("capture", {})
    host = str(camera.get("host", "")).strip()
    source = str(capture.get("source", "")).strip()
    if host and source:
        capture["source"] = _replace_url_host(source, host)
    capture["whep_url"] = whep_url
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def _start_local_webrtc_gateway(base_dir: Path, config_path: Path) -> tuple[subprocess.Popen | None, str]:
    whep_url = _default_whep_url()
    config = _ensure_portable_capture_defaults(config_path, whep_url)
    mediamtx_exe = base_dir / "tools" / "mediamtx" / "mediamtx.exe"
    if not mediamtx_exe.exists():
        _append_runtime_log(base_dir, f"mediamtx missing: {mediamtx_exe}")
        print("MediaMTX executable is missing; preview will use JPEG fallback.")
        return None, whep_url

    capture = config.get("capture", {})
    source = str(capture.get("source", "")).strip()
    if not source:
        _append_runtime_log(base_dir, "mediamtx skipped: capture source is empty")
        print("Capture source is empty; preview will use JPEG fallback.")
        return None, whep_url

    mediamtx_dir = mediamtx_exe.parent
    runtime_config_path = mediamtx_dir / "mediamtx.generated.yml"
    runtime_config_path.write_text(_mediamtx_config_text(source), encoding="utf-8")
    stdout_log_path = base_dir / "logs" / "mediamtx.stdout.log"
    stderr_log_path = base_dir / "logs" / "mediamtx.stderr.log"
    _append_runtime_log(base_dir, f"starting mediamtx from {mediamtx_exe}")
    _append_runtime_log(base_dir, f"mediamtx source {source}")
    _append_runtime_log(base_dir, f"mediamtx config {runtime_config_path}")

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with stdout_log_path.open("ab") as stdout_handle, stderr_log_path.open("ab") as stderr_handle:
        process = subprocess.Popen(
            [str(mediamtx_exe), str(runtime_config_path)],
            cwd=str(mediamtx_dir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )
    time.sleep(0.3)
    exit_code = process.poll()
    if exit_code is not None:
        _append_runtime_log(base_dir, f"mediamtx exited immediately with code {exit_code}")
        return None, whep_url
    if _wait_for_local_port(LOCAL_WHEP_PORT, timeout_sec=5.0):
        _append_runtime_log(base_dir, f"mediamtx listening on 127.0.0.1:{LOCAL_WHEP_PORT} pid={process.pid}")
    else:
        _append_runtime_log(
            base_dir,
            f"mediamtx process started but port {LOCAL_WHEP_PORT} did not open within timeout pid={process.pid}",
        )
    atexit.register(_stop_process, process)
    return process, whep_url


def _mediamtx_config_text(rtsp_source: str) -> str:
    return (
        "logLevel: warn\n"
        "rtspTransports: [tcp]\n"
        "webrtcAdditionalHosts:\n"
        "  - 127.0.0.1\n"
        "webrtcAllowOrigins:\n"
        "  - \"*\"\n"
        "paths:\n"
        "  camera:\n"
        f"    source: {rtsp_source}\n"
    )


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
    return urlunsplit(SplitResult(parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


def _append_runtime_log(base_dir: Path, message: str) -> None:
    log_path = base_dir / "logs" / "portable-server.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _wait_for_local_port(port: int, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    while time.monotonic() <= deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


if __name__ == "__main__":
    main()
