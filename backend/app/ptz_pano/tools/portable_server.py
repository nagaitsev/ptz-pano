from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn


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

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    _ensure_camera_config(config_path)

    os.environ["PTZ_PANO_CAMERA_CONFIG"] = str(config_path)
    os.environ["PTZ_PANO_TARGET_HFOV_SCALE"] = args.target_hfov_scale

    url = f"http://127.0.0.1:{args.port}/"
    print("PTZ Pano portable server starting...")
    print(f"Local:  {url}")
    print(f"Config: {config_path}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "ptz_pano.api.main:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


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


if __name__ == "__main__":
    main()
