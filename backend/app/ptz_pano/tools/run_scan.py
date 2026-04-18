from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.calibration import FovTable
from ptz_pano.models import ScanDocument
from ptz_pano.scan import ScanPlanConfig, ScanPlanner, ScanRunner
from ptz_pano.storage import ScanRepository
from ptz_pano.tools.config import build_camera, build_capture, load_app_config, load_camera_config, load_capture_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a camera scan into data/scans.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("data/scans"))
    args = parser.parse_args()

    raw_config = load_app_config(args.config)
    fov_table = None
    calibration_config = raw_config.get("calibration")
    if calibration_config and calibration_config.get("fov_table"):
        fov_table = FovTable.load(Path(calibration_config["fov_table"]))

    raw_scan_config = dict(raw_config["scan"])
    settle_sec = raw_scan_config.pop("settle_sec", 1.0)
    scan_config = ScanPlanConfig(**raw_scan_config)
    camera_config = load_camera_config(args.config)
    capture_config = load_capture_config(args.config)

    document = ScanDocument(id=args.scan_id, camera=camera_config, capture=capture_config)
    camera = build_camera(args.config)
    capture = build_capture(args.config)
    runner = ScanRunner(
        camera=camera,
        capture=capture,
        repository=ScanRepository(args.root),
        settle_sec=settle_sec,
        fov_table=fov_table,
    )

    try:
        scan_path = runner.run(document, ScanPlanner(scan_config))
    finally:
        camera.close()

    print(scan_path)


if __name__ == "__main__":
    main()
