from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.scan import ScanPlanConfig, ScanPlanner
from ptz_pano.tools.config import load_app_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Print scan poses without touching the camera.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw_scan_config = dict(load_app_config(args.config)["scan"])
    raw_scan_config.pop("settle_sec", None)
    planner = ScanPlanner(ScanPlanConfig(**raw_scan_config))

    for index, pose in enumerate(planner.poses(), start=1):
        print(f"{index:04d} pan={pose.pan} tilt={pose.tilt} zoom={pose.zoom}")


if __name__ == "__main__":
    main()

