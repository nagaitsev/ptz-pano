from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.tools.config import build_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Read current VISCA pan/tilt/zoom status.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    camera = build_camera(args.config)
    try:
        pose = camera.get_position()
    finally:
        camera.close()

    print(f"pan={pose.pan} tilt={pose.tilt} zoom={pose.zoom}")


if __name__ == "__main__":
    main()
