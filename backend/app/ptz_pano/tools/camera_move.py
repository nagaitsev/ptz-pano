from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.models import CameraPose
from ptz_pano.tools.config import build_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a VISCA camera for debugging.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--pan", type=int)
    parser.add_argument("--tilt", type=int)
    parser.add_argument("--zoom", type=int)
    args = parser.parse_args()

    camera = build_camera(args.config)
    try:
        if args.home:
            camera.home()
        elif args.stop:
            camera.stop()
        elif args.pan is not None and args.tilt is not None and args.zoom is not None:
            camera.move_absolute(CameraPose(args.pan, args.tilt, args.zoom))
        else:
            raise SystemExit("use --home, --stop, or --pan/--tilt/--zoom")
    finally:
        camera.close()


if __name__ == "__main__":
    main()

