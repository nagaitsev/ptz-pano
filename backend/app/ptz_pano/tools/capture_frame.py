from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.tools.config import build_capture


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one frame for debugging.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    capture = build_capture(args.config)
    output_path = capture.grab_frame(args.out)
    print(output_path)


if __name__ == "__main__":
    main()

