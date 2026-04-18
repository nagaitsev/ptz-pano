from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.tools.config import build_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one raw VISCA command and print responses.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--hex",
        required=True,
        help="VISCA command bytes, for example: '81 01 06 01 18 14 03 03 FF'",
    )
    args = parser.parse_args()

    camera = build_camera(args.config)
    try:
        command = bytes.fromhex(args.hex)
        responses = camera.send_raw(command)
    finally:
        camera.close()

    if not responses:
        print("no response")
    for index, response in enumerate(responses, start=1):
        print(f"response_{index}={response.hex(' ').upper()}")


if __name__ == "__main__":
    main()

