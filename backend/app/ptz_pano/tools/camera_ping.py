from __future__ import annotations

import argparse
import socket
from pathlib import Path

from ptz_pano.tools.config import load_camera_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TCP reachability for a camera.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_camera_config(args.config)
    if config.transport != "tcp":
        raise SystemExit("camera_ping currently checks TCP profiles only")

    with socket.create_connection((config.host, config.port), timeout=config.timeout_sec):
        print(f"ok tcp://{config.host}:{config.port}")


if __name__ == "__main__":
    main()

