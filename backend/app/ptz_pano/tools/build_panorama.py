from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.panorama import PanoramaBuilder
from ptz_pano.storage import ScanRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Build panorama artifacts from a saved scan.")
    parser.add_argument("--scan", type=Path, required=True)
    args = parser.parse_args()

    scan_path = args.scan.resolve()
    repository = ScanRepository(scan_path.parent)
    output_path = PanoramaBuilder(repository).build_manifest(scan_path.name)
    print(output_path)


if __name__ == "__main__":
    main()

