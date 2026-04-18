from __future__ import annotations

import argparse
import os
from pathlib import Path

from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.stitching import PanoramaBuilder
from ptz_pano.stitching.simple_compositor import SimpleCompositor
from ptz_pano.storage import ScanRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Build panorama artifacts from a saved scan.")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument(
        "--lens-calibration",
        type=Path,
        default=os.environ.get("PTZ_PANO_LENS_CALIBRATION"),
        help="Optional lens calibration JSON used to undistort frames before compositing.",
    )
    parser.add_argument(
        "--strategy",
        choices=["average", "max_weight"],
        default="average",
        help="Stitching strategy: 'average' (weighted blend) or 'max_weight' (no ghosting).",
    )
    parser.add_argument(
        "--projection",
        choices=["angular", "sphere"],
        default="angular",
        help="Projection model: 'angular' is the previous fast remap, 'sphere' uses 3D rays.",
    )
    args = parser.parse_args()

    scan_path = args.scan.resolve()
    repository = ScanRepository(scan_path.parent)
    lens_calibration = (
        None
        if args.lens_calibration is None
        else LensCalibration.from_file(Path(args.lens_calibration))
    )
    compositor = SimpleCompositor(
        lens_calibration=lens_calibration,
        strategy=args.strategy,
        projection=args.projection,
    )
    output_path = PanoramaBuilder(repository, compositor=compositor).build_manifest(scan_path.name)
    print(output_path)


if __name__ == "__main__":
    main()
