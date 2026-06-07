from __future__ import annotations

import argparse
import os
from pathlib import Path

from ptz_pano.calibration.lens_table import LensCalibration
from ptz_pano.stitching import PanoramaBuilder
from ptz_pano.stitching.feature_graph_compositor import FeatureGraphCompositor
from ptz_pano.stitching.openstitching_compositor import OpenStitchingCompositor
from ptz_pano.stitching.opencv_stitcher import OpenCvStitcherCompositor
from ptz_pano.storage import ScanRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Build panorama artifacts from a saved scan.")
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument(
        "--stitch-engine",
        choices=["opencv", "feature_graph", "openstitching"],
        default="opencv",
        help="Panorama backend: opencv keeps the existing pipeline; openstitching uses the OpenStitching package; feature_graph builds an experimental feature-based mosaic.",
    )
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
        help="Accepted for compatibility; OpenCV Stitcher is used by default.",
    )
    parser.add_argument(
        "--projection",
        choices=["angular", "sphere"],
        default="sphere",
        help="Accepted for compatibility; OpenCV Stitcher PANORAMA is used by default.",
    )
    parser.add_argument(
        "--stitch-quality",
        choices=["fast", "quality"],
        default="fast",
        help="Telemetry fallback mode: fast keeps weighted compositing; quality adds feature alignment and detail blending.",
    )
    args = parser.parse_args()

    scan_path = args.scan.resolve()
    repository = ScanRepository(scan_path.parent)
    lens_calibration = (
        None
        if args.lens_calibration is None
        else LensCalibration.from_file(Path(args.lens_calibration))
    )
    if args.stitch_engine == "feature_graph":
        compositor = FeatureGraphCompositor(lens_calibration=lens_calibration)
    elif args.stitch_engine == "openstitching":
        compositor = OpenStitchingCompositor(
            lens_calibration=lens_calibration,
            fallback_quality=args.stitch_quality,
        )
    else:
        compositor = OpenCvStitcherCompositor(
            lens_calibration=lens_calibration,
            fallback_quality=args.stitch_quality,
        )
    output_path = PanoramaBuilder(repository, compositor=compositor).build_manifest(scan_path.name)
    print(output_path)


if __name__ == "__main__":
    main()
