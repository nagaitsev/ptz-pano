from __future__ import annotations

import argparse
from pathlib import Path

from ptz_pano.calibration import FovTable
from ptz_pano.models import FovSample


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or inspect a zoom-to-FOV table.")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--add", nargs=3, metavar=("ZOOM", "HFOV", "VFOV"))
    parser.add_argument("--query", type=int)
    args = parser.parse_args()

    if args.add:
        samples = []
        if args.table.exists():
            samples.extend(FovTable.load(args.table).samples)
        zoom, hfov, vfov = args.add
        samples.append(FovSample(int(zoom), float(hfov), float(vfov)))
        table = FovTable(tuple(samples))
        table.save(args.table)
        print(args.table)
        return

    table = FovTable.load(args.table)
    if args.query is not None:
        hfov, vfov = table.fov_for_zoom(args.query)
        print(f"zoom={args.query} hfov={hfov:.4f} vfov={vfov:.4f}")
    else:
        for sample in table.samples:
            print(f"zoom={sample.zoom} hfov={sample.hfov_deg} vfov={sample.vfov_deg}")


if __name__ == "__main__":
    main()

