from __future__ import annotations

import argparse

from .batch import compute_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="landscape2d",
        description="Compute 2D landscape metrics from label matrices.",
    )
    parser.add_argument("input_dir", help="Directory containing .npy/.npz/.csv/.txt matrices.")
    parser.add_argument("output_csv", help="Destination CSV path.")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated positive class labels, for example 1,2,3.",
    )
    parser.add_argument(
        "--spacing",
        default="1,1",
        help="Pixel spacing as sx,sy. Default: 1,1.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=4,
        choices=(4, 8),
        help="2D patch connectivity. Default: 4.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=float("inf"),
        help="Search radius for PROX/SIMI/CONNECT. Default: infinite.",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Maximum possible class count for relative patch richness.",
    )
    parser.add_argument(
        "--min-patch-cells",
        type=int,
        default=0,
        help="Remove connected patches smaller than this many cells before metrics. Default: 0.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory for per-sample PNG label maps.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compute_batch(
        args.input_dir,
        args.output_csv,
        classes=args.classes,
        spacing=args.spacing,
        connectivity=args.connectivity,
        search_radius=args.search_radius,
        max_classes=args.max_classes,
        min_patch_cells=args.min_patch_cells,
        plot_dir=args.plot_dir,
        progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
