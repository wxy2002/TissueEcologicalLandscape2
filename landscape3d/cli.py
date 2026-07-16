from __future__ import annotations

import argparse

from .batch import compute_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="landscape3d",
        description="Compute 3D landscape metrics from .npy/.npz/.nii/.nii.gz label volumes.",
    )
    parser.add_argument(
        "input_dir", help="Directory containing .npy/.npz/.nii/.nii.gz volumes."
    )
    parser.add_argument("output_csv", help="Destination CSV path.")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated positive class labels, for example 1,2,3.",
    )
    parser.add_argument(
        "--spacing",
        default="1,1,1",
        help="Voxel spacing as sx,sy,sz. Default: 1,1,1.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=6,
        choices=(6, 18, 26),
        help="3D patch connectivity. Default: 6.",
    )
    parser.add_argument(
        "--min-non-background-voxels",
        type=int,
        default=0,
        help="Skip samples with fewer non-background voxels than this value.",
    )
    parser.add_argument(
        "--min-non-background-volume",
        type=float,
        default=0.0,
        help="Skip samples with less non-background volume than this value.",
    )
    parser.add_argument(
        "--min-patch-voxels",
        type=int,
        default=0,
        help="Remove patches with fewer voxels than this value before computing metrics.",
    )
    parser.add_argument(
        "--min-patch-volume",
        type=float,
        default=0.0,
        help="Remove patches with less volume than this value before computing metrics.",
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
        min_patch_voxels=args.min_patch_voxels,
        min_patch_volume=args.min_patch_volume,
        min_non_background_voxels=args.min_non_background_voxels,
        min_non_background_volume=args.min_non_background_volume,
        progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
