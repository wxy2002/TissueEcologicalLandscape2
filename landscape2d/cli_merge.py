from __future__ import annotations

import argparse

from .merge_maps import merge_patient_maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="landscape2d-merge-patient-maps",
        description="Merge multiple WSI label-map .npz files from the same patient horizontally.",
    )
    parser.add_argument("input_dir", help="Directory containing per-WSI .npz label maps.")
    grouping = parser.add_mutually_exclusive_group(required=True)
    grouping.add_argument("--patient-manifest", help="CSV with source_file,patient_id columns; preferred over prefix grouping.")
    grouping.add_argument(
        "--prefix-length",
        type=int,
        help="Group files by the first N characters of the filename stem.",
    )
    parser.add_argument(
        "--blank-cols",
        type=int,
        default=1,
        help="Blank background columns inserted between WSI maps. Default: 1.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_merged",
        help="Suffix for merged output files. Default: _merged.",
    )
    deletion = parser.add_mutually_exclusive_group()
    deletion.add_argument("--delete-originals", action="store_true", help="Explicitly delete sources after verified atomic save. Sources are kept by default.")
    deletion.add_argument("--keep-originals", action="store_true", help="Keep sources (the default; retained for compatibility).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned merges without writing or deleting files.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    merge_patient_maps(
        args.input_dir,
        prefix_length=args.prefix_length,
        blank_cols=args.blank_cols,
        output_suffix=args.output_suffix,
        delete_originals=args.delete_originals,
        patient_manifest=args.patient_manifest,
        dry_run=args.dry_run,
        progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
