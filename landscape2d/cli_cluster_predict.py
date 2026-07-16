from __future__ import annotations

import argparse

from .feature_clustering import infer_feature_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="landscape2d-cluster-predict",
        description="Apply a saved autoencoder cluster model to a new feature matrix.",
    )
    parser.add_argument(
        "input_features",
        help="Feature file or directory: .h5/.hdf5/.npy/.npz/.csv/.txt. H5 defaults to feats/coords.",
    )
    parser.add_argument("model_path", help="Saved *_cluster_model.pt from training.")
    parser.add_argument("output_dir", help="Directory for inferred labels .npz, metrics CSV, and plot.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda, cuda:0, cpu, auto, etc. Default: cuda.")
    parser.add_argument(
        "--max-silhouette-samples",
        type=int,
        default=2000,
        help="Maximum samples used for silhouette scoring.",
    )
    parser.add_argument(
        "--save-latent",
        action="store_true",
        help="Also store latent features in the output .npz.",
    )
    parser.add_argument(
        "--min-patch-cells",
        type=int,
        default=0,
        help="Remove connected patches smaller than this many cells from saved label matrices. Default: 0.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    infer_feature_file(
        args.input_features,
        args.model_path,
        args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        max_silhouette_samples=args.max_silhouette_samples,
        save_latent=args.save_latent,
        min_patch_cells=args.min_patch_cells,
        progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
