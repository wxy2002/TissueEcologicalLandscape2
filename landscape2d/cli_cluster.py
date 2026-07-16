from __future__ import annotations

import argparse

from .feature_clustering import (
    cluster_feature_file,
    parse_hidden_dims,
    parse_k_range,
    parse_optional_limit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="landscape2d-cluster-features",
        description="Reduce pathology tile feature vectors with a PyTorch autoencoder and cluster them.",
    )
    parser.add_argument(
        "input_features",
        help="Feature file or directory: .h5/.hdf5/.npy/.npz/.csv/.txt. H5 defaults to feats/coords.",
    )
    parser.add_argument("output_dir", help="Directory for labels .npz, metrics CSV, model, and plot.")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fixed cluster count K. Mutually exclusive with --k-range.",
    )
    parser.add_argument(
        "--k-range",
        default=None,
        help="Inclusive K range for automatic selection, for example 2,10.",
    )
    parser.add_argument("--latent-dim", type=int, default=32, help="Autoencoder latent dimension.")
    parser.add_argument(
        "--hidden-dims",
        default=None,
        help="Comma-separated hidden layer widths. Default is inferred from input dimension.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Autoencoder training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda, cuda:0, cpu, auto, etc. Default: cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-silhouette-samples",
        type=int,
        default=2000,
        help="Maximum samples used for silhouette scoring.",
    )
    parser.add_argument(
        "--max-train-tiles",
        default="200000",
        help="Maximum total tiles sampled for autoencoder training and K search. Use 0 for no limit.",
    )
    parser.add_argument(
        "--max-train-tiles-per-file",
        default="2000",
        help="Maximum tiles sampled per feature file before the global cap. Use 0 for no per-file limit.",
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
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization before autoencoder training.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress messages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_clusters is not None and args.k_range is not None:
        raise SystemExit("--n-clusters and --k-range are mutually exclusive")
    cluster_feature_file(
        args.input_features,
        args.output_dir,
        n_clusters=args.n_clusters,
        k_range=parse_k_range(args.k_range),
        latent_dim=args.latent_dim,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        standardize=not args.no_standardize,
        max_silhouette_samples=args.max_silhouette_samples,
        max_train_tiles=parse_optional_limit(args.max_train_tiles),
        max_train_tiles_per_file=parse_optional_limit(args.max_train_tiles_per_file),
        min_patch_cells=args.min_patch_cells,
        save_latent=args.save_latent,
        progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
