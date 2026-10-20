from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import json
import warnings
import os
import sys
from typing import TextIO

import numpy as np
import pandas as pd
from numbers import Integral

from .filters import filter_small_patches
from .grid import labels_to_matrix
# Re-export input and scoring helpers for callers of the original module.
from .feature_io import (
    FeatureBlock, _file_digest, _with_grid_overrides, feature_count, iter_feature_files,
    load_feature_block, load_feature_sample, load_features, load_h5_feature_block,
    validate_features, validate_coords,
)
from .cluster_quality import (
    clustering_metrics, _silhouette_score, _calinski_harabasz_score, _davies_bouldin_score,
)


@dataclass(frozen=True)
class FeatureClusteringResult:
    labels: np.ndarray
    sampled_latent: np.ndarray
    sampled_labels: np.ndarray
    sample_source_files: np.ndarray
    sample_row_indices: np.ndarray
    metrics: pd.DataFrame
    selected_k: int
    labels_path: Path
    metrics_path: Path
    plot_path: Path | None
    model_path: Path | None = None

    @property
    def latent(self) -> np.ndarray:
        """Compatibility alias. Rows align with sampled_labels, not full labels."""
        warnings.warn("Use sampled_latent and sampled_labels; full labels cover all tiles.", DeprecationWarning, stacklevel=2)
        return self.sampled_latent


def _validate_clustering_options(min_patch_cells: int, max_samples: int, batch_size: int) -> None:
    for name, value in (("min_patch_cells", min_patch_cells), ("max_silhouette_samples", max_samples)):
        if not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


def cluster_feature_file(
    input_path: str | Path,
    output_dir: str | Path,
    n_clusters: int | None = None,
    k_range: tuple[int, int] | None = None,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
    standardize: bool = True,
    max_silhouette_samples: int = 2000,
    max_train_tiles: int | None = 200_000,
    max_train_tiles_per_file: int | None = 2_000,
    min_patch_cells: int = 0,
    save_latent: bool = False,
    progress: bool = True,
    progress_stream: TextIO | None = None,
    grid_stride: tuple[float, float] | None = None,
    grid_origin: tuple[float, float] | None = None,
    coordinate_level: int | None = None,
    coord_downsample: float | None = None,
    mpp: tuple[float, float] | None = None,
) -> FeatureClusteringResult:
    _validate_clustering_options(min_patch_cells, max_silhouette_samples, batch_size)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = progress_stream or sys.stderr

    feature_files = iter_feature_files(input_path)
    train_blocks = _load_training_sample_blocks(
        feature_files,
        max_train_tiles=max_train_tiles,
        max_train_tiles_per_file=max_train_tiles_per_file,
        seed=seed,
        progress=progress,
        stream=stream,
    )
    features = np.concatenate([block.features for block in train_blocks], axis=0)
    k_values = _resolve_k_values(n_clusters, k_range, features.shape[0])
    _show_progress(
        progress,
        stream,
        f"Training autoencoder: files={len(feature_files)}, sampled_n={features.shape[0]}, m={features.shape[1]}, latent_dim={latent_dim}",
    )
    latent, model_payload = _fit_autoencoder(
        features,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
        standardize=standardize,
        progress=progress,
        progress_stream=stream,
    )

    metric_rows = []
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_score: tuple[float, float, float] | None = None
    selected_k = k_values[0]
    k_progress = _ProgressBar(
        total=len(k_values),
        label="Searching K",
        enabled=progress,
        stream=stream,
    )
    for k in k_values:
        k_progress.update(message=f"k={k}")
        labels_zero, centers, inertia = kmeans(
            latent,
            n_clusters=k,
            random_state=seed,
        )
        metrics = clustering_metrics(
            latent,
            labels_zero,
            centers=centers,
            inertia=inertia,
            max_silhouette_samples=max_silhouette_samples,
            random_state=seed,
        )
        score = (
            _metric_for_sort(metrics["silhouette_score"], larger_is_better=True),
            _metric_for_sort(metrics["calinski_harabasz_score"], larger_is_better=True),
            _metric_for_sort(metrics["davies_bouldin_score"], larger_is_better=False),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_labels = labels_zero
            best_centers = centers
            selected_k = k
        metric_rows.append(_metrics_row(k, labels_zero, metrics))
        k_progress.advance(message=f"k={k} done")
    k_progress.close(message=f"selected_k={selected_k}")

    if best_labels is None or best_centers is None:
        raise RuntimeError("failed to cluster features")
    labels = (best_labels.astype(np.int64) + 1).reshape(-1, 1)
    dataframe = pd.DataFrame(metric_rows)
    dataframe["selected"] = dataframe["k"] == selected_k
    ordered = ["k", "selected"] + [column for column in dataframe.columns if column not in {"k", "selected"}]
    dataframe = dataframe[ordered]

    metrics_path = output_dir / "cluster_metrics.csv"
    dataframe.to_csv(metrics_path, index=False)

    model_path = output_dir / "cluster_model.pt"
    _save_cluster_model(
        model_path,
        model_payload=model_payload,
        centers=best_centers,
        selected_k=selected_k,
        seed=seed,
        blocks=train_blocks,
    )

    labels_path = output_dir / "all_clusters.npz"
    full_labels, _, _, _, _, _ = _predict_and_save_feature_files(
        feature_files=feature_files,
        model_path=model_path,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        save_latent=save_latent,
        aggregate_path=labels_path,
        progress=progress,
        progress_stream=stream,
        min_patch_cells=min_patch_cells,
        max_metric_samples=max_silhouette_samples,
        grid_stride=grid_stride, grid_origin=grid_origin, coordinate_level=coordinate_level,
        coord_downsample=coord_downsample, mpp=mpp,
    )

    plot_path = output_dir / "latent_scatter.png"
    saved_plot = save_latent_scatter(latent, labels.ravel(), plot_path)
    _show_progress(progress, stream, f"Saved cluster labels: {output_dir}")
    _show_progress(progress, stream, f"Saved cluster metrics: {metrics_path}")
    _show_progress(progress, stream, f"Saved cluster model: {model_path}")
    if saved_plot:
        _show_progress(progress, stream, f"Saved latent plot: {saved_plot}")
    return FeatureClusteringResult(
        labels=full_labels,
        sampled_latent=latent,
        sampled_labels=labels.ravel(),
        sample_source_files=np.concatenate([np.repeat(str(block.source_path), block.features.shape[0]) for block in train_blocks]),
        sample_row_indices=np.concatenate([block.row_indices if block.row_indices is not None else np.arange(block.features.shape[0]) for block in train_blocks]),
        metrics=dataframe,
        selected_k=selected_k,
        labels_path=labels_path,
        metrics_path=metrics_path,
        plot_path=saved_plot,
        model_path=model_path,
    )


def infer_feature_file(
    input_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 256,
    device: str = "cuda",
    max_silhouette_samples: int = 2000,
    save_latent: bool = False,
    min_patch_cells: int = 0,
    progress: bool = True,
    progress_stream: TextIO | None = None,
    grid_stride: tuple[float, float] | None = None,
    grid_origin: tuple[float, float] | None = None,
    coordinate_level: int | None = None,
    coord_downsample: float | None = None,
    mpp: tuple[float, float] | None = None,
) -> FeatureClusteringResult:
    _validate_clustering_options(min_patch_cells, max_silhouette_samples, batch_size)
    input_path = Path(input_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = progress_stream or sys.stderr

    feature_files = iter_feature_files(input_path)
    _show_progress(progress, stream, f"Loading cluster model: {model_path}")
    labels_path = output_dir / "all_clusters.npz"
    labels, dataframe, latent_for_plot, labels_for_plot, selected_k, sample_indices = _predict_and_save_feature_files(
        feature_files=feature_files,
        model_path=model_path,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        save_latent=save_latent,
        aggregate_path=labels_path,
        progress=progress,
        progress_stream=stream,
        max_metric_samples=max_silhouette_samples,
        min_patch_cells=min_patch_cells,
        grid_stride=grid_stride, grid_origin=grid_origin, coordinate_level=coordinate_level,
        coord_downsample=coord_downsample, mpp=mpp,
    )

    metrics_path = output_dir / "cluster_metrics.csv"
    dataframe.to_csv(metrics_path, index=False)
    plot_path = output_dir / "latent_scatter.png"
    saved_plot = save_latent_scatter(latent_for_plot, labels_for_plot, plot_path) if latent_for_plot.shape[0] else None
    _show_progress(progress, stream, f"Saved inferred labels: {output_dir}")
    _show_progress(progress, stream, f"Saved inference metrics: {metrics_path}")
    if saved_plot:
        _show_progress(progress, stream, f"Saved latent plot: {saved_plot}")
    with np.load(labels_path) as aggregate:
        offsets = np.cumsum(np.r_[0, aggregate["sample_counts"]])
        source_indices = np.searchsorted(offsets[1:], sample_indices, side="right")
        sample_sources = aggregate["source_files"][source_indices]
        sample_rows = sample_indices - offsets[source_indices]
    return FeatureClusteringResult(
        labels=labels,
        sampled_latent=latent_for_plot,
        sampled_labels=labels_for_plot,
        sample_source_files=sample_sources,
        sample_row_indices=sample_rows,
        metrics=dataframe,
        selected_k=selected_k,
        labels_path=labels_path,
        metrics_path=metrics_path,
        plot_path=saved_plot,
        model_path=model_path,
    )


def reduce_features_autoencoder(
    features: np.ndarray,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
    standardize: bool = True,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> np.ndarray:
    latent, _ = _fit_autoencoder(
        features,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
        standardize=standardize,
        progress=progress,
        progress_stream=progress_stream,
    )
    return latent


def _fit_autoencoder(
    features: np.ndarray,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
    standardize: bool = True,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "Autoencoder feature clustering requires PyTorch. Install torch or run on a server with PyTorch available."
        ) from exc

    features = validate_features(features)
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if latent_dim >= features.shape[1]:
        raise ValueError("latent_dim must be smaller than the input feature dimension")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(seed)
    x = features.astype(np.float32, copy=True)
    if standardize:
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = x.std(axis=0, keepdims=True).astype(np.float32)
        std[std == 0] = 1.0
        x = (x - mean) / std
    else:
        mean = np.zeros((1, x.shape[1]), dtype=np.float32)
        std = np.ones((1, x.shape[1]), dtype=np.float32)

    torch_device = _torch_device(device, torch)
    input_dim = x.shape[1]
    hidden = hidden_dims or _default_hidden_dims(input_dim, latent_dim)
    model = _Autoencoder(input_dim, latent_dim, hidden, nn).to(torch_device)
    tensor = torch.from_numpy(x)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=min(batch_size, x.shape[0]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    stream = progress_stream or sys.stderr
    model.train()
    epoch_progress = _ProgressBar(
        total=epochs,
        label="Training autoencoder",
        enabled=progress,
        stream=stream,
    )
    for epoch in range(1, epochs + 1):
        losses = []
        batch_progress = _ProgressBar(
            total=len(loader),
            label=f"Epoch {epoch}/{epochs}",
            enabled=progress,
            stream=stream,
            leave=False,
        )
        for (batch,) in loader:
            batch = batch.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_progress.advance()
        mean_loss = float(np.mean(losses))
        batch_progress.close(message=f"loss={mean_loss:.6f}")
        epoch_progress.advance(message=f"loss={mean_loss:.6f}")
    epoch_progress.close(message="done")

    model.eval()
    encoded_batches = []
    encode_progress = _ProgressBar(
        total=math.ceil(x.shape[0] / batch_size),
        label="Encoding latent features",
        enabled=progress,
        stream=stream,
    )
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(torch_device)
            encoded_batches.append(model.encode(batch).cpu().numpy())
            encode_progress.advance()
    encode_progress.close(message="done")
    latent = np.concatenate(encoded_batches, axis=0).astype(np.float32)
    if not np.all(np.isfinite(latent)):
        raise RuntimeError("autoencoder produced non-finite latent features")
    model_payload: dict[str, object] = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dims": tuple(int(value) for value in hidden),
        "standardize": bool(standardize),
        "mean": mean,
        "std": std,
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
    }
    return latent, model_payload


def encode_features_with_model(
    features: np.ndarray,
    model_path: str | Path,
    batch_size: int = 256,
    device: str = "cuda",
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "Autoencoder feature inference requires PyTorch. Install torch or run on a server with PyTorch available."
        ) from exc
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    features = validate_features(features)
    input_dim = int(payload["input_dim"])
    if features.shape[1] != input_dim:
        raise ValueError(
            f"feature dimension mismatch: model expects {input_dim}, got {features.shape[1]}"
        )
    x = features.astype(np.float32, copy=True)
    if bool(payload.get("standardize", True)):
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        x = (x - mean) / std
    torch_device = _torch_device(device, torch)
    model = _Autoencoder(
        input_dim=input_dim,
        latent_dim=int(payload["latent_dim"]),
        hidden_dims=tuple(int(value) for value in payload["hidden_dims"]),
        nn=nn,
    ).to(torch_device)
    model.encoder.load_state_dict(payload["encoder_state"])
    model.decoder.load_state_dict(payload["decoder_state"])
    model.eval()
    encoded_batches = []
    encode_progress = _ProgressBar(
        total=math.ceil(x.shape[0] / batch_size),
        label="Encoding new features",
        enabled=progress,
        stream=progress_stream or sys.stderr,
    )
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(torch_device)
            encoded_batches.append(model.encode(batch).cpu().numpy())
            encode_progress.advance()
    encode_progress.close(message="done")
    latent = np.concatenate(encoded_batches, axis=0).astype(np.float32)
    centers = np.asarray(payload["centers"], dtype=np.float32)
    selected_k = int(payload["selected_k"])
    return latent, centers, selected_k


def _save_cluster_model(
    model_path: str | Path,
    model_payload: dict[str, object],
    centers: np.ndarray,
    selected_k: int,
    seed: int,
    blocks: list[FeatureBlock] | None = None,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Saving the autoencoder cluster model requires PyTorch."
        ) from exc
    payload = dict(model_payload)
    payload.update(
        {
            "centers": centers.astype(np.float32),
            "selected_k": int(selected_k),
            "seed": int(seed),
            "sample_ids": [block.sample_id for block in blocks] if blocks else [],
            "source_files": [str(block.source_path) for block in blocks] if blocks else [],
            "sample_counts": [int(block.features.shape[0]) for block in blocks] if blocks else [],
        }
    )
    torch.save(payload, model_path)


def kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = validate_features(data)
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if n_clusters > x.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of samples")
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_centers = None
    best_inertia = math.inf
    for _ in range(n_init):
        centers = _kmeans_plus_plus(x, n_clusters, rng)
        labels = np.zeros(x.shape[0], dtype=np.int64)
        previous_inertia = math.inf
        for _iteration in range(max_iter):
            distances = _squared_distances(x, centers)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.sum(distances[np.arange(x.shape[0]), labels]))
            new_centers = centers.copy()
            for cluster in range(n_clusters):
                members = x[labels == cluster]
                if members.size:
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    new_centers[cluster] = x[rng.integers(0, x.shape[0])]
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if abs(previous_inertia - inertia) <= tol or shift <= tol:
                break
            previous_inertia = inertia
        distances = _squared_distances(x, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(x.shape[0]), labels]))
        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
    if best_labels is None or best_centers is None:
        raise RuntimeError("k-means failed to initialize")
    return best_labels, best_centers, best_inertia


def save_latent_scatter(
    latent: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> Path | None:
    try:
        Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    output_path = Path(output_path)
    coords = _pca_2d(latent)
    plt.figure(figsize=(7, 6), dpi=160)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab20", alpha=0.85)
    plt.xlabel("Latent PC1")
    plt.ylabel("Latent PC2")
    plt.title("Autoencoder Latent Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


class _Autoencoder:
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: tuple[int, ...], nn):
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*layers)

        decoder_layers = []
        previous = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            previous = hidden
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        return self

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, x):
        return self.encoder(x)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


def _default_hidden_dims(input_dim: int, latent_dim: int) -> tuple[int, ...]:
    first = min(512, max(latent_dim * 4, input_dim // 2))
    second = min(256, max(latent_dim * 2, first // 2))
    return tuple(dim for dim in (first, second) if dim > latent_dim)


def _torch_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu or --device auto."
        )
    return torch.device(device)


def _resolve_k_values(
    n_clusters: int | None,
    k_range: tuple[int, int] | None,
    n_samples: int,
) -> list[int]:
    if n_clusters is None and k_range is None:
        raise ValueError("provide either n_clusters or k_range")
    if n_clusters is not None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples")
        return [int(n_clusters)]
    if k_range is None:
        raise ValueError("k_range is required when n_clusters is not provided")
    start, stop = k_range
    if start < 2 or stop < start:
        raise ValueError("k_range must be like (2, 10)")
    stop = min(stop, n_samples)
    if stop < start:
        raise ValueError("k_range contains no feasible K for the sampled training tiles")
    return list(range(start, stop + 1))


def parse_k_range(value: str | None) -> tuple[int, int] | None:
    if value is None or not value.strip():
        return None
    items = [int(item.strip()) for item in value.split(",")]
    if len(items) != 2:
        raise ValueError("k_range must have two integers, for example 2,10")
    return (items[0], items[1])


def parse_hidden_dims(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_optional_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    limit = int(value)
    if limit < 0:
        raise ValueError("sample limits must be non-negative; use 0 for no limit")
    return None if limit == 0 else limit


def _metrics_row(k: int, labels: np.ndarray, metrics: dict[str, float]) -> dict[str, float]:
    row: dict[str, float] = {"k": float(k), **metrics}
    counts = np.bincount(labels, minlength=k)
    for index, count in enumerate(counts, start=1):
        row[f"cluster_{index}_count"] = float(count)
        row[f"cluster_{index}_proportion"] = float(count / labels.size)
    return row


def _metric_for_sort(value: float, larger_is_better: bool) -> float:
    if math.isnan(value):
        return -math.inf
    return value if larger_is_better else -value


def _show_progress(progress: bool, stream: TextIO, message: str) -> None:
    if progress:
        print(f"[landscape2d-cluster] {message}", file=stream, flush=True)


def _load_training_sample_blocks(
    feature_files: list[Path],
    max_train_tiles: int | None,
    max_train_tiles_per_file: int | None,
    seed: int,
    progress: bool,
    stream: TextIO,
) -> list[FeatureBlock]:
    for value in (max_train_tiles, max_train_tiles_per_file):
        if value is not None and (not isinstance(value, Integral) or value <= 0):
            raise ValueError("training limits must be positive integers or None")
    rng = np.random.default_rng(seed)
    counts = [feature_count(path) for path in feature_files]
    per_file_indices = []
    for count in counts:
        file_limit = count if max_train_tiles_per_file is None else min(count, max_train_tiles_per_file)
        per_file_indices.append(_sample_indices(count, file_limit, rng))

    total = sum(indices.size for indices in per_file_indices)
    if max_train_tiles is not None and total > max_train_tiles:
        keep = _sample_indices(total, max_train_tiles, rng)
        offsets = np.cumsum([0] + [indices.size for indices in per_file_indices])
        reduced = []
        for index, indices in enumerate(per_file_indices):
            local_mask = (keep >= offsets[index]) & (keep < offsets[index + 1])
            local_positions = keep[local_mask] - offsets[index]
            reduced.append(indices[local_positions])
        per_file_indices = reduced

    bar = _ProgressBar(
        total=len(feature_files),
        label="Loading sampled training tiles",
        enabled=progress,
        stream=stream,
    )
    blocks = []
    for path, indices in zip(feature_files, per_file_indices):
        if indices.size == 0:
            bar.advance(message=f"{path.name} n=0")
            continue
        block = load_feature_sample(path, indices)
        blocks.append(block)
        bar.advance(message=f"{path.name} n={block.features.shape[0]}")
    bar.close(message=f"sampled_n={sum(block.features.shape[0] for block in blocks)}")
    if not blocks:
        raise ValueError("no training tiles were sampled")
    if sum(block.features.shape[0] for block in blocks) < 2:
        raise ValueError("at least two sampled training tiles are required")
    return blocks


def _sample_indices(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit >= count:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(count, size=limit, replace=False).astype(np.int64))


def _predict_and_save_feature_files(
    feature_files: list[Path],
    model_path: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    save_latent: bool,
    aggregate_path: Path,
    progress: bool,
    progress_stream: TextIO,
    max_metric_samples: int = 2000,
    min_patch_cells: int = 0,
    grid_stride: tuple[float, float] | None = None,
    grid_origin: tuple[float, float] | None = None,
    coordinate_level: int | None = None,
    coord_downsample: float | None = None,
    mpp: tuple[float, float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, int, np.ndarray]:
    all_labels = []
    all_latent = []
    sample_ids = []
    source_files = []
    sample_counts = []
    total_inertia = 0.0
    selected_k = 0
    latent_dim = 0
    metric_latent_parts = []
    metric_label_parts = []
    metric_seen = 0
    metric_index_parts: list[np.ndarray] = []
    model_id = _file_digest(model_path)
    map_paths = []
    rng = np.random.default_rng(0)

    bar = _ProgressBar(
        total=len(feature_files),
        label="Inferring full feature files",
        enabled=progress,
        stream=progress_stream,
    )
    for path in feature_files:
        block = _with_grid_overrides(load_feature_block(path), grid_stride, grid_origin, coordinate_level, coord_downsample, mpp)
        latent, centers, selected_k = encode_features_with_model(
            block.features,
            model_path,
            batch_size=batch_size,
            device=device,
            progress=False,
            progress_stream=progress_stream,
        )
        latent_dim = latent.shape[1]
        distances = _squared_distances(latent, centers)
        labels_zero = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(latent.shape[0]), labels_zero]))
        total_inertia += inertia
        labels = (labels_zero.astype(np.int64) + 1)
        all_labels.append(labels)
        if save_latent:
            all_latent.append(latent.astype(np.float32))
        _update_metric_sample(
            metric_latent_parts,
            metric_label_parts,
            latent,
            labels_zero,
            max_metric_samples,
            metric_seen,
            rng,
            metric_index_parts,
        )
        metric_seen += latent.shape[0]
        map_path = _save_single_label_output(
            output_dir=output_dir,
            block=block,
            labels=labels,
            latent=latent,
            selected_k=selected_k,
            latent_dim=latent_dim,
            save_latent=save_latent,
            min_patch_cells=min_patch_cells,
            model_id=model_id,
        )
        if map_path is not None:
            map_paths.append(map_path.name)
        sample_ids.append(block.sample_id)
        source_files.append(str(block.source_path))
        sample_counts.append(block.features.shape[0])
        bar.advance(message=f"{path.name} n={block.features.shape[0]}")
    bar.close(message="done")

    tile_labels = np.concatenate(all_labels).astype(np.int64)
    aggregate_payload: dict[str, np.ndarray] = {
        "tile_labels": tile_labels,
        "selected_k": np.asarray([selected_k], dtype=np.int64),
        "latent_dim": np.asarray([latent_dim], dtype=np.int64),
        "sample_ids": np.asarray(sample_ids),
        "source_files": np.asarray(source_files),
        "sample_counts": np.asarray(sample_counts, dtype=np.int64),
    }
    if save_latent and all_latent:
        aggregate_payload["latent"] = np.concatenate(all_latent).astype(np.float32)
    aggregate_payload["artifact_type"] = np.asarray("tile_label_summary")
    aggregate_payload["model_id"] = np.asarray(model_id)
    np.savez(aggregate_path, **aggregate_payload)
    (output_dir / "label_maps.json").write_text(json.dumps({"label_maps": map_paths}, indent=2) + "\n", encoding="utf-8")

    if metric_latent_parts:
        metric_latent = np.concatenate(metric_latent_parts).astype(np.float32)
        metric_labels = np.concatenate(metric_label_parts).astype(np.int64)
    else:
        metric_latent = np.empty((0, max(1, latent_dim)), dtype=np.float32)
        metric_labels = np.empty((0,), dtype=np.int64)
    if metric_latent.shape[0] > 0:
        metrics = clustering_metrics(metric_latent, metric_labels, centers, total_inertia,
                                     max_silhouette_samples=max_metric_samples)
    else:
        metrics = {"inertia": total_inertia, "silhouette_score": float("nan"),
                   "calinski_harabasz_score": float("nan"), "davies_bouldin_score": float("nan")}
    metrics["metric_sample_count"] = float(metric_latent.shape[0])
    row = _metrics_row(selected_k, (tile_labels - 1).astype(np.int64), metrics)
    dataframe = pd.DataFrame([{**row, "selected": True}])
    ordered = ["k", "selected"] + [
        column for column in dataframe.columns if column not in {"k", "selected"}
    ]
    dataframe = dataframe[ordered]
    sample_indices = np.concatenate(metric_index_parts) if metric_index_parts else np.empty(0, dtype=np.int64)
    return tile_labels.reshape(-1, 1), dataframe, metric_latent, metric_labels + 1, selected_k, sample_indices


def _save_single_label_output(
    output_dir: Path,
    block: FeatureBlock,
    labels: np.ndarray,
    latent: np.ndarray,
    selected_k: int,
    latent_dim: int,
    save_latent: bool,
    min_patch_cells: int = 0,
    model_id: str = "",
) -> Path | None:
    payload: dict[str, np.ndarray] = {
        "tile_labels": labels.astype(np.int64),
        "selected_k": np.asarray(selected_k, dtype=np.int64),
        "latent_dim": np.asarray(latent_dim, dtype=np.int64),
        "source_file": np.asarray(str(block.source_path)),
        "model_id": np.asarray(model_id),
        "artifact_type": np.asarray("tile_labels"),
    }
    spatial = block.coords is not None and block.grid_stride is not None
    if spatial:
        origin = block.grid_origin or tuple(block.coords.min(axis=0))
        label_matrix = labels_to_matrix(labels, block.coords, stride=block.grid_stride, origin=origin)
        label_matrix = filter_small_patches(label_matrix, min_patch_cells)
        payload.update(matrix=label_matrix, labels=label_matrix, artifact_type=np.asarray("label_map"),
                       grid_stride=np.asarray(block.grid_stride), grid_origin=np.asarray(origin),
                       coordinate_level=np.asarray(block.coordinate_level), coord_downsample=np.asarray(block.coord_downsample))
        scale = np.asarray(block.mpp if block.mpp is not None else (1.0, 1.0))
        payload["spacing"] = np.asarray(block.grid_stride) * block.coord_downsample * scale
        payload["spacing_unit"] = np.asarray("um" if block.mpp is not None else "coordinate_units")
        if block.mpp is not None:
            payload["mpp"] = np.asarray(block.mpp)
    elif min_patch_cells > 0:
        raise ValueError("small-patch filtering requires coords and grid_stride")
    if block.coords is not None:
        payload["coords"] = block.coords
    if save_latent:
        payload["latent"] = latent.astype(np.float32)
    path = output_dir / f"{block.sample_id}_clusters.npz"
    np.savez(path, **payload)
    return path if spatial else None


def _update_metric_sample(
    latent_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
    latent: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seen_before: int,
    rng: np.random.Generator,
    index_parts: list[np.ndarray] | None = None,
) -> None:
    if max_samples <= 0:
        return
    existing = sum(part.shape[0] for part in latent_parts)
    take = min(max_samples - existing, latent.shape[0])
    if take > 0:
        latent_parts.append(latent[:take].astype(np.float32))
        label_parts.append(labels[:take].astype(np.int64))
        if index_parts is not None:
            index_parts.append(np.arange(seen_before, seen_before + take, dtype=np.int64))
    if take == latent.shape[0]:
        return
    sample_latent = np.concatenate(latent_parts)
    sample_labels = np.concatenate(label_parts)
    sample_indices = np.concatenate(index_parts) if index_parts is not None else None
    # Continue through the tail of the same file that filled the reservoir.
    for local_index in range(take, latent.shape[0]):
        global_index = seen_before + local_index
        replacement = int(rng.integers(0, global_index + 1))
        if replacement < max_samples:
            sample_latent[replacement] = latent[local_index]
            sample_labels[replacement] = labels[local_index]
            if sample_indices is not None:
                sample_indices[replacement] = global_index
    latent_parts[:] = [sample_latent]
    label_parts[:] = [sample_labels]
    if index_parts is not None:
        index_parts[:] = [sample_indices]


class _ProgressBar:
    def __init__(
        self,
        total: int,
        label: str,
        enabled: bool,
        stream: TextIO,
        width: int = 28,
        leave: bool = True,
    ) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled
        self.stream = stream
        self.width = width
        self.leave = leave
        self.current = 0
        self._last_length = 0
        self._closed = False
        if self.enabled:
            self.render()

    def advance(self, step: int = 1, message: str = "") -> None:
        if self._closed:
            return
        self.current = min(self.total, self.current + step)
        self.render(message)

    def update(self, current: int | None = None, message: str = "") -> None:
        if self._closed:
            return
        if current is not None:
            self.current = min(self.total, max(0, int(current)))
        self.render(message)

    def close(self, message: str = "") -> None:
        if self._closed:
            return
        self.current = self.total
        if self.enabled:
            self.render(message)
            if self.leave:
                print(file=self.stream, flush=True)
            else:
                print("\r" + " " * self._last_length + "\r", end="", file=self.stream, flush=True)
        self._closed = True

    def render(self, message: str = "") -> None:
        if not self.enabled:
            return
        fraction = self.current / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(round(100 * fraction))
        text = (
            f"\r[landscape2d-cluster] {self.label}: "
            f"[{bar}] {self.current}/{self.total} {percent:3d}%"
        )
        if message:
            text += f" | {message}"
        padding = " " * max(0, self._last_length - len(text))
        print(text + padding, end="", file=self.stream, flush=True)
        self._last_length = len(text)


def _kmeans_plus_plus(data: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((n_clusters, data.shape[1]), dtype=np.float32)
    centers[0] = data[rng.integers(0, data.shape[0])]
    closest = _squared_distances(data, centers[:1]).ravel()
    for index in range(1, n_clusters):
        total = float(closest.sum())
        if total <= 0:
            centers[index] = data[rng.integers(0, data.shape[0])]
        else:
            probabilities = closest / total
            centers[index] = data[rng.choice(data.shape[0], p=probabilities)]
        closest = np.minimum(closest, _squared_distances(data, centers[index : index + 1]).ravel())
    return centers


def _squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(data * data, axis=1, keepdims=True)
        - 2.0 * data @ centers.T
        + np.sum(centers * centers, axis=1)
    )
    return np.maximum(distances, 0.0)


def _pca_2d(data: np.ndarray) -> np.ndarray:
    x = validate_features(data)
    centered = x - x.mean(axis=0, keepdims=True)
    if x.shape[1] == 1:
        return np.column_stack([centered[:, 0], np.zeros(x.shape[0], dtype=np.float32)])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] == 1:
        return np.column_stack([centered @ components[:, 0], np.zeros(x.shape[0], dtype=np.float32)])
    return centered @ components
