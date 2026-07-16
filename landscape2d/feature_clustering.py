from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import sys
from typing import TextIO

import numpy as np
import pandas as pd

from .filters import filter_small_patches


@dataclass(frozen=True)
class FeatureClusteringResult:
    labels: np.ndarray
    latent: np.ndarray
    metrics: pd.DataFrame
    selected_k: int
    labels_path: Path
    metrics_path: Path
    plot_path: Path | None
    model_path: Path | None = None


@dataclass(frozen=True)
class FeatureBlock:
    sample_id: str
    source_path: Path
    features: np.ndarray
    coords: np.ndarray | None = None


def load_features(path: str | Path) -> np.ndarray:
    return load_feature_block(path).features


def load_feature_block(path: str | Path) -> FeatureBlock:
    path = Path(path)
    if path.suffix == ".npy":
        return FeatureBlock(_sample_id(path), path, validate_features(np.load(path)))
    if path.suffix == ".npz":
        with np.load(path) as data:
            for key in ("features", "x", "array"):
                if key in data:
                    coords = validate_coords(data["coords"]) if "coords" in data else None
                    return FeatureBlock(
                        _sample_id(path),
                        path,
                        validate_features(data[key]),
                        coords=coords,
                    )
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} must contain one array or an array named 'features'"
                )
            return FeatureBlock(_sample_id(path), path, validate_features(data[keys[0]]))
    if path.suffix in {".csv", ".txt"}:
        delimiter = "," if path.suffix == ".csv" else None
        return FeatureBlock(
            _sample_id(path),
            path,
            validate_features(np.loadtxt(path, delimiter=delimiter)),
        )
    if path.suffix in {".h5", ".hdf5"}:
        return load_h5_feature_block(path)
    raise ValueError(f"unsupported feature file: {path}")


def load_feature_sample(
    path: str | Path,
    indices: np.ndarray | None = None,
) -> FeatureBlock:
    path = Path(path)
    if indices is None:
        return load_feature_block(path)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("sample indices cannot be empty")
    if path.suffix in {".h5", ".hdf5"}:
        return load_h5_feature_block(path, indices=indices)
    block = load_feature_block(path)
    return FeatureBlock(
        block.sample_id,
        block.source_path,
        validate_features(block.features[indices]),
        coords=validate_coords(block.coords[indices]) if block.coords is not None else None,
    )


def load_h5_feature_block(
    path: str | Path,
    feature_key: str = "feats",
    coords_key: str = "coords",
    indices: np.ndarray | None = None,
) -> FeatureBlock:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading .h5 feature files requires h5py.") from exc
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if feature_key not in handle:
            raise ValueError(f"{path} does not contain dataset {feature_key!r}")
        if indices is None:
            features = validate_features(handle[feature_key][()])
            coords = validate_coords(handle[coords_key][()]) if coords_key in handle else None
        else:
            order = np.argsort(indices)
            sorted_indices = np.asarray(indices[order], dtype=np.int64)
            inverse = np.argsort(order)
            features = validate_features(handle[feature_key][sorted_indices][inverse])
            coords = (
                validate_coords(handle[coords_key][sorted_indices][inverse])
                if coords_key in handle
                else None
            )
    return FeatureBlock(_sample_id(path), path, features, coords=coords)


def iter_feature_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    suffixes = (".npy", ".npz", ".csv", ".txt", ".h5", ".hdf5")
    if path.is_file():
        if path.name.lower().endswith(suffixes):
            return [path]
        raise ValueError(f"unsupported feature file: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    files = sorted(item for item in path.iterdir() if item.name.lower().endswith(suffixes))
    if not files:
        raise ValueError(f"no feature files found in {path}")
    return files


def feature_count(path: str | Path) -> int:
    path = Path(path)
    if path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading .h5 feature files requires h5py.") from exc
        with h5py.File(path, "r") as handle:
            if "feats" not in handle:
                raise ValueError(f"{path} does not contain dataset 'feats'")
            return int(handle["feats"].shape[0])
    return int(load_feature_block(path).features.shape[0])


def validate_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {array.shape!r}")
    if array.shape[0] < 2:
        raise ValueError("features must contain at least two samples")
    if array.shape[1] < 1:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must be finite")
    return array


def validate_coords(coords: np.ndarray) -> np.ndarray:
    array = np.asarray(coords)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"coords must have shape n x 2, got {array.shape!r}")
    if not np.all(np.isfinite(array)):
        raise ValueError("coords must be finite")
    return array


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
) -> FeatureClusteringResult:
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
    full_labels, _, _, _, _ = _predict_and_save_feature_files(
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
        latent=latent,
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
) -> FeatureClusteringResult:
    input_path = Path(input_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = progress_stream or sys.stderr

    feature_files = iter_feature_files(input_path)
    _show_progress(progress, stream, f"Loading cluster model: {model_path}")
    labels_path = output_dir / "all_clusters.npz"
    labels, dataframe, latent_for_plot, labels_for_plot, selected_k = _predict_and_save_feature_files(
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
    )

    metrics_path = output_dir / "cluster_metrics.csv"
    dataframe.to_csv(metrics_path, index=False)
    plot_path = output_dir / "latent_scatter.png"
    saved_plot = save_latent_scatter(latent_for_plot, labels_for_plot, plot_path)
    _show_progress(progress, stream, f"Saved inferred labels: {output_dir}")
    _show_progress(progress, stream, f"Saved inference metrics: {metrics_path}")
    if saved_plot:
        _show_progress(progress, stream, f"Saved latent plot: {saved_plot}")
    return FeatureClusteringResult(
        labels=labels,
        latent=latent_for_plot,
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

    rng = np.random.default_rng(seed)
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
    order_check = rng.integers(0, max(1, latent.shape[0]))
    if not np.all(np.isfinite(latent[order_check])):
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


def clustering_metrics(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    inertia: float,
    max_silhouette_samples: int = 2000,
    random_state: int = 0,
) -> dict[str, float]:
    x = validate_features(data)
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels)
    if unique.size < 2:
        silhouette = float("nan")
        ch = float("nan")
        db = float("nan")
    else:
        silhouette = _silhouette_score(
            x, labels, max_samples=max_silhouette_samples, random_state=random_state
        )
        ch = _calinski_harabasz_score(x, labels, centers, inertia)
        db = _davies_bouldin_score(x, labels, centers)
    return {
        "inertia": float(inertia),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
    }


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
    return None if limit <= 0 else limit


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


def _sample_id(path: Path) -> str:
    name = path.name
    for suffix in (".hdf5", ".h5", ".npy", ".npz", ".csv", ".txt"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _block_slices(blocks: list[FeatureBlock]) -> list[slice]:
    slices = []
    start = 0
    for block in blocks:
        stop = start + block.features.shape[0]
        slices.append(slice(start, stop))
        start = stop
    return slices


def _save_label_outputs(
    output_dir: Path,
    blocks: list[FeatureBlock],
    slices: list[slice],
    labels: np.ndarray,
    latent: np.ndarray,
    selected_k: int,
    latent_dim: int,
    save_latent: bool,
    aggregate_path: Path,
    progress: bool = False,
    progress_stream: TextIO | None = None,
    min_patch_cells: int = 0,
) -> None:
    sample_ids = np.asarray([block.sample_id for block in blocks])
    source_files = np.asarray([str(block.source_path) for block in blocks])
    sample_counts = np.asarray([block.features.shape[0] for block in blocks], dtype=np.int64)
    aggregate_payload: dict[str, np.ndarray] = {
        "tile_labels": labels.ravel().astype(np.int64),
        "selected_k": np.asarray([selected_k], dtype=np.int64),
        "latent_dim": np.asarray([latent_dim], dtype=np.int64),
        "sample_ids": sample_ids,
        "source_files": source_files,
        "sample_counts": sample_counts,
    }
    if save_latent:
        aggregate_payload["latent"] = latent.astype(np.float32)
    np.savez(aggregate_path, **aggregate_payload)

    save_progress = _ProgressBar(
        total=len(blocks),
        label="Saving sample matrices",
        enabled=progress,
        stream=progress_stream or sys.stderr,
    )
    for block, block_slice in zip(blocks, slices):
        block_labels = labels[block_slice].ravel().astype(np.int64)
        label_matrix = labels_to_matrix(block_labels, block.coords)
        label_matrix = filter_small_patches(label_matrix, min_patch_cells)
        payload: dict[str, np.ndarray] = {
            "labels": label_matrix,
            "matrix": label_matrix,
            "tile_labels": block_labels,
            "selected_k": np.asarray([selected_k], dtype=np.int64),
            "latent_dim": np.asarray([latent_dim], dtype=np.int64),
            "source_file": np.asarray([str(block.source_path)]),
        }
        if block.coords is not None:
            payload["coords"] = block.coords
        if save_latent:
            payload["latent"] = latent[block_slice].astype(np.float32)
        np.savez(output_dir / f"{block.sample_id}_clusters.npz", **payload)
        save_progress.advance(message=block.sample_id)
    save_progress.close(message="done")


def labels_to_matrix(
    labels: np.ndarray,
    coords: np.ndarray | None,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if coords is None:
        return labels.reshape(1, -1)
    coords = validate_coords(coords)
    if coords.shape[0] != labels.shape[0]:
        raise ValueError(
            f"coords and labels length mismatch: {coords.shape[0]} vs {labels.shape[0]}"
        )
    x_values = np.unique(coords[:, 0])
    y_values = np.unique(coords[:, 1])
    x_to_col = {value: index for index, value in enumerate(x_values.tolist())}
    y_to_row = {value: index for index, value in enumerate(y_values.tolist())}
    matrix = np.zeros((len(y_values), len(x_values)), dtype=np.int64)
    for label, (x_coord, y_coord) in zip(labels.tolist(), coords.tolist()):
        matrix[y_to_row[y_coord], x_to_col[x_coord]] = int(label)
    return matrix


def _show_progress(progress: bool, stream: TextIO, message: str) -> None:
    if progress:
        print(f"[landscape2d-cluster] {message}", file=stream, flush=True)


def _load_feature_blocks(
    feature_files: list[Path],
    progress: bool,
    stream: TextIO,
) -> list[FeatureBlock]:
    bar = _ProgressBar(
        total=len(feature_files),
        label="Loading feature files",
        enabled=progress,
        stream=stream,
    )
    blocks = []
    for path in feature_files:
        block = load_feature_block(path)
        blocks.append(block)
        bar.advance(message=f"{path.name} n={block.features.shape[0]}")
    bar.close(message=f"files={len(blocks)}")
    return blocks


def _load_training_sample_blocks(
    feature_files: list[Path],
    max_train_tiles: int | None,
    max_train_tiles_per_file: int | None,
    seed: int,
    progress: bool,
    stream: TextIO,
) -> list[FeatureBlock]:
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
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, int]:
    all_labels = []
    all_latent = []
    sample_ids = []
    source_files = []
    sample_counts = []
    counts_by_cluster: np.ndarray | None = None
    total_inertia = 0.0
    selected_k = 0
    latent_dim = 0
    metric_latent_parts = []
    metric_label_parts = []
    metric_seen = 0
    rng = np.random.default_rng(0)

    bar = _ProgressBar(
        total=len(feature_files),
        label="Inferring full feature files",
        enabled=progress,
        stream=progress_stream,
    )
    for path in feature_files:
        block = load_feature_block(path)
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
        counts = np.bincount(labels_zero, minlength=selected_k)
        counts_by_cluster = counts if counts_by_cluster is None else counts_by_cluster + counts
        _update_metric_sample(
            metric_latent_parts,
            metric_label_parts,
            latent,
            labels_zero,
            max_metric_samples,
            metric_seen,
            rng,
        )
        metric_seen += latent.shape[0]
        _save_single_label_output(
            output_dir=output_dir,
            block=block,
            labels=labels,
            latent=latent,
            selected_k=selected_k,
            latent_dim=latent_dim,
            save_latent=save_latent,
            min_patch_cells=min_patch_cells,
        )
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
    np.savez(aggregate_path, **aggregate_payload)

    if metric_latent_parts:
        metric_latent = np.concatenate(metric_latent_parts).astype(np.float32)
        metric_labels = np.concatenate(metric_label_parts).astype(np.int64)
    else:
        metric_latent = np.empty((0, max(1, latent_dim)), dtype=np.float32)
        metric_labels = np.empty((0,), dtype=np.int64)
    metrics = {
        "inertia": total_inertia,
        "silhouette_score": float("nan"),
        "calinski_harabasz_score": float("nan"),
        "davies_bouldin_score": float("nan"),
    }
    row = _metrics_row(selected_k, (tile_labels - 1).astype(np.int64), metrics)
    dataframe = pd.DataFrame([{**row, "selected": True}])
    ordered = ["k", "selected"] + [
        column for column in dataframe.columns if column not in {"k", "selected"}
    ]
    dataframe = dataframe[ordered]
    return tile_labels.reshape(-1, 1), dataframe, metric_latent, metric_labels + 1, selected_k


def _save_single_label_output(
    output_dir: Path,
    block: FeatureBlock,
    labels: np.ndarray,
    latent: np.ndarray,
    selected_k: int,
    latent_dim: int,
    save_latent: bool,
    min_patch_cells: int = 0,
) -> None:
    label_matrix = labels_to_matrix(labels, block.coords)
    label_matrix = filter_small_patches(label_matrix, min_patch_cells)
    payload: dict[str, np.ndarray] = {
        "labels": label_matrix,
        "matrix": label_matrix,
        "tile_labels": labels.astype(np.int64),
        "selected_k": np.asarray([selected_k], dtype=np.int64),
        "latent_dim": np.asarray([latent_dim], dtype=np.int64),
        "source_file": np.asarray([str(block.source_path)]),
    }
    if block.coords is not None:
        payload["coords"] = block.coords
    if save_latent:
        payload["latent"] = latent.astype(np.float32)
    np.savez(output_dir / f"{block.sample_id}_clusters.npz", **payload)


def _update_metric_sample(
    latent_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
    latent: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seen_before: int,
    rng: np.random.Generator,
) -> None:
    if max_samples <= 0:
        return
    existing = sum(part.shape[0] for part in latent_parts)
    remaining = max_samples - existing
    if remaining > 0:
        take = min(remaining, latent.shape[0])
        latent_parts.append(latent[:take].astype(np.float32))
        label_parts.append(labels[:take].astype(np.int64))
        return
    # Reservoir replacement keeps the plotting/metric sample bounded.
    if not latent_parts:
        return
    sample_latent = np.concatenate(latent_parts)
    sample_labels = np.concatenate(label_parts)
    for local_index in range(latent.shape[0]):
        global_index = seen_before + local_index
        replacement = rng.integers(0, global_index + 1)
        if replacement < max_samples:
            sample_latent[replacement] = latent[local_index]
            sample_labels[replacement] = labels[local_index]
    latent_parts[:] = [sample_latent]
    label_parts[:] = [sample_labels]


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


def _silhouette_score(
    data: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> float:
    if data.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(data.shape[0], size=max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
    unique = np.unique(labels)
    if unique.size < 2 or unique.size == data.shape[0]:
        return float("nan")
    distances = np.sqrt(np.maximum(_squared_distances(data, data), 0.0))
    scores = []
    for index in range(data.shape[0]):
        same = labels == labels[index]
        same[index] = False
        if np.any(same):
            a = float(distances[index, same].mean())
        else:
            a = 0.0
        b_values = [
            float(distances[index, labels == other].mean())
            for other in unique
            if other != labels[index] and np.any(labels == other)
        ]
        b = min(b_values) if b_values else float("nan")
        denominator = max(a, b)
        if denominator > 0 and not math.isnan(denominator):
            scores.append((b - a) / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def _calinski_harabasz_score(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    inertia: float,
) -> float:
    n_samples = data.shape[0]
    unique = np.unique(labels)
    k = unique.size
    if k < 2 or n_samples <= k or inertia <= 0:
        return float("nan")
    overall = data.mean(axis=0)
    between = 0.0
    for cluster in unique:
        count = int(np.count_nonzero(labels == cluster))
        center = centers[cluster]
        between += count * float(np.sum((center - overall) ** 2))
    return float((between / (k - 1)) / (inertia / (n_samples - k)))


def _davies_bouldin_score(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    scatters = []
    for cluster in unique:
        members = data[labels == cluster]
        if members.size == 0:
            scatters.append(0.0)
        else:
            distances = np.linalg.norm(members - centers[cluster], axis=1)
            scatters.append(float(distances.mean()))
    scatters_array = np.asarray(scatters)
    center_distances = np.linalg.norm(
        centers[unique][:, None, :] - centers[unique][None, :, :], axis=2
    )
    np.fill_diagonal(center_distances, np.inf)
    ratios = (scatters_array[:, None] + scatters_array[None, :]) / center_distances
    return float(np.max(ratios, axis=1).mean())


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
