"""Sample-based cluster quality metrics with explicit degenerate cases."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist

from .feature_io import validate_features


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
    if labels.shape != (x.shape[0],) or np.any(labels < 0):
        raise ValueError("labels must be a non-negative vector aligned with data")
    unique, remapped = np.unique(labels, return_inverse=True)
    if unique.size < 2 or unique.size >= x.shape[0]:
        silhouette = float("nan")
        ch = float("nan")
        db = float("nan")
    else:
        silhouette = _silhouette_score(
            x, labels, max_samples=max_silhouette_samples, random_state=random_state
        )
        # Quality scores describe this sample's clusters, not distances to the
        # training centroids. Inertia in the output still covers all input tiles.
        sample = x.astype(float)
        sample_centers = np.asarray([sample[remapped == index].mean(axis=0) for index in range(unique.size)])
        sample_inertia = float(np.sum((sample - sample_centers[remapped]) ** 2))
        ch = _calinski_harabasz_score(sample, remapped, sample_centers, sample_inertia)
        db = _davies_bouldin_score(sample, remapped, sample_centers)
    return {
        "inertia": float(inertia),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
    }


def _silhouette_score(
    data: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> float:
    if max_samples <= 0:
        return float("nan")
    if data.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(data.shape[0], size=max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
    unique = np.unique(labels)
    if unique.size < 2 or unique.size == data.shape[0]:
        return float("nan")
    distances = cdist(data, data)
    scores = []
    for index in range(data.shape[0]):
        same = labels == labels[index]
        same[index] = False
        if np.any(same):
            a = float(distances[index, same].mean())
        else:
            scores.append(0.0)
            continue
        b_values = [
            float(distances[index, labels == other].mean())
            for other in unique
            if other != labels[index] and np.any(labels == other)
        ]
        b = min(b_values) if b_values else float("nan")
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 0 else 0.0)
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
    if k < 2 or n_samples <= k:
        return float("nan")
    if inertia == 0:
        return 1.0
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
    if np.allclose(scatters_array, 0):
        return 0.0
    center_distances[center_distances == 0] = np.inf
    ratios = (scatters_array[:, None] + scatters_array[None, :]) / center_distances
    return float(np.max(ratios, axis=1).mean())


