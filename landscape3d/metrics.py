from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from numbers import Integral

from landscape_common.aggregation import contagion, interspersion, clumpiness
from landscape_common.spatial import nearest_boundary_distances

from .adjacency import surface_and_contact_areas
from .io import parse_classes, parse_spacing, validate_volume
from .patches import PatchStats, connectivity_structure, label_patches, summarize_patches


CLASS_METRICS = [
    "voxel_count",
    "volume",
    "pland",
    "patch_count",
    "patch_density",
    "largest_patch_volume",
    "largest_patch_index",
    "patch_volume_mean",
    "patch_volume_sd",
    "patch_volume_cv",
    "patch_volume_range",
    "total_surface_area",
    "surface_density",
    "patch_surface_area_mean",
    "patch_surface_area_sd",
    "patch_surface_area_cv",
    "patch_surface_area_range",
    "boundary_to_background_area",
    "boundary_to_other_classes_area",
    "surface_volume_ratio_mean",
    "surface_volume_ratio_sd",
    "surface_volume_ratio_cv",
    "shape_index_mean",
    "shape_index_sd",
    "shape_index_cv",
    "sphericity_mean",
    "sphericity_sd",
    "sphericity_cv",
    "radius_gyration_mean",
    "radius_gyration_sd",
    "radius_gyration_cv",
    "like_adjacency_percent",
    "clumpiness",
    "splitting_index",
    "effective_mesh_size",
    "patch_volume_concentration",
    "nearest_neighbor_mean",
    "nearest_neighbor_sd",
    "nearest_neighbor_cv",
    "unlike_boundary_area",
    "unlike_boundary_percent",
]


def compute_metrics(
    volume: np.ndarray,
    classes: list[int] | str | None = None,
    spacing: tuple[float, float, float] | str = (1, 1, 1),
    connectivity: int = 6,
    min_patch_voxels: int = 0,
    min_patch_volume: float = 0.0,
    missing_value: float = np.nan,
) -> dict[str, float]:
    labels = validate_volume(volume)
    spacing_tuple = parse_spacing(spacing)
    connectivity_structure(connectivity)
    if not isinstance(min_patch_voxels, Integral) or min_patch_voxels < 0:
        raise ValueError("min_patch_voxels must be non-negative")
    if not np.isfinite(min_patch_volume) or min_patch_volume < 0:
        raise ValueError("min_patch_volume must be non-negative")

    class_values = parse_classes(classes)
    if class_values is None:
        class_values = sorted(int(value) for value in np.unique(labels) if value > 0)
    if not set(int(value) for value in np.unique(labels) if value > 0).issubset(class_values):
        raise ValueError("classes must include every observed non-background label")
    labels = _filter_small_patches(
        labels,
        class_values,
        spacing_tuple,
        connectivity,
        min_patch_voxels,
        min_patch_volume,
    )

    non_background_mask = labels > 0
    non_background_voxels = int(np.count_nonzero(non_background_mask))
    voxel_volume = float(np.prod(spacing_tuple))
    total_non_background_volume = non_background_voxels * voxel_volume
    landscape_extent_volume = labels.size * voxel_volume

    adjacency, pairwise = surface_and_contact_areas(labels, class_values, spacing_tuple)

    result: dict[str, float] = {}
    class_patch_stats: dict[int, list[PatchStats]] = {}
    class_volumes: dict[int, float] = {}

    for cls in class_values:
        mask = labels == cls
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            for metric in CLASS_METRICS:
                result[f"class_{cls}_{metric}"] = missing_value
            class_patch_stats[cls] = []
            class_volumes[cls] = 0.0
            continue

        patch_labels, patch_count = label_patches(mask, connectivity)
        patch_stats = summarize_patches(patch_labels, patch_count, spacing_tuple)
        class_patch_stats[cls] = patch_stats
        class_volume = voxel_count * voxel_volume
        class_volumes[cls] = class_volume
        surface_area = adjacency[cls]["total_surface_area"]
        other_area = adjacency[cls]["boundary_to_other_classes_area"]
        like_count = adjacency[cls]["like_adjacency_count"]
        adjacency_count = adjacency[cls]["total_adjacency_count"]

        patch_volumes = [patch.volume for patch in patch_stats]
        patch_surfaces = [patch.surface_area for patch in patch_stats]
        sv_ratios = [_safe_divide(patch.surface_area, patch.volume) for patch in patch_stats]
        nearest = nearest_boundary_distances(patch_stats)

        result.update(
            {
                f"class_{cls}_voxel_count": float(voxel_count),
                f"class_{cls}_volume": class_volume,
                f"class_{cls}_pland": _safe_percent(
                    class_volume, total_non_background_volume
                ),
                f"class_{cls}_patch_count": float(patch_count),
                f"class_{cls}_patch_density": _safe_divide(
                    patch_count, total_non_background_volume
                ),
                f"class_{cls}_largest_patch_volume": max(patch_volumes),
                f"class_{cls}_largest_patch_index": _safe_percent(
                    max(patch_volumes), total_non_background_volume
                ),
                f"class_{cls}_total_surface_area": surface_area,
                f"class_{cls}_surface_density": _safe_divide(
                    surface_area, total_non_background_volume
                ),
                f"class_{cls}_boundary_to_background_area": adjacency[cls][
                    "boundary_to_background_area"
                ],
                f"class_{cls}_boundary_to_other_classes_area": other_area,
                f"class_{cls}_like_adjacency_percent": _safe_percent(
                    like_count, adjacency_count
                ) if voxel_count > 1 else float("nan"),
                f"class_{cls}_clumpiness": clumpiness(like_count, adjacency_count, class_volume / landscape_extent_volume),
                f"class_{cls}_splitting_index": _splitting_index(
                    patch_volumes, total_non_background_volume
                ),
                f"class_{cls}_effective_mesh_size": _effective_mesh_size(
                    patch_volumes, total_non_background_volume
                ),
                f"class_{cls}_patch_volume_concentration": _patch_volume_concentration(patch_volumes),
                f"class_{cls}_unlike_boundary_area": adjacency[cls]["unlike_boundary_area"],
                f"class_{cls}_unlike_boundary_percent": _safe_percent(
                    other_area, surface_area
                ),
            }
        )
        _add_summary(result, f"class_{cls}_patch_volume", patch_volumes)
        _add_summary(result, f"class_{cls}_patch_surface_area", patch_surfaces)
        _add_summary(result, f"class_{cls}_surface_volume_ratio", sv_ratios, include_range=False)
        _add_summary(result, f"class_{cls}_shape_index", [p.shape_index for p in patch_stats], include_range=False)
        _add_summary(result, f"class_{cls}_sphericity", [p.sphericity for p in patch_stats], include_range=False)
        _add_summary(result, f"class_{cls}_radius_gyration", [p.radius_gyration for p in patch_stats], include_range=False)
        _add_summary(result, f"class_{cls}_nearest_neighbor", nearest, include_range=False)

    _add_pairwise_columns(result, pairwise, class_values, prefix="landscape")
    for a, b in combinations(class_values, 2):
        result[f"class_{a}_to_{b}_boundary_area"] = pairwise.get((a, b), 0.0)

    result.update(
        _landscape_metrics(
            labels=labels,
            class_values=class_values,
            class_patch_stats=class_patch_stats,
            class_volumes=class_volumes,
            total_non_background_volume=total_non_background_volume,
            landscape_extent_volume=landscape_extent_volume,
            pairwise=pairwise,
            adjacency=adjacency,
        )
    )
    return dict(sorted(result.items()))


def _filter_small_patches(
    labels: np.ndarray,
    class_values: list[int],
    spacing: tuple[float, float, float],
    connectivity: int,
    min_patch_voxels: int,
    min_patch_volume: float,
) -> np.ndarray:
    if min_patch_voxels <= 0 and min_patch_volume <= 0:
        return labels

    voxel_volume = float(np.prod(spacing))
    filtered = labels.copy()
    for cls in class_values:
        patch_labels, patch_count = label_patches(labels == cls, connectivity)
        sizes = np.bincount(patch_labels.ravel(), minlength=patch_count + 1)
        remove = (sizes < min_patch_voxels) | (sizes * voxel_volume < min_patch_volume)
        remove[0] = False
        filtered[remove[patch_labels]] = 0
    return filtered


def _landscape_metrics(
    labels: np.ndarray,
    class_values: list[int],
    class_patch_stats: dict[int, list[PatchStats]],
    class_volumes: dict[int, float],
    total_non_background_volume: float,
    landscape_extent_volume: float,
    pairwise: dict[tuple[int, int], float],
    adjacency: dict[int, dict[str, float]],
) -> dict[str, float]:
    all_patches = [patch for patches in class_patch_stats.values() for patch in patches]
    patch_volumes = [patch.volume for patch in all_patches]
    present_volumes = [value for value in class_volumes.values() if value > 0]
    proportions = [
        volume / total_non_background_volume
        for volume in present_volumes
        if total_non_background_volume > 0
    ]
    sum_class_surfaces = sum(values["total_surface_area"] for values in adjacency.values())
    total_unlike = sum(pairwise.values())
    total_surface_area = sum_class_surfaces - total_unlike
    total_adjacencies = sum(values["total_adjacency_count"] for values in adjacency.values())
    like_adjacencies = sum(values["like_adjacency_count"] for values in adjacency.values())
    diversity_count = len(proportions)
    shannon = -sum(p * math.log(p) for p in proportions if p > 0)
    simpson = 1.0 - sum(p * p for p in proportions)
    modified_simpson = -math.log(sum(p * p for p in proportions)) if proportions else float("nan")

    result = {
        "landscape_total_non_background_volume": total_non_background_volume,
        "landscape_class_richness": float(diversity_count),
        "landscape_shannon_diversity": shannon if proportions else float("nan"),
        "landscape_simpson_diversity": simpson if proportions else float("nan"),
        "landscape_modified_simpson_diversity": modified_simpson,
        "landscape_shannon_evenness": _safe_divide(shannon, math.log(diversity_count))
        if diversity_count > 1
        else float("nan"),
        "landscape_simpson_evenness": _safe_divide(simpson, 1.0 - 1.0 / diversity_count)
        if diversity_count > 1
        else float("nan"),
        "landscape_total_surface_area": total_surface_area,
        "landscape_sum_class_surface_area": sum_class_surfaces,
        "landscape_surface_density": _safe_divide(
            total_surface_area, total_non_background_volume
        ),
        "landscape_largest_patch_index": _safe_percent(
            max(patch_volumes) if patch_volumes else 0.0,
            total_non_background_volume,
        ),
        "landscape_patch_count": float(len(all_patches)),
        "landscape_patch_density": _safe_divide(
            len(all_patches), total_non_background_volume
        ),
        "landscape_shape_index_mean": _mean([p.shape_index for p in all_patches]),
        "landscape_sphericity_mean": _mean([p.sphericity for p in all_patches]),
        "landscape_contagion": _contagion(labels, class_values),
        "landscape_interspersion_juxtaposition": interspersion(pairwise, diversity_count),
        "landscape_effective_mesh_size": _effective_mesh_size(
            patch_volumes, total_non_background_volume
        ),
        "landscape_splitting_index": _splitting_index(
            patch_volumes, total_non_background_volume
        ),
        "landscape_landscape_division": 1.0
        - _safe_divide(sum(v * v for v in patch_volumes), total_non_background_volume**2),
        "landscape_like_face_adjacency_percent": _safe_percent(like_adjacencies, total_adjacencies),
        "landscape_total_unlike_boundary_area": total_unlike,
        "landscape_unlike_boundary_percent": _safe_percent(total_unlike, total_surface_area),
        "landscape_extent_volume": landscape_extent_volume,
    }
    return result


def _add_pairwise_columns(
    result: dict[str, float],
    pairwise: dict[tuple[int, int], float],
    class_values: list[int],
    prefix: str,
) -> None:
    for a, b in combinations(class_values, 2):
        result[f"{prefix}_class_{a}_to_{b}_boundary_area"] = pairwise.get((a, b), 0.0)


def _add_summary(
    result: dict[str, float],
    prefix: str,
    values: list[float],
    include_range: bool = True,
) -> None:
    clean = [value for value in values if not math.isnan(value)]
    result[f"{prefix}_mean"] = _mean(clean)
    result[f"{prefix}_sd"] = _sd(clean)
    result[f"{prefix}_cv"] = _cv(clean)
    if include_range:
        result[f"{prefix}_range"] = (max(clean) - min(clean)) if clean else float("nan")


def _patch_volume_concentration(patch_volumes: list[float]) -> float:
    if not patch_volumes:
        return float("nan")
    total = sum(patch_volumes)
    return _safe_percent(math.sqrt(sum(v * v for v in patch_volumes)), total)


def _effective_mesh_size(patch_volumes: list[float], total_volume: float) -> float:
    return _safe_divide(sum(volume * volume for volume in patch_volumes), total_volume)


def _splitting_index(patch_volumes: list[float], total_volume: float) -> float:
    denominator = sum(volume * volume for volume in patch_volumes)
    return _safe_divide(total_volume * total_volume, denominator)


def _contagion(labels: np.ndarray, class_values: list[int]) -> float:
    """Six-face extension of the documented 2D adjacency entropy formula."""
    return contagion(labels)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def _cv(values: list[float]) -> float:
    mean = _mean(values)
    sd = _sd(values)
    return _safe_percent(sd, mean)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return float(numerator / denominator)


def _safe_percent(numerator: float, denominator: float) -> float:
    value = _safe_divide(numerator, denominator)
    return value * 100.0 if not math.isnan(value) else value
