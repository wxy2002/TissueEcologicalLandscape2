from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from numbers import Integral
from collections import defaultdict

from landscape_common.aggregation import aggregation_index, clumpiness, contagion, interspersion, max_like_edges_2d
from landscape_common.spatial import spatial_statistics

from .adjacency import edge_and_contact_lengths
from .filters import filter_small_patches
from .io import iter_regions, parse_classes, parse_spacing, validate_matrix, validate_regions
from .patches import PatchStats, label_patches, summarize_patches


CLASS_METRICS = [
    "cell_count",
    "area",
    "pland",
    "patch_count",
    "patch_density",
    "largest_patch_area",
    "largest_patch_index",
    "patch_area_mean",
    "patch_area_am",
    "patch_area_md",
    "patch_area_sd",
    "patch_area_cv",
    "patch_area_range",
    "total_edge_length",
    "edge_density",
    "patch_perimeter_mean",
    "patch_perimeter_am",
    "patch_perimeter_md",
    "patch_perimeter_sd",
    "patch_perimeter_cv",
    "patch_perimeter_range",
    "boundary_to_background_length",
    "boundary_to_other_classes_length",
    "perimeter_area_ratio_mean",
    "perimeter_area_ratio_am",
    "perimeter_area_ratio_md",
    "perimeter_area_ratio_sd",
    "perimeter_area_ratio_cv",
    "landscape_shape_index",
    "normalized_landscape_shape_index",
    "shape_index_2d_mean",
    "shape_index_2d_am",
    "shape_index_2d_md",
    "shape_index_2d_sd",
    "shape_index_2d_cv",
    "compactness_mean",
    "compactness_am",
    "compactness_md",
    "compactness_sd",
    "compactness_cv",
    "fractal_dimension_2d_mean",
    "fractal_dimension_2d_am",
    "fractal_dimension_2d_md",
    "fractal_dimension_2d_sd",
    "fractal_dimension_2d_cv",
    "related_circumscribing_circle_mean",
    "related_circumscribing_circle_am",
    "related_circumscribing_circle_md",
    "related_circumscribing_circle_sd",
    "related_circumscribing_circle_cv",
    "contiguity_index_mean",
    "contiguity_index_am",
    "contiguity_index_md",
    "contiguity_index_sd",
    "contiguity_index_cv",
    "perimeter_area_fractal_dimension",
    "radius_gyration_mean",
    "radius_gyration_am",
    "radius_gyration_md",
    "radius_gyration_sd",
    "radius_gyration_cv",
    "like_adjacency_percent",
    "clumpiness",
    "splitting_index",
    "effective_mesh_size",
    "cohesion_2d",
    "connectance_index",
    "proximity_index_mean",
    "proximity_index_am",
    "proximity_index_md",
    "proximity_index_sd",
    "proximity_index_cv",
    "similarity_index_mean",
    "similarity_index_am",
    "similarity_index_md",
    "similarity_index_sd",
    "similarity_index_cv",
    "nearest_neighbor_mean",
    "nearest_neighbor_am",
    "nearest_neighbor_md",
    "nearest_neighbor_sd",
    "nearest_neighbor_cv",
    "unlike_boundary_length",
    "unlike_boundary_percent",
    "edge_contrast_index",
    "contrast_weighted_edge_density",
    "total_edge_contrast_index",
]


def compute_metrics(
    matrix: np.ndarray,
    classes: list[int] | str | None = None,
    spacing: tuple[float, float] | str = (1, 1),
    connectivity: int = 4,
    search_radius: float = math.inf,
    contrast_weights: dict[tuple[int, int], float] | None = None,
    max_classes: int | None = None,
    min_patch_cells: int = 0,
    missing_value: float = np.nan,
    region_ids: np.ndarray | None = None,
    similarity_weights: dict[tuple[int, int], float] | None = None,
) -> dict[str, float]:
    labels = validate_matrix(matrix)
    spacing_tuple = parse_spacing(spacing)
    if math.isnan(search_radius) or search_radius < 0:
        raise ValueError("search_radius must be non-negative (positive infinity is allowed)")
    if max_classes is not None and (not isinstance(max_classes, Integral) or max_classes <= 0):
        raise ValueError("max_classes must be a positive integer")
    _validate_weights(similarity_weights, "similarity_weights")
    _validate_weights(contrast_weights, "contrast_weights")
    if region_ids is not None:
        region_ids = validate_regions(labels, region_ids)
    labels = filter_small_patches(labels, min_patch_cells, connectivity, region_ids)
    class_values = parse_classes(classes)
    if class_values is None:
        class_values = sorted(int(value) for value in np.unique(labels) if value > 0)
    present = set(int(value) for value in np.unique(labels) if value > 0)
    if not present.issubset(class_values):
        raise ValueError("classes must include every observed non-background label")
    if max_classes is not None and max_classes < len(present):
        raise ValueError("max_classes cannot be smaller than the observed class count")

    non_background_cells = int(np.count_nonzero(labels > 0))
    cell_area = float(np.prod(spacing_tuple))
    total_non_background_area = non_background_cells * cell_area
    landscape_extent_area = (labels.size if region_ids is None else np.count_nonzero(region_ids)) * cell_area

    # Complete every class and every independent slide before any cross-class
    # calculation. Region borders are treated as separate landscape boundaries.
    class_patch_stats: dict[int, list[PatchStats]] = {cls: [] for cls in class_values}
    adjacency = None
    pairwise: dict[tuple[int, int], float] = defaultdict(float)
    class_max_like = defaultdict(int)
    for region_id, local, bounds in iter_regions(labels, region_ids):
        local_adjacency, local_pairs = edge_and_contact_lengths(local, class_values, spacing_tuple)
        if adjacency is None:
            adjacency = local_adjacency
        else:
            for cls, values in local_adjacency.items():
                for key, value in values.items():
                    adjacency[cls][key] += value
        for pair, length in local_pairs.items():
            pairwise[pair] += length
        for cls in class_values:
            patch_labels, count = label_patches(local == cls, connectivity)
            patches = summarize_patches(patch_labels, count, spacing_tuple, region_id,
                                        (bounds[0].start, bounds[1].start))
            class_patch_stats[cls].extend(patches)
            class_max_like[cls] += max_like_edges_2d(sum(patch.cell_count for patch in patches))
    if adjacency is None:
        adjacency, _ = edge_and_contact_lengths(labels, class_values, spacing_tuple)
    class_areas = {cls: sum(patch.area for patch in patches) for cls, patches in class_patch_stats.items()}
    proximity_by_patch, nearest_by_patch, similarity_by_patch, connected_counts, possible_pairs = spatial_statistics(
        class_patch_stats, search_radius, similarity_weights
    )
    spatial = (proximity_by_patch, nearest_by_patch, similarity_by_patch, connected_counts, possible_pairs)
    result: dict[str, float] = {}
    for cls in class_values:
        patch_stats = class_patch_stats[cls]
        cell_count = sum(patch.cell_count for patch in patch_stats)
        if cell_count == 0:
            for metric in CLASS_METRICS:
                result[f"class_{cls}_{metric}"] = missing_value
            continue
        patch_count = len(patch_stats)
        class_area = class_areas[cls]
        edge_length = adjacency[cls]["total_edge_length"]
        other_length = adjacency[cls]["boundary_to_other_classes_length"]
        like_count = adjacency[cls]["like_adjacency_count"]
        adjacency_count = adjacency[cls]["total_adjacency_count"]

        patch_areas = [patch.area for patch in patch_stats]
        patch_perimeters = [patch.perimeter for patch in patch_stats]
        perimeter_area_ratios = [
            _safe_divide(patch.perimeter, patch.area) for patch in patch_stats
        ]
        proximity = [proximity_by_patch[id(patch)] for patch in patch_stats]
        similarity = [similarity_by_patch[id(patch)] for patch in patch_stats]
        nearest = [nearest_by_patch[id(patch)] for patch in patch_stats]
        contrast_edge = _class_contrast_edge(cls, pairwise, contrast_weights)

        result.update(
            {
                f"class_{cls}_cell_count": float(cell_count),
                f"class_{cls}_area": class_area,
                f"class_{cls}_pland": _safe_percent(
                    class_area, total_non_background_area
                ),
                f"class_{cls}_patch_count": float(patch_count),
                f"class_{cls}_patch_density": _safe_divide(
                    patch_count, total_non_background_area
                ),
                f"class_{cls}_largest_patch_area": max(patch_areas),
                f"class_{cls}_largest_patch_index": _safe_percent(
                    max(patch_areas), total_non_background_area
                ),
                f"class_{cls}_total_edge_length": edge_length,
                f"class_{cls}_edge_density": _safe_divide(
                    edge_length, total_non_background_area
                ),
                f"class_{cls}_boundary_to_background_length": adjacency[cls][
                    "boundary_to_background_length"
                ],
                f"class_{cls}_boundary_to_other_classes_length": other_length,
                f"class_{cls}_like_adjacency_percent": _safe_percent(
                    like_count, adjacency_count
                ) if cell_count > 1 else float("nan"),
                f"class_{cls}_clumpiness": clumpiness(
                    like_count, adjacency_count, _safe_divide(class_area, landscape_extent_area)
                ),
                f"class_{cls}_splitting_index": _splitting_index(
                    patch_areas, total_non_background_area
                ),
                f"class_{cls}_effective_mesh_size": _effective_mesh_size(
                    patch_areas, total_non_background_area
                ),
                f"class_{cls}_cohesion_2d": _cohesion_2d(patch_stats, non_background_cells),
                f"class_{cls}_landscape_shape_index": _landscape_shape_index(
                    edge_length, class_area
                ),
                f"class_{cls}_normalized_landscape_shape_index": _normalized_landscape_shape_index(
                    edge_length, class_area, cell_count, spacing_tuple
                ),
                f"class_{cls}_perimeter_area_fractal_dimension": _perimeter_area_fractal_dimension(
                    patch_areas, patch_perimeters
                ),
                f"class_{cls}_connectance_index": _connectance(connected_counts.get(cls, 0), possible_pairs.get(cls, 0)),
                f"class_{cls}_unlike_boundary_length": adjacency[cls][
                    "unlike_boundary_length"
                ],
                f"class_{cls}_unlike_boundary_percent": _safe_percent(
                    other_length, edge_length
                ),
                f"class_{cls}_edge_contrast_index": _safe_percent(
                    contrast_edge, other_length
                ),
                f"class_{cls}_contrast_weighted_edge_density": _safe_divide(
                    contrast_edge, total_non_background_area
                ),
                f"class_{cls}_total_edge_contrast_index": _safe_percent(
                    contrast_edge, edge_length
                ),
            }
        )
        _add_summary(result, f"class_{cls}_patch_area", patch_areas, patch_areas)
        _add_summary(result, f"class_{cls}_patch_perimeter", patch_perimeters, patch_areas)
        _add_summary(
            result,
            f"class_{cls}_perimeter_area_ratio",
            perimeter_area_ratios,
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_shape_index_2d",
            [p.shape_index_2d for p in patch_stats],
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_compactness",
            [p.compactness for p in patch_stats],
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_fractal_dimension_2d",
            [p.fractal_dimension_2d for p in patch_stats],
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_related_circumscribing_circle",
            [p.related_circumscribing_circle for p in patch_stats],
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_contiguity_index",
            [p.contiguity_index for p in patch_stats],
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_radius_gyration",
            [p.radius_gyration for p in patch_stats],
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_proximity_index",
            proximity,
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_similarity_index",
            similarity,
            patch_areas,
            include_range=False,
        )
        _add_summary(
            result,
            f"class_{cls}_nearest_neighbor",
            nearest,
            patch_areas if len(nearest) == len(patch_areas) else None,
            include_range=False,
        )

    _add_pairwise_columns(result, pairwise, class_values, prefix="landscape")
    for a, b in combinations(class_values, 2):
        result[f"class_{a}_to_{b}_boundary_length"] = pairwise.get((a, b), 0.0)

    result.update(
        _landscape_metrics(
            labels=labels,
            class_values=class_values,
            class_patch_stats=class_patch_stats,
            class_areas=class_areas,
            total_non_background_area=total_non_background_area,
            landscape_extent_area=landscape_extent_area,
            pairwise=pairwise,
            adjacency=adjacency,
            spacing=spacing_tuple,
            search_radius=search_radius,
            contrast_weights=contrast_weights,
            max_classes=max_classes,
            spatial=spatial,
            region_ids=region_ids,
            class_max_like=class_max_like,
        )
    )
    return dict(sorted(result.items()))


def _landscape_metrics(
    labels: np.ndarray,
    class_values: list[int],
    class_patch_stats: dict[int, list[PatchStats]],
    class_areas: dict[int, float],
    total_non_background_area: float,
    landscape_extent_area: float,
    pairwise: dict[tuple[int, int], float],
    adjacency: dict[int, dict[str, float]],
    spacing: tuple[float, float],
    search_radius: float,
    contrast_weights: dict[tuple[int, int], float] | None,
    max_classes: int | None,
    spatial: tuple,
    region_ids: np.ndarray | None,
    class_max_like: dict[int, int],
) -> dict[str, float]:
    all_patches = [patch for patches in class_patch_stats.values() for patch in patches]
    patch_areas = [patch.area for patch in all_patches]
    present_areas = [value for value in class_areas.values() if value > 0]
    proportions = [
        area / total_non_background_area
        for area in present_areas
        if total_non_background_area > 0
    ]
    sum_class_edges = sum(values["total_edge_length"] for values in adjacency.values())
    total_unlike = sum(pairwise.values())
    total_edge_length = sum_class_edges - total_unlike
    total_cell_count = int(round(_safe_divide(total_non_background_area, float(np.prod(spacing)))))
    weighted_contrast_edge = _weighted_contrast_edge(pairwise, contrast_weights)
    diversity_count = len(proportions)
    shannon = -sum(p * math.log(p) for p in proportions if p > 0)
    simpson = 1.0 - sum(p * p for p in proportions)
    modified_simpson = -math.log(sum(p * p for p in proportions)) if proportions else float("nan")
    patch_perimeters = [patch.perimeter for patch in all_patches]
    patch_shapes = [patch.shape_index_2d for patch in all_patches]
    patch_compactness = [patch.compactness for patch in all_patches]
    patch_fractal = [patch.fractal_dimension_2d for patch in all_patches]
    patch_circle = [patch.related_circumscribing_circle for patch in all_patches]
    patch_contiguity = [patch.contiguity_index for patch in all_patches]
    patch_gyration = [patch.radius_gyration for patch in all_patches]
    proximity, nearest, similarity, connected, possible = spatial
    all_proximity = [proximity[id(patch)] for patch in all_patches]
    all_similarity = [similarity[id(patch)] for patch in all_patches]
    class_cells = {cls: sum(patch.cell_count for patch in patches) for cls, patches in class_patch_stats.items()}
    single_likes = {cls: values["like_adjacency_count"] / 2 for cls, values in adjacency.items()}

    result = {
        "landscape_total_non_background_area": total_non_background_area,
        "landscape_extent_area": landscape_extent_area,
        "landscape_class_richness": float(diversity_count),
        "landscape_patch_richness_density": _safe_divide(
            diversity_count, landscape_extent_area
        ),
        "landscape_relative_patch_richness": _safe_percent(
            diversity_count, float(max_classes)
        )
        if max_classes
        else float("nan"),
        "landscape_shannon_diversity": shannon if proportions else float("nan"),
        "landscape_simpson_diversity": simpson if proportions else float("nan"),
        "landscape_modified_simpson_diversity": modified_simpson,
        "landscape_shannon_evenness": _safe_divide(shannon, math.log(diversity_count))
        if diversity_count > 1
        else float("nan"),
        "landscape_simpson_evenness": _safe_divide(simpson, 1.0 - 1.0 / diversity_count)
        if diversity_count > 1
        else float("nan"),
        "landscape_modified_simpson_evenness": _safe_divide(
            modified_simpson, math.log(diversity_count)
        )
        if diversity_count > 1
        else float("nan"),
        "landscape_total_edge_length": total_edge_length,
        "landscape_sum_class_edge_length": sum_class_edges,
        "landscape_edge_density": _safe_divide(
            total_edge_length, total_non_background_area
        ),
        "landscape_largest_patch_index": _safe_percent(
            max(patch_areas) if patch_areas else 0.0,
            total_non_background_area,
        ),
        "landscape_patch_count": float(len(all_patches)),
        "landscape_patch_density": _safe_divide(
            len(all_patches), total_non_background_area
        ),
        "landscape_landscape_shape_index": _landscape_shape_index(
            total_edge_length, total_non_background_area
        ),
        "landscape_normalized_landscape_shape_index": _normalized_landscape_shape_index(
            total_edge_length, total_non_background_area, total_cell_count, spacing
        ),
        "landscape_perimeter_area_fractal_dimension": _perimeter_area_fractal_dimension(
            patch_areas, patch_perimeters
        ),
        "landscape_shape_index_2d_mean": _mean(patch_shapes),
        "landscape_compactness_mean": _mean(patch_compactness),
        "landscape_contagion": contagion(labels, region_ids),
        "landscape_interspersion_juxtaposition": interspersion(pairwise, diversity_count),
        "landscape_effective_mesh_size": _effective_mesh_size(
            patch_areas, total_non_background_area
        ),
        "landscape_splitting_index": _splitting_index(
            patch_areas, total_non_background_area
        ),
        "landscape_landscape_division": 1.0
        - _safe_divide(sum(area * area for area in patch_areas), total_non_background_area**2),
        "landscape_aggregation_index": aggregation_index(class_cells, single_likes, class_max_like),
        "landscape_connectance_index": _connectance(sum(connected.values()), sum(possible.values())),
        "landscape_total_unlike_boundary_length": total_unlike,
        "landscape_unlike_boundary_percent": _safe_percent(
            total_unlike, total_edge_length
        ),
        "landscape_contrast_weighted_edge_density": _safe_divide(
            weighted_contrast_edge, total_non_background_area
        ),
        "landscape_total_edge_contrast_index": _safe_percent(
            weighted_contrast_edge, total_edge_length
        ),
    }
    _add_summary(result, "landscape_patch_area", patch_areas, patch_areas)
    _add_summary(result, "landscape_patch_perimeter", patch_perimeters, patch_areas)
    _add_summary(result, "landscape_shape_index_2d", patch_shapes, patch_areas, include_range=False)
    _add_summary(result, "landscape_compactness", patch_compactness, patch_areas, include_range=False)
    _add_summary(result, "landscape_fractal_dimension_2d", patch_fractal, patch_areas, include_range=False)
    _add_summary(result, "landscape_related_circumscribing_circle", patch_circle, patch_areas, include_range=False)
    _add_summary(result, "landscape_contiguity_index", patch_contiguity, patch_areas, include_range=False)
    _add_summary(result, "landscape_radius_gyration", patch_gyration, patch_areas, include_range=False)
    _add_summary(result, "landscape_proximity_index", all_proximity, patch_areas, include_range=False)
    _add_summary(result, "landscape_similarity_index", all_similarity, patch_areas, include_range=False)
    return result


def _add_pairwise_columns(
    result: dict[str, float],
    pairwise: dict[tuple[int, int], float],
    class_values: list[int],
    prefix: str,
) -> None:
    for a, b in combinations(class_values, 2):
        result[f"{prefix}_class_{a}_to_{b}_boundary_length"] = pairwise.get((a, b), 0.0)


def _add_summary(
    result: dict[str, float],
    prefix: str,
    values: list[float],
    weights: list[float] | None = None,
    include_range: bool = True,
) -> None:
    clean_pairs = [
        (value, weights[index] if weights is not None and index < len(weights) else None)
        for index, value in enumerate(values)
        if not math.isnan(value)
    ]
    clean = [value for value, _ in clean_pairs]
    clean_weights = [weight for _, weight in clean_pairs if weight is not None]
    result[f"{prefix}_mean"] = _mean(clean)
    result[f"{prefix}_am"] = _weighted_mean(clean, clean_weights)
    result[f"{prefix}_md"] = _median(clean)
    result[f"{prefix}_sd"] = _sd(clean)
    result[f"{prefix}_cv"] = _cv(clean)
    if include_range:
        result[f"{prefix}_range"] = (max(clean) - min(clean)) if clean else float("nan")


def _connectance(connected: int, possible: int) -> float:
    return 100.0 * connected / possible if possible else 0.0


def _cohesion_2d(patches: list[PatchStats], landscape_cells: int) -> float:
    if not patches:
        return float("nan")
    if landscape_cells <= 1:
        return 0.0
    perimeter_sum = sum(patch.perimeter_cells for patch in patches)
    weighted_sum = sum(patch.perimeter_cells * math.sqrt(patch.cell_count) for patch in patches)
    return 100.0 * (1.0 - perimeter_sum / weighted_sum) / (1.0 - 1.0 / math.sqrt(landscape_cells))


def _validate_weights(weights, name: str) -> None:
    if weights is None:
        return
    for pair, weight in weights.items():
        if len(pair) != 2 or any(not isinstance(cls, Integral) or cls <= 0 for cls in pair):
            raise ValueError(f"{name} keys must be pairs of positive class labels")
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"{name} values must be finite in [0, 1]")
        if pair[::-1] in weights and weights[pair[::-1]] != weight:
            raise ValueError(f"{name} must be symmetric")


def _landscape_shape_index(edge_length: float, area: float) -> float:
    if edge_length <= 0 or area <= 0:
        return float("nan")
    return _safe_divide(edge_length, 4.0 * math.sqrt(area))


def _normalized_landscape_shape_index(
    edge_length: float,
    area: float,
    cell_count: int,
    spacing: tuple[float, float],
) -> float:
    if edge_length <= 0 or area <= 0 or cell_count <= 0:
        return float("nan")
    minimum = 4.0 * math.sqrt(area)
    maximum = 2.0 * (spacing[0] + spacing[1]) * cell_count
    return _safe_divide(edge_length - minimum, maximum - minimum)


def _perimeter_area_fractal_dimension(
    patch_areas: list[float],
    patch_perimeters: list[float],
) -> float:
    pairs = [
        (area, perimeter)
        for area, perimeter in zip(patch_areas, patch_perimeters)
        if area > 1 and perimeter > 0
    ]
    if len(pairs) < 2:
        return float("nan")
    log_area = np.log([area for area, _ in pairs])
    log_perimeter = np.log([perimeter for _, perimeter in pairs])
    if np.allclose(log_area, log_area[0]):
        return float("nan")
    slope = float(np.polyfit(log_area, log_perimeter, 1)[0])
    return 2.0 * slope


def _class_contrast_edge(
    cls: int,
    pairwise: dict[tuple[int, int], float],
    contrast_weights: dict[tuple[int, int], float] | None,
) -> float:
    total = 0.0
    for pair, length in pairwise.items():
        if cls in pair:
            total += length * _contrast_weight(pair[0], pair[1], contrast_weights)
    return total


def _weighted_contrast_edge(
    pairwise: dict[tuple[int, int], float],
    contrast_weights: dict[tuple[int, int], float] | None,
) -> float:
    return sum(
        length * _contrast_weight(a, b, contrast_weights)
        for (a, b), length in pairwise.items()
    )


def _contrast_weight(
    first: int,
    second: int,
    contrast_weights: dict[tuple[int, int], float] | None,
) -> float:
    if first == second:
        return 0.0
    if contrast_weights is None:
        return 1.0
    return float(
        contrast_weights.get((first, second), contrast_weights.get((second, first), 1.0))
    )


def _effective_mesh_size(patch_areas: list[float], total_area: float) -> float:
    return _safe_divide(sum(area * area for area in patch_areas), total_area)


def _splitting_index(patch_areas: list[float], total_area: float) -> float:
    denominator = sum(area * area for area in patch_areas)
    return _safe_divide(total_area * total_area, denominator)


def _contagion(labels: np.ndarray, class_values: list[int]) -> float:
    # Requested output columns do not change the classes present in the map.
    return contagion(labels)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or len(values) != len(weights):
        return float("nan")
    total_weight = sum(weights)
    if total_weight <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


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
