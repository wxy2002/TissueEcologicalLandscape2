from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

from .adjacency import edge_and_contact_lengths
from .filters import filter_small_patches
from .io import parse_classes, parse_spacing, validate_matrix
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
) -> dict[str, float]:
    labels = validate_matrix(matrix)
    spacing_tuple = parse_spacing(spacing)
    labels = filter_small_patches(labels, min_patch_cells, connectivity)
    class_values = parse_classes(classes)
    if class_values is None:
        max_class = int(labels.max(initial=0))
        class_values = list(range(1, max_class + 1))

    non_background_cells = int(np.count_nonzero(labels > 0))
    cell_area = float(np.prod(spacing_tuple))
    total_non_background_area = non_background_cells * cell_area
    landscape_extent_area = labels.size * cell_area

    adjacency, pairwise = edge_and_contact_lengths(labels, class_values, spacing_tuple)

    result: dict[str, float] = {}
    class_patch_stats: dict[int, list[PatchStats]] = {}
    class_areas: dict[int, float] = {}

    for cls in class_values:
        mask = labels == cls
        cell_count = int(np.count_nonzero(mask))
        if cell_count == 0:
            for metric in CLASS_METRICS:
                result[f"class_{cls}_{metric}"] = missing_value
            class_patch_stats[cls] = []
            class_areas[cls] = 0.0
            continue

        patch_labels, patch_count = label_patches(mask, connectivity)
        patch_stats = summarize_patches(patch_labels, patch_count, spacing_tuple)
        class_patch_stats[cls] = patch_stats
        class_area = cell_count * cell_area
        class_areas[cls] = class_area
        edge_length = adjacency[cls]["total_edge_length"]
        other_length = adjacency[cls]["boundary_to_other_classes_length"]
        like_length = adjacency[cls]["like_internal_edge_length"]
        internal_length = adjacency[cls]["total_internal_edge_length"]

        patch_areas = [patch.area for patch in patch_stats]
        patch_perimeters = [patch.perimeter for patch in patch_stats]
        perimeter_area_ratios = [
            _safe_divide(patch.perimeter, patch.area) for patch in patch_stats
        ]
        proximity = _proximity_indices(patch_stats, patch_stats, search_radius)
        similarity = _similarity_indices(
            patch_stats, class_patch_stats, class_areas, search_radius
        )
        nearest = _nearest_neighbor_distances(patch_stats)
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
                    like_length, internal_length
                ),
                f"class_{cls}_clumpiness": _clumpiness(
                    like_length,
                    internal_length,
                    class_area,
                    total_non_background_area,
                ),
                f"class_{cls}_splitting_index": _splitting_index(
                    patch_areas, total_non_background_area
                ),
                f"class_{cls}_effective_mesh_size": _effective_mesh_size(
                    patch_areas, total_non_background_area
                ),
                f"class_{cls}_cohesion_2d": _cohesion_2d(patch_areas),
                f"class_{cls}_landscape_shape_index": _landscape_shape_index(
                    edge_length, class_area
                ),
                f"class_{cls}_normalized_landscape_shape_index": _normalized_landscape_shape_index(
                    edge_length, class_area, cell_count, spacing_tuple
                ),
                f"class_{cls}_perimeter_area_fractal_dimension": _perimeter_area_fractal_dimension(
                    patch_areas, patch_perimeters
                ),
                f"class_{cls}_connectance_index": _connectance_index(
                    patch_stats, patch_stats, search_radius
                ),
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
) -> dict[str, float]:
    all_patches = [patch for patches in class_patch_stats.values() for patch in patches]
    patch_areas = [patch.area for patch in all_patches]
    present_areas = [value for value in class_areas.values() if value > 0]
    proportions = [
        area / total_non_background_area
        for area in present_areas
        if total_non_background_area > 0
    ]
    total_edge_length = sum(values["total_edge_length"] for values in adjacency.values())
    total_unlike = sum(pairwise.values()) * 2.0
    total_internal = sum(values["total_internal_edge_length"] for values in adjacency.values())
    same_internal = sum(values["like_internal_edge_length"] for values in adjacency.values())
    total_cell_count = int(round(_safe_divide(total_non_background_area, float(np.prod(spacing)))))
    weighted_contrast_edge = _weighted_contrast_edge(pairwise, contrast_weights) * 2.0
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
    all_proximity = _proximity_indices(all_patches, all_patches, search_radius)
    all_similarity = _similarity_indices(
        all_patches, class_patch_stats, class_areas, search_radius
    )

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
        "landscape_contagion": _contagion(labels, class_values),
        "landscape_interspersion_juxtaposition": _safe_percent(
            total_unlike, total_internal
        ),
        "landscape_effective_mesh_size": _effective_mesh_size(
            patch_areas, total_non_background_area
        ),
        "landscape_splitting_index": _splitting_index(
            patch_areas, total_non_background_area
        ),
        "landscape_landscape_division": 1.0
        - _safe_divide(sum(area * area for area in patch_areas), total_non_background_area**2),
        "landscape_aggregation_index": _safe_percent(same_internal, total_internal),
        "landscape_connectance_index": _connectance_index(
            all_patches, all_patches, search_radius
        ),
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


def _nearest_neighbor_distances(patches: list[PatchStats]) -> list[float]:
    if len(patches) < 2:
        return []
    centroids = np.asarray([patch.centroid for patch in patches], dtype=float)
    distances = cdist(centroids, centroids)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1).tolist()


def _proximity_indices(
    target_patches: list[PatchStats],
    candidate_patches: list[PatchStats],
    search_radius: float,
) -> list[float]:
    if len(candidate_patches) < 2:
        return [float("nan") for _ in target_patches]
    values = []
    for patch in target_patches:
        total = 0.0
        found = False
        for other in candidate_patches:
            if other is patch:
                continue
            distance = _centroid_distance(patch, other)
            if distance <= 0 or distance > search_radius:
                continue
            total += other.area / (distance * distance)
            found = True
        values.append(total if found else 0.0)
    return values


def _similarity_indices(
    target_patches: list[PatchStats],
    class_patch_stats: dict[int, list[PatchStats]],
    class_areas: dict[int, float],
    search_radius: float,
) -> list[float]:
    all_candidates = [
        (cls, patch)
        for cls, patches in class_patch_stats.items()
        for patch in patches
    ]
    if len(all_candidates) < 2:
        return [float("nan") for _ in target_patches]
    patch_class = {
        id(patch): cls for cls, patches in class_patch_stats.items() for patch in patches
    }
    values = []
    for patch in target_patches:
        own_class = patch_class.get(id(patch))
        own_area = class_areas.get(own_class, 0.0)
        total = 0.0
        found = False
        for other_class, other in all_candidates:
            if other is patch:
                continue
            distance = _centroid_distance(patch, other)
            if distance <= 0 or distance > search_radius:
                continue
            similarity = _safe_divide(min(own_area, class_areas[other_class]), max(own_area, class_areas[other_class]))
            if math.isnan(similarity):
                continue
            total += similarity * other.area / (distance * distance)
            found = True
        values.append(total if found else 0.0)
    return values


def _connectance_index(
    target_patches: list[PatchStats],
    candidate_patches: list[PatchStats],
    search_radius: float,
) -> float:
    if len(target_patches) < 2:
        return float("nan")
    total_pairs = len(target_patches) * (len(target_patches) - 1) / 2.0
    connected = 0
    for first, second in combinations(target_patches, 2):
        if first in candidate_patches and second in candidate_patches:
            if _centroid_distance(first, second) <= search_radius:
                connected += 1
    return _safe_percent(float(connected), total_pairs)


def _centroid_distance(first: PatchStats, second: PatchStats) -> float:
    return float(np.linalg.norm(np.asarray(first.centroid) - np.asarray(second.centroid)))


def _clumpiness(
    like_length: float,
    internal_length: float,
    class_area: float,
    total_area: float,
) -> float:
    gii = _safe_divide(like_length, internal_length)
    pi = _safe_divide(class_area, total_area)
    if math.isnan(gii) or math.isnan(pi):
        return float("nan")
    if gii < pi:
        return (gii - pi) / pi if pi > 0 else float("nan")
    return (gii - pi) / (1.0 - pi) if pi < 1 else 1.0


def _cohesion_2d(patch_areas: list[float]) -> float:
    if not patch_areas:
        return float("nan")
    total = sum(patch_areas)
    return _safe_percent(math.sqrt(sum(area * area for area in patch_areas)), total)


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
    class_to_index = {cls: index for index, cls in enumerate(class_values)}
    matrix = np.zeros((len(class_values), len(class_values)), dtype=float)
    for axis in range(2):
        left_slices = [slice(None)] * 2
        right_slices = [slice(None)] * 2
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)
        left = labels[tuple(left_slices)]
        right = labels[tuple(right_slices)]
        valid = (left > 0) & (right > 0)
        for a, b in zip(left[valid].tolist(), right[valid].tolist()):
            ia = class_to_index.get(int(a))
            ib = class_to_index.get(int(b))
            if ia is not None and ib is not None:
                matrix[ia, ib] += 1
                matrix[ib, ia] += 1
    total = matrix.sum()
    if total <= 0 or len(class_values) <= 1:
        return float("nan")
    probabilities = matrix[matrix > 0] / total
    entropy = -sum(float(p) * math.log(float(p)) for p in probabilities)
    maximum = 2.0 * math.log(len(class_values))
    return (1.0 - entropy / maximum) * 100.0


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
