"""Adjacency counting and entropy metrics; see docs/metric-definitions.md."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def directed_adjacencies(labels: np.ndarray, regions: np.ndarray | None = None) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for axis in range(labels.ndim):
        first = [slice(None)] * labels.ndim
        second = first.copy()
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        a, b = labels[tuple(first)], labels[tuple(second)]
        valid = (a > 0) & (b > 0)
        if regions is not None:
            ra, rb = regions[tuple(first)], regions[tuple(second)]
            valid &= (ra > 0) & (ra == rb)
        pairs, frequencies = np.unique(np.column_stack((a[valid], b[valid])), axis=0, return_counts=True)
        for (left, right), count in zip(pairs, frequencies):
            counts[(int(left), int(right))] += int(count)
            counts[(int(right), int(left))] += int(count)
    return dict(counts)


def contagion(labels: np.ndarray, regions: np.ndarray | None = None) -> float:
    classes, cells = np.unique(labels[labels > 0], return_counts=True)
    m = len(classes)
    if m < 2:
        return float("nan")
    counts = directed_adjacencies(labels, regions)
    if not counts:
        return float("nan")
    row_sums: dict[int, int] = defaultdict(int)
    for (first, _), count in counts.items():
        row_sums[first] += count
    proportions = dict(zip(classes.tolist(), (cells / cells.sum()).tolist()))
    entropy_terms = []
    for (first, _), count in counts.items():
        q = proportions[first] * count / row_sums[first]
        entropy_terms.append(q * math.log(q))
    return float(100.0 * (1.0 + sum(entropy_terms) / (2.0 * math.log(m))))


def interspersion(pairwise: dict[tuple[int, int], float], present_classes: int) -> float:
    if present_classes < 3:
        return float("nan")
    total = sum(pairwise.values())
    if total <= 0:
        return float("nan")
    entropy = -sum((value / total) * math.log(value / total) for value in pairwise.values() if value > 0)
    return float(100.0 * entropy / math.log(present_classes * (present_classes - 1) / 2))


def max_like_edges_2d(cells: int) -> int:
    if cells <= 1:
        return 0
    n = math.isqrt(cells)
    remainder = cells - n * n
    maximum = 2 * n * (n - 1)
    if remainder:
        maximum += 2 * remainder - (1 if remainder <= n else 2)
    return maximum


def aggregation_index(class_cells: dict[int, int], single_like_counts: dict[int, float], maxima: dict[int, int] | None = None) -> float:
    total_cells = sum(class_cells.values())
    capacities = maxima if maxima is not None else {cls: max_like_edges_2d(count) for cls, count in class_cells.items()}
    if total_cells <= 0 or not any(capacities.values()):
        return float("nan")
    score = 0.0
    for cls, count in class_cells.items():
        maximum = capacities.get(cls, 0)
        if maximum:
            score += (count / total_cells) * single_like_counts.get(cls, 0.0) / maximum
    return float(100.0 * score)


def clumpiness(like: float, total: float, proportion: float) -> float:
    if total <= 0 or proportion <= 0:
        return float("nan")
    g = like / total
    if proportion >= 1:
        return 1.0 if g == 1 else float("nan")
    denominator = proportion if g < proportion and proportion < 0.5 else 1.0 - proportion
    return float((g - proportion) / denominator)
