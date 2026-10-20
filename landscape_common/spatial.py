"""Exact boundary-cell distances without a dense patch-by-patch matrix."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
from scipy.spatial import cKDTree


class PatchIndex:
    """Index one independent slide and, for ENN/PROX, one class of patches."""

    def __init__(self, patches):
        self.patches = list(patches)
        self.centers = np.asarray([patch.centroid for patch in patches], dtype=float)
        self.radii = np.asarray([
            np.linalg.norm(patch.boundary_points - self.centers[index], axis=1).max(initial=0)
            for index, patch in enumerate(patches)
        ])
        self.max_radius = float(self.radii.max(initial=0))
        self.tree = cKDTree(self.centers) if patches else None
        self.boundary_trees: dict[int, cKDTree] = {}

    def distance(self, first: int, second: int) -> float:
        a = self.patches[first].boundary_points
        b = self.patches[second].boundary_points
        if len(a) > len(b):
            first, second, a, b = second, first, b, a
        tree = self.boundary_trees.get(second)
        if tree is None:
            tree = self.boundary_trees[second] = cKDTree(b)
        distances, _ = tree.query(a, k=1)
        return float(np.min(distances))

    def pairs(self, radius: float):
        for first in range(len(self.patches)):
            candidates = range(first + 1, len(self.patches)) if math.isinf(radius) else self.tree.query_ball_point(
                self.centers[first], np.nextafter(radius + self.radii[first] + self.max_radius, math.inf)
            )
            for second in candidates:
                if second <= first:
                    continue
                lower_bound = np.linalg.norm(self.centers[first] - self.centers[second]) - self.radii[first] - self.radii[second]
                if lower_bound > np.nextafter(radius, math.inf):
                    continue
                distance = self.distance(first, second)
                if 0 < distance <= radius:
                    yield first, second, distance

    def nearest(self) -> list[float]:
        if len(self.patches) < 2:
            return [float("nan")] * len(self.patches)
        values = []
        for first in range(len(self.patches)):
            _, nearest = self.tree.query(self.centers[first], k=2)
            second = next(int(value) for value in np.atleast_1d(nearest) if int(value) != first)
            best = self.distance(first, second)
            candidates = self.tree.query_ball_point(
                self.centers[first], np.nextafter(best + self.radii[first] + self.max_radius, math.inf)
            )
            for second in candidates:
                if second == first:
                    continue
                lower_bound = np.linalg.norm(self.centers[first] - self.centers[second]) - self.radii[first] - self.radii[second]
                if lower_bound < best:
                    best = min(best, self.distance(first, second))
            values.append(best)
        return values


def spatial_statistics(class_patches, radius, similarity_weights=None):
    """Return aligned per-patch values and counts, pooled from within-slide pairs."""
    proximity, nearest, similarity = {}, {}, {}
    connected_by_class = defaultdict(int)
    pairs_by_class = defaultdict(int)
    by_region = defaultdict(list)
    patch_classes = {}
    for cls, patches in class_patches.items():
        groups = defaultdict(list)
        for patch in patches:
            groups[patch.region_id].append(patch)
            by_region[patch.region_id].append(patch)
            patch_classes[id(patch)] = cls
        for group in groups.values():
            index = PatchIndex(group)
            n = len(group)
            pairs_by_class[cls] += n * (n - 1) // 2
            values = np.zeros(n)
            for first, second, distance in index.pairs(radius):
                values[first] += group[second].area / distance**2
                values[second] += group[first].area / distance**2
                connected_by_class[cls] += 1
            for patch, prox, enn in zip(group, values, index.nearest()):
                proximity[id(patch)] = float(prox)
                nearest[id(patch)] = enn
                # The default is identity similarity: SIMI equals PROX.
                similarity[id(patch)] = float(prox)
    if similarity_weights is not None:
        similarity = {id(patch): 0.0 for group in by_region.values() for patch in group}
        for group in by_region.values():
            index = PatchIndex(group)
            for first, second, distance in index.pairs(radius):
                a, b = patch_classes[id(group[first])], patch_classes[id(group[second])]
                weight = similarity_weights.get((a, b), similarity_weights.get((b, a), 1.0 if a == b else 0.0))
                if weight:
                    similarity[id(group[first])] += weight * group[second].area / distance**2
                    similarity[id(group[second])] += weight * group[first].area / distance**2
    return proximity, nearest, similarity, dict(connected_by_class), dict(pairs_by_class)


def nearest_boundary_distances(patches):
    return PatchIndex(patches).nearest()
