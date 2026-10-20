import math
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from landscape2d.metrics import compute_metrics as metrics2
from landscape3d.metrics import compute_metrics as metrics3
from landscape2d.io import parse_spacing as spacing2
from landscape3d.io import parse_spacing as spacing3
from landscape2d.merge_maps import merge_matrices_horizontally
from landscape2d.patches import label_patches, summarize_patches
from landscape3d.patches import label_patches as label3, summarize_patches as summarize3
from landscape_common.spatial import PatchIndex


class MetricRegressionTests(unittest.TestCase):
    def test_anisotropic_geometry(self):
        result = metrics2(np.array([[1, 1, 0], [0, 2, 0]]), spacing=(2, 3), classes=[1, 2, 3])
        for key, value in dict(class_1_area=12, class_2_area=6, class_1_total_edge_length=14,
                               class_1_to_2_boundary_length=2, landscape_total_non_background_area=18).items():
            self.assertEqual(result[key], value)
        self.assertTrue(math.isnan(result['class_3_area']))
        result = metrics3(np.ones((2, 3, 4), int), spacing=(2, 3, 5))
        self.assertEqual(result['class_1_volume'], 720)
        self.assertEqual(result['class_1_total_surface_area'], 592)

    def test_similarity_relabeling_and_coefficients(self):
        original = metrics2(np.array([[1, 0, 2]]), similarity_weights={(1, 2): 1})
        renamed = metrics2(np.array([[2, 0, 1]]), similarity_weights={(1, 2): 1})
        self.assertEqual(original['class_1_similarity_index_mean'], .25)
        self.assertEqual(original['class_1_similarity_index_mean'], renamed['class_2_similarity_index_mean'])
        default = metrics2(np.array([[1, 0, 2]]))
        self.assertEqual(default['class_1_similarity_index_mean'], 0)
        self.assertEqual(default['landscape_proximity_index_mean'], 0)

    def test_contagion_formula_and_absent_classes(self):
        for function, matrix in ((metrics2, np.array([[1, 1, 1, 2]])),
                                 (metrics3, np.array([[[1, 1, 1, 2]]]))):
            a = function(matrix, classes=[1, 2])['landscape_contagion']
            b = function(matrix, classes=[1, 2, 99])['landscape_contagion']
            self.assertAlmostEqual(a, 32.36379021876728)
            self.assertEqual(a, b)
            self.assertAlmostEqual(function(np.where(matrix == 1, 99, 4))['landscape_contagion'], a)

    def test_standard_aggregation_metrics(self):
        result = metrics2(np.array([[1, 2, 3]]))
        self.assertAlmostEqual(result['landscape_interspersion_juxtaposition'], 100 * math.log(2) / math.log(3))
        self.assertEqual(metrics2(np.array([[1, 1, 2, 2]]))['landscape_aggregation_index'], 100)
        self.assertEqual(metrics2(np.array([[1, 0, 2]]))['class_1_cohesion_2d'], 0)
        self.assertEqual(metrics2(np.ones((1, 1), int))['class_1_cohesion_2d'], 0)
        # A two-cell class has two directed like joins out of eight sides.
        result = metrics2(np.array([[1, 1, 2, 2]]))
        self.assertEqual(result['class_1_like_adjacency_percent'], 25)
        self.assertAlmostEqual(result['class_1_clumpiness'], -.5)
        # Four cells form one square: COHESION is 100 when it is the entire map.
        self.assertAlmostEqual(metrics2(np.ones((2, 2), int))['class_1_cohesion_2d'], 100)

    def test_unique_landscape_boundaries(self):
        result = metrics2(np.array([[1, 2]]))
        self.assertEqual(result['landscape_sum_class_edge_length'], 8)
        self.assertEqual(result['landscape_total_edge_length'], 7)
        self.assertEqual(result['landscape_total_unlike_boundary_length'], 1)
        result = metrics3(np.array([[[1, 2]]]))
        self.assertEqual(result['landscape_sum_class_surface_area'], 12)
        self.assertEqual(result['landscape_total_surface_area'], 11)
        self.assertNotIn('landscape_aggregation_index', result)
        self.assertNotIn('class_1_cohesion_3d', result)
        self.assertFalse(any('fractal_dimension_3d' in key for key in result))

    def test_boundary_distance_and_radius(self):
        matrix = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1]])
        result = metrics2(matrix, search_radius=4)
        self.assertEqual(result['class_1_nearest_neighbor_mean'], 4)
        self.assertEqual(result['class_1_proximity_index_mean'], 3 / 16)
        self.assertEqual(result['class_1_connectance_index'], 100)
        self.assertEqual(metrics2(matrix, search_radius=3.9)['class_1_connectance_index'], 0)
        self.assertEqual(metrics2(np.ones((2, 2), int))['class_1_proximity_index_mean'], 0)
        result = metrics3(matrix.reshape(1, 1, -1), spacing=(2, 3, 5))
        self.assertEqual(result['class_1_nearest_neighbor_mean'], 20)

    def test_spatial_index_matches_brute_force(self):
        rng = np.random.default_rng(8)
        for dimensions in (2, 3):
            mask = rng.random((12, 10) if dimensions == 2 else (5, 6, 7)) < .2
            if dimensions == 2:
                labels, count = label_patches(mask, 4)
                patches = summarize_patches(labels, count, (2, 3))
            else:
                labels, count = label3(mask, 6)
                patches = summarize3(labels, count, (2, 3, 4))
            index = PatchIndex(patches)
            expected = [min(float(cdist(a.boundary_points, b.boundary_points).min())
                            for b in patches if b is not a) for a in patches]
            np.testing.assert_allclose(index.nearest(), expected)
            actual = {(a, b) for a, b, distance in index.pairs(6)}
            reference = {(a, b) for a in range(len(patches)) for b in range(a + 1, len(patches))
                         if cdist(patches[a].boundary_points, patches[b].boundary_points).min() <= 6}
            self.assertEqual(actual, reference)

    def test_patient_results_ignore_canvas_layout(self):
        slides = [np.ones((2, 2), int), np.array([[1, 0, 1], [2, 0, 2]])]
        reference = None
        for gap in (0, 1, 3, 10):
            matrix = merge_matrices_horizontally(slides, gap)
            regions = merge_matrices_horizontally([np.full(slide.shape, i, int) for i, slide in enumerate(slides, 1)], gap)
            result = metrics2(matrix, region_ids=regions, search_radius=5, similarity_weights={(1, 2): .5})
            if reference is None:
                reference = result
            else:
                self.assertEqual(result.keys(), reference.keys())
                for key in result:
                    if math.isnan(reference[key]):
                        self.assertTrue(math.isnan(result[key]), key)
                    else:
                        self.assertAlmostEqual(result[key], reference[key], msg=key)
            self.assertEqual(result['landscape_extent_area'], 10)
            self.assertEqual(result['class_1_patch_count'], 3)
            self.assertEqual(result['class_1_area'], 6)
            self.assertEqual(result['class_1_nearest_neighbor_mean'], 2)
            self.assertEqual(result['class_1_connectance_index'], 100)

    def test_patient_composition_pools_area(self):
        matrix = np.array([[1, 1, 1, 1, 2]])
        result = metrics2(matrix, region_ids=np.array([[1, 1, 1, 1, 2]]))
        self.assertEqual(result['class_1_pland'], 80)
        self.assertEqual(result['class_2_pland'], 20)
        self.assertAlmostEqual(result['landscape_shannon_diversity'], -.8 * math.log(.8) - .2 * math.log(.2))
        self.assertEqual(result['class_1_boundary_to_other_classes_length'], 0)
        self.assertEqual(result['landscape_connectance_index'], 0)

    def test_parameter_validation(self):
        for function, valid in ((spacing2, (1, 1)), (spacing3, (1, 1, 1))):
            for value in (float('nan'), float('inf'), 0, -1):
                with self.assertRaises(ValueError):
                    function((value, *valid[1:]))
        for kwargs in ({'min_patch_cells': -1}, {'min_patch_cells': 1.5}, {'search_radius': -1},
                       {'search_radius': float('nan')}, {'max_classes': 0}, {'max_classes': 1.5},
                       {'max_classes': 1}, {'connectivity': 5}, {'classes': [1]},
                       {'similarity_weights': {(1, 2): -1}}, {'similarity_weights': {(1, 2): 1, (2, 1): .5}}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                metrics2(np.array([[1, 2]]), **kwargs)
        for kwargs in ({'min_patch_volume': float('inf')}, {'min_patch_voxels': 1.5}, {'connectivity': 8}):
            with self.assertRaises(ValueError):
                metrics3(np.ones((1, 1, 1), int), **kwargs)

    def test_all_background_and_sparse_labels(self):
        result = metrics2(np.zeros((2, 2), int), classes=[1, 2])
        self.assertEqual(result['landscape_patch_count'], 0)
        self.assertTrue(math.isnan(result['landscape_contagion']))
        result = metrics2(np.array([[1, 99999999]]))
        self.assertIn('class_99999999_area', result)
        self.assertNotIn('class_2_area', result)


if __name__ == '__main__':
    unittest.main()
