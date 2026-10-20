import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from landscape2d.batch import compute_batch
from landscape2d.feature_clustering import (
    _load_training_sample_blocks, _silhouette_score, _update_metric_sample,
    cluster_feature_file, clustering_metrics, infer_feature_file, load_feature_block,
)
from landscape2d.grid import labels_to_matrix


class ClusteringRegressionTests(unittest.TestCase):
    def test_grid_preserves_gaps_and_origin(self):
        coords = np.array([[0, 0], [256, 0], [768, 0]])
        result = labels_to_matrix(np.ones(3, int), coords, stride=(256, 256))
        np.testing.assert_array_equal(result, [[1, 1, 0, 1]])
        result = labels_to_matrix(np.array([1, 2]), np.array([[512, 256], [512, 768]]),
                                  stride=(256, 256), origin=(256, 0))
        np.testing.assert_array_equal(result, [[0, 0], [0, 1], [0, 0], [0, 2]])

    def test_grid_rejects_unknown_or_ambiguous_coordinates(self):
        for coords, kwargs in ((None, {}), (np.array([[0, 0], [256, 0]]), {}),
                               (np.zeros((2, 2)), {'stride': (256, 256)}),
                               (np.array([[0, 0], [300, 0]]), {'stride': (256, 256)}),
                               (np.zeros((1, 2)), {'stride': (256, 256)}),
                               (np.array([[0, 0], [256, 0]]), {'stride': (float('inf'), 1)})):
            with self.assertRaises(ValueError):
                labels_to_matrix(np.array([1, 2]), coords, **kwargs)

    def test_singleton_and_zero_distance_quality(self):
        x = np.array([[0], [1], [10]], np.float32)
        labels = np.array([0, 0, 1])
        expected = ((1 - 1 / 10) + (1 - 1 / 9) + 0) / 3
        self.assertAlmostEqual(_silhouette_score(x, labels, 2000, 0), expected)
        scores = clustering_metrics(x, labels, centers=np.array([[100], [200]]), inertia=999)
        self.assertAlmostEqual(scores['calinski_harabasz_score'], 120 + 1 / 3)
        self.assertAlmostEqual(scores['davies_bouldin_score'], 1 / 19)
        self.assertEqual(scores['inertia'], 999)
        scores = clustering_metrics(np.zeros((4, 1)), np.array([0, 0, 1, 1]), np.zeros((2, 1)), 0)
        self.assertEqual(scores['silhouette_score'], 0)
        self.assertEqual(scores['davies_bouldin_score'], 0)
        self.assertEqual(scores['calinski_harabasz_score'], 1)

    def test_reservoir_visits_tail_and_tracks_rows(self):
        reference = list(range(10))
        rng = np.random.default_rng(0)
        for index in range(10, 100):
            replacement = int(rng.integers(0, index + 1))
            if replacement < 10:
                reference[replacement] = index
        for sizes in ((100,), (3, 7, 90), (4, 30, 66)):
            parts, labels, indices = [], [], []
            rng = np.random.default_rng(0)
            start = 0
            for size in sizes:
                values = np.arange(start, start + size)
                _update_metric_sample(parts, labels, values[:, None], values, 10, start, rng, indices)
                start += size
            np.testing.assert_array_equal(np.concatenate(parts).ravel(), reference)
            np.testing.assert_array_equal(np.concatenate(labels), reference)
            np.testing.assert_array_equal(np.concatenate(indices), reference)

    def test_single_row_per_file_is_a_valid_global_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(3):
                np.save(root / f'{i}.npy', np.full((4, 3), i, np.float32))
            blocks = _load_training_sample_blocks(sorted(root.glob('*.npy')), None, 1, 0, False, io.StringIO())
            self.assertEqual(sum(block.features.shape[0] for block in blocks), 3)
            self.assertTrue(all(block.row_indices.shape == (1,) for block in blocks))
            np.save(root / 'single.npy', np.ones((1, 3), np.float32))
            self.assertEqual(load_feature_block(root / 'single.npy').features.shape, (1, 3))

    def test_training_inference_batch_and_sample_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'features'
            source.mkdir()
            rng = np.random.default_rng(17)
            for i in range(2):
                with h5py.File(source / f'case{i}.h5', 'w') as handle:
                    handle['feats'] = np.r_[rng.normal(-2, .05, (8, 8)), rng.normal(2, .05, (8, 8))].astype(np.float32)
                    handle['coords'] = [(c * 256, r * 256) for r in range(4) for c in range(4)]
                    handle.attrs['grid_stride'] = (256, 256)
                    handle.attrs['mpp'] = (.5, .5)
            training = cluster_feature_file(source, root / 'train', n_clusters=2, latent_dim=2,
                                            hidden_dims=(4,), epochs=2, batch_size=8, device='cpu',
                                            max_train_tiles=16, max_train_tiles_per_file=8, progress=False)
            self.assertEqual(training.labels.shape, (32, 1))
            self.assertEqual(training.sampled_latent.shape, (16, 2))
            self.assertEqual(training.sampled_labels.shape, (16,))
            self.assertEqual(len(training.sample_row_indices), 16)
            inference = infer_feature_file(source, training.model_path, root / 'predict', device='cpu',
                                           batch_size=8, max_silhouette_samples=20, progress=False)
            np.testing.assert_array_equal(inference.labels, training.labels)
            metrics = inference.metrics.iloc[0]
            for key in ('silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'):
                self.assertTrue(math.isfinite(metrics[key]), key)
            self.assertEqual(metrics['metric_sample_count'], 20)
            self.assertEqual(inference.sampled_latent.shape, (20, 2))
            for label, path, row in zip(inference.sampled_labels, inference.sample_source_files, inference.sample_row_indices):
                case_index = int(Path(path).stem[-1])
                self.assertEqual(label, inference.labels[case_index * 16 + row, 0])
            with np.load(root / 'predict' / 'case0_clusters.npz') as label_map:
                np.testing.assert_array_equal(label_map['spacing'], [128, 128])
                self.assertEqual(label_map['spacing_unit'].item(), 'um')
                self.assertTrue(label_map['model_id'].item())
            manifest = json.loads((root / 'predict' / 'label_maps.json').read_text())
            self.assertEqual(len(manifest['label_maps']), 2)
            # Metrics CSV can live in its own input directory and be rerun.
            first = compute_batch(root / 'predict', root / 'predict' / 'metrics.csv', search_radius=1000)
            second = compute_batch(root / 'predict', root / 'predict' / 'metrics.csv', search_radius=1000)
            self.assertEqual(len(first), 2)
            np.testing.assert_allclose(first.select_dtypes('number'), second.select_dtypes('number'), equal_nan=True)
            np.save(root / 'single.npy', np.zeros((1, 8), np.float32))
            single = infer_feature_file(root / 'single.npy', training.model_path, root / 'single', device='cpu', progress=False)
            self.assertEqual(single.labels.shape, (1, 1))
            with np.load(root / 'single' / 'single_clusters.npz') as output:
                self.assertNotIn('matrix', output)
                self.assertIn('tile_labels', output)
            with self.assertRaises(ValueError):
                compute_batch(root / 'single', root / 'single' / 'metrics.csv')


if __name__ == '__main__':
    unittest.main()
