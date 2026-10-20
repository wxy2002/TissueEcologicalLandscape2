import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import nibabel as nib
import numpy as np

from landscape2d.batch import compute_batch
from landscape3d.batch import compute_batch as batch3
from landscape2d.io import load_label_map, load_matrix
from landscape3d.io import load_volume
from landscape2d.merge_maps import merge_patient_maps


def save_map(path, matrix, model='fixed-model', spacing=(2, 3)):
    np.savez(path, matrix=np.asarray(matrix, dtype=int), model_id=np.asarray(model),
             spacing=np.asarray(spacing), spacing_unit=np.asarray('um'))


class IoAndMergeRegressionTests(unittest.TestCase):
    def test_batch_skips_cluster_summaries_and_its_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(2):
                np.savez(root / f'case{index}_clusters.npz', matrix=np.array([[1, 1, 2]]))
            np.savez(root / 'all_clusters.npz', tile_labels=np.ones(6), sample_ids=['a', 'b'])
            (root / 'cluster_metrics.csv').write_text('k,silhouette_score\n2,.5\n')
            for _ in range(2):
                result = compute_batch(root, root / 'metrics.csv')
                self.assertEqual(len(result), 2)
            self.assertEqual(len(compute_batch(root, root / 'other_metrics.csv')), 2)

    def test_new_patient_revision_replaces_only_manifest_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix in 'AB':
                save_map(root / f'P_{suffix}.npz', [[1]])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old = merge_patient_maps(root, 1, progress=False)[0].output_path
            old_bytes = old.read_bytes()
            for suffix in 'CD':
                save_map(root / f'P_{suffix}.npz', [[1]])
            manifest = root / 'patients.csv'
            manifest.write_text('source_file,patient_id\n' + ''.join(f'P_{suffix}.npz,P\n' for suffix in 'ABCD'))
            new = merge_patient_maps(root, patient_manifest=manifest, output_suffix='_merged_v2', progress=False)[0].output_path
            self.assertEqual(old.read_bytes(), old_bytes)
            self.assertEqual(json.loads((root / 'label_maps.json').read_text())['label_maps'], [new.name])
            result = compute_batch(root, root / 'metrics.csv')
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]['class_1_area'], 24)

    def test_merge_preserves_sources_and_rejects_incremental_overwrite(self):
        for delete in (False, True):
            with self.subTest(delete=delete), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                save_map(root / 'P_A.npz', [[1]])
                save_map(root / 'P_B.npz', [[2]])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = merge_patient_maps(root, 1, delete_originals=delete, progress=False)[0]
                self.assertEqual((root / 'P_A.npz').exists(), not delete)
                saved = result.output_path.read_bytes()
                merged = load_label_map(result.output_path)
                with self.assertRaises(ValueError):
                    load_matrix(result.output_path)
                np.testing.assert_array_equal(merged.matrix, [[1, 0, 2]])
                np.testing.assert_array_equal(merged.region_ids, [[1, 0, 2]])
                self.assertEqual(merged.spacing, (2, 3))
                save_map(root / 'P_C.npz', [[3]])
                save_map(root / 'P_D.npz', [[4]])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with self.assertRaises(FileExistsError):
                        merge_patient_maps(root, 1, delete_originals=delete, progress=False)
                self.assertEqual(result.output_path.read_bytes(), saved)
                self.assertTrue((root / 'P_C.npz').exists())
                self.assertTrue((root / 'P_D.npz').exists())

    def test_manifest_grouping_and_patient_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_map(root / 'C3L-00004-21_clusters.npz', [[1, 1], [1, 1]])
            save_map(root / 'C3L-00004-26_clusters.npz', [[1, 1]])
            (root / 'label_maps.json').write_text(json.dumps({'label_maps': [path.name for path in root.glob('*.npz')]}))
            manifest = root / 'patients.csv'
            manifest.write_text('source_file,patient_id\nC3L-00004-21_clusters.npz,C3L-00004\nC3L-00004-26_clusters.npz,C3L-00004\n')
            results = merge_patient_maps(root, patient_manifest=manifest, blank_cols=0, progress=False)
            self.assertEqual(len(results), 1)
            self.assertTrue((root / 'C3L-00004-21_clusters.npz').exists())
            result = compute_batch(root, root / 'metrics.csv')
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]['class_1_patch_count'], 2)
            self.assertEqual(result.iloc[0]['class_1_area'], 36)
            self.assertEqual(result.iloc[0]['landscape_extent_area'], 36)
            self.assertEqual(result.iloc[0]['class_1_proximity_index_mean'], 0)
            with self.assertRaises(ValueError):
                compute_batch(root, root / 'metrics.csv', spacing=(1, 1))

    def test_model_and_scale_validation_retains_originals(self):
        for second_model, second_spacing in (('other', (2, 3)), ('fixed-model', (1, 1))):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                save_map(root / 'P_A.npz', [[1]])
                save_map(root / 'P_B.npz', [[1]], model=second_model, spacing=second_spacing)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with self.assertRaises(ValueError):
                        merge_patient_maps(root, 1, delete_originals=True, progress=False)
                self.assertTrue((root / 'P_A.npz').exists())
                self.assertFalse((root / 'P_merged.npz').exists())

    def test_atomic_save_failure_does_not_delete_or_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_map(root / 'P_A.npz', [[1]])
            save_map(root / 'P_B.npz', [[1]])
            with warnings.catch_warnings(), patch('landscape2d.merge_maps.os.link', side_effect=OSError('disk error')):
                warnings.simplefilter('ignore')
                with self.assertRaises(OSError):
                    merge_patient_maps(root, 1, delete_originals=True, progress=False)
            self.assertTrue((root / 'P_A.npz').exists())
            self.assertTrue((root / 'P_B.npz').exists())
            self.assertFalse((root / 'P_merged.npz').exists())
            self.assertFalse(list(root.glob('*.tmp')))

    def test_legacy_merge_is_not_accepted_as_one_slide(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'legacy_merged.npz'
            np.savez(path, matrix=np.array([[1, 0, 2]]), source_files=['a', 'b'])
            with self.assertRaises(ValueError):
                load_label_map(path)

    def test_text_single_row_and_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'row.csv').write_text('1,2,3\n')
            (root / 'column.txt').write_text('1\n2\n3\n')
            self.assertEqual(load_matrix(root / 'row.csv').shape, (1, 3))
            self.assertEqual(load_matrix(root / 'column.txt').shape, (3, 1))

    def test_3d_streaming_batch_and_nifti_spacing_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = np.ones((2, 2, 2), np.int16)
            path = root / 'sample.nii.gz'
            nib.save(nib.Nifti1Image(volume, np.diag([2, 3, 5, 1])), path)
            np.testing.assert_array_equal(load_volume(path), volume)
            result = batch3(root, root / 'metrics.csv', spacing=(2, 3, 5))
            self.assertEqual(result.iloc[0]['class_1_volume'], 240)


if __name__ == '__main__':
    unittest.main()
