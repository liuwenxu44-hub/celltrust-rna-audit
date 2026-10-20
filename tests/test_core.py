"""Artificial data only; these tests do not estimate study performance."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from celltrust_audit import core, metrics, statistics
from celltrust.m4_compat.scoring import standardize_outer_compat

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


class MetricTests(unittest.TestCase):
    def test_original_metric_regressions(self):
        self.assertTrue(all(metrics.validate_metric().values()))

    def test_missing_support_is_not_zero_filled(self):
        result = metrics.one([], [], np.array([], dtype=str), 10)
        self.assertEqual(result['AP_status'], 'NO_FINITE_SCORE')
        self.assertTrue(np.isnan(result['AP']))
        self.assertEqual(result['score_coverage'], 0)

    def test_bootstrap_is_donor_clustered_and_reproducible(self):
        x = np.array([[.1, .2], [.3, .4], [.5, .6]])
        a = statistics.paired_effects(x, seed=123)
        b = statistics.paired_effects(x, seed=123)
        np.testing.assert_array_equal(a['bootstrap_draws'], b['bootstrap_draws'])
        np.testing.assert_allclose(a['estimate'], [.3, .4, .35])
        self.assertEqual(a['bootstrap_draws'].shape, (2000, 3))
        self.assertFalse(a['refitted'])

    def test_absent_stratum_invalidates_overall(self):
        x = np.array([[.1, np.nan], [.3, np.nan]])
        result = statistics.paired_effects(x, seed=123)
        self.assertTrue(np.isnan(result['estimate'][-1]))
        self.assertEqual(result['valid_bootstrap'][-1], 0)


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        dest = cls.folder / 'inputs'
        dest.mkdir()
        rng = np.random.default_rng(811)
        n, p = 120, 80
        donor = np.repeat([f'artificial_donor_{i}' for i in range(6)], 20)
        outer = np.repeat([f'O{i+1}' for i in range(6)], 20)
        data = {'cell_id': [f'artificial_cell_{i:04}' for i in range(n)],
                'donor_id': donor, 'outer_fold': outer}
        for i in range(6):
            remaining = [j for j in range(6) if j != i]
            data[f'inner_O{i+1}'] = np.repeat([
                'OUTER_QUERY' if j == i else f'I{remaining.index(j)+1}'
                for j in range(6)], 20)
        cls.cells = pd.DataFrame(data)
        cls.cells.to_csv(dest / 'method_cells.tsv', sep='\t', index=False)
        cls.labels = np.tile(np.repeat(core.LABELS, 5), 6)
        counts = rng.poisson(2.0, size=(n, p)).astype(float)
        for j, label in enumerate(core.LABELS):
            counts[cls.labels == label, j*10:j*10+5] += rng.poisson(3, size=(30, 5))
        cls.rna = sparse.csr_matrix(counts)
        sparse.save_npz(dest / 'RNA_counts.npz', cls.rna)
        sparse.save_npz(dest / 'ADT_counts.npz', sparse.csr_matrix(rng.poisson(20, size=(n, 8))))
        pd.DataFrame({'feature_id': [f'gene_{i}' for i in range(p)], 'modality': 'RNA',
                      'original_feature_index': np.arange(p)}).to_csv(dest / 'RNA_features.tsv', sep='\t', index=False)
        pd.DataFrame({'feature_id': [f'antibody_{i}' for i in range(8)], 'modality': 'ADT',
                      'original_feature_index': np.arange(p, p+8)}).to_csv(dest / 'ADT_features.tsv', sep='\t', index=False)
        cls.markers = pd.DataFrame([
            {'label': label, 'allowed_for_primary': 'TRUE', 'marker_role': role,
             'ensembl_gene_id': f'gene_{index}'}
            for j, label in enumerate(core.LABELS)
            for role, indices in [('CORE_POSITIVE', range(j*10, j*10+5)),
                                  ('EXPLICIT_NEGATIVE', range(((j+1)%4)*10, ((j+1)%4)*10+2))]
            for index in indices])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rna_first_does_not_need_adt_file(self):
        path = self.folder / 'inputs/ADT_counts.npz'
        hidden = path.with_suffix('.hidden')
        path.rename(hidden)
        try:
            view = core.load_condition(self.folder, 'A')
            self.assertTrue(np.array_equal(view['counts'].toarray(), self.rna.toarray()))
        finally:
            hidden.rename(path)

    def test_late_filter_multiplier_identity(self):
        a = core.load_condition(self.folder, 'A')
        c = core.load_condition(self.folder, 'C')
        expected = np.repeat(a['denominator']/c['denominator'], np.diff(a['linear'].indptr))
        np.testing.assert_allclose(c['linear'].data/a['linear'].data, expected, rtol=1e-12, atol=1e-12)

    def test_all_six_scores_and_donor_separation(self):
        view = core.load_condition(self.folder, 'A')
        rep = core.representation(view, 'O1')
        inner = [core.representation(view, f'O1_I{i}') for i in range(1, 6)]
        mark = core.marker_components(view, self.markers, True)
        result = core.score_outer(self.labels, self.cells, rep, inner, mark, 'A')
        self.assertEqual(result.shape, (20, 6))
        self.assertTrue(np.isfinite(result).any(axis=0).all())
        self.assertFalse(np.isin(rep['neighbors'], rep['query']).any())
        self.assertFalse(set(self.cells.donor_id.iloc[rep['train']]) & set(self.cells.donor_id.iloc[rep['query']]))
        self.assertEqual(mark['scaled_rank_max_error'], 0.0)

    def test_cli_validates_design_and_seed(self):
        cli = module('public_score_cli', ROOT / 'scripts/score.py')
        self.assertEqual(len(cli.validate_design(self.cells)), 36)
        expected = int.from_bytes(hashlib.sha256(b'D3_Zenodo7555405\tO1\tHVG2000\tCT001B_PCA_v1').digest()[:8], 'big') % 2**32
        self.assertEqual(cli.d3_seed('O1'), expected)
        bad = self.cells.copy()
        bad.loc[0, 'outer_fold'] = 'O2'
        with self.assertRaisesRegex(ValueError, 'DONOR_SPLIT'):
            cli.validate_design(bad)

    def test_cli_end_to_end_synthetic(self):
        labels = self.folder / 'labels.tsv'
        markers = self.folder / 'markers.tsv'
        result = self.folder / 'synthetic_scores.npz'
        pd.DataFrame({'cell_id': self.cells.cell_id, 'observed_label': self.labels}).to_csv(labels, sep='\t', index=False)
        self.markers.to_csv(markers, sep='\t', index=False)
        cmd = [sys.executable, '-B', str(ROOT / 'scripts/score.py'), '--inputs', str(self.folder),
               '--labels', str(labels), '--markers', str(markers), '--dataset', 'D4',
               '--condition', 'A', '--output', str(result)]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
        with np.load(result, allow_pickle=False) as z:
            self.assertEqual(z['scores'].shape, (120, 6))
            np.testing.assert_array_equal(z['cell_ids'], self.cells.cell_id)
        repeat = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn('OUTPUT_EXISTS_REFUSE_OVERWRITE', repeat.stderr)

    def test_zero_scale_compatibility_only_fills_missing_old_values(self):
        ref = {key: np.zeros(24) for key in ['S', 'G', 'H']}
        target = {key: np.array([0., 1.]) for key in ['S', 'G', 'H']}
        comp, status, _, used, old, _ = standardize_outer_compat(ref, target, np.zeros(24, int), np.zeros(2, int))
        self.assertTrue(np.isnan(old['Z_S']).all())
        self.assertTrue(np.isfinite(comp['Z_S']).all())
        self.assertTrue(used['S'].all())


class BoundaryGuardTests(unittest.TestCase):
    def test_counterfactuals_and_bad_metadata(self):
        guard = module('public_modality_guard', ROOT / 'scripts/modality_guard.py')
        counts = sparse.csr_matrix([[2, 5, 0, 200], [7, 1, 3, 4], [1, 4, 6, 20], [5, 0, 2, 2]])
        report, _ = guard.audit(counts, ['RNA']*3+['ADT'], ['x', 'x', 'y', 'y'], 'Artificial metadata by construction')
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(len(report['canary_checks']), 3)
        with self.assertRaisesRegex(ValueError, 'PROVENANCE_REQUIRED'):
            guard.audit(counts, ['RNA']*3+['ADT'], ['x', 'x', 'y', 'y'], '')

    def test_fractional_counts_require_explicit_authority(self):
        guard = module('public_modality_guard_fractional', ROOT / 'scripts/modality_guard.py')
        counts = sparse.csr_matrix([[1, 2.5], [2, 1]])
        with self.assertRaisesRegex(ValueError, 'FRACTIONAL_COUNTS'):
            guard.audit(counts, ['RNA', 'ADT'], ['x', 'x'], 'Artificial metadata')

    def test_realpath_checker_abstains_without_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'missing.json'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/realpath_guard.py'),
                                     '--contract', str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'NOT_VERIFIABLE')

    def test_execution_checker_distinguishes_conflict_and_absence(self):
        guard = module('public_execution_guard', ROOT / 'scripts/realpath_guard.py')
        with self.assertRaises(guard.Decision) as caught:
            guard.assert_signature({'shape': [2]}, {'shape': [3]}, 'synthetic')
        self.assertEqual(caught.exception.status, 'FAIL')
        with self.assertRaises(guard.Decision) as caught:
            guard.require({}, 'missing', 'synthetic')
        self.assertEqual(caught.exception.status, 'NOT_VERIFIABLE')


if __name__ == '__main__':
    unittest.main()
