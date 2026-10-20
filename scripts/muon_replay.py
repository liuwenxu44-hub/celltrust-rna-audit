#!/usr/bin/env python3
"""Bounded server-only replay of the pinned official MUON E03 RNA branch.

Original notebook cells are validated as text and never executed wholesale.
Only RNA preprocessing through neighbors is reproduced. Modern package versions
are pinned and recorded; this is not historical byte-exact notebook execution.
The public adapter independently checks normalization and actual consumer
bindings, but does not independently reimplement the Scanpy graph algorithm.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS')
for _key in THREAD_KEYS:
    os.environ.setdefault(_key, '2')
    if int(os.environ[_key]) not in (1, 2):
        raise RuntimeError('PUBLIC_E03_THREAD_BOUND_EXCEEDED:' + _key)

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


PROJECT = Path(os.environ.get("CELLTRUST_AUDIT_ROOT", str(Path.cwd()))).resolve()
NOTEBOOK_HASH = '39cefe1d1404829fff8d930bb88f2ba0bab779e06e02a15681b6f7b0b8883753'
COMMIT = '62bdec09381286a1e4b8236726fad42c772c59f1'
CASES = ('original', 'fresh_rebuild', 'adt_deleted', 'adt_scaled_x2',
         'adt_rows_shuffled', 'injected_late_adt')
VERSIONS = {'muon': '0.1.9', 'mudata': '0.4.1', 'protobuf': '7.36.1',
            'scanpy': '1.12.4', 'anndata': '0.13.3.post0', 'numpy': '2.5.3',
            'scipy': '1.18.1', 'pandas': '3.0.5', 'scikit-learn': '1.9.0'}
RTOL = 2e-6
ATOL = 2e-6
SEED = 2026090905
SELECTED_CELLS = (9, 10, 12, 40, 43, 46, 48, 53, 54, 57, 59, 62, 64, 68, 74)
RECIPE_LINES = {
    12: 'mdata = mu.read_10x_mtx(os.path.join(data_dir, "filtered_feature_bc_matrix"))',
    40: "rna = mdata.mod['rna']",
    43: "rna.var['mt'] = rna.var_names.str.startswith('MT-')",
    46: "mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)",
    48: "mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))",
    53: 'sc.pp.normalize_total(rna, target_sum=1e4)',
    54: 'sc.pp.log1p(rna)',
    57: 'sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)',
    62: 'rna.raw = rna',
    64: 'sc.pp.scale(rna, max_value=10)',
    68: "sc.tl.pca(rna, svd_solver='arpack')",
    74: 'sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def signature(value):
    value = np.ascontiguousarray(value)
    if value.dtype.kind not in 'biufc':
        raise TypeError('NUMERIC_SIGNATURE_REQUIRES_NUMERIC_BUFFER')
    return {'shape': list(value.shape), 'dtype': value.dtype.str,
            'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def identities(values):
    digest = hashlib.sha256()
    count = 0
    for value in values:
        text = str(value).encode('utf-8')
        digest.update(len(text).to_bytes(8, 'big'))
        digest.update(text)
        count += 1
    return {'count': count, 'sha256': digest.hexdigest()}


def matrix_signature(value):
    if not sparse.issparse(value):
        return {'storage': 'dense', 'array': signature(value)}
    if value.format not in ('csr', 'csc'):
        raise RuntimeError('UNSUPPORTED_SPARSE_FORMAT:' + value.format)
    return {'storage': value.format, 'shape': list(value.shape),
            'dtype': value.dtype.str, 'data': signature(value.data),
            'indices': signature(value.indices), 'indptr': signature(value.indptr)}


def memory_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def check_budget(start):
    if memory_bytes() > 32 * (1 << 30):
        raise RuntimeError('PUBLIC_E03_MEMORY_BOUND_EXCEEDED')
    if time.perf_counter() - start > 7200:
        raise RuntimeError('PUBLIC_E03_WALLTIME_BOUND_EXCEEDED')


def load_authority(assets):
    notebook = assets / 'original_notebook.ipynb'
    if sha(notebook) != NOTEBOOK_HASH:
        raise RuntimeError('PINNED_NOTEBOOK_HASH_CONFLICT')
    document = json.loads(notebook.read_text(encoding='utf-8'))
    selected = json.loads((assets / 'selected_original_code_cells.json').read_text())
    for index in SELECTED_CELLS:
        original = ''.join(document['cells'][index]['source'])
        if document['cells'][index]['cell_type'] != 'code' or selected[str(index)] != original:
            raise RuntimeError('ARCHIVED_ORIGINAL_CODE_CELL_CONFLICT:' + str(index))
        if index in RECIPE_LINES and RECIPE_LINES[index] not in original:
            raise RuntimeError('REPLAY_CALL_NOT_IN_PINNED_CELL:' + str(index))
    acquisition_path = assets / 'acquisition.json'
    acquisition = json.loads(acquisition_path.read_text())
    if acquisition.get('notebook_commit') != COMMIT:
        raise RuntimeError('ACQUISITION_NOTEBOOK_COMMIT_CONFLICT')
    source_files = {}
    for item in acquisition['files']:
        path = (assets / item['relative_path']).resolve(strict=True)
        if not path.is_relative_to(assets):
            raise RuntimeError('ACQUISITION_PATH_ESCAPES_ASSETS')
        observed = record(path)
        if observed['sha256'] != item['sha256'] or observed['bytes'] != item['bytes']:
            raise RuntimeError('ACQUIRED_FILE_IDENTITY_CONFLICT:' + item['relative_path'])
        source_files[item['relative_path']] = observed
    for name in ('original_notebook.ipynb', 'selected_original_code_cells.json'):
        if name not in source_files:
            raise RuntimeError('ACQUISITION_REQUIRED_SOURCE_MISSING:' + name)
    source_files['acquisition.json'] = record(acquisition_path)
    observed_versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if observed_versions != VERSIONS:
        raise RuntimeError('PUBLIC_PINNED_ENVIRONMENT_CONFLICT:' + json.dumps(observed_versions))
    environment = {'packages': observed_versions, 'python': sys.version,
                   'executable': sys.executable, 'prefix': sys.prefix,
                   'platform': platform.platform(),
                   'threads': {key: os.environ[key] for key in THREAD_KEYS},
                   'PYTHONPATH': os.environ.get('PYTHONPATH'),
                   'runtime_claim': 'MODERN_PINNED_REPLAY_NOT_HISTORICAL_BYTE_EXACT'}
    return source_files, environment


def metadata_modality_check(mdata, matrix_dir):
    feature_path = matrix_dir / 'features.tsv.gz'
    if not feature_path.exists():
        feature_path = matrix_dir / 'features.tsv'
    table = pd.read_csv(feature_path, sep='\t', header=None, dtype=str)
    if table.shape[1] != 3:
        raise RuntimeError('PUBLIC_FEATURE_TYPE_METADATA_REQUIRED')
    source_types = table.groupby(0)[2].agg(lambda values: set(values))
    result = {}
    for modality, declared in (('rna', 'Gene Expression'), ('prot', 'Antibody Capture')):
        if modality not in mdata.mod:
            raise RuntimeError('ACTUAL_MUON_MODALITY_MISSING:' + modality)
        obj = mdata.mod[modality]
        if 'gene_ids' not in obj.var:
            raise RuntimeError('ACTUAL_MUON_FEATURE_ID_METADATA_MISSING:' + modality)
        for feature_id in obj.var.gene_ids.astype(str):
            if feature_id not in source_types or source_types[feature_id] != {declared}:
                raise RuntimeError('MUON_MODALITY_CONFLICT_WITH_EXPLICIT_FEATURE_TYPES')
        if 'feature_types' in obj.var and not (obj.var.feature_types == declared).all():
            raise RuntimeError('MUON_RETAINED_FEATURE_TYPE_CONFLICT')
        result[modality] = {'declared_10x_type': declared,
                            'feature_ids': identities(obj.var.gene_ids),
                            'feature_names': identities(obj.var_names),
                            'cell_ids': identities(obj.obs_names),
                            'shape': list(obj.shape)}
    return result


def independent_normalization(raw, observed):
    # The denominator is the post-public-gene-filter RNA feature universe,
    # not the original library total used earlier by the notebook's cell QC.
    counts = sparse.csr_matrix(raw, dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.sort_indices()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0) or not np.isfinite(counts.data).all():
        raise RuntimeError('PUBLIC_FILTERED_RNA_COUNTS_NOT_VALID')
    expected = counts.copy()
    expected.data = np.log1p(expected.data * np.repeat(10000.0 / totals,
                                                      np.diff(expected.indptr)))
    actual = sparse.csr_matrix(observed, dtype=np.float64, copy=True)
    actual.sum_duplicates()
    actual.sort_indices()
    delta = (actual - expected).tocsr()
    delta.eliminate_zeros()
    max_abs = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    # Sparse entries absent in either matrix still participate in this check.
    absolute = abs(delta)
    tolerance = abs(expected).multiply(RTOL)
    residual = absolute - tolerance
    failures = int(np.count_nonzero(residual.data > ATOL))
    passed = failures == 0 and np.isfinite(actual.data).all()
    return {'method': 'INDEPENDENT_NUMPY_SCIPY_RNA_FILTERED_FEATURE_DENOMINATOR',
            'status': 'PASS' if passed else 'FAIL_NUMERIC_NORMALIZATION',
            'rtol': RTOL, 'atol': ATOL, 'max_abs_difference': max_abs,
            'outside_tolerance_entries': failures,
            'raw_counts': matrix_signature(raw),
            'denominator': signature(totals),
            'normalized_observed': matrix_signature(observed),
            'cells': raw.shape[0], 'features': raw.shape[1]}


@contextmanager
def capture_pca(events):
    module = importlib.import_module('scanpy.preprocessing._pca')
    original = module._get_obs_rep

    def observed(obj, *args, **kwargs):
        value = original(obj, *args, **kwargs)
        events.append({'function': 'scanpy.preprocessing._pca._get_obs_rep',
                       'layer': kwargs.get('layer'), 'obsm': kwargs.get('obsm'),
                       'array': signature(value), 'cell_ids': identities(obj.obs_names),
                       'feature_ids': identities(obj.var_names)})
        return value

    module._get_obs_rep = observed
    try:
        yield
    finally:
        module._get_obs_rep = original


@contextmanager
def capture_neighbors(events):
    # Neighbors.compute_neighbors imports this function locally from _utils;
    # patch the owning module before the real call, not an unused alias.
    module = importlib.import_module('scanpy.tools._utils')
    original = module._choose_representation

    def observed(obj, *args, **kwargs):
        value = original(obj, *args, **kwargs)
        events.append({'function': 'scanpy.tools._utils._choose_representation',
                       'use_rep': kwargs.get('use_rep'), 'n_pcs': kwargs.get('n_pcs'),
                       'array': signature(value), 'cell_ids': identities(obj.obs_names)})
        return value

    module._choose_representation = observed
    try:
        yield
    finally:
        module._choose_representation = original


def actual_binding_check(rna, pca_events, neighbor_events):
    hvg = rna.var.highly_variable.to_numpy()
    failures = []
    if len(pca_events) != 1:
        failures.append('PCA_GETTER_EVENT_COUNT')
    else:
        event = pca_events[0]
        if event['array'] != signature(rna.X[:, hvg]):
            failures.append('PCA_CONSUMED_ARRAY_NOT_SAVED_SCALED_HVG_X')
        if event['cell_ids'] != identities(rna.obs_names):
            failures.append('PCA_CONSUMED_CELL_IDENTITY')
        if event['feature_ids'] != identities(rna.var_names[hvg]):
            failures.append('PCA_CONSUMED_FEATURE_IDENTITY')
        if event['layer'] is not None or event['obsm'] is not None:
            failures.append('PCA_UNDECLARED_LAYER_OR_REPRESENTATION')
    if len(neighbor_events) != 1:
        failures.append('NEIGHBOR_GETTER_EVENT_COUNT')
    else:
        event = neighbor_events[0]
        if event['array'] != signature(rna.obsm['X_pca'][:, :20]):
            failures.append('NEIGHBORS_CONSUMED_ARRAY_NOT_PCA20')
        if event['cell_ids'] != identities(rna.obs_names):
            failures.append('NEIGHBORS_CONSUMED_CELL_IDENTITY')
        if event['n_pcs'] != 20 or event['use_rep'] is not None:
            failures.append('NEIGHBORS_CALL_SELECTION_CONFLICT')
    for key in ('distances', 'connectivities'):
        graph = rna.obsp[key]
        if graph.shape != (rna.n_obs, rna.n_obs) or not np.isfinite(graph.data).all():
            failures.append('INVALID_ACTUAL_OUTPUT_GRAPH:' + key)
    return {'status': 'PASS' if not failures else 'FAIL_ACTUAL_CONSUMER_BINDING',
            'failures': failures,
            'limitation': 'No independent numerical recreation of the public Scanpy graph algorithm.'}


def execute_case(case, assets, out, source_files, environment, start):
    began = time.perf_counter()
    matrix_dir = assets / 'filtered_feature_bc_matrix'
    # Original public reader, with no replacement reader or silent format adapter.
    mdata = mu.read_10x_mtx(str(matrix_dir))
    modalities = metadata_modality_check(mdata, matrix_dir)
    if mdata.mod['rna'].n_obs > 8000 or mdata.mod['rna'].n_vars > 40000:
        raise RuntimeError('FIXED_PUBLIC_INPUT_SHAPE_BOUND_EXCEEDED')
    mutation = {'type': case, 'before_protein': matrix_signature(mdata.mod['prot'].X),
                'protein_metadata': modalities['prot']}
    if case == 'adt_deleted':
        del mdata.mod['prot']
    elif case == 'adt_scaled_x2':
        mdata.mod['prot'].X = mdata.mod['prot'].X * 2
    elif case == 'adt_rows_shuffled':
        permutation = np.random.default_rng(SEED).permutation(mdata.mod['prot'].n_obs)
        mdata.mod['prot'].X = mdata.mod['prot'].X[permutation, :].copy()
        mutation.update({'seed': SEED, 'row_permutation': signature(permutation),
                          'cell_metadata_held_fixed': True})
    if 'prot' in mdata.mod:
        mutation['after_protein'] = matrix_signature(mdata.mod['prot'].X)

    # Exact authored RNA branch: cell-QC columns are not recomputed after
    # filter_var, matching the original notebook's ordering and semantics.
    rna = mdata.mod['rna']
    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None,
                               log1p=False, inplace=True)
    mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
    mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
    mu.pp.filter_obs(rna, 'total_counts', lambda x: (x > 1500) & (x < 15000))
    mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)
    filtered_raw = rna.X.copy()
    if rna.n_obs < 51 or rna.n_vars < 51:
        raise RuntimeError('PUBLIC_QC_OUTPUT_OUTSIDE_FROZEN_RECIPE_DIMENSIONS')
    if case == 'injected_late_adt':
        protein = mdata.mod['prot']
        positions = protein.obs_names.get_indexer(rna.obs_names)
        if np.any(positions < 0) or not protein.obs_names.is_unique:
            raise RuntimeError('INJECTED_ADT_ALIGNMENT_UNRESOLVED')
        combined = ad.AnnData(X=sparse.hstack(
            [rna.X, protein.X[positions, :]], format='csr'))
        # Only the intended denominator fault differs. No second RNA-only
        # normalisation is performed after this explicit RNA slice.
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)
        rna.X = combined.X[:, :rna.n_vars].copy()
        del combined
    else:
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
    normalization = independent_normalization(filtered_raw, rna.X)
    sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
    rna.raw = rna
    sc.pp.scale(rna, max_value=10)
    pca_events = []
    with capture_pca(pca_events):
        sc.tl.pca(rna, svd_solver='arpack')
    neighbor_events = []
    with capture_neighbors(neighbor_events):
        sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
    binding = actual_binding_check(rna, pca_events, neighbor_events)
    hvg = rna.var.highly_variable.to_numpy()
    stages = {'normalized_RNA': matrix_signature(rna.raw.X),
              'HVG': signature(hvg), 'scaled_RNA': matrix_signature(rna.X),
              'scaled_HVG': signature(rna.X[:, hvg]),
              'PCA': signature(rna.obsm['X_pca']),
              'PCA_components': signature(rna.varm['PCs']),
              'distances': matrix_signature(rna.obsp['distances']),
              'connectivities': matrix_signature(rna.obsp['connectivities'])}
    trace = {'schema_version': 1, 'case': case, 'assay_scope': 'RNA',
             'notebook_commit': COMMIT, 'notebook_sha256': NOTEBOOK_HASH,
             'original_code_cell_indices': list(SELECTED_CELLS),
             'source_files': source_files, 'producer': record(__file__),
             'environment': environment, 'modality_metadata': modalities,
             'protein_counterfactual': mutation,
             'cell_ids': identities(rna.obs_names),
             'feature_ids': identities(rna.var.gene_ids),
             'feature_names': identities(rna.var_names),
             'filtered_raw_counts': matrix_signature(filtered_raw),
             'pca_get_obs_rep': pca_events,
             'neighbor_choose_representation': neighbor_events,
             'stage_signatures': stages,
             'actual_PCA_parameters': dict(rna.uns['pca']['params']),
             'actual_neighbor_parameters': dict(rna.uns['neighbors']['params'])}
    case_path = out / (case + '.h5ad')
    rna.write_h5ad(case_path, compression='gzip', compression_opts=4)
    # Reopen serialized output for actual-file binding, not just in-memory claims.
    saved = ad.read_h5ad(case_path)
    saved_stages = {'normalized_RNA': matrix_signature(saved.raw.X),
                    'HVG': signature(saved.var.highly_variable.to_numpy()),
                    'scaled_RNA': matrix_signature(saved.X),
                    'scaled_HVG': signature(saved.X[:, saved.var.highly_variable.to_numpy()]),
                    'PCA': signature(saved.obsm['X_pca']),
                    'PCA_components': signature(saved.varm['PCs']),
                    'distances': matrix_signature(saved.obsp['distances']),
                    'connectivities': matrix_signature(saved.obsp['connectivities'])}
    if saved_stages != stages:
        raise RuntimeError('SERIALIZED_PUBLIC_STATE_DIFFERS_FROM_ACTUAL_OUTPUT')
    trace['serialized_state_stage_signatures'] = saved_stages
    trace_path = out / (case + '.trace.json')
    check_path = out / (case + '.normalization_check.json')
    write_json(trace_path, trace)
    write_json(check_path, {'normalization': normalization, 'actual_bindings': binding})
    guard_status = ('PASS' if normalization['status'] == 'PASS' and binding['status'] == 'PASS'
                    else 'FAIL')
    result = {'case': case, 'adapter_status': guard_status,
              'normalization': normalization, 'actual_bindings': binding,
              'stage_signatures': stages, 'cell_ids': identities(rna.obs_names),
              'feature_ids': identities(rna.var.gene_ids),
              'files': {'state': record(case_path), 'trace': record(trace_path),
                        'normalization_check': record(check_path)},
              'dimensions': {'cells': rna.n_obs, 'features': rna.n_vars,
                             'HVG': int(hvg.sum()), 'PCs': rna.obsm['X_pca'].shape[1]},
              'elapsed_seconds': time.perf_counter() - began,
              'process_peak_rss_bytes': memory_bytes()}
    write_json(out / (case + '.receipt.json'), result)
    del saved, rna, mdata, filtered_raw
    gc.collect()
    check_budget(start)
    print(json.dumps({'public_case': case, 'adapter_status': guard_status,
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return result


def compare_states(baseline, candidate):
    first = ad.read_h5ad(baseline)
    second = ad.read_h5ad(candidate)
    rows = []
    if not np.array_equal(first.obs_names, second.obs_names) or not np.array_equal(
            first.var.gene_ids, second.var.gene_ids):
        return {'status': 'FAIL', 'reason': 'CELL_OR_FEATURE_IDENTITY_CHANGED'}
    for name, left, right in (
        ('normalized_RNA', first.raw.X, second.raw.X),
        ('HVG', first.var.highly_variable.to_numpy(), second.var.highly_variable.to_numpy()),
        ('scaled_RNA', first.X, second.X),
        ('PCA', first.obsm['X_pca'], second.obsm['X_pca']),
        ('PCA_components', first.varm['PCs'], second.varm['PCs']),
        ('distances', first.obsp['distances'], second.obsp['distances']),
        ('connectivities', first.obsp['connectivities'], second.obsp['connectivities']),
    ):
        exact = matrix_signature(left) == matrix_signature(right)
        if sparse.issparse(left) and sparse.issparse(right):
            difference = (left.astype(np.float64) - right.astype(np.float64)).tocsr()
            maximum = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
            residual = abs(difference) - abs(left.astype(np.float64)).multiply(RTOL)
            close = not np.any(residual.data > ATOL)
        else:
            maximum = float(np.max(np.abs(np.asarray(left, dtype=float) -
                                           np.asarray(right, dtype=float))))
            close = bool(np.allclose(left, right, rtol=RTOL, atol=ATOL))
        if name == 'HVG':
            close = exact
        rows.append({'stage': name, 'exact': exact, 'within_tolerance': bool(close),
                     'max_abs_difference': maximum})
    del first, second
    gc.collect()
    return {'status': 'PASS' if all(row['within_tolerance'] for row in rows) else 'FAIL',
            'all_exact': all(row['exact'] for row in rows),
            'rtol': RTOL, 'atol': ATOL, 'stages': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets.resolve(strict=True)
    out = args.out.resolve()
    if sys.platform != 'linux' or any(not path.is_relative_to(PROJECT)
                                      for path in (assets, out)):
        raise RuntimeError('SERVER_PROJECT_ONLY_PUBLIC_EXECUTION')
    if out.exists() or args.out.is_symlink() or not out.parent.is_dir():
        raise RuntimeError('PUBLIC_OUTPUT_MUST_BE_NEW_WITH_PREFLIGHTED_PARENT')
    source_files, environment = load_authority(assets)
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        cases = {case: execute_case(case, assets, out, source_files, environment, start)
                 for case in CASES}
        comparisons = {
            case: compare_states(out / 'original.h5ad', out / (case + '.h5ad'))
            for case in ('fresh_rebuild', 'adt_deleted', 'adt_scaled_x2', 'adt_rows_shuffled')}
        original_authored = cases['original']['adapter_status']
        injected = cases['injected_late_adt']['normalization']['status']
        completed = (all(cases[case]['adapter_status'] == 'PASS'
                         for case in CASES if case != 'injected_late_adt')
                     and all(result['status'] == 'PASS' for result in comparisons.values())
                     and injected == 'FAIL_NUMERIC_NORMALIZATION'
                     and cases['injected_late_adt']['actual_bindings']['status'] == 'PASS')
        inventory = [record(out / (case + suffix)) for case in CASES
                     for suffix in ('.h5ad', '.trace.json', '.normalization_check.json', '.receipt.json')]
        total_bytes = sum(row['bytes'] for row in inventory)
        if total_bytes > 8 * (1 << 30):
            raise RuntimeError('PUBLIC_OUTPUT_DISK_BOUND_EXCEEDED')
        summary = {'schema_version': 1, 'stage': 'CT-MOD-005',
                   'public_workflow': 'E03_OFFICIAL_MUON_5K_PBMC_RNA_THROUGH_NEIGHBORS',
                   'status': 'PASS_WITH_LIMITATIONS' if completed else 'FAIL_PUBLIC_SOFTWARE_CHECKS',
                   'created_utc': datetime.now(timezone.utc).isoformat(),
                   'original_authored_workflow_adapter_status': original_authored,
                   'injected_fault_normalization_status': injected,
                   'case_statuses': {case: result['adapter_status'] for case, result in cases.items()},
                   'rebuild_and_counterfactual_comparisons': comparisons,
                   'source_files': source_files, 'environment': environment,
                   'producer': record(__file__), 'command': sys.argv,
                   'files': inventory, 'output_bytes': total_bytes,
                   'elapsed_seconds': time.perf_counter() - start,
                   'peak_rss_bytes': memory_bytes(),
                   'claims': {'biological_scoring_cohort': False,
                              'annotation_or_label_evaluation': False,
                              'full_notebook_execution': False,
                              'author_error_inferred_from_injection': False},
                   'limitations': [
                       'One fixed public RNA branch and one downloaded filtered-count resource only.',
                       'Original authored behavior is separate from intentionally injected faults.',
                       'Actual MUON/Scanpy calls are run in the recorded contemporary environment; not historical byte-exact reproduction.',
                       'Independent NumPy/SciPy normalization plus actual getter/file binding; no independent full numerical graph-algorithm recreation.',
                       'Protein/DSB/MOFA/UMAP/clustering/annotations and biological performance are out of scope.',
                       'Process peak RSS is a cumulative high-water mark, not an isolated per-case peak.',
                   ]}
        write_json(out / 'summary.json', summary)
        print(json.dumps({'status': summary['status'], 'out': str(out),
                          'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
        return 0 if completed else 2
    except Exception as error:
        write_json(out / 'failure.json', {
            'status': 'PUBLIC_EXECUTION_BLOCKED_PARTIAL_OUTPUT_RETAINED',
            'exception_type': type(error).__name__, 'message': str(error),
            'elapsed_seconds': time.perf_counter() - start,
            'peak_rss_bytes': memory_bytes()})
        raise


if __name__ == '__main__':
    sys.exit(main())
