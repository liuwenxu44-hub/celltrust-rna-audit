#!/usr/bin/env python3
"""Server-only, bounded D3/O1 replay of the frozen CT-MOD-002 RNA pathway.

This producer has no detector, expected decisions, labels or performance code.
It writes a new state.h5, execution_trace.json and receipt.json.  The clean
recipe is the RNA-only branch of scanpy_reference_v001.py, restricted to O1.
The two explicitly requested software-test variants change only the matrix
normalised before RNA slicing; wrong_layer additionally constructs correct
RNA-only layers while the real Scanpy calls continue to consume X.

Numeric signature: SHA-256 of C-contiguous native-dtype bytes, with dtype.str
and shape recorded separately. Identity signature: concatenate UTF-8 strings,
each preceded by its byte length as an unsigned eight-byte big-endian integer.
No cache loading or state mutation cases are implemented in this producer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


# Set bounded defaults before importing numerical libraries. The orchestrator
# remains responsible for process-level resource and wall-time enforcement.
THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
)
for _thread_variable in THREAD_VARIABLES:
    os.environ.setdefault(_thread_variable, '2')
    if int(os.environ[_thread_variable]) not in (1, 2):
        raise RuntimeError('THREAD_BOUND_EXCEEDED:' + _thread_variable)

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import distance


PROJECT_ROOT = Path(os.environ.get("CELLTRUST_AUDIT_ROOT", str(Path.cwd()))).resolve()
REFERENCE_SOURCE = Path(__file__).resolve().parents[1] / 'reference/scanpy_reference.py'
REFERENCE_SHA256 = '3e675f87aab882a2cf6a6b0fe4571e8e665c1caaf97bbb543b78f2f5508a9b6e'
RECIPE_ID = 'd3-o1-scanpy-reference-v1'
SCOPE = 'BOUNDED_REAL_D3_O1_REPRESENTATION_SOFTWARE_AUDIT_NO_PERFORMANCE'
FROZEN_INPUTS = {
    'RNA_counts.npz': '464cad79562bc0f638c45f6b275e44926c8a8bb5137f93a715e89ca5bbe17e00',
    'RNA_features.tsv': '22f39498279d969f9382bf89b21e2202703198a387cc9e0d216f9d7dca802bf2',
    'method_cells.tsv': '1fe0fc86410c55e666c48d03c2ec426c0ce133973bf45454407ac86c065f04d4',
    'ADT_counts.npz': 'aaf04a29073d26f07a3a67c68d184b89461fa13806ebe898a86d5a90688eec42',
    'ADT_features.tsv': '7895bd2d7e0a5d3c92460256a42e6d50371237915f44de8ae8b8f3027f1976de',
}
EXPECTED_ENVIRONMENT = {
    'scanpy': '1.12.4', 'anndata': '0.13.3.post0', 'numpy': '2.5.3',
    'scipy': '1.18.1', 'scikit-learn': '1.9.0', 'h5py': '3.16.0',
    'pandas': '3.0.5',
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': file_hash(path)}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def array_signature(value):
    if sparse.issparse(value):
        raise TypeError('NUMERIC_ARRAY_SIGNATURE_REQUIRES_ACTUAL_DENSE_BUFFER')
    value = np.ascontiguousarray(value)
    if value.dtype.kind not in 'biufc':
        raise TypeError('NUMERIC_ARRAY_SIGNATURE_REQUIRES_NUMERIC_DTYPE')
    return {'shape': list(value.shape), 'dtype': value.dtype.str,
            'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def identity_signature(values):
    digest = hashlib.sha256()
    count = 0
    for value in values:
        encoded = str(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        count += 1
    return {'count': count, 'sha256': digest.hexdigest()}


def matrix_signature(value):
    if not sparse.issparse(value):
        return {'storage': 'dense', 'array': array_signature(value)}
    # Do not densify an entire RNA matrix merely to make a trace. Preserve
    # actual sparse storage identity, including index ordering and duplicates.
    if value.format not in ('csr', 'csc'):
        raise TypeError('UNSUPPORTED_OBSERVED_SPARSE_STORAGE:' + value.format)
    return {'storage': value.format, 'shape': list(value.shape),
            'dtype': value.dtype.str, 'data': array_signature(value.data),
            'indices': array_signature(value.indices),
            'indptr': array_signature(value.indptr)}


def matrix_event(obj, value, *, layer=None):
    return {'input_key': 'X' if layer is None else 'layer:' + layer,
            'matrix': matrix_signature(value),
            'cell_ids': identity_signature(obj.obs_names),
            'feature_ids': identity_signature(obj.var_names)}


def peak_rss_bytes():
    # This executable is restricted to Linux on server223: ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def budget_check(started):
    if peak_rss_bytes() > 32 * (1 << 30):
        raise RuntimeError('PEAK_RSS_EXCEEDS_32_GIB_RETAIN_PARTIAL_OUTPUTS')
    if time.perf_counter() - started > 2 * 3600:
        raise RuntimeError('WALLTIME_EXCEEDS_TWO_HOURS_RETAIN_PARTIAL_OUTPUTS')


def project_path(path, *, existing):
    result = Path(path).resolve(strict=existing)
    if not result.is_relative_to(PROJECT_ROOT.resolve(strict=True)):
        raise RuntimeError('SERVER_PROJECT_OWNED_PATH_REQUIRED:' + str(result))
    return result


def environment_record():
    versions = {name: importlib.metadata.version(name)
                for name in EXPECTED_ENVIRONMENT}
    if versions != EXPECTED_ENVIRONMENT:
        raise RuntimeError('FROZEN_ENVIRONMENT_CONFLICT:' + json.dumps(versions))
    if sys.platform != 'linux':
        raise RuntimeError('SERVER223_LINUX_EXECUTION_REQUIRED')
    return {'packages': versions, 'python': sys.version,
            'executable': sys.executable, 'prefix': sys.prefix,
            'platform': platform.platform(),
            'threads': {name: os.environ[name] for name in THREAD_VARIABLES}}


def verify_sources(inputs, mode):
    names = ['RNA_counts.npz', 'RNA_features.tsv', 'method_cells.tsv']
    if mode != 'clean':
        names += ['ADT_counts.npz', 'ADT_features.tsv']
    records = {}
    for name in names:
        path = project_path(inputs / name, existing=True)
        record = file_record(path)
        if record['sha256'] != FROZEN_INPUTS[name]:
            raise RuntimeError('FROZEN_INPUT_HASH_CONFLICT:' + name)
        records[name] = record
    reference = file_record(REFERENCE_SOURCE)
    if reference['sha256'] != REFERENCE_SHA256:
        raise RuntimeError('FROZEN_REFERENCE_SOURCE_HASH_CONFLICT')
    return records, reference


def make_object(counts, features, ids):
    names = np.asarray([
        f'{modality}:{index}:{feature}'
        for index, (modality, feature) in enumerate(
            zip(features.modality, features.feature_id, strict=True))
    ], dtype=str)
    obj = ad.AnnData(X=counts.copy())
    obj.obs_names = ids
    obj.var_names = names
    return obj


def normalize(obj, trace, label):
    before = matrix_event(obj, obj.X)
    totals = np.asarray(obj.X.sum(axis=1)).ravel().copy()
    sc.pp.normalize_total(obj, target_sum=10000,
                          exclude_highly_expressed=False, inplace=True)
    sc.pp.log1p(obj)
    trace.append({'operation': 'scanpy.normalize_total_then_log1p',
                  'branch': label, 'target_sum': 10000,
                  'exclude_highly_expressed': False,
                  'before': before, 'after': matrix_event(obj, obj.X),
                  'actual_denominator': array_signature(totals)})
    return totals


@contextmanager
def observe_real_pca_getter(events):
    module = importlib.import_module('scanpy.preprocessing._pca')
    original = module._get_obs_rep

    def observed(actual_obj, *args, **kwargs):
        actual = original(actual_obj, *args, **kwargs)
        if args:
            raise RuntimeError('UNSUPPORTED_POSITIONAL_PCA_GETTER_SELECTION')
        layer = kwargs.get('layer')
        obsm = kwargs.get('obsm')
        events.append({
            'function': 'scanpy.preprocessing._pca._get_obs_rep',
            'layer': layer, 'obsm': obsm,
            'input_key': ('obsm:' + obsm if obsm is not None else
                          'layer:' + layer if layer is not None else 'X'),
            'array': array_signature(actual),
            'cell_ids': identity_signature(actual_obj.obs_names),
            'feature_ids': identity_signature(actual_obj.var_names),
        })
        return actual

    module._get_obs_rep = observed
    try:
        yield
    finally:
        module._get_obs_rep = original


def traced_cdist(actual_query, actual_training, *, query_start, query_stop,
                 query_ids, training_ids, events):
    # Capture signatures from the exact objects passed to scipy, immediately
    # around the real call; no nominal parent object substitutes for them.
    query_signature = array_signature(actual_query)
    training_signature = array_signature(actual_training)
    result = distance.cdist(actual_query, actual_training, metric='sqeuclidean')
    events.append({
        'function': 'scipy.spatial.distance.cdist',
        'query_start': query_start, 'query_stop': query_stop,
        'metric': 'sqeuclidean', 'query_array': query_signature,
        'training_array': training_signature,
        'distance_array': array_signature(result),
        'query_cell_ids': identity_signature(query_ids),
        'training_cell_ids': identity_signature(training_ids),
    })
    return result


def run(inputs, out, mode, started):
    environment = environment_record()
    source_files, reference = verify_sources(inputs, mode)
    cells = pd.read_csv(inputs / 'method_cells.tsv', sep='\t',
                        usecols=['pilot_cell_index', 'cell_id', 'donor_id',
                                 'outer_fold'])
    ids = cells.cell_id.to_numpy()
    n = len(cells)
    if n != 7200 or n > 8000 or len(set(ids)) != n:
        raise RuntimeError('FROZEN_D3_CELL_IDENTITY_BOUNDARY_CONFLICT')
    if not np.array_equal(cells.pilot_cell_index.to_numpy(), np.arange(n)):
        raise RuntimeError('FROZEN_D3_CELL_ROW_ORDER_CONFLICT')
    rna = sparse.load_npz(inputs / 'RNA_counts.npz').astype(np.float64)
    features = pd.read_csv(inputs / 'RNA_features.tsv', sep='\t')
    if rna.shape != (n, 16825) or rna.shape[1] > 22000:
        raise RuntimeError('FROZEN_D3_RNA_DIMENSION_CONFLICT')
    if len(features) != rna.shape[1] or not (features.modality == 'RNA').all():
        raise RuntimeError('FROZEN_EXPLICIT_RNA_FEATURE_IDENTITY_CONFLICT')
    if not np.isfinite(rna.data).all() or np.any(rna.data < 0):
        raise RuntimeError('RNA_COUNTS_NOT_FINITE_NONNEGATIVE')
    rna_totals = np.asarray(rna.sum(axis=1)).ravel()
    if np.any(rna_totals <= 0):
        raise RuntimeError('ZERO_RNA_TOTAL_NO_IMPLICIT_CELL_REMOVAL')

    normalization_events = []
    correct = None
    if mode == 'clean':
        obj = make_object(rna, features, ids)
        normalization_totals = normalize(obj, normalization_events, 'actual_X')
    else:
        adt = sparse.load_npz(inputs / 'ADT_counts.npz').astype(np.float64)
        adt_features = pd.read_csv(inputs / 'ADT_features.tsv', sep='\t')
        if adt.shape != (n, 175) or not (adt_features.modality == 'ADT').all():
            raise RuntimeError('FROZEN_D3_ADT_IDENTITY_CONFLICT')
        if not np.isfinite(adt.data).all() or np.any(adt.data < 0):
            raise RuntimeError('ADT_COUNTS_NOT_FINITE_NONNEGATIVE')
        all_features = pd.concat([features, adt_features], ignore_index=True)
        counts = sparse.hstack([rna, adt], format='csr')
        order = np.argsort(all_features.original_feature_index.to_numpy(),
                           kind='stable')
        counts = counts[:, order]
        all_features = all_features.iloc[order].reset_index(drop=True)
        if counts.shape[1] > 22000:
            raise RuntimeError('FEATURE_SHAPE_BOUND_EXCEEDED')
        obj = make_object(counts, all_features, ids)
        normalization_totals = normalize(obj, normalization_events, 'actual_X')
        keep = all_features.modality.to_numpy() == 'RNA'
        obj = obj[:, keep].copy()
        features = all_features.loc[keep].reset_index(drop=True)
        if not np.array_equal(features.feature_id.to_numpy(),
                              pd.read_csv(inputs / 'RNA_features.tsv',
                                          sep='\t').feature_id.to_numpy()):
            raise RuntimeError('RNA_ORDER_CHANGED_BY_EXPLICIT_MODALITY_SLICE')
        del counts, all_features, adt
        if mode == 'wrong_layer':
            correct = make_object(rna, features, ids)
            normalize(correct, normalization_events, 'correct_RNA_layer_source')
            # These are real AnnData layers, not a declared but absent layer.
            obj.layers['rna_only_normalized'] = correct.X.copy()

    tr = np.flatnonzero(cells.outer_fold.to_numpy() != 'O1')
    qu = np.flatnonzero(cells.outer_fold.to_numpy() == 'O1')
    if len(tr) < 31 or len(qu) == 0:
        raise RuntimeError('D3_O1_PARTITION_BOUNDARY_CONFLICT')
    if set(cells.donor_id.iloc[tr]) & set(cells.donor_id.iloc[qu]):
        raise RuntimeError('QUERY_DONOR_ENTERS_TRAINING')
    minimum = max(10, math.ceil(.01 * len(tr)))
    filter_mask, _ = sc.pp.filter_genes(rna[tr], min_cells=minimum,
                                       inplace=False)
    eligible = np.flatnonzero(filter_mask)
    train = obj[tr, eligible].copy()
    n_top = min(2000, len(eligible))
    sc.pp.highly_variable_genes(train, n_top_genes=n_top, flavor='seurat',
                                n_bins=20, batch_key=None, subset=False,
                                inplace=True)
    chosen_local = np.flatnonzero(train.var.highly_variable.to_numpy())
    chosen = eligible[chosen_local]
    train = train[:, chosen_local].copy()
    normalized_selected = obj.X[:, chosen].toarray()
    sc.pp.scale(train, zero_center=True, max_value=None)
    center = train.var['mean'].to_numpy()
    scale = train.var['std'].to_numpy()
    seed = int.from_bytes(hashlib.sha256(
        b'D3_SCANPY_REFERENCE|O1|CTMOD002_V1').digest()[:8], 'big') % (2 ** 32)
    k = min(30, len(tr) - 1, len(chosen) - 1)
    if k != 30 or len(chosen) != 2000:
        raise RuntimeError('FROZEN_D3_O1_REPRESENTATION_DIMENSION_CONFLICT')
    additional_arrays = {}
    if correct is not None:
        correct_selected = correct.X[:, chosen].toarray()
        correct_train = correct[tr, chosen].copy()
        sc.pp.scale(correct_train, zero_center=True, max_value=None)
        train.layers['rna_only_scaled'] = correct_train.X.copy()
        additional_arrays = {
            'layer_correct_normalized_selected': correct_selected,
            'layer_correct_scaled_training': np.asarray(
                train.layers['rna_only_scaled']),
        }
        del correct_train, correct
    pca_events = []
    with observe_real_pca_getter(pca_events):
        # Exactly the frozen call: do not add mask_var or change its defaults.
        sc.pp.pca(train, n_comps=k, zero_center=True, svd_solver='arpack',
                  random_state=seed, dtype='float64')
    if len(pca_events) != 1:
        raise RuntimeError('PCA_ACTUAL_GETTER_TRACE_NOT_EXACTLY_ONE_READ')
    scaled_training = np.asarray(train.X)
    if pca_events[0]['array'] != array_signature(scaled_training):
        raise RuntimeError('PCA_GETTER_BUFFER_NOT_BOUND_TO_SAVED_TRAINING_X')
    components = train.varm['PCs']
    pca_center = np.asarray(train.X.mean(axis=0)).ravel()
    scaled = (normalized_selected - center) / scale
    coords = (scaled - pca_center) @ components
    training_pca = train.obsm['X_pca']
    if not np.allclose(coords[tr], training_pca, rtol=1e-8, atol=1e-8):
        raise RuntimeError('SCANPY_QUERY_PROJECTION_MISMATCH')
    budget_check(started)

    cdist_events = []
    neighbors = np.empty((len(qu), 30), dtype=np.int32)
    for begin in range(0, len(qu), 192):
        stop = min(begin + 192, len(qu))
        actual_query = coords[qu[begin:stop]]
        actual_training = coords[tr]
        ds = traced_cdist(actual_query, actual_training,
                          query_start=begin, query_stop=stop,
                          query_ids=ids[qu[begin:stop]], training_ids=ids[tr],
                          events=cdist_events)
        for offset, row in enumerate(ds):
            kth = np.partition(row, 29)[29]
            candidates = np.flatnonzero(row <= kth)
            ordering = np.lexsort((ids[tr[candidates]], row[candidates]))[:30]
            neighbors[begin + offset] = tr[candidates[ordering]]
    if np.isin(neighbors, qu).any():
        raise RuntimeError('QUERY_CELL_ENTERS_TRAINING_NEIGHBORHOOD')

    arrays = {
        'train': tr, 'query': qu, 'eligible': eligible, 'selected': chosen,
        'normalized_selected': normalized_selected,
        'scaled_training': scaled_training, 'center': center, 'scale': scale,
        'pca_center': pca_center, 'components': components,
        'coordinates': coords, 'neighbors': neighbors,
        'training_pca': training_pca, 'rna_totals': rna_totals,
        'normalization_totals': normalization_totals, **additional_arrays,
    }
    signatures = {name: array_signature(value) for name, value in arrays.items()}
    identity_values = {
        'cell_ids': ids, 'feature_ids': features.feature_id.to_numpy(),
        'feature_names': np.asarray(obj.var_names),
    }
    parameters = {
        'normalization': {'target_sum': 10000,
                          'exclude_highly_expressed': False, 'log1p': True},
        'filter': {'min_cells': minimum, 'training_only': True},
        'hvg': {'n_top_genes': n_top, 'flavor': 'seurat', 'n_bins': 20,
                'batch_key': None, 'subset': False},
        'scale': {'zero_center': True, 'max_value': None},
        'pca': {'n_comps': k, 'zero_center': True, 'svd_solver': 'arpack',
                'random_state': seed, 'dtype': 'float64', 'layer': None,
                'obsm': None},
        'neighbors': {'n_neighbors': 30, 'metric': 'sqeuclidean',
                      'tie_break': 'cell_id_lexicographic', 'chunk_size': 192},
    }
    trace = {
        'schema_version': 1, 'recipe_id': RECIPE_ID, 'assay_scope': 'RNA',
        'producer_code_sha256': file_hash(__file__),
        'mode': mode, 'dataset': 'D3', 'outer_context': 'O1',
        'array_signature_encoding': 'sha256(C-contiguous native-dtype bytes)',
        'identity_signature_encoding': 'uint64-big-endian-length-prefixed-UTF8',
        'array_signatures': signatures,
        'identities': {name: identity_signature(values)
                       for name, values in identity_values.items()},
        'normalization': normalization_events,
        'pca_get_obs_rep': pca_events,
        'pca_output': {'coordinates': array_signature(training_pca),
                       'components': array_signature(components)},
        'cdist_calls': cdist_events, 'source_files': source_files,
        'reference_source': reference, 'parameters': parameters,
        'wrong_layer_available_layers': list(train.layers.keys()),
        'environment': environment,
    }
    with h5py.File(out / 'state.h5', 'x') as handle:
        handle.attrs.update({
            'schema_version': 1, 'recipe_id': RECIPE_ID, 'assay_scope': 'RNA',
            'mode': mode, 'dataset': 'D3', 'outer_context': 'O1', 'seed': seed,
            'scanpy_version': environment['packages']['scanpy'],
        })
        for name, value in arrays.items():
            handle.create_dataset(name, data=value, compression='gzip',
                                   compression_opts=4, shuffle=True)
        for name, values in identity_values.items():
            handle.create_dataset(name, data=np.asarray(values, dtype=object),
                                   dtype=h5py.string_dtype(encoding='utf-8'))
    write_json(out / 'execution_trace.json', trace)
    # Check input bytes again so a changed source cannot inherit an earlier
    # intake identity. This is a fixed allowlist, never a directory traversal.
    for name, record in source_files.items():
        if file_hash(inputs / name) != record['sha256']:
            raise RuntimeError('INPUT_CHANGED_DURING_REPLAY:' + name)
    budget_check(started)
    output_files = {
        'state': file_record(out / 'state.h5'),
        'trace': file_record(out / 'execution_trace.json'),
        'producer': file_record(__file__), 'frozen_recipe': reference,
    }
    receipt = {
        'schema_version': 1, 'recipe_id': RECIPE_ID, 'mode': mode,
        'scope': SCOPE, 'status': 'PRODUCER_COMPLETED_NOT_A_GUARD_VERDICT',
        'dataset': 'D3', 'outer_context': 'O1', 'assay_scope': 'RNA',
        'version': environment['packages']['scanpy'],
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'source_files': source_files,
        'files': output_files, 'environment': environment,
        'timing': {'elapsed_seconds': time.perf_counter() - started},
        'peak_rss_bytes': peak_rss_bytes(),
        'dimensions': {'cells': n, 'RNA_features': len(features),
                       'training_cells': len(tr), 'query_cells': len(qu),
                       'eligible_features': len(eligible),
                       'selected_features': len(chosen), 'pcs': k,
                       'neighbors': 30},
        'parameters': parameters,
        'labels_instances_overlays_scores_targets_read': False,
        'biological_performance_evaluated': False,
        'limitations': [
            'Only frozen real D3/O1 RNA representation is in scope.',
            'Mode labels document execution and are not detector evidence.',
            'Receipt is non-self-hashing; external contract binds its bytes.',
        ],
    }
    write_json(out / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'mode': mode,
                      'out': str(out), 'elapsed_seconds':
                      receipt['timing']['elapsed_seconds'],
                      'peak_rss_bytes': receipt['peak_rss_bytes']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--mode', required=True,
                        choices=['clean', 'late_adt', 'wrong_layer'])
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = project_path(args.inputs, existing=True)
    out = project_path(args.out, existing=False)
    if out.exists() or args.out.is_symlink():
        raise RuntimeError('EXISTING_OUTPUT_REFUSED_NO_OVERWRITE:' + str(out))
    if not out.parent.is_dir():
        raise RuntimeError('OUTPUT_PARENT_MUST_BE_PREFLIGHTED_BY_ORCHESTRATOR')
    out.mkdir(exist_ok=False)
    try:
        run(inputs, out, args.mode, started)
    except Exception as error:
        write_json(out / 'failure.json', {
            'status': 'PRODUCER_FAILED_PARTIAL_OUTPUT_RETAINED',
            'mode': args.mode, 'scope': SCOPE,
            'exception_type': type(error).__name__, 'message': str(error),
            'elapsed_seconds': time.perf_counter() - started,
            'peak_rss_bytes': peak_rss_bytes(),
        })
        raise


if __name__ == '__main__':
    main()
