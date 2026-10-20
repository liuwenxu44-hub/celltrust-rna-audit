#!/usr/bin/env python3
"""Bounded D3 side audit; never replace main truth, refit thresholds, or score AP.

The hash-verified historical classifier is imported unchanged. Its threshold
fitter is replaced by an identity-checked lookup of the frozen exported values;
its bottom-one-percent quantile is replaced by legacy numeric cutoffs in BOTH
replays. A full-population exact categorical replay is a fail-closed prerequisite
to extracting fractional ADT values. Numeric margins are not claimed identical:
the historical threshold export has eight significant digits.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from collections import Counter

# Importing the historical module must not create a historical __pycache__.
sys.dont_write_bytecode = True

import h5py
import numpy as np
import yaml

STORAGE = Path(os.environ.get("CELLTRUST_AUDIT_ROOT", str(Path.cwd()))).resolve()
CANONICAL_ROOT = STORAGE / 'workspaces/CellTrust_R3'
MAX_SECONDS = 4 * 3600
MAX_BYTES = 1024 ** 3
MAX_MEMORY = 32 * 1024 ** 3
GENERATOR = 'scripts/m6_truth/derive_adt_truth.py'
CONFIG = 'configs/phase1/m6_adt_truth.yaml'
LEGACY = 'data/m6_prerequisite/interim/D3_adt_only.h5'
TRUTH = 'data/m6_prerequisite/truth/D3_adt_level1_truth.tsv'
THRESHOLDS = 'data/m6_prerequisite/truth/D3_adt_thresholds.tsv'
SUMMARY = 'data/m6_prerequisite/truth/D3_adt_truth_summary.json'
RAW = 'data/m6_prerequisite/raw/zenodo7555405_PBMC_vaccine_CITE.rds'
EXPECTED = {
    GENERATOR: '17b1d685e04da3f058a241046931a38e5c797e428021d352fa3da241adcd015c',
    CONFIG: 'dbf627bd36e662c655fe9e6334221b502b29f20cf32faf923b66c95927c8cf60',
    LEGACY: 'c0df61847370731f1fbf70cf496aa6ad413fe761ee29ea676db22fa9b0d975bd',
    TRUTH: '08d16f3bba5801c04856e8fa45014d20c091492ff2a1d18bd7f256df3ee259b6',
    THRESHOLDS: '4a1611a068f74509c790ab360b64b6d0af0c752ab025a2c838763e25fc370593',
    SUMMARY: '2a3c0047dd577084ddf57d7a1187c15c8a97cc7a3956050876111e8201588f26',
    RAW: 'b0da2cffb4fe1f5bb07087ce69f69fd2f3708fe5670ba5fc736eae2850cf98d4',
}
CATEGORICAL = ('cell_id', 'donor_id', 'sample_id', 'truth_level',
               'truth_label', 'truth_status', 'forced_assignment')
RESOLVED = 'RESOLVED_ADT_LEVEL1'
LIMITATIONS = [
    'EXPLORATORY_NONINDEPENDENT_ADT_TRUTH_SIDE_AUDIT_NOT_NEW_GROUND_TRUTH',
    'Historical marker thresholds were exported at eight significant digits; '
    'exact categorical legacy replay does not recover unexported full-precision '
    'thresholds or prove counterfactual equivalence at that unavailable precision.',
    'Both replays use original float32 classifier arithmetic and fixed legacy '
    'sample-specific numeric low-library cutoffs; no fractional quantile refit.',
    'All 175 ADT channels enter rowwise log1p centering. Fractional entries in '
    'CD72/fake-S1 can indirectly alter other marker CLR values within that cell.',
    'Selected 7200 cells, labels, donor splits, targets, and all main results '
    'remain frozen. Eligibility implications below are hypothetical flags only.',
    'No AP rescoring, uncertainty estimates, threshold fitting, model fitting, '
    'population reselection, or independent biological validation is performed.',
]


class NotVerifiable(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise NotVerifiable(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n')


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def tsv(path, data, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t',
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def check_budget(out, start):
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    require(size <= MAX_BYTES, 'STAGE_DISK_LIMIT_1_GIB')
    require(time.monotonic() - start <= MAX_SECONDS, 'FOUR_HOUR_TIME_LIMIT')
    return size


def decode(dataset):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                       for x in dataset], dtype=object)


def identity(path, expected=None, authority=None):
    observed = sha(path)
    require(expected is None or observed == expected, 'HASH_MISMATCH:' + str(path))
    return {'path': str(path), 'sha256': observed, 'bytes': path.stat().st_size,
            'authority': authority or 'current source identity'}


def read_inputs(root, out):
    mod2 = root / 'research/CT-MOD-002'
    freeze_path = mod2 / 'scientific_evidence_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    require(freeze['freeze_status'] == 'frozen', 'MOD2_NOT_FROZEN')
    frozen = {entry['path']: entry for entry in freeze['files']}
    need = ['intake/D3_prior_truth_dependency_identity.json',
            'intake/D3_raw_identity.json', 'intake/D3_count_semantics_audit.json',
            'D3/inputs/evaluation_cells.tsv', 'D3/inputs/selected_source_counts.h5']
    identities = [identity(freeze_path)]
    for rel in need:
        require(rel in frozen, 'MOD2_REQUIRED_SOURCE_NOT_FROZEN:' + rel)
        identities.append(identity(mod2 / rel, frozen[rel]['sha256'],
                                   str(freeze_path) + '#files[path=' + rel + ']'))
    prior = json.loads((mod2 / need[0]).read_text())
    require(prior['status'] == 'PASS', 'PRIOR_DEPENDENCY_AUDIT_NOT_PASS')
    prior_files = {x['path']: x for x in prior['files']}
    raw_receipt = json.loads((mod2 / need[1]).read_text())
    require(raw_receipt['sha256'] == EXPECTED[RAW] and
            raw_receipt['bytes'] == 1594039156, 'RAW_RECEIPT_CONTRACT')
    historical_path = root / 'evidence/CT-001H_output_sha256.tsv'
    history_rows = rows(historical_path)
    # Historical manifest schema uses relative_path (not a guessed path alias).
    path_key = next((k for k in ('relative_path', 'path', 'file_path')
                     if history_rows and k in history_rows[0]), None)
    require(path_key is not None, 'HISTORICAL_MANIFEST_PATH_SCHEMA')
    historical = {x[path_key]: x for x in history_rows}
    identities.append(identity(historical_path))
    for rel in (GENERATOR, CONFIG):
        require(rel in historical and historical[rel]['sha256'] == EXPECTED[rel],
                'HISTORICAL_CODE_CONFIG_AUTHORITY:' + rel)
    for rel in (LEGACY, TRUTH, THRESHOLDS, SUMMARY):
        require(prior_files[rel]['sha256'] == EXPECTED[rel],
                'PRIOR_TRUTH_AUTHORITY:' + rel)
    for rel, expected in EXPECTED.items():
        # One sequential full RDS hash is bounded IO, not rereading/fitting data.
        identities.append(identity(root / rel, expected,
            'CT-001H frozen output manifest / MOD2 frozen prior and raw identity'))
    for source in (Path(__file__).resolve(), Path(__file__).with_name('d3_fractional_entries.R').resolve()):
        identities.append(identity(source))
    dump(out / 'source_identity.json', {'status': 'PASS', 'files': identities,
         'path_rebinding': 'SERVER223_DIRECTORY_MAP_20260908_V2',
         'source_counts': 'raw RDS ADT slot; fractional coordinates extracted only after exact replay'})
    return mod2, identities


def replay(root, out, tag, adt_path, threshold_table, cutoff, sample_sizes):
    """Run unchanged, hash-bound classifier with two narrow pure replacements."""
    spec = importlib.util.spec_from_file_location('ct001h_d3_side_' + tag, root / GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen_thresholds, seen_cutoffs = [], []

    def frozen_threshold(x, min_sep):
        frame = inspect.currentframe().f_back
        key = (str(frame.f_locals['s']), str(frame.f_locals['name']))
        require(key not in seen_thresholds and key in threshold_table,
                'THRESHOLD_LOOKUP_IDENTITY_OR_DUPLICATE')
        require(len(x) == sample_sizes[key[0]], 'THRESHOLD_SAMPLE_POPULATION')
        seen_thresholds.append(key)
        item = threshold_table[key]
        return (float(item['threshold']), item['threshold_method'],
                item['convergence'] == 'TRUE', float(item['component_separation_sd']))

    class FixedCutoffNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def quantile(self, a, q, *args, **kwargs):
            require(q == .01 and not args and not kwargs, 'UNEXPECTED_QUANTILE_CALL')
            sample = str(inspect.currentframe().f_back.f_locals['s'])
            require(sample in cutoff and sample not in seen_cutoffs and
                    len(a) == sample_sizes[sample], 'LOW_GATE_SAMPLE_IDENTITY')
            seen_cutoffs.append(sample)
            return cutoff[sample]

    module.gmm_threshold = frozen_threshold
    module.np = FixedCutoffNumpy()
    truth_path = out / (tag + '_replayed_calls.tsv')
    old_argv = sys.argv
    sys.argv = [str(root / GENERATOR), '--adt-only-h5', str(adt_path),
                '--config', str(root / CONFIG), '--truth-out', str(truth_path),
                '--threshold-out', str(out / (tag + '_fixed_threshold_receipt.tsv')),
                '--summary-out', str(out / (tag + '_replay_summary.json'))]
    try:
        with (out / (tag + '_replay.log')).open('w') as handle, contextlib.redirect_stdout(handle):
            module.main()
    finally:
        sys.argv = old_argv
    require(set(seen_thresholds) == set(threshold_table) and
            set(seen_cutoffs) == set(cutoff), 'REPLAY_LOOKUP_COVERAGE')
    return rows(truth_path)


def compare_reference(original, replayed, out):
    require(len(original) == len(replayed) == 113897, 'FULL_POPULATION_SIZE')
    mismatch = []
    field_counts = Counter()
    for old, new in zip(original, replayed):
        different = [k for k in CATEGORICAL if old[k] != new[k]]
        field_counts.update(different)
        if different:
            mismatch.append({'cell_id': old['cell_id'], 'fields': '|'.join(different),
                'original_label': old['truth_label'], 'replayed_label': new['truth_label'],
                'original_status': old['truth_status'], 'replayed_status': new['truth_status']})
    tsv(out / 'legacy_replay_mismatches.tsv', mismatch,
        ['cell_id', 'fields', 'original_label', 'replayed_label', 'original_status', 'replayed_status'])
    result = {'population_cells': len(original), 'mismatch_cells': len(mismatch),
              'field_mismatches': dict(field_counts), 'categorical_fields': list(CATEGORICAL),
              'numeric_margin_string_mismatches': sum(a['top_margin'] != b['top_margin']
                                                     for a, b in zip(original, replayed)),
              'evidence_string_mismatches': sum(a['adt_evidence'] != b['adt_evidence']
                                                for a, b in zip(original, replayed)),
              'full_precision_threshold_recovery_claimed': False}
    dump(out / 'legacy_replay_gate.json', result)
    require(not mismatch, 'NOT_VERIFIABLE_EXPORTED_THRESHOLDS_CANNOT_EXACTLY_REPLAY_LEGACY_CATEGORICAL_TRUTH')
    return result


def write_comparisons(original, integer, fractional, selected, changed, cutoff,
                      old_lib, new_lib, cfg, out):
    require(len(original) == len(integer) == len(fractional) == 113897,
            'FRACTIONAL_FULL_POPULATION_COVERAGE')
    selected_ids = {r['cell_id'] for r in selected}
    require(len(selected_ids) == len(selected) == 7200, 'FIXED_SELECTED_POPULATION')
    all_ids = {r['cell_id']: i for i, r in enumerate(original)}
    for r in selected:
        i = all_ids[r['cell_id']]
        require(int(r['source_cell_index']) == i and r['clean_label'] == original[i]['truth_label']
                and r['donor_id'] == original[i]['donor_id'] and
                r['sample_id'] == original[i]['sample_id'], 'FIXED_SELECTED_IDENTITY')
    data, summary = [], Counter()
    for i, (a, b, c) in enumerate(zip(original, integer, fractional)):
        require(all(a[k] == c[k] for k in ('cell_id', 'donor_id', 'sample_id',
                                          'truth_level', 'forced_assignment')),
                'FRACTIONAL_QUERY_IDENTITY')
        require(i in changed or (a['truth_label'] == c['truth_label'] and
                                a['truth_status'] == c['truth_status']),
                'UNCHANGED_COUNT_CELL_CHANGED_UNDER_FIXED_ROWWISE_RULES')
        old_ok = a['truth_status'] == RESOLVED and a['truth_label'] in cfg['primary_labels']
        new_ok = c['truth_status'] == RESOLVED and c['truth_label'] in cfg['primary_labels']
        item = {'source_cell_index': i, 'cell_id': a['cell_id'], 'donor_id': a['donor_id'],
                'sample_id': a['sample_id'], 'fixed_selected': int(a['cell_id'] in selected_ids),
                'fractional_channels': '|'.join(sorted(changed.get(i, []))),
                'original_label': a['truth_label'], 'integer_label': b['truth_label'],
                'fractional_label': c['truth_label'], 'original_status': a['truth_status'],
                'integer_status': b['truth_status'], 'fractional_status': c['truth_status'],
                'label_discordant': int(a['truth_label'] != c['truth_label']),
                'status_discordant': int(a['truth_status'] != c['truth_status']),
                'original_resolved_eligible': int(old_ok), 'fractional_resolved_eligible': int(new_ok),
                'resolved_eligibility_changed': int(old_ok != new_ok),
                'legacy_library_float32': float(old_lib[i]),
                'fractional_library_float32': float(new_lib[i]),
                'fixed_legacy_low_cutoff': float(cutoff[a['sample_id']]),
                'legacy_low_flag': int(old_lib[i] <= cutoff[a['sample_id']]),
                'fractional_fixed_cutoff_low_flag': int(new_lib[i] <= cutoff[a['sample_id']])}
        data.append(item)
        for scope in ('ALL_ORIGINAL', 'FIXED_SELECTED' if item['fixed_selected'] else 'NOT_SELECTED'):
            for unit, value in [('overall', 'ALL'), ('donor', a['donor_id']),
                                ('sample', a['sample_id']), ('original_label', a['truth_label'] or 'NO_LABEL'),
                                ('fractional_channel_set', item['fractional_channels'] or 'NONE')]:
                for metric, value_count in [('cells', 1), ('label_discordant', item['label_discordant']),
                        ('status_discordant', item['status_discordant']),
                        ('resolved_eligibility_changed', item['resolved_eligibility_changed']),
                        ('low_gate_changed', int(item['legacy_low_flag'] != item['fractional_fixed_cutoff_low_flag']))]:
                    summary[(scope, unit, value, metric)] += value_count
    tsv(out / 'cell_label_comparison.tsv', data, list(data[0]))
    grouped = [dict(zip(['scope', 'group_type', 'group', 'metric'], key), count=value)
               for key, value in sorted(summary.items())]
    tsv(out / 'grouped_discordance.tsv', grouped, ['scope', 'group_type', 'group', 'metric', 'count'])
    channel_rows = []
    for channel in ('CD72', 'fake-S1'):
        for scope in ('ALL_ORIGINAL', 'FIXED_SELECTED'):
            exposed = [r for r in data if channel in r['fractional_channels'].split('|')
                       and (scope == 'ALL_ORIGINAL' or r['fixed_selected'])]
            channel_rows.append({'channel': channel, 'scope': scope, 'exposed_cells': len(exposed),
                'label_discordant': sum(r['label_discordant'] for r in exposed),
                'status_discordant': sum(r['status_discordant'] for r in exposed),
                'resolved_eligibility_changed': sum(r['resolved_eligibility_changed'] for r in exposed),
                'interpretation': 'overlapping_exposures_not_independent_channel_effects'})
    tsv(out / 'channel_exposure_discordance.tsv', channel_rows, list(channel_rows[0]))
    donor_label = []
    for scope in ('ALL_ORIGINAL', 'FIXED_SELECTED'):
        for branch, branch_rows in [('ORIGINAL_FROZEN', original), ('INTEGER_REPLAY', integer),
                                    ('FRACTIONAL_FIXED_RULE_SIDE_AUDIT', fractional)]:
            counts = Counter((r['donor_id'], r['truth_label']) for r in branch_rows
                if r['truth_status'] == RESOLVED and r['truth_label'] in cfg['primary_labels']
                and (scope == 'ALL_ORIGINAL' or r['cell_id'] in selected_ids))
            for donor in sorted({r['donor_id'] for r in original}):
                for label in cfg['primary_labels']:
                    n = counts[(donor, label)]
                    donor_label.append({'scope': scope, 'branch': branch, 'donor_id': donor,
                                        'label': label, 'cells': n,
                                        'hypothetical_ge20_flag': int(n >= cfg['quality_gates']['primary_cells_per_donor_label_min'])})
    tsv(out / 'donor_label_eligibility_implications.tsv', donor_label, list(donor_label[0]))
    gate_rows = []
    quality = cfg['quality_gates']
    for branch, branch_rows in [('ORIGINAL_FROZEN', original), ('INTEGER_REPLAY', integer),
                                ('FRACTIONAL_FIXED_RULE_SIDE_AUDIT', fractional)]:
        valid_fraction = sum(r['truth_status'] in (RESOLVED, 'OUTSIDE_PRIMARY_LABEL_SET')
                             for r in branch_rows) / len(branch_rows)
        for label in cfg['primary_labels']:
            matching = [r for r in donor_label if r['scope'] == 'ALL_ORIGINAL'
                        and r['branch'] == branch and r['label'] == label]
            n_cells = sum(r['cells'] for r in matching)
            n_donors = sum(r['hypothetical_ge20_flag'] for r in matching)
            gate_rows.append({'branch': branch, 'label': label, 'cells': n_cells,
                'donors_ge20': n_donors, 'resolved_or_explicit_nonprimary_fraction': valid_fraction,
                'primary_donor_gate': int(n_donors >= quality['primary_donors_per_label_min']),
                'primary_cell_gate': int(n_cells >= quality['primary_cells_per_label_min']),
                'primary_fraction_gate': int(valid_fraction >= quality['primary_resolved_or_explicit_nonprimary_fraction_min']),
                'external_total_donor_gate': int(len({r['donor_id'] for r in branch_rows}) >= quality['external_donors_min']),
                'external_label_donor_gate': int(n_donors >= quality['external_donors_per_label_min']),
                'external_label_cell_gate': int(n_cells >= quality['external_cells_per_label_min']),
                'role': 'D3_external_only_primary_flags_descriptive_not_role_promotion'})
    tsv(out / 'population_quality_gate_implications.tsv', gate_rows, list(gate_rows[0]))
    return {metric: summary[('ALL_ORIGINAL', 'overall', 'ALL', metric)]
            for metric in ('cells', 'label_discordant', 'status_discordant',
                           'resolved_eligibility_changed', 'low_gate_changed')}


def run(root, out, start):
    require(np.__version__ == '2.3.2' and h5py.__version__ == '3.14.0' and yaml.__version__ == '6.0.2',
            'HISTORICAL_NUMPY_H5PY_YAML_ENVIRONMENT_REQUIRED')
    mod2, sources = read_inputs(root, out)
    cfg = yaml.safe_load((root / CONFIG).read_text())
    original = rows(root / TRUTH)
    with h5py.File(root / LEGACY, 'r') as handle:
        raw_shape = handle['counts/matrix'].shape
        stored = np.asarray(handle['counts/matrix'], dtype=np.float32)
        ids, donor, samples, antibodies = [decode(handle[key]) for key in
            ('cells/cell_id', 'cells/donor_id', 'cells/sample_id', 'antibodies/name')]
    counts = stored.T if raw_shape == (len(antibodies), len(ids)) else stored
    require(counts.shape == (113897, 175) and len(set(donor)) == 6 and len(set(samples)) == 24,
            'ORIGINAL_ADT_POPULATION')
    require(all(r['cell_id'] == ids[i] and r['donor_id'] == donor[i] and
                r['sample_id'] == samples[i] for i, r in enumerate(original)), 'LEGACY_TRUTH_IDENTITY')
    old_lib = counts.sum(1)
    cutoff = {s: np.quantile(old_lib[samples == s], .01) for s in sorted(set(samples))}
    sample_sizes = {s: int(np.sum(samples == s)) for s in cutoff}
    cutoff_rows = [{'sample_id': s, 'original_cells': sample_sizes[s],
                    'cutoff': float(cutoff[s]), 'rule': 'float32_legacy_library_sum_quantile_0.01_inclusive',
                    'fractional_cutoff_refit': False} for s in cutoff]
    tsv(out / 'fixed_legacy_quality_cutoffs.tsv', cutoff_rows, list(cutoff_rows[0]))
    threshold_rows = rows(root / THRESHOLDS)
    thresholds = {(r['sample_donor'], r['antibody']): r for r in threshold_rows}
    require(len(thresholds) == len(threshold_rows) == 24 * 175 and set(thresholds) ==
            {(s, a) for s in cutoff for a in antibodies}, 'THRESHOLD_IDENTITY_COVERAGE')
    require(all(r['normalization'] == 'WITHIN_SAMPLE_CLR' and np.isfinite(float(r['threshold']))
                for r in threshold_rows), 'THRESHOLD_NORMALIZATION_OR_FINITE')
    integer = replay(root, out, 'integer', root / LEGACY, thresholds, cutoff, sample_sizes)
    gate = compare_reference(original, integer, out)
    check_budget(out, start)
    scratch = out / 'runtime_tmp'
    scratch.mkdir()
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    script_r = Path(__file__).resolve().with_name('d3_fractional_entries.R')
    with (out / 'fractional_extraction.log').open('w') as handle:
        subprocess.run(['Rscript', '--vanilla', str(script_r), str(root), str(out)],
            check=True, cwd=out, env=env, stdout=handle, stderr=subprocess.STDOUT,
            timeout=max(1, MAX_SECONDS - int(time.monotonic() - start)))
    ledger = rows(out / 'fractional_entries.tsv')
    require(len(ledger) == 544, 'FRACTIONAL_COORDINATE_COUNT')
    changed, positions = {}, set()
    float32_rounding_count = 0
    for item in ledger:
        i, j = int(item['source_cell_index']), int(item['antibody_index'])
        require((i, j) not in positions and 0 <= i < len(ids) and 0 <= j < len(antibodies),
                'FRACTIONAL_COORDINATE_IDENTITY')
        positions.add((i, j))
        value = float(item['fractional_value'])
        require(item['cell_id'] == ids[i] and item['antibody'] == antibodies[j]
                and item['donor_id'] == donor[i] and item['sample_id'] == samples[i]
                and counts[i, j] == float(item['legacy_integer']) == np.trunc(value)
                and np.isfinite(value) and value >= 0 and value != np.floor(value),
                'FRACTIONAL_LEDGER_VALUE_OR_IDENTITY')
        counts[i, j] = np.float32(value)
        float32_rounding_count += int(float(np.float32(value)) != value)
        changed.setdefault(i, []).append(str(antibodies[j]))
    require(Counter(r['antibody'] for r in ledger) == {'CD72': 272, 'fake-S1': 272},
            'FRACTIONAL_CHANNEL_COUNTS')
    selected = rows(mod2 / 'D3/inputs/evaluation_cells.tsv')
    selection_index = np.asarray([int(r['source_cell_index']) for r in selected], dtype=int)
    # Independently bind the patched full source to the already-frozen 7200-cell export.
    with h5py.File(mod2 / 'D3/inputs/selected_source_counts.h5', 'r') as handle:
        group = handle['ADT']
        require(list(decode(group['cell_ids'])) == [r['cell_id'] for r in selected]
                and list(decode(group['feature_names'])) == list(antibodies), 'SELECTED_H5_IDENTITY')
        from scipy.sparse import csr_matrix
        dims = np.asarray(group['Dim'])
        selected_source = csr_matrix((np.asarray(group['x']), np.asarray(group['i']),
                                     np.asarray(group['p'])), shape=(int(dims[1]), int(dims[0])))
        require(np.array_equal(counts[selection_index], selected_source.toarray().astype(np.float32)),
                'PATCHED_FULL_SOURCE_VS_FROZEN_SELECTED_SOURCE')
    selected_ids = {r['cell_id'] for r in selected}
    require(sum(r['cell_id'] in selected_ids for r in ledger) == 98,
            'SELECTED_FRACTIONAL_ENTRY_COUNT')
    dump(out / 'fractional_patch_receipt.json', {
        'ledger_sha256': sha(out / 'fractional_entries.tsv'),
        'fractional_entries': len(ledger), 'fractional_cells': len(changed),
        'selected_fractional_entries': 98,
        'float64_to_float32_changed_entries': float32_rounding_count,
        'input_precision': '17_digit_float64_ledger',
        'classifier_precision': 'historical_float32_preserved',
        'stored_axis_shape': list(raw_shape),
        'patched_full_source_equals_frozen_selected_float32': True,
        'no_unlisted_count_positions_modified': True})
    check_budget(out, start)
    # Preserve the original on-disk axis orientation and float32 calculation layout.
    fractional_path = out / 'D3_fractional_adt_only.h5'
    with h5py.File(fractional_path, 'w') as handle:
        handle.create_dataset('counts/matrix', data=stored, compression='gzip', compression_opts=1)
        dt = h5py.string_dtype('utf-8')
        for key, value in [('cells/cell_id', ids), ('cells/donor_id', donor),
                           ('cells/sample_id', samples), ('antibodies/name', antibodies)]:
            handle.create_dataset(key, data=value, dtype=dt)
    new_lib = counts.sum(1)
    fractional = replay(root, out, 'fractional', fractional_path, thresholds, cutoff, sample_sizes)
    result = write_comparisons(original, integer, fractional, selected, changed, cutoff,
                               old_lib, new_lib, cfg, out)
    # No source file is modified; close the same identity chain at completion.
    for item in sources:
        require(sha(item['path']) == item['sha256'], 'INPUT_DRIFT_DURING_AUDIT:' + item['path'])
    return {'status': 'PASS_WITH_LIMITATIONS', 'legacy_categorical_replay': gate,
            'discordance': result, 'fractional_entries': len(ledger),
            'fractional_cells': len(changed), 'selected_cells_unchanged': 7200,
            'threshold_fits': 0, 'fractional_quality_quantile_fits': 0,
            'main_truth_writes': 0, 'AP_rescoring': 0,
            'stage_bytes': check_budget(out, start)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    require(root == CANONICAL_ROOT, 'CANONICAL_223_ROOT_REQUIRED')
    require(out.is_relative_to(root / 'research/CT-MOD-004') and
            out != root / 'research/CT-MOD-004', 'NEW_MOD004_SUBDIRECTORY_REQUIRED')
    require(not out.exists(), 'OUTPUT_ALREADY_EXISTS_NO_OVERWRITE')
    # Execution uses the 223 orchestrator and V2 rebind, not a new hostname gate.
    subprocess.run([sys.executable, str(STORAGE / '.directory-policy/check_output_paths.py'),
                    'CellTrust', str(out)], check=True)
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    def timeout_handler(signum, frame):
        raise NotVerifiable('FOUR_HOUR_TIME_LIMIT')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_SECONDS)
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    report = {'audit_id': 'CT-MOD-004_WP3_D3_FIXED_CLASSIFIER_SIDE_AUDIT',
              'limitations': LIMITATIONS, 'source_authoring_only_until_223_execution': False,
              'environment': {'python': sys.version, 'executable': sys.executable,
                              'numpy': np.__version__, 'h5py': h5py.__version__,
                              'yaml': yaml.__version__, 'platform': platform.platform()},
              'resource_caps': {'wall_seconds': MAX_SECONDS, 'address_space_bytes': MAX_MEMORY,
                                'stage_bytes': MAX_BYTES}}
    exit_code = 0
    try:
        report.update(run(root, out, start))
    except Exception as exc:
        report.update(status='NOT_VERIFIABLE', reason=type(exc).__name__ + ':' + str(exc),
                      main_truth_writes=0, AP_rescoring=0, threshold_fits=0)
        exit_code = 2
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - start
        dump(out / 'audit_report.json', report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
