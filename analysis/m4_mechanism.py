"""Accepted posthoc numerical diagnostics, without internal orchestration.
Functions consume supplied fixed representations/references, not new fits.
AA/AB/BA/BB are raw-first/reference-second diagnostics, not causal effects.
"""
from __future__ import annotations

import argparse

import dataclasses

import hashlib

import itertools

import json

import platform

import resource

import sys

import time

from pathlib import Path

import numpy as np

import pandas as pd

COMBOS = ("AA", "AB", "BA", "BB")

COMPONENTS = ("S", "G", "H")

LABELS = ("T_cell", "B_cell", "NK_cell", "monocyte")

STRATA = ("Within 10%", "Within 30%", "Systematic 10%", "Systematic 30%",
          "T_cell omitted", "B_cell omitted", "NK_cell omitted", "monocyte omitted")

SEED = 2026090702

N_BOOT = 2000

STATUS = "POSTHOC_MECHANISTIC_DIAGNOSTIC"

FALLBACK_GROUPS = ("ALL", "NO_EMPIRICAL_FALLBACK_EITHER_REFERENCE",
                   "EMPIRICAL_FALLBACK_REFERENCE_A_ONLY",
                   "EMPIRICAL_FALLBACK_REFERENCE_B_ONLY",
                   "EMPIRICAL_FALLBACK_BOTH_REFERENCES")

def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def ap_one(y, score, eligible_cells):
    """Exact stepwise/tie AP and undefined states from CT-MOD-001 evaluate.one.

    Inputs are already on an explicit finite support. An all-positive AP of 1 is
    retained for reconstruction but must not enter discrimination comparisons.
    """
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    require(len(y) == len(score), "AP_LENGTH_MISMATCH")
    require(np.isin(y, [0, 1]).all(), "AP_NONBINARY_FROZEN_TARGET")
    require(np.isfinite(score).all(), "AP_SUPPORT_CONTAINS_NONFINITE_SCORE")
    n = len(y)
    positives = int(y.sum())
    negatives = n - positives
    state = ("NO_FINITE_SCORE" if n == 0 else "UNDEFINED_NO_POSITIVES"
             if positives == 0 else "NO_DISCRIMINATION_INFORMATION"
             if negatives == 0 else "DEFINED")
    value = np.nan
    if n and positives:
        order = np.argsort(-score, kind="stable")
        ordered_score, ordered_y = score[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), n - 1]
        true_positive = np.cumsum(ordered_y)[ends]
        recall = true_positive / positives
        value = float(np.sum((recall - np.r_[0., recall[:-1]])
                             * true_positive / (ends + 1)))
    return {"AP": value, "AP_status": state, "scored_cells": n,
            "errors": positives, "correct_cells": negatives,
            "eligible_cells": int(eligible_cells),
            "excluded_cells": int(eligible_cells - n),
            "score_coverage": n / eligible_cells if eligible_cells else np.nan}

def same_query(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(np.array_equal(a, b), "CROSS_CONDITION_QUERY_IDENTITY_MISMATCH")
    require(len(np.unique(a)) == len(a), "DUPLICATE_QUERY_IDENTITY")

def cross_standardize(refs, raw, inner_codes, outer_codes, inner_ids, outer_ids,
                      standardizer):
    """No targets/eligibility are accepted here; only frozen method-side inputs."""
    return {combo: standardizer(refs[combo[1]], raw[combo[0]], inner_codes,
                                outer_codes, 12, inner_ids, outer_ids)
            for combo in COMBOS}

def affine_control(y, score):
    """Test the actual float64 2*s+1, including its tie partition, not just AP."""
    score = np.asarray(score, dtype=np.float64)
    require(np.isfinite(score).all(), "AFFINE_CONTROL_REQUIRES_EXPLICIT_FINITE_SET")
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = 2.0 * score + 1.0
    finite = bool(np.isfinite(transformed).all())
    order = np.argsort(-score, kind="stable")
    t_order = np.argsort(-transformed, kind="stable")
    order_preserved = finite and bool(np.array_equal(order, t_order))
    tie_preserved = finite and bool(np.array_equal(
        np.diff(score[order]) == 0, np.diff(transformed[order]) == 0))
    original_ap = ap_one(y, score, len(y))["AP"]
    transformed_ap = ap_one(y, transformed, len(y))["AP"] if finite else np.nan
    ap_equal = bool(np.isclose(original_ap, transformed_ap, rtol=0, atol=1e-15,
                              equal_nan=True))
    valid = finite and order_preserved and tie_preserved
    require(not valid or ap_equal, "AP_MONOTONE_INVARIANCE_IMPLEMENTATION_FAILURE")
    state = ("PASS_ORDER_TIES_AND_AP" if valid else
             "NOT_VERIFIABLE_FLOAT64_OVERFLOW" if not finite else
             "NOT_VERIFIABLE_FLOAT64_ORDER_OR_TIE_CHANGE")
    return {"transform": "2*s+1_FLOAT64", "cells": len(score),
            "AP_original": original_ap, "AP_transformed": transformed_ap,
            "finite": finite, "order_preserved": order_preserved,
            "ties_preserved": tie_preserved, "AP_equal": ap_equal,
            "control_status": state}

def self_test():
    """Toy-only contract tests: no project paths, files, targets, or models."""
    checks = {}
    checks["ties_at_one_threshold"] = ap_one([1, 0], [1., 1.], 2)["AP"] == .5
    checks["tied_positive_AP"] = ap_one([0, 1, 1], [.1, .9, .9], 3)["AP"] == 1.
    checks["no_positive_is_NA"] = np.isnan(ap_one([0, 0], [1., 2.], 2)["AP"])
    checks["all_positive_not_ranking"] = (ap_one([1, 1], [1., 2.], 2)["AP_status"]
                                           == "NO_DISCRIMINATION_INFORMATION")
    checks["affine_AP_and_ties"] = affine_control(
        np.array([1, 0, 1, 0]), np.array([3., 3., 1., -1.]))["control_status"] == "PASS_ORDER_TIES_AND_AP"
    tiny = np.nextafter(0., 1.)
    checks["floating_tie_collapse_detected"] = not affine_control(
        np.array([0, 1]), np.array([0., tiny]))["ties_preserved"]
    toy = np.array([[1., np.nan, 3.], [2., 5., np.nan]])
    support = np.isfinite(toy).all(axis=0)
    checks["common_support_no_imputation"] = (support.tolist() == [True, False, False]
                                               and int(np.isnan(toy).sum()) == 2)
    query = np.array(["q1", "q2"])
    same_query(query, query.copy())
    try:
        same_query(query, query[::-1])
    except RuntimeError:
        checks["query_permutation_rejected"] = True
    else:
        checks["query_permutation_rejected"] = False
    refs = {"A": {"S": np.array([1., 3.])}, "B": {"S": np.array([10., 14.])}}
    raw = {"A": {"S": np.array([5., 7.])}, "B": {"S": np.array([20., 25.])}}
    def toy_standardizer(ref, observed, *unused):
        return observed["S"] - np.mean(ref["S"])
    crossed = cross_standardize(refs, raw, [0, 0], [0, 0], ["i1", "i2"], query,
                                toy_standardizer)
    checks["raw_first_reference_second_mapping"] = all(np.array_equal(crossed[c], x)
        for c, x in {"AA": [3., 5.], "AB": [-7., -5.],
                     "BA": [18., 23.], "BB": [8., 13.]}.items())
    require(all(checks.values()), "TOY_SELF_TEST_FAILURE:" + repr(checks))
    print(json.dumps({"status": "PASS_TOY_ONLY", "checks": {k: bool(v) for k, v in checks.items()},
                      "project_data_read": False, "files_written": False}, indent=2))

def reconstruct_instance(ins, codes, reps, cells, diagnostics, standardizer, scale_audit):
    n, ids = len(cells), cells.cell_id.to_numpy()
    raw = np.full((2, n, 3), np.nan)
    standardized = np.full((4, n, 3), np.nan)
    used = np.zeros((4, n, 3), dtype=bool)
    statuses = np.full((4, n, 3), "", dtype=object)
    reasons = np.full((4, n, 3), "", dtype=object)
    reference_rows, reference_arrays = [], {}
    diag = {c: {context: diagnostics(codes[r["query"]], codes[r["neighbors"]],
                                    np.full(r["neighbors"].shape, 1 / 30), 4)
                for context, r in reps[c].items()} for c in "AB"}
    for outer in [f"O{i}" for i in range(1, 6)]:
        q = reps["A"][outer]["query"]
        inners = [f"{outer}_I{i}" for i in range(1, 5)]
        ix = np.concatenate([reps["A"][i]["query"] for i in inners])
        require(np.array_equal(np.sort(ix), np.sort(reps["A"][outer]["train"])), "INNER_REFERENCE_PARTITION_MISMATCH")
        require(not np.intersect1d(ix, q).size, "OUTER_QUERY_IN_INNER_REFERENCE")
        for c in "AB":
            same_query(ix, np.concatenate([reps[c][i]["query"] for i in inners]))
        refs = {c: {key: np.concatenate([diag[c][i][key] for i in inners]) for key in COMPONENTS} for c in "AB"}
        outer_raw = {c: {key: diag[c][outer][key] for key in COMPONENTS} for c in "AB"}
        crossed = cross_standardize(refs, outer_raw, codes[ix], codes[q], ids[ix], ids[q], standardizer)
        reference_arrays[f"{outer}__inner_query_indices"] = ix
        for ci, c in enumerate("AB"):
            raw[ci, q, :] = np.column_stack([outer_raw[c][key] for key in COMPONENTS])
            reference_arrays[f"{outer}__{c}__reference_SGH"] = np.column_stack([refs[c][key] for key in COMPONENTS])
            for key in COMPONENTS:
                for code, label in enumerate(LABELS):
                    mask = codes[ix] == code
                    audit = dataclasses.asdict(scale_audit(refs[c][key][mask], refs[c][key]))
                    reference_rows.append({"instance_id": ins.instance_id, "analysis_role": ins.analysis_role,
                        "stratum": ins.stratum, "outer_context": outer, "reference_condition": c,
                        "observed_label": label, "component": key, "query_cells": int((codes[q] == code).sum()),
                        "reference_support_ge12": audit["n"] >= 12,
                        "reference_source": "FOUR_FROZEN_INNER_QUERY_DIAGNOSTICS_OUTER_TRAIN_DONORS_ONLY", **audit})
        for j, combo in enumerate(COMBOS):
            z, state, why, fallback, _old, _old_status = crossed[combo]
            standardized[j, q, :] = np.column_stack([z["Z_" + key] for key in COMPONENTS])
            used[j, q, :] = np.column_stack([fallback[key] for key in COMPONENTS])
            statuses[j, q, :] = np.column_stack([state[key] for key in COMPONENTS])
            reasons[j, q, :] = np.column_stack([why[key] for key in COMPONENTS])
    # fmax is the authoritative M4 rule. One finite component is not NA imputation.
    risk = np.fmax(standardized[:, :, 0], standardized[:, :, 1])
    return raw, standardized, risk, used, statuses, reasons, reference_rows, reference_arrays

def declared_contrasts():
    rows = []
    for metric in ("M4_RISK", "Z_S", "Z_G"):
        for before, after in itertools.combinations(COMBOS, 2):
            rows.append({"contrast_id": f"{metric}__{after}_MINUS_{before}",
                         "terms": {f"{after}_{metric}": 1., f"{before}_{metric}": -1.},
                         "interpretation": "PAIRED_POSTHOC_2X2_COMPARISON_NOT_UNIQUE_CAUSAL_EFFECT"})
        rows.append({"contrast_id": f"{metric}__2X2_INTERACTION",
                     "terms": {f"BB_{metric}": 1., f"BA_{metric}": -1.,
                               f"AB_{metric}": -1., f"AA_{metric}": 1.},
                     "interpretation": "DESCRIPTIVE_NONADDITIVITY_NOT_UNIQUE_CAUSAL_DECOMPOSITION"})
    rows.append({"contrast_id": "NEIGHBOR_DISAGREEMENT__B_MINUS_A",
                 "terms": {"B_NEIGHBOR_DISAGREEMENT": 1., "A_NEIGHBOR_DISAGREEMENT": -1.},
                 "interpretation": "FROZEN_RAW_STRUCTURE_CHANGE"})
    for combo in COMBOS:
        for component in ("Z_S", "Z_G"):
            rows.append({"contrast_id": f"{combo}__M4_RISK_MINUS_{component}",
                         "terms": {f"{combo}_M4_RISK": 1., f"{combo}_{component}": -1.},
                         "interpretation": "MAX_COMBINATION_DIAGNOSTIC_NOT_CANDIDATE_COMPARISON"})
    rows.append({"contrast_id": "M4_CHANGE_MINUS_RAW_DISAGREEMENT_CHANGE",
                 "terms": {"BB_M4_RISK": 1., "AA_M4_RISK": -1.,
                           "B_NEIGHBOR_DISAGREEMENT": -1., "A_NEIGHBOR_DISAGREEMENT": 1.},
                 "interpretation": "DESCRIPTIVE_DIFFERENCE_OF_AP_CHANGES_NOT_CAUSAL_ATTRIBUTION"})
    return rows

def weighted_mean(x, weights):
    finite = np.isfinite(x)
    denominator = weights @ finite.astype(float)
    numerator = weights @ np.where(finite, x, 0.)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)

def summarize(donor_contrasts, donors, draws):
    weights = np.array([np.bincount(row, minlength=len(donors)) for row in draws], dtype=float)
    frame = pd.DataFrame(donor_contrasts)
    grouped = frame[frame.analysis_role == "PRIMARY"].groupby(
        ["support", "fallback_group", "contrast_id", "interpretation", "donor_id", "stratum"], sort=False)
    donor_rows = []
    for keys, group in grouped:
        finite = group.paired_AP_difference.notna()
        donor_rows.append(dict(zip(["support", "fallback_group", "contrast_id", "interpretation", "donor_id", "stratum"], keys),
                              paired_AP_difference=group.paired_AP_difference.mean(),
                              paired_valid_instances=int(finite.sum()),
                              selected_instances=len(group)))
    donor_frame = pd.DataFrame(donor_rows)
    summary = []
    for keys, group in donor_frame.groupby(["support", "fallback_group", "contrast_id", "interpretation"], sort=False):
        by_key = {(r.donor_id, r.stratum): r.paired_AP_difference for r in group.itertuples()}
        x = np.array([[by_key.get((donor, stratum), np.nan) for stratum in STRATA] for donor in donors])
        point = weighted_mean(x, np.ones((1, len(donors))))[0]
        boot = weighted_mean(x, weights)
        subsets = [(name, [i]) for i, name in enumerate(STRATA)] + [
            ("ALL_8_PRIMARY_STRATA", list(range(8))),
            ("ORDINARY_4_PRIMARY_STRATA", list(range(4))),
            ("OMITTED_4_PRIMARY_STRATA", list(range(4, 8)))]
        for name, indices in subsets:
            selected_point = point[indices]
            estimate = float(np.mean(selected_point)) if np.isfinite(selected_point).all() else np.nan
            selected_boot = boot[:, indices]
            valid = np.isfinite(selected_boot).all(axis=1)
            values = np.mean(selected_boot[valid], axis=1)
            lower, upper = np.quantile(values, [.025, .975]) if len(values) else (np.nan, np.nan)
            summary.append(dict(zip(["support", "fallback_group", "contrast_id", "interpretation"], keys),
                dataset="D4", stratum=name, paired_AP_difference=estimate,
                ci_lower=lower, ci_upper=upper, valid_bootstrap=len(values), requested_bootstrap=N_BOOT,
                donor_count=len(donors), donors_with_any_defined_selected_stratum=int(np.isfinite(x[:, indices]).any(axis=1).sum()),
                donors_complete_selected_strata=int(np.isfinite(x[:, indices]).all(axis=1).sum()),
                complete_stratum_mean_required=True, independent_unit="DONOR", evidence_status=STATUS,
                uncertainty="FIXED_PIPELINE_PAIRED_DONOR_RESAMPLING_NOT_FULL_TRAINING",
                inference_status="UNDEFINED_MISSING_STRATUM" if not np.isfinite(estimate) else "POSTHOC_DESCRIPTIVE_INTERVAL"))
    return donor_rows, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    self_test()
