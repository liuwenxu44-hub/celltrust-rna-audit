"""Label-overlay diagnostics and robust cross-fitted M4 standardization."""

from __future__ import annotations

import math
import numpy as np


EPSILON = 1e-8


def apply_legal_neighbor_mask(weights, neighbor_indices, legal_mask):
    weights = np.asarray(weights, dtype=float).copy()
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    valid = neighbor_indices >= 0
    legal = np.zeros_like(valid)
    legal[valid] = legal_mask[neighbor_indices[valid]]
    weights[~legal] = 0.0
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    return weights, legal.sum(axis=1)


def diagnostics_from_neighbors(query_codes, neighbor_codes, weights, n_labels):
    query_codes = np.asarray(query_codes, dtype=np.int64)
    neighbor_codes = np.asarray(neighbor_codes, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if neighbor_codes.shape != weights.shape or len(query_codes) != len(weights):
        raise ValueError("M4_DIAGNOSTIC_SHAPE_MISMATCH")
    probabilities = np.zeros((len(query_codes), n_labels), dtype=np.float64)
    for label in range(n_labels):
        probabilities[:, label] = np.sum(weights * (neighbor_codes == label), axis=1)
    row_sum = probabilities.sum(axis=1)
    valid_query = (query_codes >= 0) & (query_codes < n_labels) & (row_sum > 0)
    same = np.full(len(query_codes), np.nan)
    same[valid_query] = probabilities[np.flatnonzero(valid_query), query_codes[valid_query]]
    alternative_matrix = probabilities.copy()
    alternative_matrix[np.flatnonzero(valid_query), query_codes[valid_query]] = -np.inf
    best_code = np.argmax(alternative_matrix, axis=1)
    best = alternative_matrix[np.arange(len(query_codes)), best_code]
    best[~valid_query | ~np.isfinite(best)] = np.nan
    margin = best - same
    if n_labels >= 2:
        entropy = -np.sum(probabilities * np.log(probabilities + EPSILON), axis=1) / math.log(n_labels)
    else:
        entropy = np.full(len(query_codes), np.nan)
    entropy[~valid_query] = np.nan
    concentration = best / (1.0 - same + EPSILON)
    effective = np.divide(1.0, np.sum(weights * weights, axis=1), out=np.full(len(weights), np.nan), where=np.sum(weights * weights, axis=1) > 0)
    return {
        "probabilities": probabilities,
        "S": same,
        "B": best,
        "G": margin,
        "H": entropy,
        "C_alt": concentration,
        "best_code": best_code,
        "effective_neighbor_count": effective,
    }


def robust_center_scale(values, pooled_values):
    values = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    pooled = np.asarray([x for x in pooled_values if np.isfinite(x)], dtype=float)
    if len(values) == 0:
        return None, None, "MISSING_SCALE_ESTIMATE"
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad > 0:
        return center, 1.4826 * mad + EPSILON, "MAD"
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr_scale = float((q3 - q1) / 1.349)
    if iqr_scale > 0:
        return center, iqr_scale + EPSILON, "IQR"
    if len(pooled):
        pooled_center = float(np.median(pooled))
        pooled_mad = float(np.median(np.abs(pooled - pooled_center)))
        if pooled_mad > 0:
            return center, 1.4826 * pooled_mad + EPSILON, "POOLED_MAD"
        pq1, pq3 = np.quantile(pooled, [0.25, 0.75])
        pooled_iqr = float((pq3 - pq1) / 1.349)
        if pooled_iqr > 0:
            return center, pooled_iqr + EPSILON, "POOLED_IQR"
    return center, None, "MISSING_SCALE_ESTIMATE"


def standardize_outer(inner_diag, outer_diag, inner_codes, outer_codes, minimum_inner_support=12):
    result = {key: np.full(len(outer_codes), np.nan) for key in ("Z_S", "Z_G", "Z_H")}
    statuses = {key: np.full(len(outer_codes), "", dtype=object) for key in ("S", "G", "H")}
    mappings = (("S", "Z_S", -1.0), ("G", "Z_G", 1.0), ("H", "Z_H", 1.0))
    for raw_key, z_key, direction in mappings:
        pooled = inner_diag[raw_key][np.isfinite(inner_diag[raw_key])]
        for label in sorted(set(int(x) for x in outer_codes if x >= 0)):
            ref = inner_diag[raw_key][(inner_codes == label) & np.isfinite(inner_diag[raw_key])]
            target = np.flatnonzero(outer_codes == label)
            if len(ref) < minimum_inner_support:
                statuses[raw_key][target] = "INNER_LABEL_SUPPORT_LT12"
                continue
            center, scale, status = robust_center_scale(ref, pooled)
            statuses[raw_key][target] = status
            if scale is None:
                continue
            if direction < 0:
                result[z_key][target] = (center - outer_diag[raw_key][target]) / scale
            else:
                result[z_key][target] = (outer_diag[raw_key][target] - center) / scale
    return result, statuses


def combine_multiscale(r15, r30, r50):
    values = np.vstack([r15, r30, r50]).T
    complete = np.all(np.isfinite(values), axis=1)
    median = np.full(len(values), np.nan)
    maximum = np.full(len(values), np.nan)
    median[complete] = np.median(values[complete], axis=1)
    maximum[complete] = np.max(values[complete], axis=1)
    return median, maximum
