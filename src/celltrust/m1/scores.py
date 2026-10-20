"""Frozen uncalibrated M1 risk formula."""

from __future__ import annotations

import math
import numpy as np

EPSILON = 1e-8


def _quantile(x, q):
    return float(np.quantile(np.asarray(x, dtype=float), q, method="linear"))


def robust_reference(values, pooled_values=None, epsilon=EPSILON):
    clean = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not len(clean):
        return None, None, "MISSING_SCALE_ESTIMATE"
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    if mad > 0:
        return center, 1.4826 * mad + epsilon, "MAD"
    iqr_scale = (_quantile(clean, 0.75) - _quantile(clean, 0.25)) / 1.349
    if iqr_scale > 0:
        return center, iqr_scale + epsilon, "IQR_OVER_1.349"
    pooled = np.asarray([] if pooled_values is None else [float(v) for v in pooled_values if v is not None and math.isfinite(float(v))])
    if len(pooled):
        pooled_center = float(np.median(pooled))
        pooled_mad = float(np.median(np.abs(pooled - pooled_center)))
        if pooled_mad > 0:
            return center, 1.4826 * pooled_mad + epsilon, "POOLED_MAD"
        pooled_iqr = (_quantile(pooled, 0.75) - _quantile(pooled, 0.25)) / 1.349
        if pooled_iqr > 0:
            return center, pooled_iqr + epsilon, "POOLED_IQR_OVER_1.349"
    return center, None, "MISSING_SCALE_ESTIMATE"


def combine_risk(d_obs, d_alt, abs_ref, margin_ref, epsilon=EPSILON):
    abs_center, abs_scale, abs_status = abs_ref
    if abs_scale is None:
        return {"A": None, "M": None, "R": None, "margin_raw": None, "status": "MISSING_SCALE_ESTIMATE", "abs_scale_status": abs_status, "margin_scale_status": "NOT_EVALUATED"}
    a = (float(d_obs) - abs_center) / abs_scale
    if d_alt is None or not math.isfinite(float(d_alt)):
        return {"A": a, "M": None, "R": a, "margin_raw": None, "status": "OK_NO_ALTERNATIVE", "abs_scale_status": abs_status, "margin_scale_status": "NA_NO_ALTERNATIVE"}
    margin_raw = math.log((float(d_obs) + epsilon) / (float(d_alt) + epsilon))
    margin_center, margin_scale, margin_status = margin_ref
    if margin_scale is None:
        return {"A": a, "M": None, "R": a, "margin_raw": margin_raw, "status": "OK_MARGIN_MISSING_SCALE", "abs_scale_status": abs_status, "margin_scale_status": margin_status}
    m = (margin_raw - margin_center) / margin_scale
    return {"A": a, "M": m, "R": max(a, m), "margin_raw": margin_raw, "status": "OK", "abs_scale_status": abs_status, "margin_scale_status": margin_status}
