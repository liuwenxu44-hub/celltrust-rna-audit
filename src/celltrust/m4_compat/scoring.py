"""M4 v0.2 zero-scale compatibility layer.

The frozen M4 v0.1 robust score is authoritative whenever it is finite.  This
module supplies a parameter-free empirical normal score only when the complete
v0.1 MAD -> IQR -> pooled-scale chain is unavailable for a finite, adequately
supported label-conditional reference distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from celltrust.m4.scoring import robust_center_scale, standardize_outer


MINIMUM_REFERENCE_SUPPORT = 12


@dataclass(frozen=True)
class ReferenceScaleAudit:
    n: int
    n_unique: int
    median: float | None
    mad: float | None
    iqr: float | None
    iqr_scale: float | None
    old_center: float | None
    old_scale: float | None
    old_status: str
    finite: bool


def finite_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def scale_audit(values, pooled_values) -> ReferenceScaleAudit:
    ref = finite_values(values)
    if len(ref) == 0:
        return ReferenceScaleAudit(0, 0, None, None, None, None, None, None, "MISSING_SCALE_ESTIMATE", False)
    median = float(np.median(ref))
    mad = float(np.median(np.abs(ref - median)))
    q25, q75 = np.quantile(ref, [0.25, 0.75])
    iqr = float(q75 - q25)
    center, scale, status = robust_center_scale(ref, finite_values(pooled_values))
    return ReferenceScaleAudit(
        n=len(ref),
        n_unique=int(np.unique(ref).size),
        median=median,
        mad=mad,
        iqr=iqr,
        iqr_scale=float(iqr / 1.349),
        old_center=center,
        old_scale=scale,
        old_status=status,
        finite=bool(np.all(np.isfinite(ref))),
    )


def mid_empirical_cdf(query, reference) -> np.ndarray:
    """Tie-aware mid empirical CDF with the frozen finite-sample boundary."""
    ref = finite_values(reference)
    if len(ref) == 0:
        return np.full(np.asarray(query).shape, np.nan, dtype=float)
    ordered = np.sort(ref)
    x = np.asarray(query, dtype=float)
    less = np.searchsorted(ordered, x, side="left")
    less_equal = np.searchsorted(ordered, x, side="right")
    equal = less_equal - less
    q = (less + 0.5 * equal + 0.5) / (len(ordered) + 1.0)
    q_min = 0.5 / (len(ordered) + 1.0)
    q_max = 1.0 - q_min
    q = np.clip(q, q_min, q_max)
    q[~np.isfinite(x)] = np.nan
    return q


def empirical_normal_score(query, reference, direction: str) -> np.ndarray:
    """Return a finite uncalibrated normal score in the frozen risk direction."""
    q = mid_empirical_cdf(query, reference)
    z = norm.ppf(q)
    if direction == "LOW_IS_HIGH_RISK":
        z = -z
    elif direction != "HIGH_IS_HIGH_RISK":
        raise ValueError("M4_COMPAT_INVALID_DIRECTION")
    return np.asarray(z, dtype=float)


def _old_status_name(status: str) -> str:
    if status == "MAD":
        return "ROBUST_MAD"
    if status == "IQR":
        return "ROBUST_IQR"
    if status in {"POOLED_MAD", "POOLED_IQR"}:
        return "ROBUST_POOLED"
    return status


def standardize_outer_compat(
    inner_diag: dict[str, np.ndarray],
    outer_diag: dict[str, np.ndarray],
    inner_codes,
    outer_codes,
    minimum_inner_support: int = MINIMUM_REFERENCE_SUPPORT,
    inner_clean_ids=None,
    outer_clean_ids=None,
):
    """Apply v0.1 exactly, then fill only eligible zero-scale components.

    The returned ``old`` arrays are produced by the frozen v0.1 function.  The
    compatibility arrays copy every finite v0.1 value without recomputation.
    """
    inner_codes = np.asarray(inner_codes, dtype=int)
    outer_codes = np.asarray(outer_codes, dtype=int)
    old, old_statuses = standardize_outer(
        inner_diag, outer_diag, inner_codes, outer_codes, minimum_inner_support
    )
    compat = {key: value.copy() for key, value in old.items()}
    statuses = {
        key: np.asarray([_old_status_name(str(x)) for x in old_statuses[key]], dtype=object)
        for key in ("S", "G", "H")
    }
    reasons = {key: np.full(len(outer_codes), "", dtype=object) for key in ("S", "G", "H")}
    used = {key: np.zeros(len(outer_codes), dtype=bool) for key in ("S", "G", "H")}
    mappings = (("S", "Z_S", "LOW_IS_HIGH_RISK"), ("G", "Z_G", "HIGH_IS_HIGH_RISK"), ("H", "Z_H", "HIGH_IS_HIGH_RISK"))
    pooled_by_key = {raw: finite_values(inner_diag[raw]) for raw, _, _ in mappings}
    if inner_clean_ids is not None:
        inner_clean_ids = np.asarray(inner_clean_ids, dtype=str)
    if outer_clean_ids is not None:
        outer_clean_ids = np.asarray(outer_clean_ids, dtype=str)

    for raw_key, z_key, direction in mappings:
        pooled = pooled_by_key[raw_key]
        for label in sorted(set(int(x) for x in outer_codes if x >= 0)):
            ref_mask = (inner_codes == label) & np.isfinite(inner_diag[raw_key])
            base_ref = np.asarray(inner_diag[raw_key][ref_mask], dtype=float)
            target = np.flatnonzero(outer_codes == label)
            audit = scale_audit(base_ref, pooled)
            for local_index in target:
                if np.isfinite(old[z_key][local_index]):
                    continue
                if audit.n < minimum_inner_support:
                    statuses[raw_key][local_index] = "MISSING_SCALE_ESTIMATE"
                    reasons[raw_key][local_index] = "REFERENCE_SUPPORT_LT12"
                    continue
                if not audit.finite:
                    statuses[raw_key][local_index] = "MISSING_SCALE_ESTIMATE"
                    reasons[raw_key][local_index] = "INVALID_REFERENCE_VALUES"
                    continue
                # The compatibility path is legal only after the complete v0.1
                # chain has returned no scale.  Normal nonzero scales never enter.
                if audit.old_scale is not None or audit.old_status != "MISSING_SCALE_ESTIMATE":
                    statuses[raw_key][local_index] = "MISSING_SCALE_ESTIMATE"
                    reasons[raw_key][local_index] = "OLD_SCORE_NONFINITE_WITH_VALID_SCALE"
                    continue
                ref = base_ref
                if inner_clean_ids is not None and outer_clean_ids is not None:
                    same = inner_clean_ids[ref_mask] == outer_clean_ids[local_index]
                    ref = ref[~same]
                if len(ref) < minimum_inner_support or not np.all(np.isfinite(ref)):
                    statuses[raw_key][local_index] = "MISSING_SCALE_ESTIMATE"
                    reasons[raw_key][local_index] = "REFERENCE_SUPPORT_OR_FINITE_GATE_FAILED_AFTER_DUPLICATE_EXCLUSION"
                    continue
                value = empirical_normal_score(
                    np.asarray([outer_diag[raw_key][local_index]], dtype=float), ref, direction
                )[0]
                if np.isfinite(value):
                    compat[z_key][local_index] = value
                    statuses[raw_key][local_index] = "EMPIRICAL_NORMAL_ZERO_SCALE"
                    reasons[raw_key][local_index] = ""
                    used[raw_key][local_index] = True
                else:
                    statuses[raw_key][local_index] = "MISSING_SCALE_ESTIMATE"
                    reasons[raw_key][local_index] = "EMPIRICAL_NORMAL_NONFINITE"
    return compat, statuses, reasons, used, old, old_statuses
