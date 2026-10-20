"""Paired donor-cluster summaries of fixed fitted scores; no model refitting."""

import numpy as np


def paired_effects(donor_by_stratum, *, seed, replicates=2000):
    """Equal-weight donors within stratum, then equal-weight all strata.

    Input differences must already average valid paired replicate metrics within
    donor/stratum on the prescribed common finite support. Undefined strata make
    an overall replicate undefined; finite values are never zero-imputed.
    """
    x = np.asarray(donor_by_stratum, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("EXPECTED_AT_LEAST_TWO_DONORS_AND_ONE_STRATUM")
    if np.isinf(x).any() or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("INVALID_VALUES_OR_BOOTSTRAP_COUNT")
    nd = x.shape[0]
    draws = np.random.default_rng(seed).integers(0, nd, size=(replicates, nd))
    weights = np.array([np.bincount(row, minlength=nd) for row in draws], float)
    point = weighted(x, np.ones((1, nd)))[0]
    boot = weighted(x, weights)
    overall = float(np.mean(point)) if np.isfinite(point).all() else np.nan
    overall_boot = np.where(np.isfinite(boot).all(axis=1), np.mean(boot, axis=1), np.nan)
    point = np.r_[point, overall]
    boot = np.column_stack([boot, overall_boot])
    intervals = np.asarray([ci(boot[:, j]) for j in range(boot.shape[1])])
    return {"estimate": point, "ci_lower": intervals[:, 0],
            "ci_upper": intervals[:, 1], "valid_bootstrap": intervals[:, 2].astype(int),
            "bootstrap_draws": draws, "refitted": False}

# weighted() and ci() are appended unchanged from the accepted evaluator by
# the source-release builder. They have no access to targets or file paths.


def weighted(x,w):
 fin=np.isfinite(x);den=w@fin.astype(float);num=w@np.where(fin,x,0)
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)



def ci(a):
 a=a[np.isfinite(a)];return (*np.quantile(a,[.025,.975]),len(a)) if len(a) else (np.nan,np.nan,0)

