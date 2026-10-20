#!/usr/bin/env python3
"""Independent, read-only checker for the frozen real D3/O1 RNA path.

Run on server223: python -B realpath_guard.py --contract /absolute/contract.json
stdout is one JSON report. Exit 0=PASS, 2=FAIL, 3=NOT_VERIFIABLE.

No Scanpy, AnnData, producer, labels, targets, or performance files are imported
or read. The checker independently implements the declared transformation with
NumPy/SciPy/pandas, including Seurat-flavour dispersion bins. Array signatures
bind the captured *actual* PCA getter and cdist operands/results. Instrumentation
and contracts still need an externally trusted freeze: matching hashes are not
proof against a dishonest producer or a forged future contract. These bounded
local scientific artifacts are not a hostile-file security sandbox.

The supported source is one frozen D3 count export, not arbitrary cohorts or
multimodal recipes. PASS concerns this representation path, never label quality,
biological accuracy, prevalence, or general guard sensitivity. Missing lineage,
unsupported recipes/sizes and unresolved PCA numerical conditions abstain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

# This executable's numerical work is deliberately bounded; the orchestrator
# also imposes the approved process-level CPU/RAM/time limits on server223.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, sparse
from scipy.spatial.distance import cdist


RECIPE_ID = "d3-o1-scanpy-reference-v1"
PROJECT_ROOT = Path(os.environ.get("CELLTRUST_AUDIT_ROOT", str(Path.cwd()))).resolve()
SOURCE_HASHES = {
    "RNA_counts.npz": "464cad79562bc0f638c45f6b275e44926c8a8bb5137f93a715e89ca5bbe17e00",
    "RNA_features.tsv": "22f39498279d969f9382bf89b21e2202703198a387cc9e0d216f9d7dca802bf2",
    "method_cells.tsv": "1fe0fc86410c55e666c48d03c2ec426c0ce133973bf45454407ac86c065f04d4",
}
REFERENCE_SHA256 = "3e675f87aab882a2cf6a6b0fe4571e8e665c1caaf97bbb543b78f2f5508a9b6e"
MAX_CELLS, MAX_FEATURES, MAX_HVG = 8000, 22000, 2500
EXPRESSION_RTOL, EXPRESSION_ATOL = 1e-8, 1e-10
PROJECTION_RTOL, PROJECTION_ATOL = 1e-8, 1e-8
SUBSPACE_TOL, ORTHOGONALITY_TOL, EIGENGAP_TOL = 1e-6, 1e-8, 1e-8
REQUIRED_ARRAYS = (
    "train", "query", "eligible", "selected", "normalized_selected",
    "scaled_training", "center", "scale", "pca_center", "components",
    "coordinates", "neighbors", "training_pca", "rna_totals",
    "normalization_totals",
)
IDENTITY_ARRAYS = ("cell_ids", "feature_ids", "feature_names")


class Decision(Exception):
    def __init__(self, status, reason, **details):
        super().__init__(reason)
        self.status, self.reason, self.details = status, reason, details


def fail(reason, **details):
    raise Decision("FAIL", reason, **details)


def abstain(reason, **details):
    raise Decision("NOT_VERIFIABLE", reason, **details)


def require(mapping, key, location):
    if not isinstance(mapping, dict) or key not in mapping:
        abstain("REQUIRED_EVIDENCE_MISSING", location=location, field=key)
    return mapping[key]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("AMBIGUOUS_JSON_DUPLICATE_KEY", key=key)
        result[key] = value
    return result


def read_json(path, limit=8 << 20):
    if not path.is_file():
        abstain("REQUIRED_FILE_MISSING", path=str(path))
    if path.stat().st_size > limit:
        abstain("UNSUPPORTED_JSON_SIZE", path=str(path), maximum_bytes=limit)
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=reject_duplicate_keys)
    except (ValueError, UnicodeError) as exc:
        fail("MALFORMED_JSON", path=str(path), error=str(exc))
    if not isinstance(value, dict):
        fail("JSON_ROOT_NOT_OBJECT", path=str(path))
    return value


def array_signature(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest()}


def identity_signature(values):
    digest = hashlib.sha256()
    for item in values:
        encoded = str(item).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return {"count": len(values), "sha256": digest.hexdigest()}


def assert_signature(record, expected, location):
    if not isinstance(record, dict):
        abstain("SIGNATURE_MISSING", location=location)
    for key, value in expected.items():
        observed = require(record, key, location)
        if observed != value:
            fail("EXECUTION_SIGNATURE_CONFLICT", location=location, field=key,
                 expected=value, observed=observed)


def exact(actual, expected, location):
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        fail("IDENTITY_OR_INDEX_CONFLICT", location=location,
             actual_shape=list(actual.shape), expected_shape=list(expected.shape))


def numeric(actual, expected, location, *, projection=False):
    if actual.shape != expected.shape:
        fail("SHAPE_CONFLICT", location=location,
             actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        fail("NONFINITE_NUMERIC_ARRAY", location=location)
    rtol, atol = ((PROJECTION_RTOL, PROJECTION_ATOL) if projection else
                  (EXPRESSION_RTOL, EXPRESSION_ATOL))
    maximum_error = float(np.max(np.abs(actual - expected), initial=0))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        fail("NUMERIC_RECONSTRUCTION_CONFLICT", location=location,
             maximum_absolute_error=maximum_error, rtol=rtol, atol=atol)
    return maximum_error


def bound_file(record, base, role):
    raw_path = require(record, "path", role)
    expected_hash = require(record, "sha256", role)
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        fail("INVALID_FILE_BINDING", role=role)
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        fail("INVALID_SHA256_BINDING", role=role)
    candidate = Path(raw_path)
    path = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        abstain("PATH_OUTSIDE_APPROVED_PROJECT_ROOT", role=role, path=str(path))
    if not path.is_file():
        abstain("REQUIRED_FILE_MISSING", role=role, path=str(path))
    if path.stat().st_size > (4 << 30):
        abstain("UNSUPPORTED_FILE_SIZE", role=role, path=str(path))
    observed_hash = sha256(path)
    if observed_hash != expected_hash:
        fail("BOUND_FILE_HASH_CONFLICT", role=role, expected=expected_hash,
             observed=observed_hash, path=str(path))
    if "bytes" in record and record["bytes"] != path.stat().st_size:
        fail("BOUND_FILE_BYTE_SIZE_CONFLICT", role=role)
    return path


def parameters(n_training, n_eligible, n_selected):
    seed = int.from_bytes(hashlib.sha256(
        b"D3_SCANPY_REFERENCE|O1|CTMOD002_V1").digest()[:8], "big") % (2 ** 32)
    return {
        "normalization": {"target_sum": 10000,
                          "exclude_highly_expressed": False, "log1p": True},
        "filter": {"min_cells": max(10, math.ceil(.01 * n_training)),
                   "training_only": True},
        "hvg": {"n_top_genes": min(2000, n_eligible), "flavor": "seurat",
                "n_bins": 20, "batch_key": None, "subset": False},
        "scale": {"zero_center": True, "max_value": None},
        "pca": {"n_comps": min(30, n_training - 1, n_selected - 1),
                "zero_center": True, "svd_solver": "arpack", "random_state": seed,
                "dtype": "float64", "layer": None, "obsm": None},
        "neighbors": {"n_neighbors": 30, "metric": "sqeuclidean",
                      "tie_break": "cell_id_lexicographic", "chunk_size": 192},
    }


def independent_hvg(normalized_training, eligible):
    """NumPy/SciPy/pandas reconstruction, not a Scanpy function call.

    Reverse log1p, obtain unbiased variance, log variance/mean dispersion,
    20 equal-width mean bins, within-bin mean/SD, and inclusive nth threshold.
    Scanpy permits more than 2000 genes when the cutoff ties; retain all ties.
    """
    expression = normalized_training.copy().astype(np.float64)
    expression.data = np.expm1(expression.data)
    count = expression.shape[0]
    means = np.asarray(expression.sum(axis=0)).ravel() / count
    squares = expression.copy()
    squares.data **= 2
    variance = (np.asarray(squares.sum(axis=0)).ravel() / count - means ** 2)
    variance *= count / (count - 1)
    means[means == 0] = 1e-12
    dispersions = variance / means
    dispersions[dispersions == 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        dispersions = np.log(dispersions)
    table = pd.DataFrame({"mean": np.log1p(means), "dispersion": dispersions})
    table["bin"] = pd.cut(table["mean"], bins=20)
    grouped = table.groupby("bin", observed=True)["dispersion"].agg(["mean", "std"])
    singleton = grouped["std"].isna()
    grouped.loc[singleton, "std"] = grouped.loc[singleton, "mean"]
    grouped.loc[singleton, "mean"] = 0
    per_feature = grouped.loc[table["bin"]].reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = ((table["dispersion"].to_numpy() - per_feature["mean"].to_numpy()) /
                      per_feature["std"].to_numpy())
    valid = normalized[~np.isnan(normalized)]
    if len(valid) == 0:
        abstain("HVG_DISPERSION_UNRESOLVED")
    top = min(2000, len(eligible), len(valid))
    threshold = np.sort(valid)[-top]
    # This reproduces the declared inclusive tie rule, including numpy's
    # explicit nan replacement used by the frozen Scanpy recipe.
    keep = np.nan_to_num(normalized, nan=-np.inf) >= threshold
    chosen = eligible[np.flatnonzero(keep)]
    if len(chosen) > MAX_HVG:
        abstain("UNSUPPORTED_HVG_TIE_SIZE", selected=len(chosen), maximum=MAX_HVG)
    if len(chosen) < 2:
        abstain("INSUFFICIENT_SELECTED_FEATURES", selected=len(chosen))
    return chosen, float(threshold)


def inspect(contract_path, report):
    contract = read_json(contract_path)
    report["contract_sha256"] = sha256(contract_path)
    if require(contract, "schema_version", "contract") != 1:
        abstain("UNSUPPORTED_CONTRACT_SCHEMA")
    if require(contract, "recipe_id", "contract") != RECIPE_ID:
        abstain("UNSUPPORTED_RECIPE")
    if require(contract, "assay_scope", "contract") != "RNA":
        abstain("UNSUPPORTED_ASSAY_RECIPE")
    if require(contract, "provenance_required", "contract") is not True:
        abstain("PROVENANCE_NOT_REQUIRED_BY_CONTRACT")
    source_records = require(contract, "sources", "contract")
    artifact_records = require(contract, "artifacts", "contract")
    source_paths = {}
    for name, frozen_hash in SOURCE_HASHES.items():
        record = require(source_records, name, "contract.sources")
        source_paths[name] = bound_file(record, contract_path.parent, "source:" + name)
        if record["sha256"] != frozen_hash:
            abstain("UNSUPPORTED_FROZEN_SOURCE_IDENTITY", source=name)
    artifacts = {}
    for name in ("state", "trace", "producer"):
        record = require(artifact_records, name, "contract.artifacts")
        artifacts[name] = bound_file(record, contract_path.parent, "artifact:" + name)
    trace = read_json(artifacts["trace"])
    for key, expected in (("schema_version", 1), ("recipe_id", RECIPE_ID),
                          ("dataset", "D3"), ("outer_context", "O1"),
                          ("assay_scope", "RNA")):
        if require(trace, key, "trace") != expected:
            fail("TRACE_RECIPE_CONFLICT", field=key)
    if require(trace, "producer_code_sha256", "trace") != artifact_records["producer"]["sha256"]:
        fail("PRODUCER_LINEAGE_CONFLICT")
    traced_sources = require(trace, "source_files", "trace")
    for name, frozen_hash in SOURCE_HASHES.items():
        record = require(traced_sources, name, "trace.source_files")
        if require(record, "sha256", "trace.source_files." + name) != frozen_hash:
            fail("SOURCE_LINEAGE_CONFLICT", source=name)
        traced_path = bound_file(record, artifacts["trace"].parent, "trace-source:" + name)
        if traced_path != source_paths[name]:
            fail("SOURCE_PATH_LINEAGE_CONFLICT", source=name)
    reference = require(trace, "reference_source", "trace")
    if require(reference, "sha256", "trace.reference_source") != REFERENCE_SHA256:
        fail("FROZEN_RECIPE_SOURCE_LINEAGE_CONFLICT")
    bound_file(reference, artifacts["trace"].parent, "frozen-recipe-source")
    report["checks"].append({"name": "bound_files_and_lineage", "status": "PASS"})

    # Read only the allowlisted cell/design fields; no supplied/target labels.
    cell_columns = {"pilot_cell_index", "cell_id", "donor_id", "outer_fold"}
    feature_columns = {"original_feature_index", "feature_id", "modality",
                       "source_assay", "source_identity"}
    cells = pd.read_csv(source_paths["method_cells.tsv"], sep="\t", dtype=str,
                        keep_default_na=False, usecols=lambda column: column in cell_columns)
    features = pd.read_csv(source_paths["RNA_features.tsv"], sep="\t", dtype=str,
                           keep_default_na=False, usecols=lambda column: column in feature_columns)
    if not cell_columns.issubset(cells.columns) or not feature_columns.issubset(features.columns):
        abstain("SOURCE_ASSAY_OR_IDENTITY_METADATA_MISSING")
    if not (features["modality"].eq("RNA").all() and features["source_assay"].eq("RNA").all()):
        fail("TRUSTED_SOURCE_ASSAY_CONFLICT")
    if not features["source_identity"].eq("NAMED_SOURCE_SEURAT_ASSAY_NOT_NAME_GUESS").all():
        abstain("UNSUPPORTED_SOURCE_ASSAY_AUTHORITY")
    ids = cells.cell_id.to_numpy(dtype=str)
    feature_ids = features.feature_id.to_numpy(dtype=str)
    if len(set(ids)) != len(ids) or np.any(ids == "") or np.any(feature_ids == ""):
        fail("SOURCE_IDENTITY_AMBIGUOUS")
    try:
        exact(cells.pilot_cell_index.to_numpy(dtype=np.int64), np.arange(len(cells)),
              "source.pilot_cell_index")
        original_feature_index = features.original_feature_index.to_numpy(dtype=np.int64)
    except ValueError:
        fail("SOURCE_INDEX_METADATA_INVALID")
    if len(np.unique(original_feature_index)) != len(features) or np.any(np.diff(original_feature_index) <= 0):
        fail("SOURCE_FEATURE_ORDER_CONFLICT")
    if len(cells) > MAX_CELLS or len(features) > MAX_FEATURES:
        abstain("UNSUPPORTED_SOURCE_SIZE", cells=len(cells), features=len(features))
    with zipfile.ZipFile(source_paths["RNA_counts.npz"]) as archive:
        if sum(item.file_size for item in archive.infolist()) > (2 << 30):
            abstain("UNSUPPORTED_DECOMPRESSED_SOURCE_SIZE")
    counts = sparse.load_npz(source_paths["RNA_counts.npz"]).tocsr().astype(np.float64)
    if counts.shape != (len(cells), len(features)):
        fail("SOURCE_MATRIX_METADATA_SHAPE_CONFLICT")
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        fail("SOURCE_COUNTS_INVALID")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    train = np.flatnonzero(cells.outer_fold.to_numpy() != "O1")
    query = np.flatnonzero(cells.outer_fold.to_numpy() == "O1")
    if len(train) < 31 or len(query) < 1:
        abstain("INSUFFICIENT_TRAINING_OR_QUERY_CELLS")
    if set(cells.donor_id.iloc[train]) & set(cells.donor_id.iloc[query]):
        fail("HELD_OUT_DONOR_LEAKAGE")
    minimum = max(10, math.ceil(.01 * len(train)))
    detection = np.asarray((counts[train] > 0).sum(axis=0)).ravel()
    eligible = np.flatnonzero(detection >= minimum)
    if len(eligible) < 2:
        abstain("INSUFFICIENT_ELIGIBLE_FEATURES")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        abstain("ZERO_RNA_LIBRARY")
    normalized = counts.copy()
    # Follow the declared divide-by-library/10000 arithmetic independently.
    normalized.data /= np.repeat(totals / 10000, np.diff(normalized.indptr))
    np.log1p(normalized.data, out=normalized.data)
    chosen, hvg_threshold = independent_hvg(normalized[train][:, eligible], eligible)
    expected_params = parameters(len(train), len(eligible), len(chosen))
    for location, actual in (("contract.parameters", require(contract, "parameters", "contract")),
                             ("trace.parameters", require(trace, "parameters", "trace"))):
        if actual != expected_params:
            fail("DECLARED_PARAMETER_CONFLICT", location=location,
                 expected=expected_params, observed=actual)
    n, p, t, q, m, k = (len(cells), len(features), len(train), len(query),
                        len(chosen), expected_params["pca"]["n_comps"])
    expected_shapes = {
        "train": (t,), "query": (q,), "eligible": (len(eligible),), "selected": (m,),
        "normalized_selected": (n, m), "scaled_training": (t, m), "center": (m,),
        "scale": (m,), "pca_center": (m,), "components": (m, k),
        "coordinates": (n, k), "neighbors": (q, 30), "training_pca": (t, k),
        "rna_totals": (n,), "normalization_totals": (n,),
        "cell_ids": (n,), "feature_ids": (p,), "feature_names": (p,),
    }
    arrays = {}
    with h5py.File(artifacts["state"], "r") as state:
        for key, expected in (("schema_version", 1), ("recipe_id", RECIPE_ID),
                              ("assay_scope", "RNA"), ("dataset", "D3"),
                              ("outer_context", "O1"),
                              ("seed", expected_params["pca"]["random_state"]),
                              ("scanpy_version", "1.12.4")):
            if key not in state.attrs:
                abstain("STATE_ATTRIBUTE_MISSING", attribute=key)
            if state.attrs[key] != expected:
                fail("STATE_ATTRIBUTE_CONFLICT", attribute=key)
        for name, shape in expected_shapes.items():
            if name not in state:
                abstain("STATE_DATASET_MISSING", dataset=name)
            if not isinstance(state.get(name, getlink=True), h5py.HardLink):
                abstain("UNSUPPORTED_HDF5_EXTERNAL_OR_SOFT_LINK", dataset=name)
            dataset = state[name]
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != shape:
                fail("STATE_DATASET_SHAPE_CONFLICT", dataset=name, expected_shape=list(shape))
            if name in IDENTITY_ARRAYS:
                if h5py.check_string_dtype(dataset.dtype) is None:
                    fail("STATE_IDENTITY_DTYPE_CONFLICT", dataset=name)
                arrays[name] = dataset.asstr()[:]
            else:
                integer = name in {"train", "query", "eligible", "selected", "neighbors"}
                if (integer and dataset.dtype.kind not in "iu") or (
                        not integer and dataset.dtype != np.dtype("float64")):
                    fail("STATE_NUMERIC_DTYPE_CONFLICT", dataset=name)
                arrays[name] = dataset[:]
                if not np.isfinite(arrays[name]).all():
                    fail("NONFINITE_STATE", dataset=name)
    feature_names = np.array([f"RNA:{i}:{name}" for i, name in enumerate(feature_ids)])
    for name, expected in (("cell_ids", ids), ("feature_ids", feature_ids),
                           ("feature_names", feature_names), ("train", train),
                           ("query", query), ("eligible", eligible), ("selected", chosen)):
        exact(arrays[name], expected, "state." + name)
    signatures = require(trace, "array_signatures", "trace")
    for name in REQUIRED_ARRAYS:
        assert_signature(require(signatures, name, "trace.array_signatures"),
                         array_signature(arrays[name]), "trace.array_signatures." + name)
    identities = require(trace, "identities", "trace")
    for name in IDENTITY_ARRAYS:
        assert_signature(require(identities, name, "trace.identities"),
                         identity_signature(arrays[name]), "trace.identities." + name)
    report["checks"].append({"name": "source_design_hvg_state_binding", "status": "PASS",
                             "cells": n, "RNA_features": p, "training": t, "query": q,
                             "eligible": len(eligible), "selected": m,
                             "hvg_inclusive_threshold": hvg_threshold})

    expected_expression = normalized[:, chosen].toarray()
    errors = {
        "rna_totals": numeric(arrays["rna_totals"], totals, "state.rna_totals"),
        "normalization_totals": numeric(arrays["normalization_totals"], totals,
                                         "actual_RNA_normalization_denominator"),
        "normalized_selected": numeric(arrays["normalized_selected"], expected_expression,
                                        "actual_normalized_selected_expression"),
    }
    center = expected_expression[train].mean(axis=0)
    scale = expected_expression[train].std(axis=0, ddof=1)
    scale[scale == 0] = 1
    scaled = (expected_expression - center) / scale
    scaled_training = scaled[train]
    pca_center = scaled_training.mean(axis=0)
    for name, expected in (("center", center), ("scale", scale),
                           ("scaled_training", scaled_training), ("pca_center", pca_center)):
        errors[name] = numeric(arrays[name], expected, "state." + name)
    events = require(trace, "pca_get_obs_rep", "trace")
    if not isinstance(events, list) or not events:
        abstain("ACTUAL_PCA_GETTER_PROVENANCE_MISSING")
    if len(events) != 1:
        abstain("UNSUPPORTED_PCA_GETTER_EVENT_COUNT", count=len(events))
    event = events[0]
    for key, expected in (("function", "scanpy.preprocessing._pca._get_obs_rep"),
                          ("input_key", "X"), ("layer", None), ("obsm", None)):
        if require(event, key, "pca_get_obs_rep") != expected:
            fail("ACTUAL_PCA_SELECTION_CONFLICT", field=key)
    assert_signature(require(event, "array", "pca_get_obs_rep"),
                     array_signature(arrays["scaled_training"]), "actual_PCA_getter_operand")
    assert_signature(require(event, "cell_ids", "pca_get_obs_rep"),
                     identity_signature(ids[train]), "actual_PCA_getter_cell_order")
    assert_signature(require(event, "feature_ids", "pca_get_obs_rep"),
                     identity_signature(feature_names[chosen]), "actual_PCA_getter_feature_order")
    output = require(trace, "pca_output", "trace")
    assert_signature(require(output, "coordinates", "pca_output"),
                     array_signature(arrays["training_pca"]), "actual_PCA_output_coordinates")
    assert_signature(require(output, "components", "pca_output"),
                     array_signature(arrays["components"]), "actual_PCA_output_components")
    report["checks"].append({"name": "RNA_normalization_scaling_actual_PCA_input", "status": "PASS",
                             "maximum_absolute_errors": errors})

    # Independent deterministic covariance eigendecomposition. Compare the
    # retained subspace, not eigenvector signs or arbitrary within-space rotation.
    centered_training = scaled_training - pca_center
    covariance = centered_training.T @ centered_training / (t - 1)
    eigenvalues, vectors = linalg.eigh(covariance, subset_by_index=(m - k - 1, m - 1),
                                      driver="evr", check_finite=True)
    leading = float(eigenvalues[-1])
    if leading <= 0 or eigenvalues[1] <= leading * 1e-12:
        abstain("INSUFFICIENT_PCA_NUMERIC_RANK")
    gap = float((eigenvalues[1] - eigenvalues[0]) / leading)
    if gap <= EIGENGAP_TOL:
        abstain("PCA_RETAINED_BOUNDARY_EIGENGAP_UNRESOLVED", relative_eigengap=gap,
                minimum=EIGENGAP_TOL)
    reference_components = vectors[:, 1:]
    components = arrays["components"]
    orthogonality = float(np.max(np.abs(components.T @ components - np.eye(k))))
    if orthogonality > ORTHOGONALITY_TOL:
        fail("PCA_COMPONENTS_NOT_ORTHONORMAL", maximum_error=orthogonality)
    residual = reference_components - components @ (components.T @ reference_components)
    subspace_error = float(linalg.norm(residual, ord="fro") / math.sqrt(k))
    if subspace_error > SUBSPACE_TOL:
        fail("PCA_SUBSPACE_RECONSTRUCTION_CONFLICT", relative_error=subspace_error,
             tolerance=SUBSPACE_TOL)
    expected_coordinates = (scaled - pca_center) @ components
    projection_error = numeric(arrays["coordinates"], expected_coordinates,
                               "held_out_source_projection", projection=True)
    training_error = numeric(arrays["training_pca"], expected_coordinates[train],
                             "independent_training_projection", projection=True)
    numeric(arrays["coordinates"][train], arrays["training_pca"],
            "PCA_training_output_consistency", projection=True)
    report["checks"].append({"name": "independent_PCA_and_held_out_projection", "status": "PASS",
                             "relative_subspace_error": subspace_error,
                             "orthonormality_max_error": orthogonality,
                             "relative_boundary_eigengap": gap,
                             "projection_max_error": projection_error,
                             "training_projection_max_error": training_error})

    calls = require(trace, "cdist_calls", "trace")
    if not isinstance(calls, list) or not calls:
        abstain("ACTUAL_CDIST_PROVENANCE_MISSING")
    expected_calls = math.ceil(q / 192)
    if len(calls) != expected_calls:
        fail("CDIST_EVENT_COVERAGE_CONFLICT", expected=expected_calls, observed=len(calls))
    neighbors = arrays["neighbors"]
    if np.any(neighbors < 0) or np.any(neighbors >= n) or not np.isin(neighbors, train).all():
        fail("NEIGHBOR_IDENTITY_OR_HELD_OUT_CONFLICT")
    max_distance_error = 0.0
    for call, start in zip(calls, range(0, q, 192), strict=True):
        stop = min(q, start + 192)
        for key, expected in (("function", "scipy.spatial.distance.cdist"),
                              ("query_start", start), ("query_stop", stop),
                              ("metric", "sqeuclidean")):
            if require(call, key, "cdist_call") != expected:
                fail("ACTUAL_CDIST_CALL_CONFLICT", field=key, query_start=start)
        actual_query = arrays["coordinates"][query[start:stop]]
        actual_training = arrays["coordinates"][train]
        for key, expected in (
            ("query_array", array_signature(actual_query)),
            ("training_array", array_signature(actual_training)),
            ("query_cell_ids", identity_signature(ids[query[start:stop]])),
            ("training_cell_ids", identity_signature(ids[train])),
        ):
            assert_signature(require(call, key, "cdist_call"), expected,
                             f"actual_cdist.{key}.{start}")
        captured_distances = cdist(actual_query, actual_training, metric="sqeuclidean")
        assert_signature(require(call, "distance_array", "cdist_call"),
                         array_signature(captured_distances), f"actual_cdist.distance_output.{start}")
        independent_distances = cdist(expected_coordinates[query[start:stop]],
                                      expected_coordinates[train], metric="sqeuclidean")
        max_distance_error = max(max_distance_error, numeric(
            captured_distances, independent_distances, f"independent_distances.{start}", projection=True))
        for offset, row in enumerate(independent_distances):
            threshold = np.partition(row, 29)[29]
            candidates = np.flatnonzero(row <= threshold)
            ordering = np.lexsort((ids[train[candidates]], row[candidates]))[:30]
            expected_neighbors = train[candidates[ordering]]
            exact(neighbors[start + offset], expected_neighbors,
                  f"independent_NN30_query_row:{start + offset}")
    report["checks"].append({"name": "actual_cdist_and_independent_exact_NN30", "status": "PASS",
                             "calls": len(calls), "query_rows": q,
                             "distance_max_error": max_distance_error,
                             "tie_break": "distance_then_cell_id_lexicographic"})
    # Detect changes during inspection; never update expected hashes to match.
    for name, path in source_paths.items():
        if sha256(path) != source_records[name]["sha256"]:
            fail("SOURCE_CHANGED_DURING_INSPECTION", source=name)
    for name, path in artifacts.items():
        if sha256(path) != artifact_records[name]["sha256"]:
            fail("ARTIFACT_CHANGED_DURING_INSPECTION", artifact=name)
    if sha256(contract_path) != report["contract_sha256"]:
        fail("CONTRACT_CHANGED_DURING_INSPECTION")
    return "PASS", "SUPPORTED_REAL_D3_O1_PATH_RECONSTRUCTED", {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    report = {
        "schema_version": 1, "checker_id": "ct-mod-005-independent-realpath-guard-v1",
        "recipe_id": RECIPE_ID, "contract": str(args.contract.resolve()), "checks": [],
        "limitations": [
            "Only the SHA-bound frozen D3/O1 RNA recipe is supported; joint/WNN recipes abstain.",
            "Contract authority and runtime instrumentation require an externally trusted freeze; hashes do not authenticate a malicious producer.",
            "No labels, targets, AP or biological performance are read or validated.",
            "This is one dependent technical fixture, not sensitivity, prevalence or field-validation evidence.",
            "Bounded local artifacts only; arbitrary hostile HDF5/NPZ/contracts are outside the security model.",
            "NumPy/SciPy implementations are independent of Scanpy/producer code but share numerical-library infrastructure.",
        ],
        "tolerances": {"expression_rtol": EXPRESSION_RTOL, "expression_atol": EXPRESSION_ATOL,
                       "projection_rtol": PROJECTION_RTOL, "projection_atol": PROJECTION_ATOL,
                       "subspace_relative": SUBSPACE_TOL, "orthonormality": ORTHOGONALITY_TOL,
                       "minimum_relative_eigengap": EIGENGAP_TOL},
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__,
                        "pandas": pd.__version__, "h5py": h5py.__version__},
    }
    try:
        status, reason, details = inspect(args.contract.resolve(), report)
    except Decision as exc:
        status, reason, details = exc.status, exc.reason, exc.details
    except (OSError, KeyError, TypeError, ValueError, MemoryError, linalg.LinAlgError) as exc:
        status, reason = "NOT_VERIFIABLE", "CHECKER_TECHNICAL_ERROR"
        details = {"error_type": type(exc).__name__, "error": str(exc)}
    report.update(status=status, reason=reason, details=details,
                  elapsed_seconds=time.monotonic() - started)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return {"PASS": 0, "FAIL": 2, "NOT_VERIFIABLE": 3}[status]


if __name__ == "__main__":
    sys.exit(main())
