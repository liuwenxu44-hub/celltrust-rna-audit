"""Path-only CLI adapter around the accepted, unchanged scoring functions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from celltrust_audit import core


def validate_design(cells):
    required = {"cell_id", "donor_id", "outer_fold"}
    if not required.issubset(cells) or cells.empty or cells[list(required)].isna().any().any():
        raise ValueError("MISSING_CELL_DONOR_OR_OUTER_FIELDS")
    if cells.cell_id.duplicated().any() or cells[list(required)].eq("").any().any():
        raise ValueError("AMBIGUOUS_CELL_OR_DONOR_IDENTITY")
    if (cells.groupby("donor_id").outer_fold.nunique() != 1).any():
        raise ValueError("DONOR_SPLIT_ACROSS_OUTER_FOLDS")
    contexts = []
    for outer in sorted(cells.outer_fold.unique()):
        if not outer.startswith("O") or not outer[1:].isdigit():
            raise ValueError("EXPECTED_OUTER_NAMES_O1_O2_ETC")
        key = "inner_" + outer
        if key not in cells:
            raise ValueError("MISSING_INNER_ASSIGNMENTS:" + outer)
        train = cells.outer_fold != outer
        inner = cells.loc[train, key]
        if inner.isna().any() or any(not str(v).startswith("I") or not str(v)[1:].isdigit() for v in inner):
            raise ValueError("INVALID_INNER_FOLD_CODES")
        if (cells.loc[train].groupby("donor_id")[key].nunique() != 1).any():
            raise ValueError("DONOR_SPLIT_ACROSS_INNER_FOLDS")
        # The accepted D4 tables use OUTER_HOLDOUT; D3 uses OUTER_QUERY.
        # Both are non-training sentinels, never new fold assignments.
        if not cells.loc[~train, key].isin(["OUTER_QUERY", "OUTER_HOLDOUT"]).all():
            raise ValueError("OUTER_QUERY_INNER_CODE_CONFLICT")
        folds = sorted(inner.unique())
        if len(folds) < 2:
            raise ValueError("INSUFFICIENT_INNER_PARTITIONS")
        contexts.extend([outer] + [outer + "_" + k for k in folds])
    if cells.outer_fold.nunique() < 3:
        raise ValueError("INSUFFICIENT_OUTER_PARTITIONS")
    return contexts


def d3_seed(context):
    text = f"D3_Zenodo7555405\t{context}\tHVG2000\tCT001B_PCA_v1".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**32)


def validate_inputs(inputs, condition):
    cells = pd.read_csv(inputs / "method_cells.tsv", sep="\t", dtype=str, keep_default_na=False)
    contexts = validate_design(cells)
    indices = []
    for modality in (["RNA"] if condition == "A" else ["RNA", "ADT"]):
        features = pd.read_csv(inputs / f"{modality}_features.tsv", sep="\t", keep_default_na=False)
        matrix = sparse.load_npz(inputs / f"{modality}_counts.npz")
        needed = {"feature_id", "modality", "original_feature_index"}
        if not needed.issubset(features) or not features.modality.eq(modality).all():
            raise ValueError("INVALID_EXPLICIT_MODALITY_METADATA")
        if features.feature_id.eq("").any() or features.feature_id.duplicated().any():
            raise ValueError("AMBIGUOUS_FEATURE_IDENTITIES")
        order = pd.to_numeric(features.original_feature_index, errors="raise").to_numpy()
        if not np.equal(order, np.floor(order)).all():
            raise ValueError("NONINTEGER_FEATURE_ORDER")
        indices.extend(order.tolist())
        if matrix.format != "csr" or not matrix.has_canonical_format or np.any(matrix.data == 0):
            raise ValueError("CANONICAL_CSR_WITHOUT_EXPLICIT_ZEROS_REQUIRED")
        if matrix.shape != (len(cells), len(features)) or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("INVALID_RAW_COUNT_MATRIX")
    if len(indices) != len(set(indices)):
        raise ValueError("DUPLICATE_ORIGINAL_FEATURE_INDICES")
    return cells, contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Prepared dataset root, containing inputs/")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--dataset", choices=["D3", "D4"], required=True)
    parser.add_argument("--condition", choices=list("ABCD"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("OUTPUT_EXISTS_REFUSE_OVERWRITE")
    cells, contexts = validate_inputs(args.inputs / "inputs", args.condition)
    labels = pd.read_csv(args.labels, sep="\t", dtype=str, keep_default_na=False)
    if not {"cell_id", "observed_label"}.issubset(labels) or labels.cell_id.duplicated().any() or set(labels.cell_id) != set(cells.cell_id):
        raise ValueError("OBSERVED_LABEL_IDENTITY_MISMATCH")
    observed = labels.set_index("cell_id").loc[cells.cell_id, "observed_label"].to_numpy()
    if not set(observed).issubset(core.LABELS):
        raise ValueError("UNSUPPORTED_OBSERVED_LABEL")
    markers = pd.read_csv(args.markers, sep="\t", dtype=str, keep_default_na=False)
    if args.dataset == "D3":
        core.seed_for = d3_seed
        markers = markers.copy()
        markers["ensembl_gene_id"] = markers["gene"]
    view = core.load_condition(args.inputs, args.condition)
    reps = {name: core.representation(view, name) for name in contexts}
    marker = core.marker_components(view, markers)
    scores = np.full((len(cells), len(core.METHODS)), np.nan)
    for outer in sorted(cells.outer_fold.unique()):
        inner = [rep for name, rep in reps.items() if name.startswith(outer + "_")]
        rep = reps[outer]
        scores[rep["query"]] = core.score_outer(observed, cells, rep, inner, marker, args.condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, scores=scores, cell_ids=cells.cell_id.to_numpy(dtype=str),
                            methods=np.asarray(core.METHODS, dtype=str),
                            condition=args.condition, dataset=args.dataset)


if __name__ == "__main__":
    main()
