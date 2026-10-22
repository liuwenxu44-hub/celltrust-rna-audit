# CellTrust: checking RNA-only CITE-seq workflows

CITE-seq measures RNA and antibody-derived tags (ADTs) in the same cells. If RNA and ADT counts are normalized together, deleting the ADT columns afterward does not undo their effect on RNA normalization.

This repository contains code for *Auditing RNA-only CITE-seq workflows: context-dependent score changes and execution checks*. We compare preprocessing choices, examine how they change scores for detecting potentially incorrect cell-type labels, and check which data a workflow actually used.

## What does the study compare?

We keep the cells, supplied labels and donor-based training/test splits the same while changing the input:

| Workflow | Preprocessing |
|---|---|
| **A: RNA only** | Select RNA first, then normalize using RNA counts alone. |
| **B: RNA + ADT** | Normalize the combined counts and keep both types of features. |
| **C: late ADT removal** | Normalize the combined counts, then remove ADT without renormalizing RNA. |
| **D: ADT only** | Use ADT alone as a diagnostic comparison. |

The scores use three kinds of evidence: similarity to cells with the supplied label (**M1**), known RNA marker genes (**M2**), and agreement with nearby training cells (**M4**). Training and test sets are separated by donor.

B, C and D are diagnostic comparisons, not recommended RNA-only workflows. The study does not evaluate intentionally joint RNA–protein methods. The code does not automatically correct labels or make clinical decisions.

## Try the demo

From the repository directory, using Python 3.12 on Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt h5py==3.16.0
python -m pip install --no-deps -e .
python -B -m unittest discover -s tests -v
python -B scripts/modality_guard.py --demo --output runs/modality-demo
```

The demo changes or removes artificial ADT counts and checks that RNA-only normalization stays the same. It saves a report and the normalized RNA matrix in `runs/modality-demo/`, not a paper result. Use a new output directory for each run. The tests also need `h5py`, included in the installation command above.

## Check the manuscript results

The [source-data directory](source_data/README.md) contains the numbers behind Figures 2–5: donor-level metrics and their denominators, effect estimates and intervals, reference-scale diagnostics, and the recorded technical cases. It also includes the fixed selected cells, donor partitions, observed labels, evaluation targets and ADT thresholds for both study subsets. The original count matrices remain with the data providers.

After installing the requirements, run:

```bash
python -B scripts/reproduce_summaries.py --output runs/summary-check
```

This recomputes the effect summaries and confidence intervals from saved donor-level results, checks the D3 donor/leave-one-out results and technical figure data, and verifies the fixed input tables. It does not refit models or re-create the final artwork. The summary check was actually run on the study data, not just artificial arrays.

For a real-data example, [download the exact D3 RDS and prepare its fixed subset](docs/inputs.md#prepare-the-real-d3-example). That entry was tested against the original file: its RNA and ADT outputs match the matrices used in the study byte for byte. This input preparation is not a full rerun of every analysis.

## Run the scoring code

You need prepared count matrices, feature metadata, cell labels and donor splits, as described in the [input guide](docs/inputs.md). The fixed identities, labels and splits are included; count matrices must be obtained from the original providers. This is not a one-command reproduction of every result.

For example, to score one supplied label set under workflow A:

```bash
python -B scripts/score.py --inputs prepared_dataset \
  --labels observed_labels.tsv --markers config/markers.tsv \
  --dataset D4 --condition A --output runs/score-A.npz
```

This writes six scores per cell to a NumPy `.npz` file. The scoring command does not read the target labels used to evaluate errors, choose new cells or change the training/test splits.

**D4** is the COMBAT development dataset; **D3** is the vaccination dataset used for an independent check. `--dataset D3` selects its gene-symbol mapping and random-seed rule, not a data download. Use the supplied fixed tables rather than selecting cells or assigning new splits.

## Check what a workflow actually used

The basic demo checks normalization. The additional scripts inspect the data passed to principal component analysis (PCA), the resulting coordinates and the neighboring cells selected for comparison.

Install `requirements-execution.txt` in a separate environment for these additional checks. `realpath_guard.py` needs the exact study input files and a JSON description of the inputs, parameters and execution records; see the [input guide](docs/inputs.md#frozen-real-path-checker).

```bash
export CELLTRUST_AUDIT_ROOT="$(pwd)"
python -B scripts/realpath_guard.py --contract runs/case/contract.json
```

`CELLTRUST_AUDIT_ROOT` sets the directory containing your inputs and outputs; the default is the current directory. The checker supports **D3/O1**, one predefined train/test split. Its possible results are:

- **PASS:** the supplied evidence agrees with the supported RNA-only calculation.
- **FAIL:** the checker found a specific conflict.
- **NOT_VERIFIABLE:** evidence is missing, or the workflow is outside the supported scope. This is not a detected error.

The checks rely on trustworthy source metadata and execution records. They cannot rule out fabricated evidence, and a PASS does not prove that cell labels are biologically correct.

## Where to find the code

- **Scoring:** `src/celltrust_audit/core.py`, with supporting formulas in `src/celltrust/`.
- **Evaluation:** `metrics.py` and `statistics.py` in `src/celltrust_audit/` calculate average precision, the errors found when reviewing the top-scoring 10% of cells, and uncertainty by resampling donors. The 10% budget is for comparison, not a recommended action threshold.
- **Workflow checks:** `scripts/modality_guard.py`, `realpath_producer.py` and `realpath_guard.py` check normalization and the actual PCA/neighborhood calculations.
- **Public example:** `scripts/muon_replay.py` runs the specified RNA steps from a fixed version of the public MUON tutorial. Data sources and setup are in the [input guide](docs/inputs.md#public-muon-rna-branch).
- **Additional analyses:** `analysis/m4_mechanism.py` compares raw scores and the reference values used to standardize M4; this is a diagnostic, not a causal analysis. `analysis/d3_label_audit.py` checks whether non-integer ADT counts change reference labels while keeping the labeling rules and thresholds fixed. It is not a separate biological validation of those labels.
- **Parameters and tests:** `config/` holds marker definitions and analysis settings; `reference/` preserves the relevant method and label-rule source files; `tests/` uses artificial data. The additional analyses require their study-specific inputs, and the ADT label check also needs R dependencies. `reference/scanpy_reference.py` is retained for source comparison, not as a standalone command.

## How to interpret the results

Mixing inputs can change scores in either direction; adding ADT does not always improve or inflate performance. D3 models are fitted within D3, not transferred from a trained D4 model.

Uncertainty intervals resample donors without retraining. M1's internal reference calculations reuse the coordinates learned from all training donors, while excluding test donors. Marker coverage varies between cell types, and known marker genes remain available even in tests where a cell type is removed from the supplied labels.

The MUON example uses the package versions listed here, not the tutorial's original software environment. Normalization is checked separately, but the full graph calculation is not independently reimplemented. Deliberately introduced faults are our tests, not errors attributed to the tutorial. The test cases share inputs; they do not measure how often such errors occur in published work.

The code has artificial-data tests, a study-data summary check and a tested real-D3 input preparation entry. The complete raw-data-to-manuscript study has not been independently rerun from this public repository. The source tables contain public deidentified cell/donor identities but no clinical covariates. Experiment logs, internal review records and fitted model caches are not included.

## License

Original CellTrust code is available under the [MIT license](LICENSE). Third-party tools and datasets retain their own terms; see [attribution and scope](THIRD_PARTY_NOTICES.md).
