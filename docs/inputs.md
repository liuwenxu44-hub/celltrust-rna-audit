# Inputs and fixed analysis definitions

## Core scoring

Arrange one prepared, fixed population as:

```
prepared_dataset/
  inputs/
    method_cells.tsv
    RNA_features.tsv
    RNA_counts.npz
    ADT_features.tsv
    ADT_counts.npz
```

Matrices are canonical CSR: rows in `method_cells.tsv` order, columns in the corresponding feature-table order; sum duplicate entries, eliminate explicit zeros, and sort indices before saving. Values must be finite, nonnegative raw counts or explicitly source-supported ADT count estimates. Do not substitute logged, integrated or SCT expression. RNA must have a positive library total for every retained cell. Do not silently delete a cell because a denominator is zero.

`method_cells.tsv` requires unique `cell_id`, `donor_id`, `outer_fold` (e.g. `O1`) and an `inner_O1` column for each outer context. Inner values are `I1`, `I2`, etc. The original non-training sentinel is `OUTER_HOLDOUT` in D4 and `OUTER_QUERY` in D3; the CLI accepts both without changing assignments. All cells of a donor stay in the same partition.

Feature tables require `feature_id`, `modality` (`RNA` or `ADT`), and `original_feature_index` (unique combined original-column order). Modality must come from trusted assay/feature-type metadata, not guessed gene or antibody names. For D4, marker keys are the frozen Ensembl IDs; for D3 the original exact gene-symbol mapping is applied to the same marker definitions. The `ensembl_gene_id` field name in the core API is retained for compatibility even when D3 symbols are supplied.

`observed_labels.tsv` is **one** fixed observed-label vector, with exactly one row per `cell_id` and columns `cell_id`, `observed_label`. Allowed values: `T_cell`, `B_cell`, `NK_cell`, `monocyte`. The command reorders only by exact cell-ID matching, not by a fuzzy label match. Run separately for existing predefined instances; targets are not an input to scoring.

The D4 namespace is `D4_COMBAT_Zenodo5139561`; D3 is `D3_Zenodo7555405`. These are seed definitions, not machine paths. D3 uses leave-one-donor-out outer/inner assignments; D4 must use its previously fixed donor assignments. Population selection is not re-run by this source release. D4's 14,032 cells/37 donors and D3's 7,200 cells/6 donors are distinct fixed analysis sets; neither count is a runtime default.

The supplied marker file contains the four relevant lineages only. It preserves the original rows and their evidence citations; unrelated cell-line signatures are intentionally excluded. A D3 caller must use exact symbols, not infer a new marker set or fuzzy-match identifiers.

## Evaluation

`metrics.one(y, score, cell_ids, total_eligible)` accepts a finite scored support for one donor/instance/method/condition. `y` is a binary label-error target. Tied scores share an AP threshold. All-positive sets retain the raw AP value but have `NO_DISCRIMINATION_INFORMATION` status and must be excluded from ranking comparisons; no-positive sets have undefined AP.

Compute A/B/C contrasts on the intersection of finite cells **within a method**. Method-rank comparisons require the intersection across all six scores in all three conditions. Report excluded cells and coverage against the original scenario-eligible population. Do not compare method-specific supports without disclosure.

For `statistics.paired_effects`, pass a donor-by-stratum array of differences already formed from paired common-support donor/instance metrics: first average valid replicates within donor and stratum. Donors are equally weighted within a stratum; all eight primary strata have equal weight in the overall summary. Overall estimates require all strata; missing values are not zero-imputed. The default 2,000 cluster bootstrap draws resample whole donors with replacement, retaining all paired conditions, methods and instances. Use the seed for the applicable cohort from `config/study_parameters.json`. Percentile intervals do not include refitting uncertainty, are not adjusted hypothesis tests, and should not be interpreted as equivalence certification.

## Frozen real-path checker

The supplementary posthoc mechanism module exposes the unchanged numerical functions. `cross_standardize()` accepts A/B raw and reference diagnostics, fixed codes/IDs and the supplied M4 standardizer; `reconstruct_instance()` accepts the existing frozen representations, cell metadata and diagnostic/scaling callbacks. No fitting or target generation occurs in those functions. `AA/AB/BA/BB` mean raw condition first and reference condition second. They are diagnostics, not a causal decomposition. Internal source-discovery and report/finalizer orchestration are intentionally not published; `--self-test` exercises artificial arrays only.

The ADT side-audit command's source header and `--help` describe its fixed export inputs. The original classifier and rule YAML are provided under `reference/`; the audit still requires its original workspace layout and is not the new standalone reproduction entry. Existing numeric thresholds and sample-specific bottom-1% cutoffs were reused in both replays, rather than fitted again. The source-data package now includes the saved thresholds for samples contributing selected cells, and the fixed reference labels and assignments. Exported thresholds have only eight significant digits: exact categorical legacy agreement does not recover unexported full precision. The audit remains exploratory and does not authorize changing the main labels or population.

## Prepare the real D3 example

Use [Zenodo record 7555405](https://zenodo.org/records/7555405), file `PBMC_vaccine_CITE.rds` (1,594,039,156 bytes). Its SHA-256 is `b0da2cffb4fe1f5bb07087ce69f69fd2f3708fe5670ba5fc736eae2850cf98d4`. The exact file URL and provider checksum are in `source_data/original_sources.json`. Do not substitute `PBMC_vaccine_CITE_seuratV5.rds` from a newer record.

The preparation command needs the Python requirements plus `h5py==3.16.0`, and an `Rscript` installation with `SeuratObject`, `Matrix` and `rhdf5` available. Download the source file from the provider, then run from the repository:

```bash
python -B scripts/prepare_real_example.py \
  --rds /your/data/PBMC_vaccine_CITE.rds \
  --output runs/prepared-D3
```

The script checks the raw-file hash, extracts the supplied 7,200-cell list from the named RNA and ADT count assays, checks cell and feature order, and writes `inputs/RNA_counts.npz` and `inputs/ADT_counts.npz`. It preserves fractional ADT estimates. Both output hashes must match the original study inputs, or preparation fails. It also unpacks the fixed method/feature tables, reference labels, thresholds and evaluation targets. No population or reference label is regenerated.

The default `observed_labels.tsv` is the first fixed instance; `--instance` can select another existing ID from `selected_instances.tsv`. This is a formatting choice, not a new corruption realization. The returned folder can be passed to `score.py --inputs runs/prepared-D3 --dataset D3`, with `--labels runs/prepared-D3/observed_labels.tsv`. The scoring command does not consume the evaluation targets. A full scoring run is separate from the tested input preparation and summary reproduction.

For the existing real-path producer, use `--inputs runs/prepared-D3/inputs`, not the parent folder, and the package versions in `requirements-execution.txt`. Its contract still requires genuine execution records; do not invent a trace to obtain PASS.

## Exact D4 source and fixed tables

D4 uses [Zenodo record 5139561](https://zenodo.org/records/5139561), version 1.0.0, file `COMBAT-CITESeq-DATA.h5ad` (6,409,089,483 bytes), SHA-256 `f628a15f25b9ca2f7cdefeab271fd9d007c8d5e47eb80c4b807d5f65e86ff53d`. The count source is `layers/raw`, not the normalized `X` matrix. Use exact `cell_id` matches in the supplied D4 `method_cells.tsv`, in its row order. For columns, use `original_feature_index` in the supplied RNA/ADT feature tables and verify the source feature-type metadata. Do not infer modalities from names or reapply a new quality-control filter. There is no new D4 extraction/scoring run in this release.

Both cohorts' fixed tables are under `source_data/study_inputs/<cohort>/`. Gzip compression changes storage only; `manifest.json` records the decompressed hashes. In overlays and targets, `pilot_cell_index` refers to the row of `method_cells.tsv`. There are exactly 20 existing instances per cohort, not the earlier 261-instance development collection. `is_label_error` is a controlled target relative to `reference_labels.tsv`, not a claim that the source investigators' label was wrong. `scenario_eligible` defines each instance's eligible cells; do not treat ineligible monocytes in within-lymphoid scenarios as negatives.

## ADT reference labels

The main labels are supplied unchanged, not regenerated when preparing counts. `reference/derive_adt_truth.py` and `reference/adt_reference_rules.yaml` contain their original implementation and rules. Supplementary Methods S1.1.1 gives the equation, marker combinations and status logic. The implemented transform is per-cell mean-centering of `log1p(ADT count)`, processed within each sample. The lineage margin cutoff is 0.5 in the implemented score's units: despite the historical YAML key ending in `_sd`, the code does not divide that margin by a standard deviation.

Saved antibody-by-sample thresholds for the 43 D4 and 24 D3 contributing samples are provided. They were fitted on the original per-sample ADT populations, not refitted on the smaller selected analysis subset. The D3 reference labels used the historical integer-truncated ADT export; the new count-preparation entry preserves the original fractional estimates and does not change those labels. The post hoc fixed-threshold sensitivity audit found no categorical change at the exported precision. RNA values, author annotations and evaluation scores did not supply the ADT reference labels.

`realpath_producer.py` accepts `--inputs`, `--out` and `--mode clean|late_adt|wrong_layer`. It checks the exact recorded D3 input hashes and package versions before executing. Outputs must not exist; the parent must already exist. The three source hashes and recipe are intentionally pinned, not exposed as user-tunable arguments to force arbitrary data to pass.

`realpath_guard.py` accepts a JSON contract with `schema_version: 1`, `recipe_id: d3-o1-scanpy-reference-v1`, `assay_scope: RNA`, `provenance_required: true`, `parameters`, `sources`, and `artifacts`. Source entries are `RNA_counts.npz`, `RNA_features.tsv`, `method_cells.tsv`; artifact entries are `state`, `trace`, `producer`. Each file binding contains `path` and `sha256`, optionally `bytes`. Bind the actual supplied producer code, not an unrelated hash. The trace is produced by the instrumented replay; inventing or hand-editing one does not validate an execution. Detailed required fields and decisions are executable in the checker.

Exit codes: `0 PASS`, `2 FAIL`, `3 NOT_VERIFIABLE`. Missing execution lineage, unsupported recipes and unresolved numerical conditions abstain. Hash agreement cannot authenticate a dishonest source, producer or future contract.

## Public MUON RNA branch

Pinned upstream notebook:
https://raw.githubusercontent.com/scverse/muon-tutorials/62bdec09381286a1e4b8236726fad42c772c59f1/cite-seq/1-CITE-seq-PBMC-5k.ipynb

SHA-256: `39cefe1d1404829fff8d930bb88f2ba0bab779e06e02a15681b6f7b0b8883753`.

Official count archive:
https://cf.10xgenomics.com/samples/cell-exp/3.0.2/5k_pbmc_protein_v3/5k_pbmc_protein_v3_filtered_feature_bc_matrix.tar.gz

SHA-256: `12e507a7abed5d3d6c55b1836b24d52b311787b1b6f777f24086b9ee650ced54`.

Supply an assets directory with `original_notebook.ipynb`, `filtered_feature_bc_matrix/`, `selected_original_code_cells.json` and `acquisition.json`. The selected-cell JSON maps each index in `SELECTED_CELLS` to the exact original code text. The acquisition JSON has `notebook_commit` equal to the pinned commit and a `files` list of locally verified `{relative_path, sha256, bytes}` entries, including the notebook and selected-cell JSON. It is generated locally from the files actually acquired, not copied from someone else's machine log. Then run `scripts/muon_replay.py --assets <assets> --out <new-output>`. Assets and output must be under `CELLTRUST_AUDIT_ROOT`. Respect upstream data/software terms and do not include dependency wheels in the repository.

The adapter implements only the declared RNA branch through neighbors. It does not reproduce DSB, MOFA, joint integration, annotation or the complete notebook. It validates the original code as text and does not execute the full downloaded notebook.
