# Data behind Figures 2–5

These files are the existing study results and fixed inputs, not a new analysis. The table values retain their original precision. Blank or non-finite values mean unavailable/undefined; they are not zeros. Files ending in `.gz` are ordinary gzip-compressed UTF-8 TSVs. Standard tools and `pandas.read_csv(..., sep="\t")` can read them.

Run `python scripts/reproduce_summaries.py --output runs/summary-check` from the repository. It recomputes effect estimates and percentile intervals from the saved donor-level metrics and the original bootstrap draws. It also checks the selected input identities, D3 donor sensitivity, scale-pair counts, Figure 5 distributions and technical-case counts. It neither scores cells again nor draws the final figures.

## Find a panel's data

| Figure/panel | Files and selection |
|---|---|
| 2A | `effect_tables/{D4,D3}_CORE/paired_effect_summary.tsv.gz`: `support=METHOD_COMMON`, `metric=AP`, M4_RISK/B_minus_A and M1_RISK/C_minus_A; all eight scenario rows and ALL_8_PRIMARY_STRATA. Columns A and comparison_condition are the absolute AP values. |
| 2B | `effect_tables/{D4,D3}_SCANPY/paired_effect_summary.tsv.gz`: METHOD_COMMON, AP, ALL_8_PRIMARY_STRATA, both comparisons. |
| 2C | `effect_tables/D3_CORE/individual_donor_effects.tsv.gz`: AP, ALL_8_PRIMARY_STRATA, M1_RISK/C_minus_A and M4_RISK/B_minus_A. |
| 3A | `mechanism/paired_effect_summary.tsv.gz`: support M4_2X2_COMMON, fallback_group ALL, contrast_id M4_RISK__AB_MINUS_AA, M4_RISK__BA_MINUS_AA, M4_RISK__BB_MINUS_AA and M4_RISK__2X2_INTERACTION. |
| 3B | Same mechanism table: support DIAGNOSTIC_COMMON, fallback_group ALL, Z_S/Z_G versions of those four contrasts; ALL_8_PRIMARY_STRATA, ORDINARY_4_PRIMARY_STRATA and OMITTED_4_PRIMARY_STRATA. |
| 3C | `mechanism/risk_extra_paired_effect.tsv.gz`, both B_minus_A and C_minus_A; its donor values are in `donor_risk_extra_change.tsv.gz`. These are paired differences of changes, not differences of CI endpoints. |
| 3D | `mechanism/reference_support_and_scale.tsv.gz`: PRIMARY, components S and G; join A/B on instance_id, outer_context, observed_label and component. Plot paired positive finite old_scale values; retain the unavailable pairs in the source table. |
| 3E | `mechanism/paired_effect_summary.tsv.gz`: DIAGNOSTIC_COMMON, ALL, BB__M4_RISK_MINUS_Z_S and BB__M4_RISK_MINUS_Z_G; overall, ordinary and omitted summaries. |
| 4A | CORE effect tables: METHOD_COMMON, AP, ALL_8_PRIMARY_STRATA, M1_RISK and M1_MARGIN, both comparisons and cohorts. |
| 4B | CORE/SCANPY effect tables: METHOD_COMMON, ALL_8_PRIMARY_STRATA; M4_RISK/B_minus_A, M1_RISK/C_minus_A, Scanpy B_minus_A/C_minus_A; recall_review_10, precision_review_10 and correct_label_false_review_fraction. |
| 4C | D3 effect tables: the eight scenario rows of those three review metrics, M4_RISK/B_minus_A, M1_RISK/C_minus_A and Scanpy B_minus_A. |
| 4D | `effect_tables/D3_CORE/leave_one_donor_out.tsv.gz`: M4_RISK/B_minus_A and M1_RISK/C_minus_A; AP and the three review metrics. Donor A in the figure is donorA in the source. |
| 5A | `Figure5/A_paired_bins.csv`: all displayed two-dimensional bin counts, with the same 7,699,418 positive pairs in each branch. `RNA_invariance_audit.csv` records the 28 dependent stage comparisons. |
| 5B | `Figure5/B_full_empirical_cdf.npz` retains all 199,726 distinct observed decreases with their exact frequencies. `B_ecdf_display.csv` is the display-decimated curve; `B_quantiles.csv` contains exact order-statistic quantiles. Quantiles must be computed from the full frequency distribution, not the displayed curve. |
| 5C | `Figure5/C_diagnostic_cases.csv`: all 11 case outcomes and diagnostic objects. `C_check_matrix.csv` supports Supplementary Table S2. A missing PASS is not automatically a FAIL. |
| 5D | `Figure5/D_runtime.csv`: the nine primary single-call durations; no CI or timing replicate is implied. Missing-history controls have a different timer boundary and are excluded. |

## Donors, support and denominators

Each `paired_donor_metrics.tsv.gz` contains donor × instance × method × condition records. The metric denominator columns are `scored_cells`, `errors`, `correct_cells`, `eligible_cells`, `reviewed_cells`, `errors_reviewed` and `correct_reviewed`. Repeated instances are not independent cell populations. A row with `AP_status` other than DEFINED is excluded from AP aggregation. Counts in coverage files refer to repeated cell–instance records where indicated, not independent cells.

For each effect, the script writes `reproduced_effects_and_support.tsv`, including the source population's donor count, donors with any defined selected scenario, donors complete across selected scenarios, and the effective donor count for each scenario. `summary_support.tsv.gz` supplies the same checked output with the release. These counts are distinct: the overall estimator averages donors within each scenario and then averages all eight scenarios; it does not restrict every scenario to one complete-case donor intersection. For M4 crossed diagnostics, the original summary already contains `donors_with_any_defined_selected_stratum` and `donors_complete_selected_strata`; the denominator details are in `donor_instance_ap.tsv.gz`, `paired_instance_contrasts.tsv.gz` and `support_audit.tsv.gz`.

Use METHOD_COMMON for the displayed within-method contrasts. RANK_COMMON is a different cross-method intersection. NATIVE_SUPPORT records are retained for coverage checks and should not be substituted for paired support. The saved tables also retain sanity instances and some auxiliary ADT-only records; these are not additional primary comparisons. The primary aggregation uses 16 instances across eight scenarios. All comparisons retain their original exploratory/development/independent-source status.

The intervals use 2,000 paired donor resamples with seed 2026090702 and fixed fitted outputs. `bootstrap/` supplies the actual draws; donor order is sorted donor identity, with the M4 index also supplied. Missing scenarios are not imputed. These intervals do not include model-retraining uncertainty, multiplicity adjustment or equivalence testing.

## Fixed input files

`study_inputs/D4/` and `study_inputs/D3/` contain selected cell and public donor identities, all original outer/inner assignments, the 20 observed-label overlays and evaluation-target vectors, feature identities, reference labels, and saved ADT thresholds for the samples contributing selected cells. No clinical covariates are included. `pilot_cell_index` joins the long label/target tables to the method-cell row. These existing files remove the need to guess a selection seed, reassign folds or construct new label errors.

`original_sources.json` identifies the exact COMBAT and D3 processed-data files, provider records/versions, URLs, sizes and hashes. Provider metadata and CC BY 4.0 attribution are preserved in `original_sources/`. The original multi-gigabyte count files are not redistributed. See `docs/inputs.md` for the tested D3 count-extraction entry and exact D4 count-location instructions.

`manifest.json` binds the selected original source files and their decompressed content. The support table is a separately identified reproduction output; it adds no new fitted scores or uncertainty definition. The text tables and binary bootstrap/CDF arrays are distributed together because exact machine-readable values and index arrays are needed by the executable check; a PDF matrix or a rounded spreadsheet cannot substitute for them.

The verified scope is summary reproduction and preparation of the fixed real-D3 inputs. A full raw-data-to-all-models rerun from this public checkout, independent biological verification of the ADT labels, and a recreation of the final artwork are not claimed.
