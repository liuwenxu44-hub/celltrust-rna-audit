# CellTrust RNA workflow audit

Research code accompanying *Auditing RNA-only CITE-seq workflows: context-dependent score changes and execution checks*.

This is a focused source release, not a clinical tool or an annotation-correction package. It contains the accepted case-study implementations, not the discontinued fusion/calibration development route.

## Contents

- `src/celltrust_audit/core.py`: matched RNA-first (A), mixed RNA+ADT (B), late ADT removal (C), and ADT-only diagnostic (D); donor-held-out representations and six M1/M2/M4-related scores.
- `src/celltrust/`: the original scoring formula dependencies, including the M4 zero-scale compatibility rule.
- `src/celltrust_audit/metrics.py` and `statistics.py`: tie-aware average precision, fixed 10% review-budget summaries, and paired donor-cluster bootstrap. The review budget is an evaluation device, not a recommended operational threshold.
- `scripts/modality_guard.py`: standalone RNA-boundary counterfactual checks using explicit source modality metadata.
- `scripts/realpath_producer.py` and `realpath_guard.py`: the bounded D3/O1 instrumented representation replay and independent numerical checker.
- `scripts/muon_replay.py`: the pinned public MUON RNA-branch replay; deliberately injected faults are kept separate from original authored behavior.
- `analysis/m4_mechanism.py`: posthoc raw-condition/reference-condition cross-standardization and its affine-order control.
- `analysis/d3_label_audit.py`: the bounded exploratory fractional-ADT label-reference audit; `reference/derive_adt_truth.py` and `adt_reference_rules.yaml` preserve its classifier source and rule input. This does not establish new ground truth.
- `reference/scanpy_reference.py`: unchanged scientific reference source used by the real-path replay. It is a source anchor, not a standalone public command: its historical orchestration expects study-specific inputs.
- `config/`: required marker definitions and scientific parameters only.
- `tests/`: artificial-data smoke/regression tests; no participant records.

There are no experiment logs, internal audit records, raw/derived participant matrices, manuscripts, figures, environment copies, or backup archives in this repository. Reports created when running the tools belong in your own ignored `runs/` directory and are not distributed here.

## Quick start

Python 3.12 on Linux is the recorded execution platform. In a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -B -m unittest discover -s tests -v
python -B scripts/modality_guard.py --demo --output runs/modality-demo
```

The demo is synthetic and does not reproduce a manuscript effect size. The core source uses explicit observed labels and donor assignments; it does not read evaluation targets. See [input requirements](docs/inputs.md).

For one supplied label vector and the existing fold assignments:

```bash
python -B scripts/score.py --inputs prepared_dataset \
  --labels observed_labels.tsv --markers config/markers.tsv \
  --dataset D4 --condition A --output runs/score-A.npz
```

`--dataset D3` uses the study's exact-symbol marker adapter and D3 seed namespace. B/C/D are intentional diagnostic comparators, not recommended RNA-only workflows. No sampling, relabeling or fold selection is performed by this command.

## Execution-checking code

The real-path checker is deliberately limited to the SHA-bound D3/O1 source export and recipe. Its source identities, numerical tolerances and abstention rules are unchanged. It cannot certify arbitrary inputs; `NOT_VERIFIABLE` is not a detected error. It does not inspect label correctness or biological performance and is not a hostile-file security sandbox.

For these optional scripts, install `requirements-execution.txt` into a separate environment. These are the recorded package pins, not a claim of validation under every platform. Set `CELLTRUST_AUDIT_ROOT` to an explicit workspace containing all supplied inputs, output directories, source anchors and contracts. It defaults to the current directory; it is not a server address.

```bash
export CELLTRUST_AUDIT_ROOT="$(pwd)"
python -B scripts/realpath_guard.py --contract runs/case/contract.json
```

The public MUON adapter requires the pinned original notebook and 10x filtered-count inputs acquired by the user, plus locally generated acquisition bindings. Their source locations and required schema are in `docs/inputs.md`. No upstream notebook, dataset or third-party dependency is vendored here. Do not execute an arbitrary downloaded notebook wholesale.

`analysis/m4_mechanism.py` exposes the original numerical functions without private source-discovery and finalization code; `python -B analysis/m4_mechanism.py --self-test` is a data-free test. The ADT side-audit command preserves its study-specific input identities and needs the original RDS/exported thresholds and R dependencies; the source release does not include those participant-level files or claim a fresh execution of that audit.

## Interpretation and limits

The study compares fixed matched cells, labels, instances and folds. Score changes are context-dependent: they are not a universal inflation claim, and the external cohort is not a transfer of a frozen D4 classifier. Uncertainty is paired donor resampling of fixed fitted scores, not full training uncertainty. The M1 inner prototype reference retains the documented outer-fitted feature-space limitation. Marker availability and reference knowledge differ across lineages, including omitted-label scenarios.

The public workflow replay uses a recorded contemporary environment and is not historical byte-exact notebook execution. The checker fixtures are dependent technical tests, not field prevalence or population sensitivity estimates.

This release checks source integrity and synthetic functionality. It does **not** distribute the study-specific cell/donor/target records or claim that a fresh complete manuscript reproduction has been performed from this repository. Obtain public data from its original providers and preserve the study's explicit assay metadata, count semantics, sampling and fold definitions rather than silently substituting a new population. The repository contains no new scientific results.

## License status

No project-wide software license has yet been assigned to this initial public source release. Public availability is not a grant of an MIT, Apache or other open-source license. Third-party tools and public data remain under their own licenses and terms; they are not included in this repository.

