# Code and data terms

The MIT license applies to original CellTrust code and its documentation. It does not replace the terms of external software, source datasets, marker resources or any separately attributed material.

The fixed cell identities, donor/sample identifiers and derived labels in `source_data/study_inputs/` refer to already public, deidentified data. No names, contact details, clinical covariates or new participant linkages are added. These selected tables do not claim a broader right to the original participants' data.

- **COMBAT (D4):** COvid-19 Multi-omics Blood ATlas (COMBAT) Consortium, *A blood atlas of COVID-19 defines hallmarks of disease severity and specificity*, Cell (2022), DOI [10.1016/j.cell.2022.01.012](https://doi.org/10.1016/j.cell.2022.01.012). The exact processed-data record is [Zenodo 5139561](https://zenodo.org/records/5139561), version 1.0.0, licensed CC BY 4.0 by its provider.
- **Vaccination CITE-seq (D3):** Zhang and colleagues, with the complete creator list preserved in `source_data/original_sources/D3_zenodo_record.json`. The exact processed-data record is [Zenodo 7555405](https://zenodo.org/records/7555405), licensed CC BY 4.0 by its provider. The file used is `PBMC_vaccine_CITE.rds`, not the newer SeuratV5 file.
- **Marker knowledge:** the curated marker table retains its original resource names, PMIDs and source identifiers. MIT does not replace the source resources' terms.
- **MUON tutorial and 10x example:** upstream links, notebook revision and data hashes are recorded in `docs/inputs.md`. The upstream notebook/data are not relicensed as CellTrust software. No dependency source tree or package wheel is distributed here.

For the CC BY 4.0 datasets, this study selected a fixed subset, assigned broad ADT-supported reference labels, partitioned donors and introduced controlled label errors. These modifications are documented in the manuscript and input guide; they are not annotations supplied or endorsed by the original investigators. Obtain the original count matrices from their providers. The source-data release contains result tables and input identities, not the original matrices.
