"""Small modality-boundary audit, independent of all CellTrust manifests.

Requires numpy and scipy. Modality is supplied by trusted source metadata,
never inferred from gene/antibody names. This checks a preprocessing boundary,
not label truth, cell selection, biological validity, or every downstream model.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy import sparse

def raw_signature(x):
 x=sparse.csr_matrix(x,dtype=np.float64).copy();x.eliminate_zeros();x.sort_indices()
 h=hashlib.sha256()
 for a in [np.asarray(x.shape,np.int64),x.indptr.astype(np.int64),x.indices.astype(np.int64),x.data]:h.update(a.tobytes())
 return h.hexdigest()

def rna_preprocess(counts,modalities,target_sum=10000.0):
 """Select RNA before computing the denominator; no ADT read after selection."""
 keep=np.asarray(modalities,dtype=str)=='RNA'
 if not keep.any():raise ValueError('NO_RNA_FEATURES')
 x=sparse.csr_matrix(counts[:,keep],dtype=np.float64).copy()
 x.eliminate_zeros();x.sort_indices()
 totals=np.asarray(x.sum(axis=1)).ravel()
 if np.any(totals<=0):raise ValueError('ZERO_RNA_TOTAL_NO_SILENT_CELL_DELETION')
 normalized=x.multiply((target_sum/totals)[:,None]).tocsr()
 normalized.data=np.log1p(normalized.data)
 return x,totals,normalized

def audit(counts,modalities,donors,source_metadata,allow_count_estimates=False):
 x=sparse.csr_matrix(counts,dtype=np.float64).copy();x.eliminate_zeros();x.sort_indices()
 mod=np.asarray(modalities,dtype=str);donors=np.asarray(donors,dtype=str)
 if not source_metadata.strip():raise ValueError('SOURCE_MODALITY_PROVENANCE_REQUIRED')
 if x.shape[1]!=len(mod) or x.shape[0]!=len(donors):raise ValueError('METADATA_SHAPE_MISMATCH')
 if set(mod)-{'RNA','ADT'}:raise ValueError('UNRECOGNIZED_MODALITY_EXPLICIT_MAPPING_REQUIRED')
 if not (mod=='ADT').any():raise ValueError('NO_ADT_POSITIVE_CONTROL_UNINFORMATIVE')
 if not np.isfinite(x.data).all() or np.any(x.data<0):raise ValueError('COUNTS_NOT_FINITE_NONNEGATIVE')
 fraction=int(np.count_nonzero(x.data!=np.floor(x.data)))
 if fraction and not allow_count_estimates:raise ValueError('FRACTIONAL_COUNTS_REQUIRE_EXPLICIT_SOURCE_SUPPORTED_COUNT_ESTIMATES')
 rna,total,baseline=rna_preprocess(x,mod)
 rmask=mod=='RNA';amask=mod=='ADT';adt=x[:,amask].tocsr()
 perm=np.arange(len(donors));rng=np.random.default_rng(2026090703)
 for donor in sorted(set(donors)):
  ix=np.flatnonzero(donors==donor);perm[ix]=rng.permutation(ix)
 tests=[]
 for name,mut in [('DELETE',None),('TIMES2',adt*2),('DONOR_SHUFFLE',adt[perm])]:
  modified=rna.copy() if mut is None else sparse.hstack([rna,mut],format='csr')
  mm=np.repeat('RNA',rna.shape[1]) if mut is None else np.r_[np.repeat('RNA',rna.shape[1]),np.repeat('ADT',adt.shape[1])]
  reread,den,norm=rna_preprocess(modified,mm)
  delta=norm-baseline;error=float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
  passed=raw_signature(reread)==raw_signature(rna) and np.array_equal(den,total) and error<=1e-12
  tests.append({'perturbation':name,'raw_RNA_identical':raw_signature(reread)==raw_signature(rna),'RNA_denominator_identical':bool(np.array_equal(den,total)),'max_abs_log_normalized_error':error,'status':'PASS' if passed else 'FAIL'})
 adt_total=np.asarray(adt.sum(axis=1)).ravel();mixed_total=total+adt_total
 residual=total/mixed_total
 report={'schema_version':1,'status':'PASS' if all(t['status']=='PASS' for t in tests) else 'FAIL','source_modality_metadata':source_metadata,'source_metadata_authenticated_by_tool':False,'cells':x.shape[0],'RNA_features':int(rmask.sum()),'ADT_features':int(amask.sum()),'fractional_entries':fraction,'count_estimate_semantics_explicitly_authorized':allow_count_estimates,'RNA_total_min':float(total.min()),'RNA_total_max':float(total.max()),'mixed_denominator_differs':bool(np.any(adt_total>0)),'late_filter_RNA_linear_multiplier_min':float(residual.min()),'late_filter_RNA_linear_multiplier_max':float(residual.max()),'canary_checks':tests,'rtol':1e-10,'atol':1e-12,'limitations':['Metadata authentication remains a source-audit responsibility.','Only raw RNA, denominator and log-normalized RNA are checked by this entrypoint.','Labels, cell population and donor identifiers are held fixed, not independently validated.','No claim is made about downstream models unless separately recomputed and checked.']}
 if not report['mixed_denominator_differs']:raise ValueError('ZERO_ADT_COUNTS_UNINFORMATIVE_POSITIVE_CONTROL')
 return report,baseline

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--demo',action='store_true');p.add_argument('--counts',type=Path);p.add_argument('--features',type=Path);p.add_argument('--donors',type=Path);p.add_argument('--source-modality-metadata');p.add_argument('--allow-count-estimates',action='store_true');p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 if args.output.exists():raise FileExistsError('OUTPUT_EXISTS_REFUSE_OVERWRITE:'+str(args.output))
 if args.demo:
  x=sparse.csr_matrix([[2,5,0,200,4],[7,1,3,4,30],[1,4,6,20,1],[5,0,2,2,2]],dtype=np.float64)
  mods=['RNA']*3+['ADT']*2;donors=['demo_donor_1']*2+['demo_donor_2']*2;source='Artificial five-feature example; modalities assigned by construction, not experimental evidence.'
 else:
  if not all([args.counts,args.features,args.donors,args.source_modality_metadata]):p.error('counts/features/donors/source-modality-metadata required unless --demo')
  x=sparse.load_npz(args.counts)
  with args.features.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
  if not rows or 'modality' not in rows[0] or 'feature_id' not in rows[0]:raise ValueError('FEATURE_TSV_REQUIRES_feature_id_AND_modality')
  if len({(r['modality'],r['feature_id']) for r in rows})!=len(rows):raise ValueError('DUPLICATE_MODALITY_FEATURE_IDENTITIES')
  mods=[r['modality'] for r in rows]
  with args.donors.open() as f:dr=list(csv.DictReader(f,delimiter='\t'))
  if not dr or not {'cell_id','donor_id'}<=set(dr[0]):raise ValueError('DONOR_TSV_REQUIRES_cell_id_AND_donor_id')
  if len({r['cell_id'] for r in dr})!=len(dr):raise ValueError('DUPLICATE_CELL_IDENTITIES')
  donors=[r['donor_id'] for r in dr];source=args.source_modality_metadata
 report,normalized=audit(x,mods,donors,source,args.allow_count_estimates)
 args.output.mkdir(parents=True)
 (args.output/'modality_guard_report.json').write_text(json.dumps(report,indent=2)+'\n')
 sparse.save_npz(args.output/'RNA_log_normalized.npz',normalized)
 print(json.dumps(report,indent=2))
 if report['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
