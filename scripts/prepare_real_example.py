"""Prepare the fixed 7,200-cell D3 real example from the exact public RDS.

Does not download data, select a new population, derive labels, fit thresholds,
or run scoring. Requires Rscript with SeuratObject, Matrix and rhdf5 plus the
Python packages in requirements.txt and h5py. Existing destinations are refused.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, subprocess
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

EXPECTED={'RNA_counts.npz':'464cad79562bc0f638c45f6b275e44926c8a8bb5137f93a715e89ca5bbe17e00',
          'ADT_counts.npz':'aaf04a29073d26f07a3a67c68d184b89461fa13806ebe898a86d5a90688eec42'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def decode(a):return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in a])
def run(raw,out,root,instance=None):
    assert not out.exists(),'Use a new output directory; no existing file is overwritten'
    sources=json.loads((root/'source_data/original_sources.json').read_text())
    record=sources['D3'];assert raw.stat().st_size==record['bytes'] and sha(raw)==record['sha256'],'Wrong RDS record/version'
    manifest=json.loads((root/'source_data/manifest.json').read_text())['files']
    for f in manifest:
        if f['path'].startswith('study_inputs/D3/'):
            assert sha(root/'source_data'/f['path'])==f['sha256'],f['path']
    out.mkdir(parents=True);inputs=out/'inputs';inputs.mkdir()
    source=root/'source_data/study_inputs/D3'
    for name in ['method_cells.tsv','RNA_features.tsv','ADT_features.tsv','evaluation_targets.tsv','observed_label_overlays.tsv','reference_labels.tsv','adt_thresholds.tsv']:
        (inputs/name).write_bytes(gzip.decompress((source/(name+'.gz')).read_bytes()))
    (out/'selected_instances.tsv').write_bytes((source/'instances.tsv').read_bytes())
    intermediate=out/'selected_source_counts.h5'
    subprocess.run(['Rscript',str(root/'scripts/export_selected_d3.R'),str(raw),str(inputs/'method_cells.tsv'),str(intermediate)],check=True)
    cells=pd.read_csv(inputs/'method_cells.tsv',sep='\t');identities=[]
    with h5py.File(intermediate,'r') as h:
        for modality in ['RNA','ADT']:
            g=h[modality];dim=g['Dim'][:]
            x=sparse.csr_matrix((g['x'][:],g['i'][:],g['p'][:]),shape=(int(dim[1]),int(dim[0])))
            assert np.array_equal(decode(g['cell_ids'][:]),cells.cell_id.to_numpy())
            features=pd.read_csv(inputs/f'{modality}_features.tsv',sep='\t')
            assert np.array_equal(decode(g['feature_names'][:]),features.feature_id.to_numpy())
            assert np.isfinite(x.data).all() and (x.data>=0).all()
            x.eliminate_zeros();x.sort_indices();dest=inputs/f'{modality}_counts.npz';sparse.save_npz(dest,x)
            digest=sha(dest)
            # This exact check intentionally does not silently relax the frozen
            # realpath producer's identities to accept a different serialization.
            assert digest==EXPECTED[dest.name],('Prepared file does not match frozen bytes',dest.name,digest)
            identities.append({'file':dest.name,'sha256':digest,'shape':list(x.shape),'fractional_entries':int((x.data!=np.floor(x.data)).sum())})
    labels=pd.read_csv(inputs/'observed_label_overlays.tsv',sep='\t')
    instances=pd.read_csv(out/'selected_instances.tsv',sep='\t')
    selected=instance or instances.instance_id.iloc[0]
    lab=labels[labels.instance_id==selected].sort_values('pilot_cell_index')
    assert len(lab)==len(cells) and np.array_equal(lab.pilot_cell_index,cells.pilot_cell_index)
    pd.DataFrame({'cell_id':cells.cell_id,'observed_label':lab.observed_label.to_numpy()}).to_csv(out/'observed_labels.tsv',sep='\t',index=False)
    report={'status':'REAL_D3_INPUT_PREPARATION_PASS','raw_source_sha256':record['sha256'],'prepared_matrices':identities,
            'selected_instance':selected,'cells':len(cells),'donors':cells.donor_id.nunique(),'new_selection':False,'new_reference_labels':False,'scores_computed':False}
    (out/'input_preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rds',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--instance')
    a=p.parse_args();run(a.rds.resolve(),a.output.resolve(),Path(__file__).resolve().parents[1],a.instance)
