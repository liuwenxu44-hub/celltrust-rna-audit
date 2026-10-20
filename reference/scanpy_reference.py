"""Specified Scanpy-function reference; query donors never enter fitting."""
import hashlib,json,math,sys,time
from pathlib import Path
import anndata as ad,numpy as np,pandas as pd,scanpy as sc,yaml
from scipy import sparse
from scipy.spatial.distance import cdist
root,out=map(Path,sys.argv[1:3]);dataset,condition=sys.argv[3:5]
assert dataset in ['D4','D3'] and condition in 'ABC'
protocol=yaml.safe_load((out/'protocol_and_D3_unlock.yaml').read_text())
assert sc.__version__==protocol['scanpy_reference']['version']
old=Path(json.loads((out/'intake/D4_readonly_relocation_binding.json').read_text())['source'])
source=old if dataset=='D4' else out/'D3'
dest=out/'scanpy_reference'/dataset;dest.mkdir(parents=True,exist_ok=True)
target=dest/f'{condition}.npz';assert not target.exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
for p,s in json.loads((out/'preperformance_freeze.json').read_text())['bound_files'].items():assert sha(out/p)==s
cells=pd.read_csv(source/'inputs/method_cells.tsv',sep='\t');ids=cells.cell_id.to_numpy();n=len(cells)
instances=pd.read_csv(source/'selected_instances.tsv',sep='\t',keep_default_na=False)
overlays=pd.read_csv(source/'inputs/observed_label_overlays.tsv',sep='\t',keep_default_na=False)
rna=sparse.load_npz(source/'inputs/RNA_counts.npz').astype(np.float64);rf=pd.read_csv(source/'inputs/RNA_features.tsv',sep='\t')
reads=['RNA_counts.npz','RNA_features.tsv'];start=time.time()
if condition=='A':counts=rna;features=rf.copy()
else:
 adt=sparse.load_npz(source/'inputs/ADT_counts.npz').astype(np.float64);af=pd.read_csv(source/'inputs/ADT_features.tsv',sep='\t');reads+=['ADT_counts.npz','ADT_features.tsv']
 counts=sparse.hstack([rna,adt],format='csr');features=pd.concat([rf,af],ignore_index=True)
 order=np.argsort(features.original_feature_index.to_numpy(),kind='stable');counts=counts[:,order];features=features.iloc[order].reset_index(drop=True)
names=np.array([f'{m}:{i}:{f}' for i,(m,f) in enumerate(zip(features.modality,features.feature_id))])
obj=ad.AnnData(X=counts.copy());obj.obs_names=ids;obj.var_names=names
sc.pp.normalize_total(obj,target_sum=10000,exclude_highly_expressed=False,inplace=True)
sc.pp.log1p(obj)
if condition=='C':
 keep=features.modality.to_numpy()=='RNA';obj=obj[:,keep].copy();counts=counts[:,keep];features=features[keep].reset_index(drop=True)
scores=np.full((len(instances),n,1),np.nan);repdata={};flow=[]
for context in sorted(cells.outer_fold.unique()):
 tr=np.flatnonzero(cells.outer_fold.to_numpy()!=context);qu=np.flatnonzero(cells.outer_fold.to_numpy()==context)
 assert not set(cells.donor_id.iloc[tr])&set(cells.donor_id.iloc[qu])
 minimum=max(10,math.ceil(.01*len(tr)))
 filter_mask,_=sc.pp.filter_genes(counts[tr],min_cells=minimum,inplace=False)
 eligible=np.flatnonzero(filter_mask);train=obj[tr,eligible].copy()
 sc.pp.highly_variable_genes(train,n_top_genes=min(2000,len(eligible)),flavor='seurat',n_bins=20,batch_key=None,subset=False,inplace=True)
 chosen_local=np.flatnonzero(train.var.highly_variable.to_numpy());chosen=eligible[chosen_local]
 train=train[:,chosen_local].copy()
 sc.pp.scale(train,zero_center=True,max_value=None)
 center=train.var['mean'].to_numpy();scale=train.var['std'].to_numpy()
 seed=int.from_bytes(hashlib.sha256(f'{dataset}_SCANPY_REFERENCE|{context}|CTMOD002_V1'.encode()).digest()[:8],'big')%(2**32)
 k=min(30,len(tr)-1,len(chosen)-1);assert k>0
 sc.pp.pca(train,n_comps=k,zero_center=True,svd_solver='arpack',random_state=seed,dtype='float64')
 components=train.varm['PCs'];pca_center=np.asarray(train.X.mean(axis=0)).ravel()
 X=obj.X[:,chosen].toarray();scaled=(X-center)/scale
 coords=(scaled-pca_center)@components
 assert np.allclose(coords[tr],train.obsm['X_pca'],rtol=1e-8,atol=1e-8),'SCANPY_QUERY_PROJECTION_MISMATCH'
 neighbors=np.empty((len(qu),30),dtype=np.int32)
 for begin in range(0,len(qu),192):
  ds=cdist(coords[qu[begin:begin+192]],coords[tr],metric='sqeuclidean')
  for j,row in enumerate(ds):
   kth=np.partition(row,29)[29];cand=np.flatnonzero(row<=kth);neighbors[begin+j]=tr[cand[np.lexsort((ids[tr[cand]],row[cand]))[:30]]]
 assert not np.isin(neighbors,qu).any()
 for v,ins in enumerate(instances.itertuples()):
  labs=overlays[overlays.instance_id==ins.instance_id].sort_values('pilot_cell_index').observed_label.to_numpy();assert len(labs)==n
  scores[v,qu,0]=1-np.mean(labs[neighbors]==labs[qu,None],axis=1)
 for key,value in [('train',tr),('query',qu),('eligible',eligible),('selected',chosen),('center',center),('scale',scale),('components',components),('coordinates',coords),('neighbors',neighbors)]:repdata[context+'__'+key]=value
 flow.append({'dataset':dataset,'condition':condition,'outer_context':context,'training_cells':len(tr),'query_cells':len(qu),'RNA_HVG':int((features.modality.to_numpy()[chosen]=='RNA').sum()),'ADT_HVG':int((features.modality.to_numpy()[chosen]=='ADT').sum()),'pcs':k,'k_neighbors':30,'query_donor_fit_excluded':True,'seed':seed})
 print(json.dumps({'dataset':dataset,'condition':condition,'outer_complete':context}),flush=True)
np.savez_compressed(target,scores=scores,methods=np.array(['SCANPY_NEIGHBOR_DISAGREEMENT']),cell_ids=ids.astype(str),instance_ids=instances.instance_id.to_numpy(dtype=str))
np.savez_compressed(dest/f'{condition}_representation.npz',**repdata)
pd.DataFrame(flow).to_csv(dest/f'{condition}_flow.tsv',sep='\t',index=False)
(dest/f'{condition}_receipt.json').write_text(json.dumps({'dataset':dataset,'condition':condition,'score_sha256':sha(target),'cells':n,'instances':len(instances),'version':sc.__version__,'reads':reads,'new_target_performance_access':False,'elapsed_seconds':time.time()-start,'scope':'Specified Scanpy preprocessing followed by exact held-out query-to-training neighborhood disagreement; B/C are intentional input comparators, not Scanpy defaults.'},indent=2)+'\n')
