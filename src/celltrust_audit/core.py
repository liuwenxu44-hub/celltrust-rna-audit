"""CT-MOD-001 method side: no evaluator targets, clean labels or performance."""
from pathlib import Path
import hashlib, math, json, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from celltrust.m1.scores import robust_reference as m1_reference, combine_risk
from celltrust.m2.core import robust_reference as m2_reference
from celltrust.m4.scoring import diagnostics_from_neighbors
from celltrust.m4_compat.scoring import standardize_outer_compat
LABELS=('T_cell','B_cell','NK_cell','monocyte')
METHODS=('M1_RISK','M1_MARGIN','M4_RISK','NEIGHBOR_DISAGREEMENT','M2_RISK','MARKER_MARGIN')
def load_condition(out, condition, adt_change='ORIGINAL', seed=2026090701):
    """Always reread physical views. A never opens any ADT counts file."""
    out=Path(out); cells=pd.read_csv(out/'inputs/method_cells.tsv',sep='\t')
    object_path=out/'canary_objects'/f'{adt_change}.json'
    source=json.loads(object_path.read_text()) if object_path.exists() else {'RNA_counts':'inputs/RNA_counts.npz','ADT_counts':'inputs/ADT_counts.npz'}
    rna=sparse.load_npz(out/source['RNA_counts']).astype(np.float64)
    rf=pd.read_csv(out/'inputs/RNA_features.tsv',sep='\t')
    reads=['inputs/RNA_counts.npz','inputs/RNA_features.tsv','inputs/method_cells.tsv']
    R=np.asarray(rna.sum(axis=1)).ravel()
    if condition=='A':
        counts=rna; features=rf; denominator=R
    else:
        if source['ADT_counts'] is None:
            af=pd.read_csv(out/'inputs/ADT_features.tsv',sep='\t'); adt=sparse.csr_matrix((len(cells),len(af)),dtype=np.float64)
        else:
            adt=sparse.load_npz(out/source['ADT_counts']).astype(np.float64)
            af=pd.read_csv(out/'inputs/ADT_features.tsv',sep='\t'); reads+=['inputs/ADT_counts.npz','inputs/ADT_features.tsv']
            if object_path.exists(): pass  # Actual modified bytes were created by the input-side canary builder.
            elif adt_change=='TIMES2': adt=adt*2
            elif adt_change=='DONOR_SHUFFLE':
                rng=np.random.default_rng(seed); permutation=np.arange(len(cells))
                for donor in sorted(cells.donor_id.unique()):
                    ix=np.flatnonzero(cells.donor_id.to_numpy()==donor); permutation[ix]=rng.permutation(ix)
                adt=adt[permutation].tocsr()
            elif adt_change!='ORIGINAL': raise ValueError(adt_change)
        A=np.asarray(adt.sum(axis=1)).ravel()
        if condition=='B':
            counts=sparse.hstack([rna,adt],format='csr'); features=pd.concat([rf,af],ignore_index=True)
            order=np.argsort(features.original_feature_index.to_numpy(),kind='stable'); counts=counts[:,order]; features=features.iloc[order].reset_index(drop=True)
            denominator=R+A
        elif condition=='C': counts=rna; features=rf; denominator=R+A
        elif condition=='D': counts=adt; features=af; denominator=A
        else: raise ValueError(condition)
    assert np.all(denominator>0), 'ZERO_DENOMINATOR_NOT_DROPPED'
    linear=counts.multiply((10000./denominator)[:,None]).tocsr()
    normalized=linear.copy(); normalized.data=np.log1p(normalized.data)
    return dict(condition=condition,perturbation=adt_change,counts=counts,linear=linear,normalized=normalized,
                features=features,cells=cells,denominator=denominator,reads=reads)
def fold_arrays(cells,context):
    outer=context.split('_')[0]; train=cells.outer_fold.to_numpy()!=outer
    if '_I' not in context: return np.flatnonzero(train), np.flatnonzero(~train)
    inner=context.split('_')[1]; codes=cells[f'inner_{outer}'].to_numpy()
    return np.flatnonzero(train & (codes!=inner)),np.flatnonzero(train & (codes==inner))
def seed_for(context):
    x=f'D4_COMBAT_Zenodo5139561\t{context}\tHVG2000\tCT001B_PCA_v1'.encode()
    return int.from_bytes(hashlib.sha256(x).digest()[:8],'big')%(2**32)
def mm(x,c,s,right): return np.asarray(x@(right/s[:,None])-(c/s)@right,dtype=np.float64)
def tmm(x,c,s,right): return np.asarray((x.T@right-c[:,None]*right.sum(axis=0)[None,:])/s[:,None],dtype=np.float64)
def representation(view,context):
    started=time.time(); counts=view['counts']; norm=view['normalized']; features=view['features']; cells=view['cells']
    train,query=fold_arrays(cells,context); minimum=max(10,math.ceil(.01*len(train)))
    eligible=np.flatnonzero(np.asarray(counts[train].getnnz(axis=0)).ravel()>=minimum)
    assert len(eligible)>0
    x=norm[train][:,eligible]; means=np.asarray(x.mean(axis=0)).ravel(); second=np.asarray(x.power(2).mean(axis=0)).ravel()
    var=np.maximum(0.,second-means*means); names=features.feature_id.to_numpy()[eligible]
    chosen=eligible[np.lexsort((names,-var))[:min(2000,len(eligible))]]
    x=norm[train][:,chosen].tocsr(); center=np.asarray(x.mean(axis=0)).ravel(); second=np.asarray(x.power(2).mean(axis=0)).ravel()
    scale=np.sqrt(np.maximum(0.,second-center*center)); keep=scale>0
    chosen=chosen[keep]; center=center[keep]; scale=scale[keep]; x=x[:,keep]
    k=min(30,len(train)-1,len(chosen)); assert k>0
    oversampled=min(x.shape[0],x.shape[1],k+10)
    omega=np.random.default_rng(seed_for(context)).standard_normal((x.shape[1],oversampled))
    y=mm(x,center,scale,omega)
    for _ in range(2): y=mm(x,center,scale,tmm(x,center,scale,y))
    q,_=np.linalg.qr(y,mode='reduced'); b=tmm(x,center,scale,q).T
    _,singular,vt=np.linalg.svd(b,full_matrices=False); comp=vt[:k]
    coords=mm(norm[:,chosen].tocsr(),center,scale,comp.T)
    # Exact query-to-training graph. No query-to-query or self reference.
    nbr=np.empty((len(query),30),dtype=np.int32); distances=np.empty(nbr.shape,dtype=np.float64)
    ids=cells.cell_id.to_numpy(); assert len(train)>=30
    ref=coords[train]
    for start in range(0,len(query),192):
        stop=min(len(query),start+192); ds=cdist(coords[query[start:stop]],ref,metric='sqeuclidean')
        for local,row in enumerate(ds):
            kth=np.partition(row,29)[29]; cand=np.flatnonzero(row<=kth); chosen_local=cand[np.lexsort((ids[train[cand]],row[cand]))[:30]]
            nbr[start+local]=train[chosen_local]; distances[start+local]=np.sqrt(row[chosen_local])
    assert not np.isin(nbr,query).any()
    assert not set(cells.donor_id.iloc[query]) & set(cells.donor_id.iloc[train])
    selected_modality=features.modality.to_numpy()[chosen]
    return dict(context=context,train=train,query=query,eligible=eligible,selected=chosen,center=center,scale=scale,
                components=comp,coordinates=coords,neighbors=nbr,distances=distances,singular=singular,
                RNA_selected=int(np.sum(selected_modality=='RNA')),ADT_selected=int(np.sum(selected_modality=='ADT')),
                runtime_seconds=time.time()-started,seed=seed_for(context))
def marker_components(view,marker_rows,check_scaled_ranks=False):
    """Marker matches only explicitly RNA-typed IDs; B rank universe remains mixed."""
    f=view['features']; counts=view['counts']; n,p=counts.shape
    lookup={r.feature_id:i for i,r in enumerate(f.itertuples()) if r.modality=='RNA'}
    bylabel={l:{'CORE_POSITIVE':[],'EXPLICIT_NEGATIVE':[]} for l in LABELS}
    for r in marker_rows.itertuples():
        if r.label in LABELS and str(r.allowed_for_primary)=='TRUE' and r.marker_role in bylabel[r.label] and r.ensembl_gene_id in lookup:
            bylabel[r.label][r.marker_role].append(lookup[r.ensembl_gene_id])
    for l in LABELS:
        for role in bylabel[l]: bylabel[l][role]=sorted(set(bylabel[l][role]))
    union=sorted({i for d in bylabel.values() for a in d.values() for i in a}); positions={g:i for i,g in enumerate(union)}
    gene_to_marker=np.full(p,-1,dtype=np.int32); gene_to_marker[union]=np.arange(len(union))
    ranks=np.zeros((n,len(union)),dtype=np.float64); scaled_error=0.
    for i in range(n):
        a,b=counts.indptr[i:i+2]; vals=counts.data[a:b]; idx=counts.indices[a:b]
        rr=(p-len(vals)+rankdata(vals,method='average'))/p
        pos=gene_to_marker[idx]; keep=pos>=0; ranks[i,pos[keep]]=rr[keep]
        if check_scaled_ranks:
            ss=(p-len(vals)+rankdata(view['linear'].data[a:b],method='average'))/p
            scaled_error=max(scaled_error,float(np.max(np.abs(rr-ss))) if len(rr) else 0.)
    result=np.full((n,4,3),np.nan)
    for k,l in enumerate(LABELS):
        pp=[positions[j] for j in bylabel[l]['CORE_POSITIVE']]; nn=[positions[j] for j in bylabel[l]['EXPLICIT_NEGATIVE']]
        if len(pp)<5: continue
        a=ranks[:,pp]; result[:,k,0]=a.mean(axis=1)
        if len(pp)>=4: result[:,k,1]=np.partition(a,max(0,math.ceil(.25*len(pp))-1),axis=1)[:,max(0,math.ceil(.25*len(pp))-1)]
        if len(nn)>=2: result[:,k,2]=ranks[:,nn].mean(axis=1)
    return dict(values=result,marker_ranks=ranks,matched=bylabel,scaled_rank_max_error=scaled_error)
def observed_markers(comp,labels):
    out={k:np.full(len(labels),np.nan) for k in ['P','Q','N','M']}
    for code,label in enumerate(LABELS):
        ix=np.flatnonzero(labels==label)
        for j,key in enumerate(['P','Q','N']): out[key][ix]=comp[ix,code,j]
        alternatives=np.delete(comp[ix,:,0],code,axis=1)
        available=np.isfinite(alternatives).any(axis=1)
        out['M'][ix[available]]=np.nanmax(alternatives[available],axis=1)-out['P'][ix[available]]
    return out
def m1_components(coords,labels,train,query,minimum):
    prot={l:coords[train[labels[train]==l]].mean(axis=0) for l in LABELS if np.sum(labels[train]==l)>=minimum}
    legal=sorted(prot); obs=np.full(len(query),np.nan); alt=obs.copy()
    if not legal: return obs,alt
    ds=np.column_stack([np.linalg.norm(coords[query]-prot[l],axis=1) for l in legal])
    for j,l in enumerate(legal):
        ix=np.flatnonzero(labels[query]==l); obs[ix]=ds[ix,j]
        if len(legal)>1: alt[ix]=np.min(np.delete(ds[ix],j,axis=1),axis=1)
    return obs,alt
def score_outer(labels,cells,rep,inner_reps,markers,condition):
    tr,qu=rep['train'],rep['query']; result=np.full((len(qu),6),np.nan)
    d,a=m1_components(rep['coordinates'],labels,tr,qu,20)
    result[:,1]=np.log((d+1e-8)/(a+1e-8))
    codes=np.array([LABELS.index(x) for x in labels],dtype=np.int8)
    def diag(r): return diagnostics_from_neighbors(codes[r['query']],codes[r['neighbors']],np.full(r['neighbors'].shape,1/30),4)
    diag_outer=diag(rep); result[:,3]=1-diag_outer['S']
    if condition=='D': return result
    # Frozen M1 formula uses inner-held-out prototypes in the outer-fitted space,
    # exactly as m1/runner.py; P2 four-fold membership supersedes old three-fold.
    inner_ids=[]; abs_ref=[]; alt_ref=[]; m4_refs=[]
    for r in inner_reps:
        dd,aa=m1_components(rep['coordinates'],labels,r['train'],r['query'],12)
        inner_ids.append(r['query']); abs_ref.append(dd); alt_ref.append(aa); m4_refs.append(diag(r))
    ix=np.concatenate(inner_ids); rd=np.concatenate(abs_ref); ra=np.concatenate(alt_ref); rm=np.log((rd+1e-8)/(ra+1e-8))
    assert set(ix)==set(tr) and len(ix)==len(tr)
    for l in LABELS:
        target=np.flatnonzero(labels[qu]==l); pick=labels[ix]==l
        ar=m1_reference(rd[pick],rd); mr=m1_reference(rm[pick],rm)
        for j in target:
            if not np.isfinite(d[j]): continue
            value=combine_risk(d[j],a[j] if np.isfinite(a[j]) else None,ar,mr)['R']
            if value is not None: result[j,0]=value
    observed=observed_markers(markers['values'],labels); z=np.full((4,len(qu)),np.nan)
    for j,key in enumerate(['P','Q','N','M']):
        for l in LABELS:
            target=np.flatnonzero(labels[qu]==l); refs=tr[labels[tr]==l]
            c,s,_=m2_reference(observed[key][refs],observed[key][tr])
            if np.isfinite(s): z[j,target]=(-1 if key in ['P','Q'] else 1)*(observed[key][qu[target]]-c)/s
    valid=np.isfinite(z).any(axis=0); result[valid,4]=np.nanmax(z[:,valid],axis=0); result[:,5]=observed['M'][qu]
    reference={k:np.concatenate([x[k] for x in m4_refs]) for k in ['S','G','H']}
    compat,*_=standardize_outer_compat(reference,diag_outer,codes[ix],codes[qu],12,cells.cell_id.to_numpy()[ix],cells.cell_id.to_numpy()[qu])
    zz=np.vstack([compat['Z_S'],compat['Z_G']]); valid=np.isfinite(zz).any(axis=0); result[valid,2]=np.nanmax(zz[:,valid],axis=0)
    return result
