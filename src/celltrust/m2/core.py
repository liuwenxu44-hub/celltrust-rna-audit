"""Truth-blind M2 component and cross-fitted score calculations."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.stats import rankdata

EPS=1e-8

def within_cell_ranks(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Rank all measurable genes per cell; zero=0, positive ties=average, max<=1."""
    out=counts.astype(float).tocsr(copy=True); n_gene=out.shape[1]
    for i in range(out.shape[0]):
        a,b=out.indptr[i:i+2]; values=out.data[a:b]; nz=len(values)
        if nz: out.data[a:b]=(n_gene-nz+rankdata(values,method='average'))/n_gene
    return out

def weighted_mean(v,w):
    return float(np.dot(v,w)/np.sum(w)) if len(v) and np.sum(w)>0 else np.nan

def weighted_quantile(v,w,q=.25):
    if not len(v) or np.sum(w)<=0:return np.nan
    order=np.argsort(v,kind='mergesort'); vv=np.asarray(v)[order]; ww=np.asarray(w)[order]
    return float(vv[np.searchsorted(np.cumsum(ww),q*np.sum(ww),side='left')])

def module_components(counts, ranks, library_sizes, genes, modules, weighted=False):
    """Return truth-independent cell x legal-label raw marker components."""
    pos={}; neg={}; labels=sorted({r['label'] for r in modules if r['marker_role']=='CORE_POSITIVE'})
    index={g:i for i,g in enumerate(genes)}
    for label in labels:
        for role,target in [('CORE_POSITIVE',pos),('EXPLICIT_NEGATIVE',neg)]:
            rs=[r for r in modules if r['label']==label and r['marker_role']==role and r.get('ensembl_gene_id','') in index]
            target[label]=([index[r['ensembl_gene_id']] for r in rs],np.asarray([float(r['evidence_weight']) if weighted else 1.0 for r in rs]))
    n=counts.shape[0]; out={k:np.full((n,len(labels)),np.nan) for k in ('P','Q','N','detect','logmean')}
    for j,label in enumerate(labels):
        idx,w=pos[label]
        if not idx:continue
        rv=ranks[:,idx].toarray(); cv=counts[:,idx].toarray()
        out['P'][:,j]=np.average(rv,axis=1,weights=w)
        if len(idx)>=4:out['Q'][:,j]=[weighted_quantile(x,w,.25) for x in rv]
        out['detect'][:,j]=np.dot(cv>0,w)/np.sum(w)
        norm=np.log1p(cv*10000/np.maximum(library_sizes[:,None],1.0));out['logmean'][:,j]=np.average(norm,axis=1,weights=w)
        nidx,nw=neg.get(label,([],np.asarray([])))
        if len(nidx)>=2:out['N'][:,j]=np.average(ranks[:,nidx].toarray(),axis=1,weights=nw)
    out['labels']=labels;out['positive_size']={l:len(pos[l][0]) for l in labels};out['negative_size']={l:len(neg.get(l,([],[]))[0]) for l in labels}
    return out

def observed_components(all_components, observed):
    labels=all_components['labels']; lookup={x:i for i,x in enumerate(labels)}; n=len(observed)
    out={k:np.full(n,np.nan) for k in ('P','Q','N','detect','logmean','Alt','M')};out['best_alt']=np.full(n,'',dtype=object)
    for i,y in enumerate(observed):
        if y not in lookup:continue
        j=lookup[y]
        for k in ('P','Q','N','detect','logmean'):out[k][i]=all_components[k][i,j]
        alts=[(all_components['P'][i,k],lab) for k,lab in enumerate(labels) if lab!=y and np.isfinite(all_components['P'][i,k])]
        if alts:
            alt,lab=max(alts,key=lambda x:(x[0],x[1]));out['Alt'][i]=alt;out['M'][i]=alt-out['P'][i];out['best_alt'][i]=lab
    return out

def robust_reference(values,pooled):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan,'MISSING_SCALE_ESTIMATE'
    center=float(np.median(x));mad=float(np.median(np.abs(x-center)))
    if mad>0:return center,1.4826*mad+EPS,'MAD'
    q=np.quantile(x,[.25,.75],method='linear');s=float((q[1]-q[0])/1.349)
    if s>0:return center,s+EPS,'IQR_OVER_1.349'
    p=np.asarray(pooled,dtype=float);p=p[np.isfinite(p)]
    if len(p):
        pc=float(np.median(p));pm=float(np.median(np.abs(p-pc)))
        if pm>0:return center,1.4826*pm+EPS,'POOLED_MAD'
        q=np.quantile(p,[.25,.75],method='linear');s=float((q[1]-q[0])/1.349)
        if s>0:return center,s+EPS,'POOLED_IQR_OVER_1.349'
    return center,np.nan,'MISSING_SCALE_ESTIMATE'

def crossfit_scores(comp, observed, outer_fold, legal_mask=None, self_included=False, inner_fold=None):
    """Observed-label reference standardization. No truth/evaluator input is accepted."""
    n=len(observed);legal=np.ones(n,dtype=bool) if legal_mask is None else np.asarray(legal_mask,dtype=bool)
    z={k:np.full(n,np.nan) for k in ('P','Q','N','M','logmean')};scale={k:np.full(n,'',dtype=object) for k in z}
    for fold in sorted(set(outer_fold)):
        test=np.flatnonzero(legal & (outer_fold==fold));train=np.flatnonzero(legal if self_included else (legal & (outer_fold!=fold)))
        # Marker components have no fitted parameters.  Their inner-cross-fitted
        # reference distribution is therefore exactly the union of the three
        # held-out inner partitions.  Construct that union explicitly so the
        # K=3 contract is auditable without changing its order-invariant values.
        if inner_fold is not None and not self_included:
            parts=[train[inner_fold[train]==k] for k in range(3)]
            if set(np.concatenate(parts))!=set(train):raise RuntimeError('INNER_FOLD_PARTITION_FAILURE')
            train=np.concatenate(parts)
        for key in z:
            pooled=comp[key][train]
            for y in sorted(set(observed[test])):
                ti=test[observed[test]==y];ref=comp[key][train[observed[train]==y]]
                center,s,status=robust_reference(ref,pooled);scale[key][ti]=status
                if np.isfinite(s):
                    sign=-1 if key in ('P','Q','logmean') else 1
                    z[key][ti]=sign*(comp[key][ti]-center)/s
    methods={
      'M2_LOGMEAN_POSITIVE_ONLY':['logmean'],
      'M2_RANK_POSITIVE_MARGIN':['P','M'],
      'M2_RANK_POSNEG':['P','N','M'],
      'M2_RANK_COHERENCE_POSNEG':['P','Q','N','M'],
      'M2_RANK_WEIGHTED_COHERENCE_POSNEG':['P','Q','N','M'],
    }
    risks={}
    for method,keys in methods.items():
        a=np.vstack([z[k] for k in keys]);valid=np.any(np.isfinite(a),axis=0);r=np.full(n,np.nan);r[valid]=np.nanmax(a[:,valid],axis=0);risks[method]=r
    return z,scale,risks
