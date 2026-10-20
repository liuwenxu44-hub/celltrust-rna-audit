#!/usr/bin/env python3
"""Derive CT-001H Level1 truth from a physically ADT-only HDF5 interface."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

ALLOWED_TOP_KEYS = {'counts', 'cells', 'antibodies', 'provenance'}
FORBIDDEN_TOKENS = {'rna','pca','umap','cluster','author','m1','m2','m4','m5a','celltypist','azimuth','singler','scmap','truth'}


def decode(x) -> np.ndarray:
    a=np.asarray(x)
    if a.dtype.kind in 'SO': return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in a],dtype=object)
    return a.astype(str)


def clean_marker(name: str) -> str:
    x=name.upper().replace('ANTI-','').replace('ANTI_','')
    # Preserve the biological CD number before removing panel duplicate suffixes
    # such as CD3-1 or CD56-2.
    m=re.search(r'CD[0-9]+[A-Z]?',x)
    if m: return m.group(0)
    x=re.sub(r'[^A-Z0-9]+','',x)
    x=x.replace('HUMAN','').replace('TOTAL','')
    aliases={'NCAM1':'CD56','FCGR3A':'CD16','MS4A1':'CD20','ITGAM':'CD11B','FCGR1A':'CD64','NCR1':'CD335'}
    for k,v in aliases.items():
        if k in x: return v
    return x


def gmm_threshold(x: np.ndarray, min_sep: float) -> tuple[float,str,bool,float]:
    x=x[np.isfinite(x)]
    if x.size < 100 or np.nanstd(x) < 1e-8:
        return float(np.nanquantile(x,0.75)) if x.size else float('nan'),'Q75_FALLBACK',False,0.0
    mu=np.array(np.quantile(x,[0.25,0.75]),dtype=float); var=np.repeat(max(np.var(x),1e-4),2); w=np.array([0.5,0.5]); converged=False
    for _ in range(100):
        logp=np.column_stack([np.log(w[j]+1e-12)-0.5*np.log(2*np.pi*var[j])-0.5*(x-mu[j])**2/var[j] for j in range(2)])
        logp-=logp.max(1,keepdims=True); r=np.exp(logp); r/=r.sum(1,keepdims=True)
        nk=r.sum(0)+1e-12; nmu=(r*x[:,None]).sum(0)/nk; nvar=(r*(x[:,None]-nmu)**2).sum(0)/nk; nw=nk/x.size
        if np.max(np.abs(nmu-mu)) < 1e-5: converged=True
        mu,var,w=nmu,np.maximum(nvar,1e-6),nw
        order=np.argsort(mu); mu,var,w=mu[order],var[order],w[order]
        if converged: break
    sep=float((mu[1]-mu[0])/np.sqrt(0.5*(var[0]+var[1])))
    if not converged or sep < min_sep:
        return float(np.quantile(x,0.75)),'Q75_FALLBACK',converged,sep
    # Stable grid solution of equal posterior density between component means.
    grid=np.linspace(mu[0],mu[1],2049)
    lp0=np.log(w[0]+1e-12)-0.5*np.log(var[0])-0.5*(grid-mu[0])**2/var[0]
    lp1=np.log(w[1]+1e-12)-0.5*np.log(var[1])-0.5*(grid-mu[1])**2/var[1]
    return float(grid[np.argmin(np.abs(lp0-lp1))]),'TWO_COMPONENT_GMM',True,sep


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--adt-only-h5',required=True); ap.add_argument('--config',required=True)
    ap.add_argument('--truth-out',required=True); ap.add_argument('--threshold-out',required=True); ap.add_argument('--summary-out',required=True)
    a=ap.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text())
    with h5py.File(a.adt_only_h5,'r') as h:
        top=set(h.keys())
        if not top <= ALLOWED_TOP_KEYS: raise PermissionError('CT001H_ADT_ONLY_INTERFACE_REJECTED:'+','.join(sorted(top-ALLOWED_TOP_KEYS)))
        all_names=[]
        def collect(name): all_names.append(name.lower())
        h.visit(collect)
        hits=sorted({t for n in all_names for t in FORBIDDEN_TOKENS if t in n})
        if hits: raise PermissionError('CT001H_ADT_ONLY_INTERFACE_FORBIDDEN_KEYS:'+','.join(hits))
        counts=np.asarray(h['counts/matrix'],dtype=np.float32)
        cell_id=decode(h['cells/cell_id']); donor=decode(h['cells/donor_id']); sample=decode(h['cells/sample_id']); antibody=decode(h['antibodies/name'])
    if counts.shape == (len(antibody),len(cell_id)):
        counts=counts.T
    if counts.shape != (len(cell_id),len(antibody)): raise ValueError('ADT_DIMENSION_MISMATCH')
    if len(set(cell_id)) != len(cell_id): raise ValueError('DUPLICATE_CELL_ID')
    if any(x=='' or x.lower()=='nan' for x in donor): raise ValueError('MISSING_DONOR_ID')
    marker=[clean_marker(x) for x in antibody]; mindex={}
    for i,m in enumerate(marker): mindex.setdefault(m,[]).append(i)
    # Within-sample CLR: log1p(count/geometric-mean-positive-pseudocount).
    clr=np.empty_like(counts,dtype=np.float32); lib=counts.sum(1)
    threshold_rows=[]; threshold={}; min_sep=float(cfg['thresholding']['minimum_component_separation_sd'])
    for s in sorted(set(sample)):
        ix=np.flatnonzero(sample==s); z=np.log1p(np.maximum(counts[ix],0)); z-=z.mean(1,keepdims=True); clr[ix]=z
        for j,name in enumerate(antibody):
            th,method,conv,sep=gmm_threshold(z[:,j],min_sep); threshold[(s,j)]=th
            threshold_rows.append({'antibody':name,'sample_donor':s,'normalization':'WITHIN_SAMPLE_CLR','threshold_method':method,'threshold':f'{th:.8g}','convergence':str(conv).upper(),'fallback':'NONE' if method=='TWO_COMPONENT_GMM' else method,'component_separation_sd':f'{sep:.6g}','unresolved_count':'FILLED_AFTER_CALLS'})
    low=np.zeros(len(cell_id),dtype=bool)
    for s in sorted(set(sample)):
        ix=np.flatnonzero(sample==s); low[ix]=lib[ix] <= np.quantile(lib[ix],0.01)
    lineages=list(cfg['primary_labels']); outside_rules=cfg.get('outside_primary_rules',{})
    posn=int(cfg['rule_semantics']['minimum_positive_proteins']); negn=int(cfg['rule_semantics']['minimum_negative_proteins']); margin_min=float(cfg['rule_semantics']['minimum_top_vs_second_lineage_margin_sd'])
    statuses=[]; labels=[]; margins=[]; evidence=[]
    for i in range(len(cell_id)):
        if low[i]: statuses.append('LOW_ADT_QUALITY'); labels.append(''); margins.append(''); evidence.append('library_size_bottom_1pct'); continue
        s=sample[i]; cand=[]; scores=[]; detail=[]
        for lab in lineages:
            rule=cfg['lineage_rules'][lab]
            pmarkers=list(dict.fromkeys(clean_marker(m) for m in rule['positive_aliases'] if mindex.get(clean_marker(m))))
            nmarkers=list(dict.fromkeys(clean_marker(m) for m in rule['negative_aliases'] if mindex.get(clean_marker(m))))
            if len(pmarkers)<posn or len(nmarkers)<negn: detail.append(f'{lab}:PANEL_INSUFFICIENT'); scores.append(-np.inf); continue
            # Duplicate clones/tags for one biological antigen count as one protein.
            # Positivity accepts the strongest clone; negativity requires every clone low.
            pvals=np.array([max(clr[i,j]-threshold[(s,j)] for j in mindex[m]) for m in pmarkers])
            nvals=np.array([min(threshold[(s,j)]-clr[i,j] for j in mindex[m]) for m in nmarkers])
            pc=int((pvals>=0).sum()); nc=int((nvals>=0).sum()); score=float(np.median(np.sort(pvals)[-posn:])+np.median(np.sort(nvals)[-negn:]))
            ok=pc>=posn and nc>=negn; cand.append(lab) if ok else None; scores.append(score); detail.append(f'{lab}:p{pc}/n{nc}/s{score:.3g}')
        order=np.argsort(scores)[::-1]; top=lineages[int(order[0])]; second=float(scores[int(order[1])]); first=float(scores[int(order[0])]); margin=first-second
        outside=[]
        for out_name,rule in outside_rules.items():
            pmarkers=list(dict.fromkeys(clean_marker(m) for m in rule['positive_aliases'] if mindex.get(clean_marker(m))))
            nmarkers=list(dict.fromkeys(clean_marker(m) for m in rule['negative_aliases'] if mindex.get(clean_marker(m))))
            if len(pmarkers)<posn or len(nmarkers)<negn: continue
            pvals=np.array([max(clr[i,j]-threshold[(s,j)] for j in mindex[m]) for m in pmarkers])
            nvals=np.array([min(threshold[(s,j)]-clr[i,j] for j in mindex[m]) for m in nmarkers])
            pc=int((pvals>=0).sum()); nc=int((nvals>=0).sum())
            oscore=float(np.median(np.sort(pvals)[-posn:])+np.median(np.sort(nvals)[-negn:]))
            if pc>=posn and nc>=negn: outside.append((oscore,out_name))
            detail.append(f'outside_{out_name}:p{pc}/n{nc}/s{oscore:.3g}')
        if len(cand)==1 and top==cand[0] and margin>=margin_min:
            statuses.append('RESOLVED_ADT_LEVEL1'); labels.append(top)
        elif not cand and outside:
            outside.sort(reverse=True)
            out_margin=outside[0][0]-(outside[1][0] if len(outside)>1 else max(scores))
            if out_margin>=margin_min:
                statuses.append('OUTSIDE_PRIMARY_LABEL_SET'); labels.append(''); detail.append(f'outside_call={outside[0][1]}')
            else:
                statuses.append('AMBIGUOUS_ADT'); labels.append('')
        elif len(cand)>=2:
            statuses.append('MIXED_ADT' if margin<margin_min else 'AMBIGUOUS_ADT'); labels.append('')
        elif max(scores) > 0:
            statuses.append('AMBIGUOUS_ADT'); labels.append('')
        else:
            statuses.append('UNRESOLVED_ADT'); labels.append('')
        margins.append(f'{margin:.8g}'); evidence.append('|'.join(detail))
    out=Path(a.truth_out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=['dataset_id','cell_id','donor_id','sample_id','truth_level','truth_label','truth_status','top_margin','adt_evidence','forced_assignment']
    dataset_id=Path(a.adt_only_h5).stem
    with out.open('w',encoding='utf-8',newline='') as fh:
        w=csv.writer(fh,delimiter='\t',lineterminator='\n'); w.writerow(fields)
        for row in zip(cell_id,donor,sample,labels,statuses,margins,evidence): w.writerow([dataset_id,row[0],row[1],row[2],'LEVEL1_MAJOR_LINEAGE',row[3],row[4],row[5],row[6],'FALSE'])
    unresolved_by_sample={s:int(sum((sample==s)&(np.array(statuses)!='RESOLVED_ADT_LEVEL1'))) for s in sorted(set(sample))}
    with Path(a.threshold_out).open('w',encoding='utf-8',newline='') as fh:
        fields2=list(threshold_rows[0]); w=csv.DictWriter(fh,fields2,delimiter='\t',lineterminator='\n'); w.writeheader()
        for r in threshold_rows: r['unresolved_count']=unresolved_by_sample[r['sample_donor']]; w.writerow(r)
    summary={'dataset_id':dataset_id,'cells':len(cell_id),'donors':len(set(donor)),'samples':len(set(sample)),'antibodies':len(antibody),'barcode_unique':True,
             'normalization':'WITHIN_SAMPLE_CLR','status_counts':{x:statuses.count(x) for x in cfg['output_statuses']},
             'label_counts':{x:labels.count(x) for x in lineages},'forced_assignments':0,'forbidden_inputs_seen':[]}
    Path(a.summary_out).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__': main()
