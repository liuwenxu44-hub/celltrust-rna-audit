"""Reproduce published summaries from fixed donor-level data; never fit models.

Run from a checkout: python scripts/reproduce_summaries.py --output runs/summary-check
Requires the NumPy/Pandas versions in requirements.txt. No network is used.
"""
from pathlib import Path
import argparse, gzip, hashlib, json
import numpy as np
import pandas as pd

STRATA=['Within 10%','Within 30%','Systematic 10%','Systematic 30%',
        'T_cell omitted','B_cell omitted','NK_cell omitted','monocyte omitted']
ATOL=2e-12  # absolute float64 summary roundoff; no clinical/practical equivalence claim
def read(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return bool(np.array_equal(np.isfinite(a),np.isfinite(b)) and np.allclose(a,b,atol=ATOL,rtol=0,equal_nan=True))
def weighted(x,w):
    den=w@np.isfinite(x).astype(float)
    return np.divide(w@np.where(np.isfinite(x),x,0),den,out=np.full_like(den,np.nan),where=den>0)
def summarize(x,draws,indices):
    w=np.array([np.bincount(row,minlength=len(x)) for row in draws],float)
    p=weighted(x,np.ones((1,len(x))))[0,indices]
    b=weighted(x,w)[:,indices];good=np.isfinite(b).all(axis=1)
    point=float(p.mean()) if np.isfinite(p).all() else np.nan
    ci=np.quantile(b[good].mean(axis=1),[.025,.975]) if good.any() else [np.nan,np.nan]
    return point,ci[0],ci[1],int(good.sum())
def indices(s):
    if s=='ALL_8_PRIMARY_STRATA':return list(range(8))
    if s=='ORDINARY_4_PRIMARY_STRATA':return list(range(4))
    if s=='OMITTED_4_PRIMARY_STRATA':return list(range(4,8))
    return [STRATA.index(s)]
def matrix(frame,value,donors):
    return frame.pivot(index='donor_id',columns='stratum',values=value).reindex(index=donors,columns=STRATA).to_numpy(float)
def verify_sources(root):
    manifest=json.loads((root/'manifest.json').read_text())
    derived=json.loads((root/'derived_manifest.json').read_text())
    for f in manifest['files']+derived['files']:
        p=root/f['path'];assert p.is_file() and sha(p)==f['sha256'],f['path']
        raw=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==f['content_sha256'],f['path']
    return len(manifest['files'])+len(derived['files'])
def run(root,out):
    assert not out.exists(),'Use a new output directory; existing files are not overwritten'
    nfiles=verify_sources(root);rows=[];checks=[]
    for cohort,workflow in [('D4','CORE'),('D3','CORE'),('D4','SCANPY'),('D3','SCANPY')]:
        folder=root/f'effect_tables/{cohort}_{workflow}'
        df=read(folder/'paired_donor_metrics.tsv.gz');expected=read(folder/'paired_effect_summary.tsv.gz')
        donors=sorted(df.donor_id.unique());draws=np.load(root/f'bootstrap/{cohort}_draws.npy',allow_pickle=False)
        assert draws.shape==(2000,len(donors)) and draws.min()==0 and draws.max()==len(donors)-1
        assert np.array_equal(draws,np.random.default_rng(2026090702).integers(0,len(donors),size=draws.shape))
        df=df[df.analysis_role=='PRIMARY'].copy();df.loc[df.AP_status!='DEFINED','AP']=np.nan
        for keys,g in expected.groupby(['support','method','comparison','metric'],sort=False):
            support,method,comparison,metric=keys;other=comparison[0]
            use=df[(df.support==support)&(df.method==method)]
            m=use.groupby(['donor_id','stratum','condition'],sort=False)[metric].mean().reset_index()
            a=matrix(m[m.condition=='A'],metric,donors);b=matrix(m[m.condition==other],metric,donors);delta=b-a
            for r in g.itertuples():
                ix=indices(r.stratum);calc=summarize(delta,draws,ix)
                assert equal(calc,[r.difference,r.ci_lower,r.ci_upper,r.valid_bootstrap]),(cohort,workflow,keys,r.stratum,calc)
                pa=weighted(a,np.ones((1,len(donors))))[0,ix];pb=weighted(b,np.ones((1,len(donors))))[0,ix]
                assert equal([pa.mean(),pb.mean()],[r.A,r.comparison_condition])
                rows.append({'dataset':cohort,'workflow':workflow,'support':support,'method':method,'comparison':comparison,'metric':metric,
                    'stratum':r.stratum,'estimate':calc[0],'ci_lower':calc[1],'ci_upper':calc[2],'valid_bootstrap':calc[3],
                    'population_donors':len(donors),'donors_any_selected_stratum':int(np.isfinite(delta[:,ix]).any(axis=1).sum()),
                    'donors_complete_selected_strata':int(np.isfinite(delta[:,ix]).all(axis=1).sum()),
                    'effective_donors_by_stratum':json.dumps({STRATA[i]:int(np.isfinite(delta[:,i]).sum()) for i in ix})})
            if support=='METHOD_COMMON' and cohort=='D3':
                saved=read(folder/'individual_donor_effects.tsv.gz')
                saved=saved[(saved.method==method)&(saved.comparison==comparison)&(saved.metric==metric)]
                for r in saved.itertuples():
                    v=delta[donors.index(r.donor_id),indices(r.stratum)]
                    assert equal(v.mean(),r.difference),(keys,'individual donor',r.donor_id)
                saved=read(folder/'leave_one_donor_out.tsv.gz')
                saved=saved[(saved.method==method)&(saved.comparison==comparison)&(saved.metric==metric)]
                for r in saved.itertuples():
                    w=np.ones((1,len(donors)));w[0,donors.index(r.omitted_donor)]=0
                    assert equal(weighted(delta,w).mean(),r.difference),(keys,'leave one donor out',r.omitted_donor)
        checks.append({'dataset':cohort,'workflow':workflow,'summary_rows':len(expected),'status':'MATCH'})
    frame=read(root/'mechanism/paired_donor_stratum_contrasts.tsv.gz')
    expected=read(root/'mechanism/paired_effect_summary.tsv.gz')
    donors=read(root/'mechanism/donor_bootstrap_index.tsv.gz').sort_values('donor_index').donor_id.tolist()
    draws=np.load(root/'bootstrap/M4_crossed_draws.npy',allow_pickle=False)
    for keys,g in expected.groupby(['support','fallback_group','contrast_id'],sort=False):
        use=frame[(frame.support==keys[0])&(frame.fallback_group==keys[1])&(frame.contrast_id==keys[2])]
        x=matrix(use,'paired_AP_difference',donors)
        for r in g.itertuples():
            ix=indices(r.stratum);calc=summarize(x,draws,ix)
            assert equal(calc,[r.paired_AP_difference,r.ci_lower,r.ci_upper,r.valid_bootstrap]),('M4 crossing',keys,r.stratum,calc)
            assert int(np.isfinite(x[:,ix]).any(axis=1).sum())==r.donors_with_any_defined_selected_stratum
            assert int(np.isfinite(x[:,ix]).all(axis=1).sum())==r.donors_complete_selected_strata
    checks.append({'family':'M4 crossed diagnostic','summary_rows':len(expected),'status':'MATCH'})
    df=read(root/'mechanism/donor_risk_extra_change.tsv.gz');expected=read(root/'mechanism/risk_extra_paired_effect.tsv.gz')
    for comparison,g in expected.groupby('comparison',sort=False):
        x=matrix(df[(df.comparison==comparison)&df.stratum.isin(STRATA)],'extra_AP_change',donors)
        for r in g.itertuples():
            assert equal(summarize(x,draws,indices(r.stratum)),[r.extra_M4_RISK_change_over_raw_disagreement,r.ci_lower,r.ci_upper,r.valid_bootstrap]),('Figure3C',comparison,r.stratum)
    checks.append({'family':'M4 excess over raw disagreement','summary_rows':len(expected),'status':'MATCH'})
    refs=read(root/'mechanism/reference_support_and_scale.tsv.gz')
    refs=refs[(refs.analysis_role=='PRIMARY')&refs.component.isin(['S','G'])]
    keys=['instance_id','outer_context','observed_label','component']
    pairs=refs[refs.reference_condition=='A'].merge(refs[refs.reference_condition=='B'],on=keys,validate='one_to_one',suffixes=('_A','_B'))
    complete=np.isfinite(pairs.old_scale_A)&np.isfinite(pairs.old_scale_B)&(pairs.old_scale_A>0)&(pairs.old_scale_B>0)
    assert len(pairs)==640 and int(complete.sum())==480
    assert pairs.loc[complete,'component'].value_counts().to_dict()=={'S':240,'G':240}
    assert not complete[pairs.stratum_A.str.endswith(' omitted')].any()
    checks.append({'family':'Figure3D scale pairs','candidate_pairs':640,'complete_pairs':480,'unavailable_omitted_label_pairs':160,'status':'MATCH'})
    bins=pd.read_csv(root/'Figure5/A_paired_bins.csv');assert (bins.groupby('case')['count'].sum()==7699418).all()
    cases=pd.read_csv(root/'Figure5/C_diagnostic_cases.csv');assert cases.status.value_counts().to_dict()=={'FAIL':5,'NOT_VERIFIABLE':4,'PASS':2}
    runtime=pd.read_csv(root/'Figure5/D_runtime.csv');assert len(runtime)==9
    full=np.load(root/'Figure5/B_full_empirical_cdf.npz',allow_pickle=False)
    assert int(full['count'].sum())==7699418
    assert np.array_equal(full['count'].cumsum(),full['cumulative_count'])
    assert np.all(np.diff(full['decrease'])>0)
    quantiles=pd.read_csv(root/'Figure5/B_quantiles.csv',float_precision='round_trip')
    for q in quantiles.itertuples():
        rank=int(np.ceil(q.probability*7699418))
        ix=np.searchsorted(full['cumulative_count'],rank,side='left')
        assert rank==q.rank and equal(full['decrease'][ix],q.decrease)
        assert int(full['cumulative_count'][ix])==q.count_le
    checks.append({'family':'Figure5 displayed technical data','bins':len(bins),'cases':len(cases),'timed_calls':len(runtime),'quantiles_reproduced':len(quantiles),'full_cdf_arrays':{k:list(full[k].shape) for k in full.files},'status':'IDENTITIES_COUNTS_AND_QUANTILES_MATCH'})
    # Check study selection/folds and controlled targets without scoring or relabeling.
    for cohort,n,nd in [('D4',14032,37),('D3',7200,6)]:
        folder=root/f'study_inputs/{cohort}';cells=read(folder/'method_cells.tsv.gz');truth=read(folder/'reference_labels.tsv.gz')
        assert len(cells)==n and cells.cell_id.is_unique and cells.donor_id.nunique()==nd
        assert (cells.groupby('donor_id').outer_fold.nunique()==1).all()
        assert np.array_equal(cells.cell_id,truth.cell_id) and np.array_equal(cells.pilot_cell_index,np.arange(n))
        ins=read(folder/'instances.tsv');assert len(ins)==20 and (ins.analysis_role=='PRIMARY').sum()==16
        targets=read(folder/'evaluation_targets.tsv.gz');labels=read(folder/'observed_label_overlays.tsv.gz')
        assert not targets.duplicated(['instance_id','pilot_cell_index']).any()
        assert not labels.duplicated(['instance_id','pilot_cell_index']).any()
        assert len(targets)==len(labels)==n*20
        joined=labels.merge(targets,on=['instance_id','pilot_cell_index'],validate='one_to_one')
        truthmap=truth.set_index('pilot_cell_index').clean_label
        assert np.array_equal((joined.observed_label!=joined.pilot_cell_index.map(truthmap)).astype(int),joined.is_label_error)
        checks.append({'family':'fixed study inputs','dataset':cohort,'cells':n,'donors':nd,'label_instances':20,'status':'MATCH'})
    result=pd.DataFrame(rows)
    published=read(root/'summary_support.tsv.gz')
    assert result.shape==published.shape and list(result.columns)==list(published.columns)
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            assert equal(result[column],published[column]),('published support table',column)
        else:
            assert np.array_equal(result[column],published[column]),('published support table',column)
    checks.append({'family':'published donor-support table','rows':len(result),'status':'MATCH'})
    out.mkdir(parents=True)
    result.to_csv(out/'reproduced_effects_and_support.tsv',sep='\t',index=False,float_format='%.17g')
    report={'status':'SUMMARY_REPRODUCTION_PASS','source_files_verified':nfiles,'checks':checks,'numeric_absolute_tolerance':ATOL,
            'models_refitted':False,'labels_or_thresholds_changed':False,'full_raw_to_manuscript_reproduction':False,
            'scope':'Aggregates/CIs from saved donor metrics; M4 crossed diagnostics; individual/leave-one-out D3 effects; selected-input identities; technical figure data counts. Figure aesthetics and raw model scores are not re-created.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-data',type=Path,default=Path(__file__).resolve().parents[1]/'source_data');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source_data.resolve(),a.output.resolve())
