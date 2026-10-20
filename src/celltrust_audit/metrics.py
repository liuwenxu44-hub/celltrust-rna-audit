"""Original tie-aware metrics on an explicitly finite scored support."""
import math
import numpy as np
FIELDS=["AP","recall_review_10","precision_review_10","correct_label_false_review_fraction","score_coverage"]
def one(y,s,ids,total_n):
    y=np.asarray(y,dtype=np.int8); s=np.asarray(s); n=len(y); p=int(y.sum()); neg=n-p
    result={k:np.nan for k in FIELDS}; result.update(scored_cells=n,errors=p,correct_cells=neg,eligible_cells=total_n,excluded_cells=total_n-n,reviewed_cells=0,errors_reviewed=0,correct_reviewed=0)
    result['score_coverage']=n/total_n if total_n else np.nan
    result['AP_status']='NO_FINITE_SCORE' if n==0 else 'UNDEFINED_NO_POSITIVES' if p==0 else 'NO_DISCRIMINATION_INFORMATION' if neg==0 else 'DEFINED'
    if not n: return result
    order=np.argsort(-s,kind='stable'); ss=s[order]; yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),n-1]; tp=np.cumsum(yy)[ends]
    if p:
        recall=tp/p; result['AP']=float(np.sum((recall-np.r_[0.,recall[:-1]])*tp/(ends+1)))
    k=min(n,math.floor(.10*n+.5)); review=np.lexsort((ids,-s))[:k]; caught=int(y[review].sum())
    result.update(reviewed_cells=k,errors_reviewed=caught,correct_reviewed=k-caught,
                  precision_review_10=caught/k if k else np.nan,recall_review_10=caught/p if p else np.nan,
                  correct_label_false_review_fraction=(k-caught)/neg if neg else np.nan)
    return result


def validate_metric():
    assert abs(one([0,1,1],[.1,.9,.9],np.array(['a','b','c']),3)['AP']-1)<1e-15
    assert abs(one([1,0],[1,1],np.array(['a','b']),2)['AP']-.5)<1e-15
    assert np.isnan(one([0,0],[1,2],np.array(['a','b']),2)['AP'])
    assert one([1,1],[1,2],np.array(['a','b']),2)['AP_status']=='NO_DISCRIMINATION_INFORMATION'
    assert one([1,0,0,0,0],[1,1,1,1,1],np.array(['e','d','c','b','a']),5)['errors_reviewed']==0
    return {'ties_AP':True,'no_positive_NA':True,'all_positive_nonranking':True,'review_half_up_and_cell_id_tie':True}
