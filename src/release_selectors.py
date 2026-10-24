"""Frozen distance-margin and hard-onehot EnsV definitions; no target truth."""
import numpy as np
from scipy.spatial.distance import cdist

def distance_margin(Zs,ys,Zt,prediction,dist=None):
    dist=cdist(Zt,Zs) if dist is None else dist
    margins=np.empty(len(Zt),dtype=float)
    for c in np.unique(prediction):
        mask=prediction==c;source=ys==c
        if not source.any() or source.all():raise ValueError('Margin requires same and other source classes')
        margins[mask]=dist[mask][:,~source].min(axis=1)-dist[mask][:,source].min(axis=1)
    return margins

def ensemble(predictions,mus,expected_candidates=11):
 p=np.asarray(predictions);mus=np.asarray(mus)
 if p.ndim!=2 or p.shape[0]!=expected_candidates or mus.shape!=(expected_candidates,) or len(set(mus))!=expected_candidates or p.shape[1]==0 or not np.isin(p,np.arange(1,7)).all():raise ValueError('Invalid candidate pool')
 votes=np.stack([(p==c).sum(axis=0) for c in range(1,7)],axis=1)
 role=votes.argmax(axis=1)+1
 agreement=(p==role).mean(axis=1)
 best=np.flatnonzero(agreement==agreement.max());selected=float(mus[best[np.argmin(mus[best])]])
 freq=votes/len(p);entropy=-np.sum(np.where(freq>0,freq*np.log(np.where(freq>0,freq,1)),0),axis=1)
 pairs=np.array([np.mean(p[i]!=p[j]) for i in range(len(p)) for j in range(i+1,len(p))])
 nz=pairs[pairs>0]
 diversity=dict(mean_pairwise_disagreement=float(pairs.mean()),max_pairwise_disagreement=float(pairs.max()),
  min_nonzero_disagreement=float(nz.min()) if len(nz) else None,mean_vote_entropy=float(entropy.mean()),
  mean_majority_vote_strength=float(freq.max(axis=1).mean()),unanimous_prediction_fraction=float(np.mean(votes.max(axis=1)==len(p))))
 return dict(role=role,agreement=agreement,selected_mu=selected,entropy=entropy,strength=freq.max(axis=1),diversity=diversity)

def choose_margin(mus, scores):
    mus,scores=np.asarray(mus),np.asarray(scores)
    if mus.shape!=scores.shape or not len(mus) or not np.isfinite(scores).all():
        raise ValueError('Invalid margin scores')
    return float(mus[np.lexsort((mus,-scores))[0]])
