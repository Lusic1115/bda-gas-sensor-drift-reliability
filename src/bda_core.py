"""Fixed-mu standard primal BDA. No target labels, metrics or file access.

Uses a low-rank MMD factorization equivalent to the dense generalized
problem. Source classes are inferred from ys, never from target labels.
"""
import numpy as np
from scipy.linalg import eig
from sklearn.neighbors import KNeighborsClassifier

class NumericalFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics

def discrepancy_factors(ys, nt, pseudo, mu):
    ns=len(ys); classes=np.unique(ys)
    marginal=np.r_[np.full(ns,1/ns),np.full(nt,-1/nt)]
    columns=[np.sqrt((1-mu)*len(classes))*marginal]
    if pseudo is not None:
        for c in classes:
            source=ys==c; target=pseudo==c
            v=np.zeros(ns+nt)
            v[:ns][source]=1/source.sum()
            if target.any(): v[ns:][target]=-1/target.sum()
            columns.append(np.sqrt(mu)*v)
    return np.column_stack(columns)

def _matrix_diagnostics(a):
    values=np.linalg.eigvalsh((a+a.T)/2)
    condition=float(np.linalg.cond(a))
    return dict(shape=list(a.shape),condition=condition if np.isfinite(condition) else None,
                condition_nonfinite=not np.isfinite(condition),rank=int(np.linalg.matrix_rank(a)),
                eigenvalue_min=float(values[0]),eigenvalue_max=float(values[-1]))

def _normalize_columns(a):
    norms=np.linalg.norm(a,axis=0)
    if not np.isfinite(norms).all() or np.any(norms==0):
        raise ValueError('Zero or nonfinite sample norm; no normalization repair applied')
    return a/norms

class BDA:
    def __init__(self, dim=30, lamb=1, mu=.5, iterations=10):
        if not isinstance(dim,int) or dim<1 or not isinstance(iterations,int) or iterations<1:
            raise ValueError('Positive integer dim and iterations required')
        if not np.isfinite(lamb) or lamb<=0 or not np.isfinite(mu) or not 0<=mu<=1:
            raise ValueError('Require lambda>0 and mu in [0,1]')
        self.dim,self.lamb,self.mu,self.iterations=dim,lamb,mu,iterations

    def fit_predict(self, Xs, ys, Xt):
        Xs,Xt=np.asarray(Xs,dtype=float),np.asarray(Xt,dtype=float)
        ys=np.asarray(ys)
        if Xs.ndim!=2 or Xt.ndim!=2 or Xs.shape[1]!=Xt.shape[1] or not len(Xs) or not len(Xt):
            raise ValueError('Require nonempty aligned source/target feature matrices')
        if ys.shape!=(len(Xs),) or not np.isin(ys,[1,2,3,4,5,6]).all():
            raise ValueError('Invalid source labels')
        if self.dim>Xs.shape[1] or not np.isfinite(Xs).all() or not np.isfinite(Xt).all():
            raise ValueError('Invalid dimension or nonfinite features')
        self.diagnostics_,self.pseudo_labels_ = [],[]
        diagnostic={'stage':'input_normalization','input_shape':[len(Xs)+len(Xt),Xs.shape[1]]}
        try:
            x=_normalize_columns(np.vstack([Xs,Xt]).T)
            centered=x-x.mean(axis=1,keepdims=True)
            b=centered@centered.T
            pseudo=None
            for iteration in range(1,self.iterations+1):
                diagnostic={'iteration':iteration,'B':_matrix_diagnostics(b)}
                f=discrepancy_factors(ys,len(Xt),pseudo,self.mu)
                mnorm=np.linalg.norm(f.T@f,'fro')
                if mnorm==0 or not np.isfinite(mnorm): raise ValueError('Degenerate MMD norm')
                xf=x@f
                a=xf@xf.T/mnorm+self.lamb*np.eye(x.shape[0])
                diagnostic['A']=_matrix_diagnostics(a)
                values,vectors=eig(a,b)
                finite=np.isfinite(values)
                diagnostic['finite_eigenpair_count']=int(finite.sum())
                diagnostic['infinite_eigenpair_count']=int((~finite).sum())
                if finite.sum()<self.dim: raise ValueError('Insufficient finite generalized eigenpairs')
                if np.any(np.abs(values[finite].imag)>1e-8*np.maximum(1,np.abs(values[finite].real))):
                    raise ValueError('Significant imaginary generalized eigenvalues')
                indices=np.flatnonzero(finite)[np.argsort(values[finite].real)[:self.dim]]
                projection=vectors[:,indices]
                if np.max(np.abs(projection.imag))>1e-8: raise ValueError('Complex projection')
                projection=projection.real
                selected=values[indices].real
                residual=np.linalg.norm(a@projection-(b@projection)*selected)/max(np.linalg.norm(a@projection),np.finfo(float).tiny)
                if not np.isfinite(residual) or residual>1e-6: raise ValueError('Generalized eigen residual too large')
                z=_normalize_columns(projection.T@x).T
                if not np.isfinite(z).all(): raise ValueError('Nonfinite embedding')
                source,target=z[:len(Xs)],z[len(Xs):]
                predicted=KNeighborsClassifier(n_neighbors=1).fit(source,ys).predict(target)
                diagnostic.update(selected_eigenvalues=selected.tolist(),eigen_residual=float(residual),
                    pseudo_label_changes=None if pseudo is None else int(np.sum(predicted!=pseudo)),
                    pseudo_class_counts={str(c):int(np.sum(predicted==c)) for c in np.unique(ys)},
                    projection_shape=list(projection.shape),finite=True)
                self.diagnostics_.append(diagnostic)
                self.pseudo_labels_.append(predicted.copy())
                pseudo=predicted
            self.projection_=projection
            self.source_embedding_,self.target_embedding_=source,target
            return pseudo.copy()
        except (ValueError, np.linalg.LinAlgError) as exc:
            diagnostic.update(exception=f'{type(exc).__name__}: {exc}',
                suggestion='Inspect recorded matrices and rank; review any repair explicitly before changing parameters.')
            raise NumericalFailure(str(exc),diagnostic) from exc
