"""BDA-v1.1-paper-init: initial 1NN pseudo labels before the first projection.

Only initialization differs from BDA-v1-reference-init. Shared MMD, norm,
diagnostic helpers and constructor stay frozen; iteration body is unchanged.
"""
import numpy as np
from scipy.linalg import eig
from sklearn.neighbors import KNeighborsClassifier
from bda_core import (BDA as ReferenceBDA, NumericalFailure, discrepancy_factors,
                      _matrix_diagnostics, _normalize_columns)

class BDA(ReferenceBDA):
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
            # Paper Algorithm 1: bootstrap on the actual pre-projection BDA
            # representation: source-only standardization (caller), then L2.
            pseudo=KNeighborsClassifier(n_neighbors=1).fit(x[:, :len(Xs)].T,ys).predict(x[:, len(Xs):].T)
            self.initial_pseudo_labels_=pseudo.copy()
            conditional=discrepancy_factors(ys,len(Xt),pseudo,1.)
            conditional_norm=float(np.linalg.norm(conditional.T@conditional,'fro'))
            self.initial_diagnostics_=dict(
                representation='source-standardized then sample-wise L2; before any eigendecomposition',
                sample_count=len(Xt),pseudo_class_counts={str(c):int(np.sum(pseudo==c)) for c in np.unique(ys)},
                missing_pseudo_classes=[int(c) for c in np.unique(ys) if not np.any(pseudo==c)],
                conditional_mmd_frobenius_norm=conditional_norm)
            if conditional_norm==0 or not np.isfinite(conditional_norm):
                raise ValueError('Degenerate initial conditional MMD')
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
