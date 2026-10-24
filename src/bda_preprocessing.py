"""One shared source-only scaling protocol for both experiments."""
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_source_target(Xs, Xt):
    if not np.isfinite(Xs).all() or not np.isfinite(Xt).all():
        raise ValueError('Nonfinite input features')
    scaler = StandardScaler().fit(Xs)
    return scaler.transform(Xs), scaler.transform(Xt), scaler
