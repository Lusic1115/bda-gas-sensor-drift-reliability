"""Euclidean 1NN control on source-standardized features."""
from sklearn.neighbors import KNeighborsClassifier

def fit_predict(Xs, ys, Xt):
    return KNeighborsClassifier(n_neighbors=1).fit(Xs, ys).predict(Xt)
