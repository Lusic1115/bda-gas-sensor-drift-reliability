"""Retrospective metrics. Target truth is loaded by the independent freeze gate."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def evaluate_predictions(yt, predictions, classes=(1,2,3,4,5,6)):
    yt,predictions=np.asarray(yt),np.asarray(predictions)
    if yt.ndim!=1 or yt.shape!=predictions.shape or not len(yt) or not np.isin(yt,classes).all() or not np.isin(predictions,classes).all():
        raise ValueError('Invalid evaluation arrays')
    return dict(accuracy=float(accuracy_score(yt,predictions)),
        balanced_accuracy=float(balanced_accuracy_score(yt,predictions)),
        macro_f1=float(f1_score(yt,predictions,average='macro',zero_division=0)),
        fixed_class_macro_f1=float(f1_score(yt,predictions,labels=classes,average='macro',zero_division=0)),
        target_present_macro_f1=float(f1_score(yt,predictions,labels=sorted(np.unique(yt)),average='macro',zero_division=0)))
