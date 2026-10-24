"""SVMLight feature loading; target label tokens are never interpreted here."""
from pathlib import Path
import numpy as np

N_FEATURES = 128

def _load(path, source):
    rows, labels = [], []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        tokens = line.split('#', 1)[0].split()
        if not tokens:
            raise ValueError(f'Empty sample at line {line_number}')
        if source:
            label = float(tokens[0])
            if not np.isfinite(label) or label not in (1,2,3,4,5,6):
                raise ValueError(f'Invalid source label at line {line_number}')
            labels.append(int(label))
        row = np.zeros(N_FEATURES)
        previous = 0
        for token in tokens[1:]:
            index, value = token.split(':')
            index, value = int(index), float(value)
            if not previous < index <= N_FEATURES or not np.isfinite(value):
                raise ValueError(f'Invalid feature at line {line_number}')
            row[index-1] = value
            previous = index
        rows.append(row)
    if not rows:
        raise ValueError('Empty batch')
    return np.asarray(rows), np.asarray(labels, dtype=int)

def load_source(path):
    return _load(path, source=True)

def load_target_features(path):
    return _load(path, source=False)[0]

def encode_external(values):
    mapping={'EtOH':1,'Diacetyl 0.1ppm':2,'Diacetyl 1ppm':2,
             'Phenylethanol 200ppm':3,'Phenylethanol 1000ppm':3}
    try:
        return np.asarray([mapping[s] for s in values],dtype=int)
    except KeyError as exc:
        raise ValueError('Invalid external analyte condition') from exc

def load_external_features(path, schema, assignments):
    """Feature usecols excludes Class; source Class read stops before target rows."""
    import pandas as pd
    names=pd.read_csv(schema,usecols=['feature']).feature.tolist()
    domains=pd.read_csv(assignments,usecols=['domain']).domain.to_numpy()
    if len(names)!=248 or len(names)!=len(set(names)) or 'Class' in names:
        raise ValueError('Expected frozen 248-feature schema')
    x=pd.read_csv(path,sep=' ',usecols=names)[names].to_numpy(float)
    ys=encode_external(pd.read_csv(path,sep=' ',usecols=['Class'],nrows=90)['Class'])
    if x.shape!=(700,248) or len(domains)!=700 or not np.isfinite(x).all():raise ValueError('External feature shape/values mismatch')
    if not np.array_equal(np.flatnonzero(domains=='Source'),np.arange(90)) or set(ys)!={1,2,3}:raise ValueError('Frozen source assignment mismatch')
    counts=[90,89,90,90,90,90,71]
    if any(int((domains==f'T{i}').sum())!=n for i,n in enumerate(counts,1)):raise ValueError('Frozen target assignment mismatch')
    return x[:90],ys,{f'T{i}':x[domains==f'T{i}'] for i in range(1,8)},domains
