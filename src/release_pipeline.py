"""Portable explicit Phase A / Phase B entry points; no work at import time."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MUS=[i/10 for i in range(11)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def config(name):return json.loads((ROOT/'configs'/name).read_text())
def validate_protocol(dataset):
    """Reject silent edits of the frozen data protocol, not a tuning interface."""
    if dataset=='uci':
        cfg=config('uci_protocol.yaml')
        expected=dict(source=1,targets=list(range(2,11)),
                      primary_targets=[2,6,7,8,9,10],support_subset_targets=[3,4,5],
                      class_labels=[1,2,3,4,5,6],feature_count=128)
    elif dataset=='external':
        cfg=config('external_protocol.yaml')
        expected=dict(source_n=90,targets=[f'T{i}' for i in range(1,8)],
                      target_ns=[90,89,90,90,90,90,71],class_labels=[1,2,3],
                      feature_count=248,class_mapping={'EtOH':1,'Diacetyl 0.1ppm':2,
                      'Diacetyl 1ppm':2,'Phenylethanol 200ppm':3,'Phenylethanol 1000ppm':3})
    else:
        raise ValueError('Unknown dataset protocol')
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen data protocol mismatch')
def data_paths(dataset,data):
    data=Path(data)
    if dataset=='uci':return {f'batch{i}.dat':data/f'batch{i}.dat' for i in range(1,11)}
    return {'features':data,'schema':ROOT/'data/external/feature_schema.csv','assignments':ROOT/'data/external/domain_assignments.csv'}

def phase_a(dataset,data,output):
    from bda_data import load_source,load_target_features,load_external_features
    from bda_preprocessing import scale_source_target
    from bda_core import _normalize_columns
    from bda_paper_core import BDA
    from source_only_1nn import fit_predict
    from sklearn.svm import SVC
    from release_selectors import distance_margin,ensemble,choose_margin
    validate_protocol(dataset)
    cfg=config('canonical_bda.yaml')
    if any(cfg[k]!=v for k,v in {'dim':30,'lambda':1,'iterations':10,'mu_grid':MUS,'scaler':'source_only','pre_projection_l2':True,'post_projection_l2':True,'classifier':'Euclidean1NN','early_stopping':False,'return_iteration':10}.items()):raise ValueError('Canonical protocol mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paths=data_paths(dataset,data);inputs={key:sha(path) for key,path in paths.items()}
    if dataset=='external' and inputs['features']!=config('external_protocol.yaml')['frozen_csv_sha256']:raise ValueError('External reconstructed table differs from the frozen benchmark')
    if dataset=='uci':
        xs,ys=load_source(paths['batch1.dat']);targets={f'Batch{i}':load_target_features(paths[f'batch{i}.dat']) for i in range(2,11)}
    else:xs,ys,targets,_=load_external_features(paths['features'],paths['schema'],paths['assignments'])
    # Exactly one scaler fitted to source; no target refit.
    xs,_,scaler=scale_source_target(xs,next(iter(targets.values())))
    targets={key:scaler.transform(x) for key,x in targets.items()}
    svm=SVC(kernel='rbf',C=1.,gamma='scale',class_weight='balanced').fit(xs,ys)
    records=[];scores=[];choices=[];diagnostics=[]
    def save(domain,method,pred):
        name=f'{domain}_{method}.csv';np.savetxt(output/name,pred,fmt='%d',header='prediction',comments='')
        records.append(dict(domain=domain,method=method,path=name,sample_count=len(pred)))
    for domain,xt in targets.items():
        matched=_normalize_columns(np.vstack([xs,xt]).T).T
        save(domain,'source_only_1nn',fit_predict(matched[:len(xs)],ys,matched[len(xs):]) if dataset=='external' else fit_predict(xs,ys,xt))
        save(domain,'source_only_svm',svm.predict(xt))
        pool=[];means=[];medians=[]
        for mu in MUS:
            model=BDA(dim=30,lamb=1,mu=mu,iterations=10);prediction=model.fit_predict(xs,ys,xt)
            save(domain,f'mu{mu:.1f}',prediction);pool.append(prediction)
            margin=distance_margin(model.source_embedding_,ys,model.target_embedding_,prediction)
            means.append(float(margin.mean()));medians.append(float(np.median(margin)))
            scores.append(dict(domain=domain,mu=mu,margin_mean=means[-1],margin_median=medians[-1]))
            diagnostics.append(dict(domain=domain,mu=mu,initial=model.initial_diagnostics_,iterations=model.diagnostics_))
        result=ensemble(np.stack(pool),MUS);save(domain,'multi_mu_vote',result['role'])
        for name,mu in [('fixed_mu_0.0',0.),('fixed_mu_0.5',.5),('margin_mean',choose_margin(MUS,means)),('margin_median',choose_margin(MUS,medians)),('EnsV-onehot',result['selected_mu'])]:
            choices.append(dict(domain=domain,selector=name,selected_mu=mu,prediction_path=f'{domain}_mu{mu:.1f}.csv'))
        for row,agreement in zip(scores[-11:],result['agreement']):row['ensv_agreement']=float(agreement)
    pd.DataFrame(records).to_csv(output/'prediction_index.csv',index=False)
    pd.DataFrame(scores).to_csv(output/'label_free_scores.csv',index=False)
    pd.DataFrame(choices).to_csv(output/'selections.csv',index=False)
    write_json(output/'diagnostics.json',diagnostics)
    write_json(output/'source_fit.json',dict(n_samples_seen=int(scaler.n_samples_seen_),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),target_refit=False))
    if inputs!={key:sha(path) for key,path in paths.items()}:raise RuntimeError('Input changed during Phase A')
    hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'prediction_freeze.json',dict(dataset=dataset,mu_grid=MUS,domains=list(targets),inputs=inputs,hashes=hashes,
        frozen_at=datetime.now(timezone.utc).isoformat(),target_label_parses=0,candidate_count=11*len(targets),
        source_hashes={p.name:sha(p) for p in (ROOT/'src').glob('*.py')},config_hashes={p.name:sha(p) for p in (ROOT/'configs').glob('*.yaml')}))
    return output

def verify_freeze(dataset,data,output):
    """Verify every decision/prediction and all inputs before any label parsing."""
    output=Path(output);freeze=json.loads((output/'prediction_freeze.json').read_text())
    expected_domains=[f'Batch{i}' for i in range(2,11)] if dataset=='uci' else [f'T{i}' for i in range(1,8)]
    if freeze['dataset']!=dataset or freeze['target_label_parses']!=0 or freeze['mu_grid']!=MUS or freeze['domains']!=expected_domains or freeze['candidate_count']!=11*len(expected_domains):raise RuntimeError('Incomplete protocol freeze')
    required_metadata={'prediction_index.csv','label_free_scores.csv','selections.csv',
                       'diagnostics.json','source_fit.json'}
    if not required_metadata.issubset(freeze['hashes']):
        raise RuntimeError('Incomplete frozen decision/index metadata')
    for name,digest in freeze['hashes'].items():
        if Path(name).name!=name or sha(output/name)!=digest:raise RuntimeError('Frozen artifact modified')
    for key,path in data_paths(dataset,data).items():
        if sha(path)!=freeze['inputs'][key]:raise RuntimeError('Dataset/metadata changed')
    for folder,key in [('src','source_hashes'),('configs','config_hashes')]:
        for name,digest in freeze[key].items():
            if Path(name).name!=name or sha(ROOT/folder/name)!=digest:raise RuntimeError('Frozen implementation/config changed')
    index=pd.read_csv(output/'prediction_index.csv');choices=pd.read_csv(output/'selections.csv')
    expected={(d,f'mu{mu:.1f}') for d in expected_domains for mu in MUS}|{(d,m) for d in expected_domains for m in ['source_only_1nn','source_only_svm','multi_mu_vote']}
    if len(index)!=len(expected) or set(zip(index.domain,index.method))!=expected:raise RuntimeError('Incomplete prediction index')
    if len(choices)!=5*len(expected_domains):raise RuntimeError('Incomplete selector decisions')
    for name in [*index.path,*choices.prediction_path]:
        if name not in freeze['hashes']:raise RuntimeError('Unfrozen prediction reference')
    return freeze,index,choices

def phase_b(dataset,data,output):
    from bda_evaluation import evaluate_predictions
    from bda_data import encode_external
    output=Path(output)
    if (output/'evaluation_metrics.csv').exists():raise FileExistsError('Evaluation already exists')
    freeze,index,choices=verify_freeze(dataset,data,output)
    # Sole target truth boundary: all domains have already passed the gate.
    if dataset=='uci':
        labels={f'Batch{i}':np.array([int(line.split()[0]) for line in (Path(data)/f'batch{i}.dat').read_text().splitlines()]) for i in range(2,11)};classes=(1,2,3,4,5,6)
    else:
        domains=pd.read_csv(data_paths(dataset,data)['assignments'],usecols=['domain']).domain.to_numpy()
        y=encode_external(pd.read_csv(data,sep=' ',usecols=['Class'])['Class']);labels={d:y[domains==d] for d in freeze['domains']};classes=(1,2,3)
    rows=[]
    for row in index.itertuples(index=False):
        pred=np.loadtxt(output/row.path,skiprows=1,dtype=int,ndmin=1)
        rows.append(dict(domain=row.domain,method=row.method,**evaluate_predictions(labels[row.domain],pred,classes)))
    for row in choices.itertuples(index=False):
        pred=np.loadtxt(output/row.prediction_path,skiprows=1,dtype=int,ndmin=1)
        rows.append(dict(domain=row.domain,method=row.selector,**evaluate_predictions(labels[row.domain],pred,classes)))
    pd.DataFrame(rows).to_csv(output/'evaluation_metrics.csv',index=False)
    write_json(output/'evaluation_audit.json',dict(phase_b_only=True,prediction_freeze_sha256=sha(output/'prediction_freeze.json'),target_label_parses_before_freeze=0))
