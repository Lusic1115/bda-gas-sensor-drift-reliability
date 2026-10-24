"""Synthetic fixture only: not an execution of either paper benchmark."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New fixture-output directory')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output must not exist')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    from bda_paper_core import BDA
    from bda_evaluation import evaluate_predictions
    cfg=json.loads((root/'configs/canonical_bda.yaml').read_text())
    if cfg['mu_grid']!=[i/10 for i in range(11)]:raise ValueError('Grid mismatch')
    rng=np.random.default_rng(15)
    xs=rng.normal(size=(18,5));xt=rng.normal(size=(9,5));ys=np.tile([1,2,3],6)
    # Reduced dimensionality/iteration count belongs solely to this named test fixture.
    model=BDA(dim=3,iterations=1,mu=1.)
    pred=model.fit_predict(xs,ys,xt)
    if len(pred)!=9 or not np.isfinite(model.target_embedding_).all():
        raise ValueError('Synthetic fixture failed')
    metric=evaluate_predictions(np.array([1,2,3]),np.array([1,2,3]),(1,2,3))
    if metric['accuracy']!=1.:raise ValueError('Metric fixture failed')
    args.output.mkdir(parents=True)
    (args.output/'fixture_table.csv').write_text('fixture,accuracy\nsynthetic_identity,1.0\n')
    result=dict(status='PASS',fixture_only=True,real_data_used=False,
                source_shape=[18,5],target_shape=[9,5],test_dim=3,test_iterations=1,
                prediction_count=len(pred),finite=True,config_grid_count=len(cfg['mu_grid']),
                paper_experiments_run=0)
    (args.output/'fixture_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
