import json
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bda_paper_core import BDA
from bda_core import discrepancy_factors, _normalize_columns
from bda_data import load_target_features
from bda_preprocessing import scale_source_target
from source_only_1nn import fit_predict
from release_selectors import distance_margin, ensemble, choose_margin
from bda_evaluation import evaluate_predictions

class CanonicalTests(unittest.TestCase):
 def test_bootstrap_no_mutation(self):
  rng=np.random.default_rng(15);xs=rng.normal(size=(18,5));xt=rng.normal(size=(9,5));ys=np.tile([1,2,3],6);a,b=xs.copy(),xt.copy()
  m=BDA(dim=3,iterations=2,mu=1);p=m.fit_predict(xs,ys,xt)
  z=_normalize_columns(np.vstack([xs,xt]).T).T
  np.testing.assert_array_equal(m.initial_pseudo_labels_,fit_predict(z[:18],ys,z[18:]));self.assertEqual(len(m.pseudo_labels_),2)
  self.assertEqual(p.shape,(9,));self.assertTrue(np.isfinite(m.source_embedding_).all());np.testing.assert_array_equal(a,xs);np.testing.assert_array_equal(b,xt)
 def test_missing_class_factor(self):
  f=discrepancy_factors(np.array([1,1,2,2]),3,np.ones(3),1)
  np.testing.assert_array_equal(f[:,2],[0,0,.5,.5,0,0,0]);self.assertTrue(np.isfinite(f).all())
 def test_poison_target(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'target.dat';p.write_text('POISON 1:2 128:3\nSECRET 2:4\n');self.assertEqual(load_target_features(p).shape,(2,128))
 def test_scaler_source(self):
  x=np.array([[1.,2.],[3.,6.]]);t=np.array([[100.,200.]])
  a,b,s=scale_source_target(x,t);np.testing.assert_allclose(s.mean_,[2,4]);self.assertEqual(s.n_samples_seen_,2)
 def test_selectors(self):
  p=np.tile([1,2,3],(11,1));p[:5,0]=2;p[5:10,0]=3;p[10,0]=1
  r=ensemble(p,np.arange(11)/10);self.assertEqual(r['role'][0],2);self.assertEqual(r['selected_mu'],0)
  self.assertEqual(choose_margin([.9,.1],[2.,2.]),.1)
  np.testing.assert_allclose(distance_margin(np.array([[0.],[3.]]),np.array([1,2]),np.array([[1.]]),np.array([1])),[1.])
 def test_metric_classes(self):
  r=evaluate_predictions(np.array([1,1]),np.array([1,2]),(1,2,3));self.assertAlmostEqual(r['target_present_macro_f1'],2/3);self.assertAlmostEqual(r['fixed_class_macro_f1'],2/9)
 def test_finite_and_dimension_guards(self):
  x=np.ones((4,3));y=np.array([1,2,1,2])
  for model,a in [(BDA(dim=4,iterations=1),x),(BDA(dim=2,iterations=1),x*np.nan)]:
   with self.assertRaises(ValueError):model.fit_predict(a,y,x)
 def test_independent_gate_before_truth(self):
  from unittest.mock import patch
  from release_pipeline import phase_b
  with tempfile.TemporaryDirectory() as d:
   # Even a truth parser that explodes must never be invoked before freeze validation.
   with patch('bda_data.encode_external',side_effect=AssertionError('truth touched')) as parser:
    with self.assertRaises(FileNotFoundError):phase_b('external',Path(d)/'poison',d)
    parser.assert_not_called()
 def test_complete_mock_phase_a_poison_and_tamper_gate(self):
  from unittest.mock import patch
  import release_pipeline as rp
  import bda_paper_core
  class SyntheticBDA:
   def __init__(self,**kwargs):pass
   def fit_predict(self,xs,ys,xt):
    self.source_embedding_=xs;self.target_embedding_=xt;self.initial_diagnostics_={};self.diagnostics_=[]
    return fit_predict(xs,ys,xt)
  with tempfile.TemporaryDirectory() as d:
   data=Path(d)/'data';data.mkdir();out=Path(d)/'out'
   (data/'batch1.dat').write_text('1 1:1 2:2\n2 1:3 2:4\n')
   for i in range(2,11):(data/f'batch{i}.dat').write_text('POISON 1:2.5 2:3.5\n')
   with patch.object(bda_paper_core,'BDA',SyntheticBDA):rp.phase_a('uci',data,out)
   freeze,index,choices=rp.verify_freeze('uci',data,out)
   self.assertEqual(freeze['candidate_count'],99);self.assertEqual(len(choices),45)
   (out/'selections.csv').write_text('tampered')
   with self.assertRaises(RuntimeError):rp.phase_b('uci',data,out)
 def test_external_poison_target_and_mapping(self):
  import pandas as pd
  from bda_data import load_external_features,encode_external
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);names=[f'f{i}' for i in range(248)]
   pd.DataFrame({'feature':names}).to_csv(d/'schema.csv',index=False)
   domains=['Source']*90+sum(([f'T{i}']*n for i,n in enumerate([90,89,90,90,90,90,71],1)),[])
   pd.DataFrame({'domain':domains}).to_csv(d/'domains.csv',index=False)
   frame=pd.DataFrame(np.ones((700,248)),columns=names)
   frame['Class']=['EtOH','Diacetyl 0.1ppm','Phenylethanol 200ppm']*30+['POISON']*610
   frame.to_csv(d/'features.csv',sep=' ',index=False)
   xs,ys,targets,assignments=load_external_features(d/'features.csv',d/'schema.csv',d/'domains.csv')
   self.assertEqual(xs.shape,(90,248));self.assertEqual(len(targets),7);self.assertEqual(set(ys),{1,2,3})
   np.testing.assert_array_equal(encode_external(['EtOH','Diacetyl 1ppm','Phenylethanol 1000ppm']),[1,2,3])
 def test_grid(self):
  cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/canonical_bda.yaml').read_text());self.assertEqual(cfg['mu_grid'],[i/10 for i in range(11)])
if __name__=='__main__':unittest.main()
