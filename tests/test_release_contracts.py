"""Portable packaging contracts; fixtures only, never research datasets."""
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class ReleaseContracts(unittest.TestCase):
    def test_exact_external_vocabulary_rejects_unfrozen_conditions(self):
        from bda_data import encode_external
        for value in ['Diacetyl 999ppm', 'Phenylethanol unknown']:
            with self.assertRaises(ValueError):
                encode_external([value])

    def test_adaptation_signature_has_no_truth_argument(self):
        from bda_paper_core import BDA
        self.assertEqual(list(inspect.signature(BDA.fit_predict).parameters),
                         ['self', 'Xs', 'ys', 'Xt'])

    def test_freeze_requires_decision_and_index_hashes(self):
        import release_pipeline as rp
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            domains = [f'T{i}' for i in range(1, 8)]
            methods = [f'mu{m:.1f}' for m in rp.MUS] + [
                'source_only_1nn', 'source_only_svm', 'multi_mu_vote']
            records = [dict(domain=d, method=m, path=f'{d}_{m}.csv', sample_count=1)
                       for d in domains for m in methods]
            choices = [dict(domain=d, selector=m, selected_mu=0.,
                            prediction_path=f'{d}_mu0.0.csv') for d in domains
                       for m in ['fixed_mu_0.0','fixed_mu_0.5','margin_mean',
                                 'margin_median','EnsV-onehot']]
            pd.DataFrame(records).to_csv(out/'prediction_index.csv', index=False)
            pd.DataFrame(choices).to_csv(out/'selections.csv', index=False)
            freeze = dict(dataset='external', target_label_parses=0, mu_grid=rp.MUS,
                          domains=domains, candidate_count=77, inputs={},
                          source_hashes={}, config_hashes={},
                          hashes={x['path']: '0'*64 for x in records})
            (out/'prediction_freeze.json').write_text(json.dumps(freeze))
            with patch.object(rp, 'sha', return_value='0'*64), \
                 patch.object(rp, 'data_paths', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'metadata|decision|Incomplete'):
                    rp.verify_freeze('external', 'unused', out)

    def test_frozen_protocol_rejects_edited_domains(self):
        import release_pipeline as rp
        self.assertTrue(callable(getattr(rp, 'validate_protocol', None)))
        cfg = rp.config('uci_protocol.yaml')
        cfg['targets'] = [2]
        with patch.object(rp, 'config', return_value=cfg):
            with self.assertRaises(ValueError):
                rp.validate_protocol('uci')
