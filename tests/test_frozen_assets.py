"""Level 1 uses frozen summaries and never invokes adaptation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class FrozenAssetsTests(unittest.TestCase):
    def test_tables_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            load('make_tables').generate(Path(directory))
            for n in range(1, 6):
                self.assertEqual((ROOT / f'results/paper_tables/Table{n}.csv').read_bytes(), (Path(directory) / f'Table{n}.csv').read_bytes())

    def test_figure_underlying_records_identical(self):
        module = load('make_figures')
        expected = json.loads((ROOT / 'figures/figure_values.json').read_text())
        actual = module.load_values()
        self.assertEqual(actual['figures'], expected['figures'])
        self.assertEqual(len(module.read_rows('results/canonical/fixed_mu_metrics.csv')), 99)
        self.assertEqual(len(module.read_rows('results/external/fixed_mu_metrics.csv')), 77)

    def test_default_summary_does_not_call_core(self):
        module = load('reproduce_paper')
        with tempfile.TemporaryDirectory() as directory, patch.object(module.subprocess, 'run') as run, patch.object(module, 'render_figures'), patch.object(module, 'generate_tables'):
            module.main(['--output', str(Path(directory) / 'new')])
            run.assert_not_called()
