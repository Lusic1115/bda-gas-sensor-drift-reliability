"""Synthetic driver tests; never reads or extracts the research dataset."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from src.external_reconstruction import (
    _load_notebook, _reconstruct, digest, filename_metadata, output_columns, reconstruct,
)

# Original synthetic implementation, deliberately not copied from the upstream code.
SOURCES = [
    'import os\nimport pandas as pd\n',
    'def load_data_from_main_folder(root):\n    return [pd.read_csv(os.path.join(base, name)) for base, _, names in os.walk(root) for name in names]\n',
    'class FeatureExtractor:\n    def __init__(self, families):\n        pass\n    def extract_features(self, frame):\n        return frame[["feature", "Class", "Date[yyyy-mm-dd]"]]\n',
]


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'Dataset'
        for day, name, value in [(10, 'EtOH_1.csv', 10), (2, 'Unknown_2.csv', 2)]:
            folder = self.raw / f'Day {day}'
            folder.mkdir(parents=True)
            pd.DataFrame({'feature': [value], 'Class': [name[:-4]],
                          'Date[yyyy-mm-dd]': ['25-01-01']}).to_csv(folder / name, index=False)
        self.notebook = self.root / 'fixture.ipynb'
        self.notebook.write_text(json.dumps({'cells': [{'source': s} for s in SOURCES] +
                                                    [{'source': 'raise RuntimeError("never run")'}]}))
        self.settings = dict(expected_count=2, notebook_hash=digest(self.notebook.read_bytes()),
                             cell_hashes=[digest(s.encode()) for s in SOURCES],
                             source_hash=digest('\n\n'.join(SOURCES).encode()),
                             columns=['feature', 'Class', 'Date[yyyy-mm-dd]'])

    def run_fixture(self, name):
        return _reconstruct(self.raw, self.notebook, self.root / name, **self.settings)

    def test_fixture_order_determinism_metadata_and_raw_integrity(self):
        before = {p: p.read_bytes() for p in self.raw.glob('Day */*.csv')}
        first, second = self.run_fixture('a'), self.run_fixture('b')
        self.assertEqual(first, second)
        self.assertEqual(first['rows'], 2)
        self.assertFalse(first['matches_frozen_bytes'])
        frame = pd.read_csv(self.root / 'a/reconstructed_features_700.csv', sep=' ')
        self.assertEqual(frame.feature.tolist(), [2, 10])
        self.assertEqual(frame.Class.tolist(), ['Unknown_2', 'EtOH_1'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(first['observations'][0]['relative_path'], 'Day 2/Unknown_2.csv')

    def test_existing_output_and_raw_output_rejected(self):
        self.run_fixture('a')
        with self.assertRaises(FileExistsError):
            self.run_fixture('a')
        with self.assertRaises(ValueError):
            _reconstruct(self.raw, self.notebook, self.raw / 'new', **self.settings)

    def test_production_count_and_hash_fail_closed(self):
        with self.assertRaisesRegex(ValueError, '700'):
            reconstruct(self.raw, self.notebook, self.root / 'production')
        bad = dict(self.settings, notebook_hash='0' * 64)
        with self.assertRaisesRegex(ValueError, 'no code executed'):
            _reconstruct(self.raw, self.notebook, self.root / 'bad', **bad)
        self.assertFalse((self.root / 'bad').exists())

    def test_cell_and_function_hashes_fail_closed(self):
        for key, value in [('cell_hashes', ['bad'] * 3), ('source_hash', 'bad')]:
            settings = dict(self.settings, **{key: value})
            with self.assertRaisesRegex(ValueError, 'no code executed'):
                _load_notebook(self.notebook, **{k: settings[k] for k in
                              ('notebook_hash', 'cell_hashes', 'source_hash')})

    def test_schema_order_required(self):
        settings = dict(self.settings, columns=['Class', 'feature', 'Date[yyyy-mm-dd]'])
        with self.assertRaisesRegex(ValueError, 'schema'):
            _reconstruct(self.raw, self.notebook, self.root / 'schema', **settings)
        self.assertFalse((self.root / 'schema').exists())

    def test_frozen_schema_and_literal_mapper(self):
        schema = output_columns()
        self.assertEqual(len(schema), 254)
        self.assertEqual(schema[:2], ['mean_R1[Ohm]', 'mean_R2[Ohm]'])
        self.assertEqual(schema[-3:], ['rec_lvl_R62', 'Class', 'Date[yyyy-mm-dd]'])
        self.assertEqual(filename_metadata('Phenylethanol_200ppm_3.csv'),
                         {'official_class': 'Phenylethanol 200ppm',
                          'analyte': '2-Phenylethanol', 'replicate': 3})
        self.assertEqual(filename_metadata('EtOH_1.csv')['analyte'], 'Ethanol')
        self.assertEqual(filename_metadata('Diacetyl_0.1ppm_2.csv')['analyte'], 'Diacetyl')
        with self.assertRaises(ValueError):
            filename_metadata('Unknown_1.csv')


if __name__ == '__main__':
    unittest.main()
