"""Portable driver for a user-supplied, hash-pinned official notebook.

No upstream function implementations or observations are distributed here.
"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import re
import warnings

import numpy as np
import pandas as pd

NOTEBOOK_SHA256 = '69f8f8e12abc6e3d4e1533c9ec40b8fd92079bbcee5b5aac6a058f60b98edb06'
CELL_SHA256 = (
    '1a5febfabc696bdcfbabb91274af0cb9c6e9e0accb745288f7ad4886a56fd70c',
    'a3ed7be56cf808abfeeb446c681b35b303502acb517ed5d13704dc47bc16b099',
    '84046887b24cc5fd5c71628a9f7317b1e24b8dca382a78d217717e2ad0cc11af',
)
FUNCTION_SOURCE_SHA256 = '001b0b30978ea9f6ee0f4aa52fdea46c90c82414dd1febbad4089d81fc68a2e9'
FROZEN_OUTPUT_SHA256 = '1eb02f065a25b01a54170707ece85f2faae6c6fa335ee144ea55cda20636e209'
FAMILIES = ['mean', 'rel_diff', 'sample_startslope', 'recovery_level']
ANALYTES = {'EtOH': 'Ethanol', 'Diacetyl_0.1ppm': 'Diacetyl',
            'Diacetyl_1ppm': 'Diacetyl', 'Phenylethanol_200ppm': '2-Phenylethanol',
            'Phenylethanol_1000ppm': '2-Phenylethanol'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def output_columns():
    columns = []
    for family in FAMILIES:
        prefix = {'recovery_level': 'rec_lvl'}.get(family, family)
        columns.extend(f'{prefix}_R{i}' + ('[Ohm]' if family == 'mean' else '')
                       for i in range(1, 63))
        if family in ('mean', 'rel_diff'):
            columns.extend([f'{prefix}_T01[degC]', f'{prefix}_H01[%rh]'])
    return columns + ['Class', 'Date[yyyy-mm-dd]']


def filename_metadata(filename):
    """Separate retrospective metadata mapper; never used to select raw files."""
    try:
        key, replicate = Path(filename).stem.rsplit('_', 1)
        return {'official_class': key.replace('_', ' '), 'analyte': ANALYTES[key],
                'replicate': int(replicate)}
    except (KeyError, ValueError) as exc:
        raise ValueError('Unknown external filename metadata') from exc


def _load_notebook(notebook, *, notebook_hash, cell_hashes, source_hash):
    blob = Path(notebook).read_bytes()
    if digest(blob) != notebook_hash:
        raise ValueError('Official notebook hash mismatch; no code executed')
    document = json.loads(blob)
    cells = document['cells'][:3]
    sources = [''.join(cell['source']) for cell in cells]
    if len(sources) != 3 or tuple(map(lambda s: digest(s.encode()), sources)) != tuple(cell_hashes):
        raise ValueError('Official cell hashes mismatch; no code executed')
    if digest('\n\n'.join(sources).encode()) != source_hash:
        raise ValueError('Official function source hash mismatch; no code executed')
    namespace = {}
    for index, source in enumerate(sources):
        exec(compile(source, f'official_notebook_cell_{index}', 'exec'), namespace)
    return namespace


def _reconstruct(raw_dir, notebook, out_dir, *, expected_count, notebook_hash,
                 cell_hashes, source_hash, columns):
    """Internal fixture seam; production CLI supplies only frozen constants."""
    raw_dir, notebook, out_dir = map(Path, (raw_dir, notebook, out_dir))
    raw_root, output_root = raw_dir.resolve(), out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError('Output directory already exists; refusing overwrite')
    if output_root == raw_root or raw_root in output_root.parents or output_root in raw_root.parents:
        raise ValueError('Output directory must be separate from raw dataset')
    if output_root == notebook.resolve() or output_root in notebook.resolve().parents:
        raise ValueError('Output directory overlaps the notebook')
    paths = sorted(raw_dir.glob('Day */*.csv'),
                   key=lambda p: (int(p.parent.name.split()[-1]), p.name))
    if len(paths) != expected_count or len({p.resolve() for p in paths}) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} distinct raw recordings')
    if any(p.is_symlink() or p.parent.is_symlink() for p in paths):
        raise ValueError('Symlinked raw recordings are not supported')
    before = {p: digest(p.read_bytes()) for p in paths}
    namespace = _load_notebook(notebook, notebook_hash=notebook_hash,
                               cell_hashes=cell_hashes, source_hash=source_hash)
    extractor = namespace['FeatureExtractor'](FAMILIES)
    rows, identities = [], []
    for path in paths:
        original_walk = namespace['os'].walk
        def single_walk(_):
            yield str(path.parent), [], [path.name]
        try:
            namespace['os'].walk = single_walk
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                frames = namespace['load_data_from_main_folder'](str(path.parent))
                if len(frames) != 1:
                    raise ValueError('Official loader did not return exactly one frame')
                row = extractor.extract_features(frames[0])
        finally:
            namespace['os'].walk = original_walk
        if row.shape != (1, len(columns)) or list(row.columns) != list(columns):
            raise ValueError('Extracted output schema mismatch')
        if not np.isfinite(row.iloc[:, :-2].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite extracted feature')
        rows.append(row)
        identities.append({'row': len(rows)-1, 'relative_path': path.relative_to(raw_dir).as_posix(),
                           'raw_sha256': before[path]})
    if any(digest(path.read_bytes()) != value for path, value in before.items()):
        raise ValueError('Raw recording changed during extraction')
    table = pd.concat(rows, ignore_index=True)
    payload = table.to_csv(index=False, sep=' ', lineterminator='\n').encode('utf-8')
    output_hash = digest(payload)
    manifest = {'rows': len(table), 'columns': list(table.columns),
                'output_sha256': output_hash, 'frozen_output_sha256': FROZEN_OUTPUT_SHA256,
                'matches_frozen_bytes': output_hash == FROZEN_OUTPUT_SHA256,
                'notebook_sha256': notebook_hash, 'function_source_sha256': source_hash,
                'cell_sha256': list(cell_hashes), 'raw_unchanged': True,
                'label_use': 'Literal Class metadata only; no numerical label input or label-based inclusion',
                'observations': identities}
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'reconstructed_features_700.csv').write_bytes(payload)
    (out_dir / 'reconstruction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def reconstruct(raw_dir, notebook, out_dir):
    return _reconstruct(raw_dir, notebook, out_dir, expected_count=700,
                        notebook_hash=NOTEBOOK_SHA256, cell_hashes=CELL_SHA256,
                        source_hash=FUNCTION_SOURCE_SHA256, columns=output_columns())
