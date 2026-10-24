"""Level 1 validates and renders frozen summaries; full experiments require explicit opt-in."""
import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/(name+'.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded
def generate_tables(output):
    module('make_tables').generate(output)
def render_figures(output):
    module('make_figures').render(output)
def validate():
    with (ROOT/'manifests/frozen_result_map.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            path = ROOT/row['public_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Frozen artifact hash mismatch: '+row['public_path'])
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', choices=['summaries','full'], default='summaries')
    parser.add_argument('--output', type=Path, default=Path('reproduced_paper'))
    parser.add_argument('--dataset', choices=['uci','external'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--phase', choices=['a','all'], default='a')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be a new directory')
    if args.level == 'full':
        if args.dataset is None or args.data is None:
            parser.error('--level full requires --dataset and --data')
        subprocess.run([sys.executable, str(ROOT/'scripts/run_canonical.py'), '--dataset', args.dataset, '--data', str(args.data), '--output', str(args.output), '--phase', args.phase], check=True)
        return
    validate()
    args.output.mkdir(parents=True)
    generate_tables(args.output/'tables')
    render_figures(args.output/'figures')
    print('Frozen integrity validated; five tables and six figures rendered. No adaptation run.')
if __name__ == '__main__':
    main()
