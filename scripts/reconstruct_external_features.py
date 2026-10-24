#!/usr/bin/env python3
"""Reconstruct all 700 recordings with hash-pinned author definitions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external_reconstruction import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True, help='User-extracted Dataset directory')
    parser.add_argument('--notebook', type=Path, required=True, help='Official feature_extraction.ipynb')
    parser.add_argument('--out-dir', type=Path, required=True, help='New directory outside the dataset')
    args = parser.parse_args()
    result = reconstruct(args.raw_dir, args.notebook, args.out_dir)
    print(json.dumps({key: result[key] for key in ('rows', 'output_sha256', 'matches_frozen_bytes')}, indent=2))


if __name__ == '__main__':
    main()
