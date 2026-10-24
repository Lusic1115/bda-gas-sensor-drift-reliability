"""Verify the complete prediction/decision freeze, then load retrospective truth."""
import argparse
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['uci','external'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Existing completed Phase A directory')
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from release_pipeline import phase_b
    phase_b(args.dataset,args.data,args.output)
if __name__=='__main__':main()
