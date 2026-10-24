"""Explicit opt-in reproduction; default stops after label-free prediction freeze."""
import argparse
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['uci','external'],required=True)
    parser.add_argument('--data',type=Path,required=True,help='UCI batch directory or reconstructed external feature CSV')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; must not exist')
    parser.add_argument('--phase',choices=['a','all'],default='a',help='a freezes predictions; all explicitly opts into retrospective evaluation')
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from release_pipeline import phase_a,phase_b
    phase_a(args.dataset,args.data,args.output)
    if args.phase=='all':phase_b(args.dataset,args.data,args.output)
if __name__=='__main__':main()
