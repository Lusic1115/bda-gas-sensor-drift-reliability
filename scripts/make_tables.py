"""Copy five complete frozen tables and render their literal cells as Markdown."""
import argparse
import csv
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for number in range(1, 6):
        source = ROOT / f'results/paper_tables/Table{number}.csv'
        shutil.copyfile(source, output/source.name)
        with source.open(newline='') as stream:
            rows = list(csv.reader(stream))
        def line(row):
            return '| ' + ' | '.join(cell.replace('|', '\\|').replace('\n', '<br>') for cell in row) + ' |'
        markdown = [line(rows[0]), line(['---']*len(rows[0]))] + [line(row) for row in rows[1:]]
        (output/f'Table{number}.md').write_text('\n'.join(markdown)+'\n')
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reproduced_tables'))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be a new directory')
    generate(args.output)
if __name__ == '__main__':
    main()
