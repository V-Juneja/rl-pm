import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote


def download_summary(root):
    with (root / 'membership.csv').open() as stream:
        symbols = {
            symbol.strip()
            for row in csv.DictReader(stream)
            for symbol in row['tickers'].split(',')
            if symbol.strip()
        }
    paths = list(root.glob('*_yahoo_daily.csv'))
    available = {
        symbol for symbol in symbols
        if (root / (quote(symbol, safe='') + '_yahoo_daily.csv')).exists()
    }
    manifest_path = root / 'prices-manifest.json'
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    return {
        'Historical tickers': len(symbols),
        'Price CSV files': len(paths),
        'Historical ticker files present': len(available),
        'Historical ticker files absent': len(symbols - available),
        'Last saved batch attempted': len(records),
        'Last saved batch downloaded': sum(r.get('status') == 'downloaded' for r in records),
        'Last saved batch failed': sum(r.get('status') == 'failed' for r in records),
    }


def main():
    parser = argparse.ArgumentParser(description='Download totals')
    parser.add_argument('--root', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    try:
        totals = download_summary(args.root)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Could not read download files: {error}\n')
    print('rl-pm downloads\n')
    width = max(map(len, totals))
    for label, value in totals.items():
        print(f'{label:<{width}}  {value:>7,}')
    print('\nFile counts measure presence, not price coverage.')
    print('Batch totals come from the last saved manifest.')


if __name__ == '__main__':
    main()
