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
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()
    dataset = args.root / 'final'
    metadata = dataset / 'manifest.json'
    if metadata.exists() and not args.legacy:
        report = json.loads(metadata.read_text())
        counts = report['availability_counts']
        totals = {
            'Historical securities': report['historical_securities'],
            'Securities without any history': report['securities_without_any_price_history'],
            'Daily price observations': report['daily_price_observations'],
            'Membership-day observations': report['membership_observations'],
            'Source-reported no quote': counts.get('source_reported_no_quote', 0),
            'After last available price': counts.get('after_last_available_price', 0),
            'Carried prices included': report['carried_prices_included'],
        }
        print('rl-pm data\n')
        width = max(map(len, totals))
        for label, value in totals.items():
            print(f'{label:<{width}}  {value:>10,}')
        print(f"\nPrice availability: {report['price_availability_pct']:.6f}%")
        print(f"Dates: {report['membership_start']} to {report['membership_end']}")
        print(f"Training ready: {'yes' if report['training_ready'] else 'no'}")
        print(f'Dataset: {dataset.resolve()}')
        return
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
