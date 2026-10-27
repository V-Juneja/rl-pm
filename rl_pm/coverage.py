import argparse
from bisect import bisect_right
import csv
from datetime import date
import json
from pathlib import Path
from urllib.parse import quote

from .prices import valid_dates


def audit(root, start, end):
    with (root / 'membership.csv').open() as stream:
        snapshots = [(date.fromisoformat(r['date']), set(r['tickers'].split(',')))
                     for r in csv.DictReader(stream)]
    boundaries = [d for d, _ in snapshots]
    if boundaries != sorted(set(boundaries)):
        raise ValueError('Membership dates must be unique and sorted')
    if start < boundaries[0] or end < start:
        raise ValueError('Invalid date window')
    benchmark = next((p for p in (root/'SPY_yahoo_daily.csv', root/'spy_yahoo_daily.csv') if p.exists()), None)
    if benchmark is None:
        raise ValueError('Download SPY first to supply a provisional session calendar')
    calendar, _ = valid_dates(benchmark)
    sessions = sorted(d for d in calendar if start <= d <= end)
    if not sessions:
        raise ValueError('No benchmark sessions in requested window')
    required = {}
    for day in sessions:
        for symbol in snapshots[bisect_right(boundaries, day) - 1][1]:
            required.setdefault(symbol, set()).add(day)
    details = []
    for symbol, expected in sorted(required.items()):
        path = root / (quote(symbol, safe='') + '_yahoo_daily.csv')
        observed, invalid = valid_dates(path) if path.exists() else (set(), 0)
        missing = sorted(expected - observed)
        details.append(dict(symbol=symbol, file_exists=path.exists(),
                            required_sessions=len(expected), missing_sessions=len(missing),
                            first_missing=str(missing[0]) if missing else None,
                            invalid_rows=invalid))
    missing = sum(r['missing_sessions'] for r in details)
    return dict(identity_status='unchecked', settlement_status='unchecked',
                calendar_source='SPY', status='incomplete' if missing else 'coverage_complete', start=str(sessions[0]),
                end=str(sessions[-1]), sessions=len(sessions), securities=len(details),
                absent_files=sum(not r['file_exists'] for r in details),
                required_asset_sessions=sum(len(v) for v in required.values()),
                missing_asset_sessions=missing,
                membership_last_snapshot=str(boundaries[-1]), details=details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check historical price coverage')
    parser.add_argument('--start', type=date.fromisoformat, default=date(1998, 1, 1))
    parser.add_argument('--end', type=date.fromisoformat, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    report = audit(args.root, args.start, args.end)
    destination = Path('data/coverage-report.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}, indent=2))
    print('Full report:', destination)
