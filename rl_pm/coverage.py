"""Audit observed membership/price coverage; does not certify identifiers or payouts."""
import argparse
from bisect import bisect_right
import csv
from datetime import date
import json
import math
from pathlib import Path
from urllib.parse import quote


def valid_dates(path):
    """Accept only complete, positive adjusted-close observations."""
    dates, invalid = set(), 0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                day = date.fromisoformat(row['date'])
                value = float(row['adjusted_close'])
                if not math.isfinite(value) or value <= 0 or day in dates:
                    raise ValueError('Invalid or duplicate observation')
                dates.add(day)
            except (ValueError, KeyError, TypeError):
                invalid += 1
    return dates, invalid


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
    return dict(status='BLOCKED_PENDING_DATA_VALIDATION', start=str(sessions[0]),
                end=str(sessions[-1]), sessions=len(sessions), securities=len(details),
                absent_files=sum(not r['file_exists'] for r in details),
                required_asset_sessions=sum(len(v) for v in required.values()),
                missing_asset_sessions=missing,
                warnings=['Ticker identity, terminal payouts and corporate actions remain unverified.',
                          'SPY dates are a provisional calendar; source gaps are not exchange holidays.',
                          'Membership snapshots are carried forward; last snapshot may be stale.',
                          'A missing observation may represent a halt, identity mismatch or download gap.'],
                membership_last_snapshot=str(boundaries[-1]), details=details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date.fromisoformat, default=date(1998, 1, 1))
    parser.add_argument('--end', type=date.fromisoformat, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    report = audit(args.root, args.start, args.end)
    destination = Path('data/coverage-report.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}, indent=2))
    print('Full report:', destination)
