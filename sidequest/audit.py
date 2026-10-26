"""Audit a candidate membership file without certifying its completeness."""
import csv
import hashlib
import json
from pathlib import Path


def audit(path):
    path = Path(path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    dates = [r['date'] for r in rows]
    counts = [len(set(r['tickers'].split(','))) for r in rows]
    if dates != sorted(set(dates)):
        raise ValueError('Dates must be unique and increasing')
    return dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                snapshots=len(rows), first_date=dates[0], last_date=dates[-1],
                unique_tickers=len({s for r in rows for s in r['tickers'].split(',')}),
                min_members=min(counts), max_members=max(counts),
                status='UNVERIFIED_MEMBERSHIP_ONLY',
                missing=['stable security identifiers', 'delisted total returns',
                         'independent membership verification', 'availability timestamps'])


if __name__ == '__main__':
    print(json.dumps(audit('data/raw/membership.csv'), indent=2))
