"""Fetch exploratory membership snapshots; not a complete trading dataset."""
from pathlib import Path
from urllib.request import urlopen

URL = 'https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv'
path = Path('data/raw/membership.csv')
path.parent.mkdir(parents=True, exist_ok=True)
with urlopen(URL, timeout=60) as response:
    payload = response.read()
if not payload.startswith(b'date,tickers'):
    raise ValueError('Unexpected source schema')
path.write_bytes(payload)
print(f'Downloaded {len(payload):,} bytes to {path}; run python3 -m sidequest.audit')
