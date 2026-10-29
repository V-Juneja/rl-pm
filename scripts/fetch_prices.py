"""Exploratory Yahoo daily prices. Missing/delisted coverage is NOT guaranteed.
Usage: python3 scripts/fetch_prices.py SPY AAPL MSFT
Never use a successful download subset as a survivorship-free stock universe.
"""
import csv
import datetime as dt
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

root = Path('data/raw')
root.mkdir(parents=True, exist_ok=True)
manifest = []
for symbol in sys.argv[1:] or ['SPY']:
    url = ('https://query1.finance.yahoo.com/v8/finance/chart/'
           + urllib.parse.quote(symbol, safe='')
           + f'?period1=0&period2={int(time.time())}&interval=1d')
    record = {'symbol': symbol, 'source': url, 'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'status': 'failed', 'coverage': 'EXPLORATORY_UNVERIFIED'}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)['chart']['result'][0]
        quote = result['indicators']['quote'][0]
        adjusted = result['indicators'].get('adjclose', [{}])[0].get('adjclose', [])
        filename = urllib.parse.quote(symbol, safe='') + '_yahoo_daily.csv'
        with (root / filename).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close'])
            for i, timestamp in enumerate(result['timestamp']):
                writer.writerow([dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).date(),
                                 *[quote[k][i] for k in ('open', 'high', 'low', 'close', 'volume')],
                                 adjusted[i] if adjusted else None])
        record.update(status='downloaded', rows=len(result['timestamp']), file=filename)
    except Exception as error:
        record['error'] = str(error)
    manifest.append(record)
    print(symbol, record['status'], record.get('rows', record.get('error')))
    time.sleep(.3)
(root / 'prices-manifest.json').write_text(json.dumps(manifest, indent=2))
